// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, ops::Range};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{ProveSingleOutput, bivariate_product_prover, prove_single},
};
use binius_math::{BinarySubspace, FieldBuffer, FieldVec};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};
use binius_verifier::protocols::shift::LOG_SHIFT_VARIANT_COUNT;
use bytemuck::zeroed_vec;
use itertools::izip;
use tracing::instrument;

use super::{
	SegmentWords,
	claims::PreparedOperatorClaims,
	key_collection::{KeyCollection, KeySegment, ShiftSlot, SlotRows},
	monster::{SlotWeights, build_h, build_inner_h},
};

/// The number of variables in the g (and h) multilinear of phase 1.
///
/// The axes run, from the low index positions up: the bit position within a word, the shift
/// amount, and the shift variant.
pub const SLOT_PHASE_LOG_LEN: usize = Word::LOG_BITS + Word::LOG_BITS + LOG_SHIFT_VARIANT_COUNT;

/// What one slot-binding phase produces.
///
/// A phase binds three axes: the bit position, and its slot's amount and variant.
/// It hands the next phase the point it reached, with both multilinears' evaluations there.
///
/// The factors are kept apart rather than only their product:
///
/// - the `h` evaluation becomes a weight on the next phase's keys;
/// - the product is the claim that phase proves.
pub struct SlotPhaseOutput<F> {
	/// The bit point the phase bound: `r_j` for the inner slot, `r_k` for the outer one.
	pub r_bit: Vec<F>,
	/// The amount point of the slot this phase bound.
	pub r_s: Vec<F>,
	/// The variant point of the slot this phase bound.
	pub r_v: Vec<F>,
	/// The `g` (or `G`) multilinear's evaluation at that point.
	pub g_eval: F,
	/// The `h` (or `H`) multilinear's evaluation at that point.
	pub h_eval: F,
}

impl<F: Field> SlotPhaseOutput<F> {
	/// The claim the next phase proves: the product of the two evaluations.
	pub fn claim(&self) -> F {
		self.g_eval * self.h_eval
	}
}

/// Proves the outer phase of the three-phase shift reduction.
///
/// This binds the sequence's *outer* slot, against the intermediate words.
/// Its multilinear scatters `op1(w[y], s_1)` rather than `w[y]`, so the sum over the source bit
/// index is already done and only the outer indicator remains.
/// Its weight is the oblong one.
///
/// # Why this order is forced
///
/// The oblong factor `delta_D(r_ihat, ihat)` attaches to the *output* bit index.
/// That index couples to the rest of the sum only through the outer indicator.
/// Summing it out therefore yields a function of `(k, s_2)` alone — one 15-variate sumcheck.
///
/// Binding the inner shift first would leave an output-side factor still depending on `s_2`, and
/// the sumcheck would run over `(j, s_1, s_2)` instead: 18 variates.
#[instrument(skip_all, name = "prove_outer_phase")]
pub fn prove_outer_phase<F, P, Channel, A>(
	key_collection: &KeyCollection,
	words: SegmentWords<'_>,
	prepared: &PreparedOperatorClaims<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
	alloc: &A,
) -> SlotPhaseOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	let g = sum_segment_multilinears::<F, P, A>(
		alloc,
		|segment_words, segment| {
			build_outer_g::<F, P, A>(alloc, segment_words, segment, prepared, None)
		},
		words,
		key_collection,
	);

	// BitAnd, IntMul and BinMul share the same `r_zhat_prime`.
	let h = build_h(alloc, domain_subspace, prepared.bitand.r_zhat_prime);

	run_slot_sumcheck(g, h, prepared.batched_eval(), channel, alloc)
}

/// Proves the inner phase of the three-phase shift reduction.
///
/// This binds the sequence's *inner* slot, against the witness words themselves.
/// Its weight is the bare shift indicator at the intermediate point the outer phase produced.
/// There is no oblong factor, since that phase already summed the output bit index out.
///
/// The outer phase's evaluations enter as per-key weights:
///
/// ```text
/// key scale = H_eval * eq(r_s2, s_2) * eq(r_v2, v_2)
/// ```
///
/// which is what makes this phase's claim the product the outer sumcheck closed on.
#[instrument(skip_all, name = "prove_inner_phase")]
pub fn prove_inner_phase<F, P, Channel, A>(
	key_collection: &KeyCollection,
	words: SegmentWords<'_>,
	prepared: &PreparedOperatorClaims<F>,
	outer: &SlotPhaseOutput<F>,
	channel: &mut Channel,
	alloc: &A,
) -> SlotPhaseOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	// The outer slot's weight per sequence, scaled by that phase's own evaluation.
	// Together they turn this phase's sum into the product the outer sumcheck closed on.
	let outer_weights = SlotWeights::at(&outer.r_s, &outer.r_v);
	let scale_of = |segment: &KeySegment| {
		segment
			.dense_shift_enc
			.iter()
			.map(|[_, outer_shift]| outer.h_eval * outer_weights.weight(outer_shift))
			.collect::<Vec<_>>()
	};
	let public_scale = scale_of(&key_collection.public);
	let hidden_scale = scale_of(&key_collection.hidden);

	let mut g = build_g::<F, P, A>(
		alloc,
		words.public,
		&key_collection.public,
		prepared,
		Some(&public_scale),
	);
	let hidden_g = build_g::<F, P, A>(
		alloc,
		words.hidden,
		&key_collection.hidden,
		prepared,
		Some(&hidden_scale),
	);
	for (entry, add) in iter::zip(g.as_mut(), hidden_g.as_ref()) {
		*entry += *add;
	}

	let h = build_inner_h::<F, P, A>(alloc, &outer.r_bit);

	run_slot_sumcheck(g, h, outer.claim(), channel, alloc)
}

/// Proves the single slot-binding phase of the two-phase shift reduction.
///
/// Every sequence is a lone shift here, so there is no outer slot to bind.
/// The multilinear scatters the witness words, and the weight is the oblong one.
#[instrument(skip_all, name = "prove_slot_phase")]
pub fn prove_slot_phase<F, P, Channel, A>(
	key_collection: &KeyCollection,
	words: SegmentWords<'_>,
	prepared: &PreparedOperatorClaims<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
	alloc: &A,
) -> SlotPhaseOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	let g = sum_segment_multilinears::<F, P, A>(
		alloc,
		|segment_words, segment| build_g::<F, P, A>(alloc, segment_words, segment, prepared, None),
		words,
		key_collection,
	);

	// BitAnd, IntMul and BinMul share the same `r_zhat_prime`.
	let h = build_h(alloc, domain_subspace, prepared.bitand.r_zhat_prime);

	run_slot_sumcheck(g, h, prepared.batched_eval(), channel, alloc)
}

/// Builds a phase's multilinear over both value-vector segments and sums the two.
///
/// The public and hidden words participate through their own key segment, with segment-relative key
/// ranges.
/// So each is built against its matching words, and the two results are added.
fn sum_segment_multilinears<F, P: PackedField<Scalar = F>, A: Allocator>(
	_alloc: &A,
	mut build: impl FnMut(&[Word], &KeySegment) -> FieldVec<P, A>,
	words: SegmentWords<'_>,
	key_collection: &KeyCollection,
) -> FieldVec<P, A>
where
	F: BinaryField,
{
	let mut combined = build(words.public, &key_collection.public);
	let hidden = build(words.hidden, &key_collection.hidden);
	for (entry, add) in iter::zip(combined.as_mut(), hidden.as_ref()) {
		*entry += *add;
	}
	combined
}

/// Runs the phase 1 sumcheck protocol for shift constraint verification.
///
/// One bivariate-product sumcheck over `g` and `h`, both multilinears of [`SLOT_PHASE_LOG_LEN`]
/// variables. The shift variant is the high [`LOG_SHIFT_VARIANT_COUNT`] variables of each, so the
/// sumcheck folds the variant axis rather than the prover summing a separate claim per variant.
///
/// The `g` multilinear carries the witness and the batching randomness; `h` encodes what each
/// shift does at the univariate challenge point.
///
/// # Arguments
///
/// - `sum`: the claim being proved, as the operations' batched evaluations give it. This is the
///   same value the verifier feeds its own sumcheck, so the prover proves the statement it was
///   handed rather than whatever `g · h` happens to come to.
///
/// The two agree exactly when the witness satisfies the constraint system — which is what this
/// reduction exists to establish, so it is not something the prover can check on its way in. On an
/// unsatisfying witness they differ, and that difference travels through both phases to the
/// reduction's final evaluation check, which is where verification fails.
///
/// # Returns
///
/// The point the sumcheck reached, split into its three axes, and both multilinears' evaluations
/// there. See [`SlotPhaseOutput`] for why the two evaluations are kept apart.
#[instrument(skip_all, name = "run_slot_sumcheck")]
pub fn run_slot_sumcheck<
	F: Field,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
>(
	g: FieldVec<P, A>,
	h: FieldVec<P, A>,
	sum: F,
	channel: &mut Channel,
	alloc: &A,
) -> SlotPhaseOutput<F> {
	let prover = bivariate_product_prover(alloc, [g, h], sum);

	let ProveSingleOutput {
		multilinear_evals,
		mut challenges,
	} = prove_single(prover, channel);
	challenges.reverse();

	let [g_eval, h_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");

	// The axis order of both multilinears, least significant first:
	//
	//     bit position | shift amount | shift variant
	let mut r_bit = challenges;
	let r_v = r_bit.split_off(Word::LOG_BITS * 2);
	let r_s = r_bit.split_off(Word::LOG_BITS);

	SlotPhaseOutput {
		r_bit,
		r_s,
		r_v,
		g_eval,
		h_eval,
	}
}

/// Constructs the inner phase's "g" multilinear for one key segment.
///
/// This is the shared builder on the inner slot.
/// It scatters the bits of the witness word itself, into the row its inner shift names.
///
/// # Usage
///
/// Used by the inner phase to construct the constant size g multilinears that will participate in
/// its sumcheck.
#[instrument(skip_all, name = "build_g")]
pub fn build_g<F: BinaryField, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	words: &[Word],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
	scale: Option<&[F]>,
) -> FieldVec<P, A> {
	build_slot_g(alloc, words, segment, prepared, ShiftSlot::Inner, scale)
}

/// Constructs the outer phase's "G" multilinear for one key segment.
///
/// This is the shared builder on the outer slot.
/// It scatters the bits of the *intermediate* word — the witness word with its inner shift already
/// applied — into the row the outer shift names.
///
/// The whole outer phase turns on one identity.
/// Its definition contains an inner sum over the source bit index,
///
/// ```text
/// sum_j shift-ind_op1(k, j, s_1) * w[y]_j  =  op1(w[y], s_1)_k
/// ```
///
/// and the right-hand side is just bit `k` of the intermediate word.
///
/// Every shift variant has the at-most-one-source-bit property: an output position reads exactly
/// one source position, with arithmetic right shift merely having many positions read the sign bit.
/// So the sum has at most one nonzero term, and the indicator is a permutation with holes.
/// The intermediate word therefore costs one machine shift, not 64 field products:
///
/// ```text
/// G[k][s_2] = sum_y sum_{op1, s_1} Z~_{op1,op2}(r'_x, y, s_1, s_2) * op1(w[y], s_1)_k
/// ```
///
/// # Usage
///
/// Used by the outer phase.
/// Its sumcheck has the same bivariate-product shape as the inner phase's, so one runner serves
/// both.
#[instrument(skip_all, name = "build_outer_g")]
pub fn build_outer_g<F: BinaryField, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	words: &[Word],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
	scale: Option<&[F]>,
) -> FieldVec<P, A> {
	build_slot_g(alloc, words, segment, prepared, ShiftSlot::Outer, scale)
}

/// Constructs one phase's shift multilinear for one key segment.
///
/// This builds the multilinear a phase's sumcheck runs against, over the words of a single
/// value-vector segment (public or hidden).
/// It constructs one multilinear polynomial per shift variant.
///
/// The public and hidden words participate through their own key segment, so a caller builds each
/// with the matching words and sums the two results.
///
/// The two phases differ in exactly two places, both fixed by the bound slot:
///
/// ```text
/// slot     row named by            bits scattered
/// Inner    the inner shift         w[y]
/// Outer    the outer shift         op1(w[y], s_1), the intermediate word
/// ```
///
/// Everything else is shared: the per-key accumulation, the packed-mask bit scatter, the rayon fold
/// and reduce, and the final scatter over the shift axes.
///
/// # Construction Process
///
/// 1. **Parallel Processing**: Words are processed in parallel chunks for efficiency
/// 2. **Key Processing**: For each word, iterate through its associated keys in the segment
/// 3. **Accumulation**: For each key, accumulate its contribution weighted by the r_x' tensor
/// 4. **Word Expansion**: Expand the (possibly shifted) word bitwise to populate the multilinears
/// 5. **Lambda Weighting**: Apply lambda powers to weight different operand positions
///
/// The accumulator holds one row of `Word::BITS` scalars per shift *the chosen slot takes*, not per
/// sequence.
/// Several sequences can agree on that slot, and their contributions belong in one row.
/// A scatter then spreads those rows over the full `(variant, amount)` space of the returned parts.
///
/// # Returns
///
/// One multilinear of [`SLOT_PHASE_LOG_LEN`] variables holding every shift variant's part: the
/// variant indexes the high [`LOG_SHIFT_VARIANT_COUNT`] variables, and within a variant the rows
/// of `Word::BITS` scalars run in order of shift amount.
#[instrument(skip_all, name = "build_slot_g")]
pub fn build_slot_g<F: BinaryField, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	words: &[Word],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
	slot: ShiftSlot,
	scale: Option<&[F]>,
) -> FieldVec<P, A> {
	// One row of `Word::BITS` scalars per shift the chosen slot takes.
	let slot_rows = segment.dense_shift_enc.slot_rows(slot);
	// Invariant: a scale carries one factor per shift sequence, indexed as the keys are.
	debug_assert!(
		scale.is_none_or(|scale| scale.len() == segment.dense_shift_enc.len()),
		"a per-sequence scale covers the segment's dense shift encoding"
	);
	let row_len = Word::BITS >> P::LOG_WIDTH;
	let acc_size = slot_rows.len() * row_len;

	const {
		assert!(
			P::WIDTH <= 8,
			"the optimizations below work only when the width of `P` is less than 8 (which is true for all packed 128b fields we use for now)"
		);
	}

	// Map from a u8 with `P::WIDTH` meaningful bits to the lane mask selecting exactly those lanes,
	// precomputed once and reused across every accumulator below.
	let packed_masks_map = (0..1 << P::WIDTH)
		.map(|i| P::make_mask((0..P::WIDTH).map(|bit_index| (i >> bit_index) & 1 == 1)))
		.collect::<Vec<_>>();
	// A mask for low `P::WIDTH` bits.
	let low_bits_mask = (1u8 << P::WIDTH) - 1;

	// Each word carries the keys named by the segment-relative range at its position.
	let multilinears = words
		.par_iter()
		.zip(segment.key_ranges.par_iter())
		.with_min_task(WorkPerItem::FieldMuls)
		.fold(
			|| zeroed_vec::<P>(acc_size).into_boxed_slice(),
			|mut multilinears, (word, Range { start, end })| {
				let keys = &segment.keys[*start as usize..*end as usize];

				for key in keys {
					let operator_data = &prepared[key.operation];

					let mut acc = key.accumulate(
						&segment.constraint_indices,
						operator_data.r_x_prime_tensor.as_ref(),
						&operator_data.lambda_powers,
					);
					// The other slot's weight, when a previous phase bound it: one multiply per
					// key.
					if let Some(scale) = scale {
						acc *= scale[key.dense_shift_idx as usize];
					}
					let acc_packed = P::broadcast(acc);

					// The following loop is an optimized version of the following
					// for i in 0..Word::BITS {
					//     if get_bit(scattered, i) {
					//         values[start + i] += acc;
					//     }
					// }
					debug_assert!(
						(key.dense_shift_idx as usize) < segment.dense_shift_enc.len(),
						"a key indexes the dense shift encoding of its own segment"
					);
					// The row this slot names, shared by every sequence agreeing on it.
					let start = slot_rows.row(key.dense_shift_idx) * row_len;
					let values = &mut multilinears[start..start + row_len];
					let values_per_byte = Word::BYTES >> P::LOG_WIDTH;
					// The outer slot scatters the intermediate word, formed with one machine shift.
					// The inner slot scatters the witness word untouched.
					//
					// A shift that clears the word costs nothing: the loop below skips zero bytes.
					let scattered = match slot {
						ShiftSlot::Inner => *word,
						ShiftSlot::Outer => {
							let [inner, _] =
								segment.dense_shift_enc.decode(key.dense_shift_idx as usize);
							inner.apply(*word)
						}
					};
					let mut remaining_word = scattered.0;
					let mut byte_index = 0;
					while remaining_word != 0 {
						let byte = remaining_word as u8;
						let byte_values =
							&mut values[byte_index * values_per_byte..][..values_per_byte];
						for value_index in 0..(8 >> P::LOG_WIDTH) {
							unsafe {
								let packed_mask_index =
									((byte >> (value_index * P::WIDTH)) & low_bits_mask) as usize;

								// Safety:
								// - `packed_masks_map` is guaranteed to have enough elements to be
								//   indexed with a `P::WIDTH`-bits value.
								let packed_mask = packed_masks_map.get_unchecked(packed_mask_index);

								// Safety:
								// - `values` is guaranteed to be (8 >> P::LOG_WIDTH) elements long
								//   due to the chunking
								// - `value_index` is always in bounds because we iterate over 0..(8
								//   >> P::LOG_WIDTH)
								*byte_values.get_unchecked_mut(value_index) +=
									acc_packed.select(packed_mask);
							}
						}
						remaining_word >>= 8;
						byte_index += 1;
					}
				}

				multilinears
			},
		)
		// A merge seeded with a partial that already exists never touches a buffer of zeros.
		// An identity would allocate and zero one accumulator per merge, then add all of it.
		.reduce_with(|mut acc, local| {
			izip!(acc.iter_mut(), local.iter()).for_each(|(acc, local)| {
				*acc += *local;
			});
			acc
		})
		// An empty word list yields no partials at all, and its parts are zero.
		.unwrap_or_else(|| zeroed_vec::<P>(acc_size).into_boxed_slice());

	scatter_shift_rows(alloc, &multilinears, &slot_rows)
}

/// Spreads the rows of a dense accumulator over the shift axes of a phase's multilinear.
///
/// The accumulator holds one row of `Word::BITS` scalars per shift the bound slot takes, in
/// [`SlotRows`] order.
/// Each row lands at the offset its shift's `(variant, amount)` names.
/// Every row the segment does not use stays zero.
#[instrument(skip_all, name = "scatter_shift_rows")]
fn scatter_shift_rows<P: PackedField, A: Allocator>(
	alloc: &A,
	multilinears: &[P],
	slot_rows: &SlotRows,
) -> FieldVec<P, A> {
	const {
		assert!(
			P::LOG_WIDTH <= Word::LOG_BITS,
			"P::WIDTH is not supposed to exceed 8, so this statement must hold"
		);
	}

	// A row is a whole number of packed elements, so it copies in at row alignment.
	let row_len = Word::BITS >> P::LOG_WIDTH;
	let mut g = FieldBuffer::zeros_in(alloc, SLOT_PHASE_LOG_LEN);
	for (shift, row) in iter::zip(slot_rows.shifts(), multilinears.chunks(row_len)) {
		// The variant indexes the high variables and the amount the middle ones. A slot's shifts
		// are distinct, so no two rows land in the same place and each destination is still zero.
		let offset = (shift.variant as usize * Word::BITS + shift.amount as usize) * row_len;
		g.as_mut()[offset..][..row_len].copy_from_slice(row);
	}
	g
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_core::{
		ShiftVariant,
		constraint_system::{
			AndConstraint, ConstraintSystem, InoutSegment, Operand, Shift, ShiftedValueIndex,
			ValueIndex,
		},
	};
	use binius_field::{Field, Random};
	use binius_verifier::config::B128;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::protocols::shift::{
		claims::OperatorClaims, key_collection::build_key_collection, prove::OperatorData,
	};

	/// The words the fixture's hidden segment holds, one per private value.
	const N_HIDDEN_WORDS: usize = 4;

	/// A constraint system over `N_HIDDEN_WORDS` private words whose AND operand carries the given
	/// shift sequences, one term per sequence, spread over the words in turn.
	///
	/// `build_key_collection` does not validate, so this may carry sequences the canonical form
	/// rejects — which is what lets a test pair a degenerate slot with a working one.
	fn system_with(shift_seqs: &[[Shift; 2]]) -> ConstraintSystem {
		let terms = shift_seqs
			.iter()
			.enumerate()
			.map(|(term, &shift_seq)| {
				let index = ValueIndex::private((term % N_HIDDEN_WORDS) as u32);
				ShiftedValueIndex::new(index, shift_seq)
			})
			.collect::<Operand>();
		ConstraintSystem {
			constants: Vec::new(),
			n_inout: 0,
			n_private: N_HIDDEN_WORDS,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([terms, Operand::new(), Operand::new()])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}

	/// A single AND claim at a random point, which is what weights the keys.
	fn claims(rng: &mut StdRng) -> PreparedOperatorClaims<B128> {
		let r_z = B128::random(&mut *rng);
		let claims = OperatorClaims {
			zero: OperatorData::zero_claim(r_z),
			bitand: OperatorData {
				evals: [B128::ZERO; 3],
				r_zhat_prime: r_z,
				// One AND constraint, so the constraint point is empty.
				r_x_prime: Vec::new(),
			},
			intmul: OperatorData::zero_claim(r_z),
			binmul: OperatorData::zero_claim(r_z),
		};
		claims.prepare(|| B128::random(&mut *rng))
	}

	/// The shift indicator, derived from the word operation rather than from the reduction's
	/// tables.
	///
	/// Output bit `k` reads input bit `j` exactly when shifting the one-hot word `1 << j` sets bit
	/// `k`. That is the definition of the indicator, read straight off `Shift::apply`.
	fn indicator(shift: Shift, out_bit: usize, source_bit: usize) -> bool {
		let one_hot = Word(1u64 << source_bit);
		(shift.apply(one_hot).0 >> out_bit) & 1 == 1
	}

	/// The naive reference for the outer phase's multilinear.
	///
	/// This forms the intermediate word's bit `k` as the *field sum*
	///
	/// ```text
	/// sum_j shift-ind_op1(k, j, s_1) * w[y]_j
	/// ```
	///
	/// spelled out over all 64 source bits, rather than with the machine shift `build_outer_g`
	/// uses. Two source bits reaching one output position would cancel in this sum and survive the
	/// machine shift, so agreeing with it is what pins the at-most-one-source-bit property the
	/// phase rests on.
	fn reference_outer_g(
		words: &[Word],
		segment: &KeySegment,
		prepared: &PreparedOperatorClaims<B128>,
	) -> Vec<B128> {
		let mut reference = vec![B128::ZERO; 1 << SLOT_PHASE_LOG_LEN];
		for (word, range) in iter::zip(words, &segment.key_ranges) {
			for key in &segment.keys[range.start as usize..range.end as usize] {
				let operator_data = &prepared[key.operation];
				let acc = key.accumulate(
					&segment.constraint_indices,
					operator_data.r_x_prime_tensor.as_ref(),
					&operator_data.lambda_powers,
				);
				let [inner, outer] = segment.dense_shift_enc.decode(key.dense_shift_idx as usize);

				// The outer shift names the run of bit slots, as it does in the built form.
				let base =
					(outer.variant as usize * Word::BITS + outer.amount as usize) * Word::BITS;
				for out_bit in 0..Word::BITS {
					// Bit `out_bit` of the intermediate word, as a sum over every source bit.
					let intermediate_bit = (0..Word::BITS)
						.filter(|&source_bit| indicator(inner, out_bit, source_bit))
						.filter(|&source_bit| (word.0 >> source_bit) & 1 == 1)
						.map(|_| B128::ONE)
						.sum::<B128>();
					reference[base + out_bit] += acc * intermediate_bit;
				}
			}
		}
		reference
	}

	/// Random words for the fixture's hidden segment.
	fn hidden_words(rng: &mut StdRng) -> Vec<Word> {
		(0..N_HIDDEN_WORDS)
			.map(|_| Word::from_u64(rng.random()))
			.collect()
	}

	#[test]
	fn outer_g_matches_the_naive_indicator_reference() {
		let mut rng = StdRng::seed_from_u64(0);

		// Genuine pairs across the variants: each drops bits one way and moves them back the other,
		// so none collapses to a single shift.
		let shift_seqs = [
			[Shift::srl(3), Shift::sll(3)],
			[Shift::srl(5), Shift::sll(9)],
			[Shift::sll(8), Shift::srl(2)],
			[Shift::sar(7), Shift::sll(4)],
			[Shift::rotr(1), Shift::sll(6)],
			[Shift::srl32(4), Shift::sll32(11)],
			[Shift::sra32(9), Shift::sll32(3)],
			[Shift::rotr32(5), Shift::srl32(7)],
			// Two sequences agreeing on their outer shift must land in one row.
			[Shift::srl(6), Shift::sll(3)],
			// A sequence whose inner shift clears the word contributes nothing.
			[Shift::sll(63), Shift::srl(40)],
		];
		let cs = system_with(&shift_seqs);
		let key_collection = build_key_collection(&cs, InoutSegment::Public);
		let words = hidden_words(&mut rng);
		let prepared = claims(&mut rng);

		let built = build_outer_g::<B128, B128, _>(
			&GlobalAllocator,
			&words,
			&key_collection.hidden,
			&prepared,
			None,
		);
		let reference = reference_outer_g(&words, &key_collection.hidden, &prepared);

		assert_eq!(built.as_ref(), reference.as_slice());
		// A fixture that produced nothing would pass vacuously.
		assert!(reference.iter().any(|&entry| entry != B128::ZERO));
	}

	#[test]
	fn outer_g_over_every_variant_pair_matches_the_reference() {
		let mut rng = StdRng::seed_from_u64(1);

		// Every ordered variant combination, at amounts that keep the pair irreducible for most of
		// them. A pair that happens to collapse is still a valid input here: the reference and the
		// built form must agree on it either way.
		let shift_seqs = ShiftVariant::ALL
			.into_iter()
			.flat_map(|inner_variant| {
				ShiftVariant::ALL.into_iter().map(move |outer_variant| {
					[Shift::new(inner_variant, 3), Shift::new(outer_variant, 5)]
				})
			})
			.collect::<Vec<_>>();

		let cs = system_with(&shift_seqs);
		let key_collection = build_key_collection(&cs, InoutSegment::Public);
		let words = hidden_words(&mut rng);
		let prepared = claims(&mut rng);

		let built = build_outer_g::<B128, B128, _>(
			&GlobalAllocator,
			&words,
			&key_collection.hidden,
			&prepared,
			None,
		);
		let reference = reference_outer_g(&words, &key_collection.hidden, &prepared);

		assert_eq!(built.as_ref(), reference.as_slice());
	}

	#[test]
	fn outer_g_with_a_degenerate_inner_slot_is_the_inner_builder() {
		// The compatibility property the staged rollout leans on. With the inner slot degenerate
		// the intermediate word is the witness word itself, and the row is named by the outer
		// shift — so the outer builder does exactly what the inner builder does on the mirrored
		// sequences.
		let mut rng = StdRng::seed_from_u64(2);
		let words = hidden_words(&mut rng);
		let prepared = claims(&mut rng);

		let shifts = [
			Shift::IDENTITY,
			Shift::srl(3),
			Shift::sar(7),
			Shift::rotr(1),
			Shift::sll32(4),
		];

		// `[IDENTITY, s]` for the outer builder, `[s, IDENTITY]` for the inner one.
		let outer_slotted = shifts.map(|shift| [Shift::IDENTITY, shift]);
		let inner_slotted = shifts.map(|shift| [shift, Shift::IDENTITY]);

		let outer_cs = system_with(&outer_slotted);
		let inner_cs = system_with(&inner_slotted);
		let outer_keys = build_key_collection(&outer_cs, InoutSegment::Public);
		let inner_keys = build_key_collection(&inner_cs, InoutSegment::Public);

		let from_outer = build_outer_g::<B128, B128, _>(
			&GlobalAllocator,
			&words,
			&outer_keys.hidden,
			&prepared,
			None,
		);
		let from_inner =
			build_g::<B128, B128, _>(&GlobalAllocator, &words, &inner_keys.hidden, &prepared, None);

		assert_eq!(from_outer.as_ref(), from_inner.as_ref());
	}

	#[test]
	fn slot_rows_merge_sequences_that_share_the_bound_slot() {
		// Two sequences sharing a slot shift must accumulate into one row, or the scatter would
		// land two rows on one offset and drop one of them. The dense index is per sequence, so
		// this is the projection that makes the rows per slot shift.
		let shift_seqs = [
			[Shift::srl(3), Shift::sll(3)],
			[Shift::srl(5), Shift::sll(3)],
			[Shift::srl(3), Shift::sll(9)],
		];
		let cs = system_with(&shift_seqs);
		let key_collection = build_key_collection(&cs, InoutSegment::Public);
		let encoding = &key_collection.hidden.dense_shift_enc;

		// Three sequences, but only two distinct shifts in either slot.
		assert_eq!(encoding.len(), 3);
		let inner_rows = encoding.slot_rows(ShiftSlot::Inner);
		let outer_rows = encoding.slot_rows(ShiftSlot::Outer);
		assert_eq!(inner_rows.len(), 2);
		assert_eq!(outer_rows.len(), 2);
		assert_eq!(inner_rows.shifts().collect::<Vec<_>>(), [Shift::srl(3), Shift::srl(5)]);
		assert_eq!(outer_rows.shifts().collect::<Vec<_>>(), [Shift::sll(3), Shift::sll(9)]);

		// Each sequence maps to the row of its own slot shift.
		for dense_idx in 0..encoding.len() as u16 {
			let [inner, outer] = encoding.decode(dense_idx as usize);
			assert_eq!(inner_rows.shifts().nth(inner_rows.row(dense_idx)), Some(inner));
			assert_eq!(outer_rows.shifts().nth(outer_rows.row(dense_idx)), Some(outer));
		}
	}

	#[test]
	fn has_outer_shift_reports_whether_a_third_phase_is_needed() {
		// The degeneration check: a system of lone shifts needs no outer phase.
		let lone = system_with(&[
			[Shift::srl(3), Shift::IDENTITY],
			[Shift::rotr(1), Shift::IDENTITY],
		]);
		let lone_keys = build_key_collection(&lone, InoutSegment::Public);
		assert!(!lone_keys.hidden.dense_shift_enc.has_outer_shift());

		// One genuine pair anywhere is enough to need it.
		let paired = system_with(&[
			[Shift::srl(3), Shift::IDENTITY],
			[Shift::srl(3), Shift::sll(3)],
		]);
		let paired_keys = build_key_collection(&paired, InoutSegment::Public);
		assert!(paired_keys.hidden.dense_shift_enc.has_outer_shift());

		// A segment with no keys at all has nothing to bind.
		let empty = system_with(&[]);
		let empty_keys = build_key_collection(&empty, InoutSegment::Public);
		assert!(!empty_keys.hidden.dense_shift_enc.has_outer_shift());
	}
}
