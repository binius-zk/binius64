// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, ops::Range};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
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
	key_collection::{DenseShiftEncoding, KeyCollection, KeySegment},
	monster::build_h,
};

/// The number of variables in the g (and h) multilinear of phase 1.
///
/// The axes run, from the low index positions up: the bit position within a word, the shift
/// amount, and the shift variant.
pub const PHASE_1_LOG_LEN: usize = Word::LOG_BITS + Word::LOG_BITS + LOG_SHIFT_VARIANT_COUNT;

/// Proves the first phase of the shift reduction.
///
/// Builds the g and h multilinears and runs one sumcheck over their product.
#[instrument(skip_all, name = "prover_phase_1")]
pub fn prove_phase_1<F, P, Channel, A>(
	key_collection: &KeyCollection,
	words: SegmentWords<'_>,
	prepared: &PreparedOperatorClaims<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
	alloc: &A,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	// Build the g multilinear for the public and hidden segments separately, then sum them. The
	// public words are the prefix of `words`, and each segment's key ranges are segment-relative.
	let mut g = build_g::<_, P, _>(alloc, words.public, &key_collection.public, prepared);
	let hidden_g = build_g::<_, P, _>(alloc, words.hidden, &key_collection.hidden, prepared);
	for (slot, add) in iter::zip(g.as_mut(), hidden_g.as_ref()) {
		*slot += *add;
	}

	// BitAnd, IntMul and BinMul share the same `r_zhat_prime`.
	let h = build_h(alloc, domain_subspace, prepared.bitand.r_zhat_prime);

	run_phase_1_sumcheck(g, h, prepared.batched_eval(), channel, alloc)
}

/// Runs the phase 1 sumcheck protocol for shift constraint verification.
///
/// One bivariate-product sumcheck over `g` and `h`, both multilinears of [`PHASE_1_LOG_LEN`]
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
/// `SumcheckOutput` containing the challenge vector and the final evaluation `gamma`.
#[instrument(skip_all, name = "run_sumcheck")]
pub fn run_phase_1_sumcheck<
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
) -> SumcheckOutput<F> {
	let prover = bivariate_product_prover(alloc, [g, h], sum);

	let ProveSingleOutput {
		multilinear_evals,
		mut challenges,
	} = prove_single(prover, channel);
	challenges.reverse();

	let [g_eval, h_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");

	SumcheckOutput {
		challenges,
		eval: g_eval * h_eval,
	}
}

/// Constructs the phase-1 "g" multilinear for one key segment.
///
/// This builds the g multilinear polynomials used in phase 1 of the shift protocol, over the words
/// of a single value-vector segment (public or hidden). It constructs one multilinear polynomial
/// per shift variant.
///
/// The value vector's public and hidden words participate through their own [`KeySegment`], so a
/// caller builds each segment's parts with the matching words and sums the two results.
///
/// # Construction Process
///
/// 1. **Parallel Processing**: Words are processed in parallel chunks for efficiency
/// 2. **Key Processing**: For each word, iterate through its associated keys in the segment
/// 3. **Accumulation**: For each key, accumulate its contribution weighted by the r_x' tensor
/// 4. **Word Expansion**: Expand each witness word bitwise to populate the g multilinears
/// 5. **Lambda Weighting**: Apply lambda powers to weight different operand positions
///
/// The accumulator holds one row of `Word::BITS` scalars per shift the segment uses, addressed by
/// the keys' dense shift index. A scatter through the segment's encoding then spreads those rows
/// over the full `(shift variant, shift amount)` space of the returned parts.
///
/// # Returns
///
/// One multilinear of [`PHASE_1_LOG_LEN`] variables holding every shift variant's part: the
/// variant indexes the high [`LOG_SHIFT_VARIANT_COUNT`] variables, and within a variant the rows
/// of `Word::BITS` scalars run in order of shift amount.
///
/// # Usage
///
/// Used in phase 1 to construct the constant size g multilinears
/// that will participate in the phase 1 sumcheck protocol.
#[instrument(skip_all, name = "build_g")]
pub fn build_g<F: BinaryField, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	words: &[Word],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
) -> FieldVec<P, A> {
	// One row of `Word::BITS` scalars per shift the segment uses.
	let row_len = Word::BITS >> P::LOG_WIDTH;
	let acc_size = segment.dense_shift_enc.len() * row_len;

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

					let acc = key.accumulate(
						&segment.constraint_indices,
						operator_data.r_x_prime_tensor.as_ref(),
						&operator_data.lambda_powers,
					);
					let acc_packed = P::broadcast(acc);

					// The following loop is an optimized version of the following
					// for i in 0..Word::BITS {
					//     if get_bit(word, i) {
					//         values[start + i] += acc;
					//     }
					// }
					debug_assert!(
						(key.dense_shift_idx as usize) < segment.dense_shift_enc.len(),
						"a key indexes the dense shift encoding of its own segment"
					);
					let start = key.dense_shift_idx as usize * row_len;
					let values = &mut multilinears[start..start + row_len];
					let values_per_byte = Word::BYTES >> P::LOG_WIDTH;
					let mut remaining_word = word.0;
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

	scatter_shift_rows(alloc, &multilinears, &segment.dense_shift_enc)
}

/// Spreads the rows of a dense phase-1 accumulator over the shift axes of the g multilinear.
///
/// The accumulator holds the rows of `Word::BITS` scalars the segment's keys accumulate into, one
/// per shift the segment uses and in dense shift index order. Each row lands at the offset its
/// `(variant, amount)` pair names; every row the segment does not use stays zero.
#[instrument(skip_all, name = "scatter_shift_rows")]
fn scatter_shift_rows<P: PackedField, A: Allocator>(
	alloc: &A,
	multilinears: &[P],
	dense_shift_enc: &DenseShiftEncoding,
) -> FieldVec<P, A> {
	const {
		assert!(
			P::LOG_WIDTH <= Word::LOG_BITS,
			"P::WIDTH is not supposed to exceed 8, so this statement must hold"
		);
	}

	// A row is a whole number of packed elements, so it copies in at row alignment.
	let row_len = Word::BITS >> P::LOG_WIDTH;
	let mut g = FieldBuffer::zeros_in(alloc, PHASE_1_LOG_LEN);
	for ((variant, amount), row) in iter::zip(dense_shift_enc.iter(), multilinears.chunks(row_len))
	{
		// The variant indexes the high variables and the amount the middle ones. Dense indices are
		// distinct, so no two rows land in the same place and each destination is still zero.
		let offset = (variant as usize * Word::BITS + amount as usize) * row_len;
		g.as_mut()[offset..][..row_len].copy_from_slice(row);
	}
	g
}
