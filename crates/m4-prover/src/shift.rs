// Copyright 2026 The Binius Developers

//! The batched shift-reduction prover for the data-parallel Binius64 M4 proof system.

use std::iter;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{BinarySubspace, FieldBuffer, FieldVec, multilinear::eq::eq_ind_partial_eval};
use binius_prover::{
	fold_word::fold_words,
	protocols::shift::{
		KeyCollection, KeySegment, OperatorClaims, PreparedOperatorClaims, ShiftSlot,
		has_double_shift,
		monster::{
			SlotWeights, build_h, build_inner_h, build_monster_segments, inner_h_eval,
			outer_h_evals,
		},
		slot_phase::{SLOT_PHASE_LOG_LEN, build_slot_g, run_slot_sumcheck},
		words_phase::run_sumcheck,
	},
};

use crate::witness::{FoldedWitness, FoldedWord, shift_folded_word};

/// Proves the batched shift-reduction, collapsing every operation's claims into one.
///
/// The result is a single multilinear claim on the batched witness.
///
/// This mirrors the single-instance shift reduction, with one difference.
/// The hidden witness enters already folded over the instance axis, as a [`FoldedWitness`].
/// The public words are constants shared by every instance, so they are passed as raw words.
///
/// Every phase calls the single-instance prover's own subroutines, supplying its own builder for
/// the hidden segment:
///
/// ```text
/// outer phase   build_outer_g_from_folded_words   the folded row gathered through the inner shift
/// inner phase   build_g_from_folded_words         the folded row itself
/// words phase   FoldedWitness::fold_bits          the row contracted along its bit axis
/// ```
///
/// The public segment goes through the single-instance `build_slot_g` in each case, and the two are
/// summed. `build_h`, `build_inner_h`, `build_monster_segments` and `run_sumcheck` are reused
/// unchanged.
///
/// The outer phase is skipped when no term needs both shift slots, which leaves the two-phase
/// reduction. That predicate is derived from the constraint system, so the verifier follows the
/// same branch without being told.
///
/// # Parameters
/// - `key_collection`: the prover's key collection for the constraint system.
/// - `public_words`: the public (constant) words, shared by every instance.
/// - `folded_witness`: the hidden witness, folded over the instance axis.
/// - `claims`: the operand evaluation claim of each operation.
/// - `domain_subspace`: the univariate evaluation domain.
/// - `channel`: the prover channel driving the interactive protocol.
/// - `alloc`: the allocator backing the reduction's intermediate buffers.
///
/// # Returns
/// The `SumcheckOutput` with the final challenges and the reduced witness evaluation.
pub fn prove<F, P, Channel, A>(
	key_collection: &KeyCollection,
	public_words: &[Word],
	folded_witness: &FoldedWitness<F, A>,
	claims: OperatorClaims<F>,
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
	// One batching coefficient per operation, drawn from the channel.
	// SOUNDNESS: `prepare` draws in the order the verifier draws in; do not reorder it.
	let prepared = claims.prepare(|| channel.sample());

	// Whether any term needs both shift slots, which selects the phase structure.
	// The system is public, so the verifier derives the same answer and takes the same branch.
	let three_phase = has_double_shift(key_collection);

	// Builds one phase's multilinear over both segments and sums them.
	// The public words are constants shared by every instance, so the single-instance builder folds
	// them from their bits; the hidden words arrive already folded over instances.
	let build_combined =
		|slot: ShiftSlot, public_scale: Option<&[F]>, hidden_scale: Option<&[F]>| {
			let mut g = build_slot_g::<F, F, _>(
				alloc,
				public_words,
				&key_collection.public,
				&prepared,
				slot,
				public_scale,
			);
			// The hidden segment goes through the batched builder for the slot this phase binds.
			let folded = folded_witness.words();
			let hidden_g = match slot {
				ShiftSlot::Inner => build_g_from_folded_words(
					alloc,
					folded,
					&key_collection.hidden,
					&prepared,
					hidden_scale,
				),
				ShiftSlot::Outer => build_outer_g_from_folded_words(
					alloc,
					folded,
					&key_collection.hidden,
					&prepared,
					hidden_scale,
				),
			};
			for (entry, add) in iter::zip(g.as_mut(), hidden_g.as_ref()) {
				*entry += *add;
			}
			g
		};

	let (r_j, claim, operation_scalars, inner, outer) = if three_phase {
		// The outer phase binds `(k, s_2, v_2)` against the intermediate rows.
		let g = build_combined(ShiftSlot::Outer, None, None);
		let h = build_h::<F, F, _>(alloc, domain_subspace, prepared.bitand.r_zhat_prime);
		let outer_phase =
			run_slot_sumcheck::<F, F, _, _>(g, h, prepared.batched_eval(), channel, alloc);

		// The inner phase binds the inner axes, each key weighted by what the outer phase closed
		// on.
		let outer_weights = SlotWeights::at(&outer_phase.r_s, &outer_phase.r_v);
		let scale_of = |segment: &KeySegment| {
			segment
				.dense_shift_enc
				.iter()
				.map(|[_, outer_shift]| outer_phase.h_eval * outer_weights.weight(outer_shift))
				.collect::<Vec<_>>()
		};
		let public_scale = scale_of(&key_collection.public);
		let hidden_scale = scale_of(&key_collection.hidden);

		let g = build_combined(ShiftSlot::Inner, Some(&public_scale), Some(&hidden_scale));
		let h = build_inner_h::<F, F, _>(alloc, &outer_phase.r_bit);
		let inner_phase =
			run_slot_sumcheck::<F, F, _, _>(g, h, outer_phase.claim(), channel, alloc);

		let operation_scalars = outer_h_evals(
			&prepared,
			domain_subspace,
			&outer_phase.r_bit,
			&outer_phase.r_s,
			&outer_phase.r_v,
		)
		.scaled_by(inner_h_eval(
			&outer_phase.r_bit,
			&inner_phase.r_bit,
			&inner_phase.r_s,
			&inner_phase.r_v,
		));

		(
			inner_phase.r_bit.clone(),
			inner_phase.claim(),
			operation_scalars,
			SlotWeights::at(&inner_phase.r_s, &inner_phase.r_v),
			outer_weights,
		)
	} else {
		// One slot phase binds the inner axes, so its `h` carries the oblong factor and the outer
		// slot's weight selects the identity every key carries.
		let g = build_combined(ShiftSlot::Inner, None, None);
		let h = build_h::<F, F, _>(alloc, domain_subspace, prepared.bitand.r_zhat_prime);
		let slot_phase =
			run_slot_sumcheck::<F, F, _, _>(g, h, prepared.batched_eval(), channel, alloc);

		let operation_scalars = outer_h_evals(
			&prepared,
			domain_subspace,
			&slot_phase.r_bit,
			&slot_phase.r_s,
			&slot_phase.r_v,
		);

		(
			slot_phase.r_bit.clone(),
			slot_phase.claim(),
			operation_scalars,
			SlotWeights::at(&slot_phase.r_s, &slot_phase.r_v),
			SlotWeights::identity_selecting(),
		)
	};

	// The words phase runs at the bit point the last slot phase bound.
	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

	// The witness folded at `r_j`, per segment.
	// The public fold is a raw-word fold; the hidden fold contracts the already-oblong bits.
	let public_folded = fold_words::<F, P, _>(alloc, public_words, r_j_tensor.as_ref());
	let hidden_folded = folded_witness.fold_bits::<P>(r_j_tensor.as_ref(), alloc);

	let (public_monster, hidden_monster) = build_monster_segments::<F, P, _>(
		alloc,
		key_collection,
		&prepared,
		operation_scalars,
		&inner,
		&outer,
	);

	run_sumcheck::<F, P, _, _>(
		&public_folded,
		hidden_folded,
		&public_monster,
		hidden_monster,
		public_words,
		r_j,
		claim,
		channel,
		alloc,
	)
}

/// Constructs the inner phase's "g" multilinear parts from instance-folded words.
///
/// This is [`build_slot_g_from_folded_words`] on the [`ShiftSlot::Inner`] slot: it scatters the
/// folded word itself, into the row its sequence's inner shift names.
pub fn build_g_from_folded_words<F: BinaryField, A: Allocator>(
	alloc: &A,
	folded_words: &[FoldedWord<F>],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
	scale: Option<&[F]>,
) -> FieldVec<F, A> {
	build_slot_g_from_folded_words(alloc, folded_words, segment, prepared, ShiftSlot::Inner, scale)
}

/// Constructs the outer phase's "G" multilinear parts from instance-folded words.
///
/// This is [`build_slot_g_from_folded_words`] on the [`ShiftSlot::Outer`] slot: it scatters the
/// *intermediate* folded word — the folded word with its sequence's inner shift already applied —
/// into the row the outer shift names.
///
/// **This is cleaner than the single-instance case, not harder.**
/// There the intermediate word is formed with a machine shift over packed bits.
/// Here the bits have become field elements, which a machine shift cannot touch.
///
/// It does not need to.
/// Every variant reads at most one source bit per output position, and folding over instances is
/// linear in the bits, so the two commute.
/// Forming the intermediate row is a **gather** over the 64 folded elements — exactly the
/// permutation the unfolded path applies to bits.
pub fn build_outer_g_from_folded_words<F: BinaryField, A: Allocator>(
	alloc: &A,
	folded_words: &[FoldedWord<F>],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
	scale: Option<&[F]>,
) -> FieldVec<F, A> {
	build_slot_g_from_folded_words(alloc, folded_words, segment, prepared, ShiftSlot::Outer, scale)
}

/// Constructs one phase's shift multilinear parts, one per shift variant, from instance-folded
/// words.
///
/// This is the batched analogue of the single-instance `build_slot_g`.
/// It consumes a key segment's words already folded over the instance axis.
/// So each word is a [`FoldedWord`], whose bits are field elements rather than a packed `u64`.
///
/// The single-instance builder scatters an accumulator onto a word's set bits by masking.
/// This one scales the accumulator by each folded bit with a field multiplication.
/// The two coincide when a folded bit is zero or one.
///
/// Use this for the hidden (committed) segment, whose words are folded over instances.
/// Use the single-instance builder for the public segment, whose words are shared constants.
/// Adding the two results gives the complete multilinear.
///
/// The two phases differ in exactly two places, both fixed by `slot`:
///
/// ```text
/// slot     row named by            elements scattered
/// Inner    the inner shift         the folded word
/// Outer    the outer shift         the folded word gathered through the inner shift
/// ```
///
/// # Arguments
///
/// - `folded_words`: the segment's words folded over instances, in `segment.key_ranges` order.
/// - `segment`: the shift keys of this segment, grouped by the word they act on.
/// - `prepared`: the per-operation claims, indexed by the operation each key names.
/// - `slot`: which slot of a key's shift sequence names its row.
/// - `scale`: an optional extra factor per shift sequence, as the other slot's phase weights it.
///
/// Words past the segment's key ranges are power-of-two padding, and are ignored.
///
/// # Returns
///
/// One multilinear of [`SLOT_PHASE_LOG_LEN`] variables, holding every shift variant's part.
///
/// The bound slot's variant indexes the high variables and its amount the middle ones, so a key
/// names one run of bit slots:
///
/// ```text
/// g[(variant * Word::BITS + amount) * Word::BITS + bit] += folded_bit * acc(key)
/// ```
///
/// where `acc(key)` is the key's lambda-weighted partial evaluation tensor.
/// Every word carrying that key accumulates into the same slots.
///
/// This scalar implementation ignores the single-instance builder's packing and parallelism.
pub fn build_slot_g_from_folded_words<F: BinaryField, A: Allocator>(
	alloc: &A,
	folded_words: &[FoldedWord<F>],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
	slot: ShiftSlot,
	scale: Option<&[F]>,
) -> FieldVec<F, A> {
	// Invariant: a scale carries one factor per shift sequence, indexed as the keys are.
	debug_assert!(
		scale.is_none_or(|scale| scale.len() == segment.dense_shift_enc.len()),
		"a per-sequence scale covers the segment's dense shift encoding"
	);

	// One zeroed multilinear covering every shift, drawn from the allocator.
	// A key names one `(variant, amount)` in the bound slot, and several sequences may agree on it,
	// so the scatter below accumulates rather than overwrites.
	let mut multilinear = FieldBuffer::zeros_in(alloc, SLOT_PHASE_LOG_LEN);

	// Each folded word carries the keys named by the segment-relative range at its position.
	for (word, range) in folded_words.iter().zip(&segment.key_ranges) {
		let keys = &segment.keys[range.start as usize..range.end as usize];
		for key in keys {
			let operator_data = &prepared[key.operation];

			// The lambda-weighted partial evaluation tensor for this shifted word.
			let mut acc = key.accumulate(
				&segment.constraint_indices,
				operator_data.r_x_prime_tensor.as_ref(),
				&operator_data.lambda_powers,
			);
			// The other slot's weight, when a previous phase already bound it.
			if let Some(scale) = scale {
				acc *= scale[key.dense_shift_idx as usize];
			}

			// The bound slot's variant indexes the high variables and its amount the middle ones,
			// which together name one run of bit slots.
			let [inner, outer] = segment.dense_shift_enc.decode(key.dense_shift_idx as usize);
			let row_shift = match slot {
				ShiftSlot::Inner => inner,
				ShiftSlot::Outer => outer,
			};
			let base =
				(row_shift.variant as usize * Word::BITS + row_shift.amount as usize) * Word::BITS;
			let slots = &mut multilinear.as_mut()[base..base + Word::BITS];

			// The outer slot scatters the intermediate row: the folded word gathered through the
			// inner shift. The inner slot scatters the folded word untouched.
			let scattered = match slot {
				ShiftSlot::Inner => *word,
				ShiftSlot::Outer => shift_folded_word(word, inner),
			};

			// Scatter the accumulator across this key's bit slots, scaling each by the folded bit.
			for (slot, &folded_bit) in iter::zip(slots, &scattered) {
				*slot += acc * folded_bit;
			}
		}
	}

	multilinear
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_core::{
		constraint_system::{
			AndConstraint, Composition, InoutSegment, Shift, ShiftedValueIndex, ValueIndex,
		},
		word::Word,
	};
	use binius_field::{AESTowerField8b, Field, PackedBinaryGhash1x128b, Random};
	use binius_math::{
		inner_product::inner_product_buffers,
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::random_scalars,
		univariate::lagrange_evals_scalars,
	};
	use binius_prover::protocols::shift::{OperatorData, build_key_collection, monster::build_h};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{
		config::{B128, StdChallenger},
		protocols::shift::{OperatorData as VerifierOperatorData, check_eval, verify},
	};
	use rand::prelude::*;

	use super::*;
	use crate::{
		ValueTable,
		test_utils::{N_INPUT_WORDS, crc64_circuit, populate_crc64_witness},
		witness::OperandColumns,
	};

	// The oblong evaluation of each bitand operand column A, B, C at the shift challenges.
	//
	// Builds the batched AND witness, then for each column folds its word bits by the Lagrange
	// basis at r_z and evaluates the resulting row multilinear at the (instance, constraint) point
	// r_rho || r_x. The columns are constraint-major, so r_rho (low) indexes the instance within a
	// constraint and r_x (high) indexes the constraint.
	fn evaluate_and_witness<P: PackedField<Scalar = B128>>(
		table: &ValueTable,
		constants: &[Word],
		and_constraints: &[AndConstraint],
		domain_subspace: &BinarySubspace<B128>,
		r_z: B128,
		r_x: &[B128],
		r_rho: &[B128],
	) -> [B128; 3] {
		let columns = OperandColumns::build(table, constants, and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();
		let lagrange = lagrange_evals_scalars::<B128, B128>(domain_subspace, &r_z);
		let row_point: Vec<B128> = r_rho.iter().chain(r_x).copied().collect();
		let operand_eval = |column: &[Word]| {
			let folded_column = fold_words::<B128, P, _>(&GlobalAllocator, column, &lagrange);
			evaluate(&folded_column, &row_point)
		};
		// The batch witness stores only the `A` and `B` columns.
		// The reduction reads `C` as the word-by-word AND of the two.
		// Materialize that same derived column so its evaluation matches the reduction's claim.
		let c_column: Vec<Word> = iter::zip(a, b).map(|(&a, &b)| a & b).collect();
		[operand_eval(a), operand_eval(b), operand_eval(&c_column)]
	}

	// Folds a contiguous run of value-vector words over the instance axis, one FoldedWord per word.
	// This lets the public and hidden segments be folded separately, matching how `build_g`
	// consumes one segment at a time.
	fn fold_words_over_instances(
		table: &ValueTable,
		constants: &[Word],
		r_rho: &[B128],
		words: std::ops::Range<usize>,
	) -> Vec<FoldedWord<B128>> {
		let eq = eq_ind_partial_eval_scalars::<B128>(r_rho);
		let mut folded = vec![[B128::ZERO; Word::BITS]; words.len()];
		for (rho, &weight) in eq.iter().enumerate() {
			// Reconstruct this instance independently of the fold, then fold its chosen word range.
			let vv = table.instance_value_vec(rho, constants);
			for (word, out) in vv.combined_witness()[words.clone()].iter().zip(&mut folded) {
				for (b, out_b) in out.iter_mut().enumerate() {
					if (word.0 >> b) & 1 == 1 {
						*out_b += weight;
					}
				}
			}
		}
		folded
	}

	// Shift pairs whose composition genuinely needs both slots, spread over the variants.
	//
	// The frontend emits lone shifts, so a fixture meaning to exercise the outer phase names pairs
	// itself. Each is checked irreducible, so the fixture cannot silently take the two-phase path.
	fn irreducible_pairs() -> Vec<[Shift; 2]> {
		let pairs = [
			[Shift::srl(3), Shift::sll(3)],
			[Shift::sll(8), Shift::srl(2)],
			[Shift::sar(7), Shift::sll(4)],
			[Shift::rotr(1), Shift::sll(6)],
			[Shift::srl32(4), Shift::sll32(11)],
			[Shift::sra32(9), Shift::sll32(3)],
			[Shift::rotr32(5), Shift::srl32(7)],
			// Two sequences agreeing on their outer shift must merge into one accumulator row.
			[Shift::srl(6), Shift::sll(3)],
		];
		for &[inner, outer] in &pairs {
			assert_eq!(
				Shift::compose(inner, outer),
				Composition::Pair,
				"{inner:?} then {outer:?} collapses, so it would not reach the outer phase"
			);
		}
		pairs.to_vec()
	}

	// The oblong evaluation of the three AND operand columns *as the constraints declare them*.
	//
	// The batched AND check derives its third column as the conjunction of the other two, since
	// that is what a satisfying witness gives. The shift reduction instead proves each *declared*
	// column against its claim.
	//
	// So a fixture that does not satisfy the relation must read its third claim off the declared
	// operand rather than derive it.
	fn evaluate_declared_and_operands<P: PackedField<Scalar = B128>>(
		table: &ValueTable,
		constants: &[Word],
		and_constraints: &[AndConstraint],
		domain_subspace: &BinarySubspace<B128>,
		r_z: B128,
		r_x: &[B128],
		r_rho: &[B128],
	) -> [B128; 3] {
		let columns =
			OperandColumns::<_, 3>::build(table, constants, and_constraints, &GlobalAllocator);
		let lagrange = lagrange_evals_scalars::<B128, B128>(domain_subspace, &r_z);
		let row_point: Vec<B128> = r_rho.iter().chain(r_x).copied().collect();
		columns.as_slices().map(|column| {
			let folded_column = fold_words::<B128, P, _>(&GlobalAllocator, column, &lagrange);
			evaluate(&folded_column, &row_point)
		})
	}

	// One AND constraint whose operands carry genuine shift pairs over the table's private words.
	//
	// The reduction proves that each operand column evaluates to the claim it is handed, not that
	// the AND relation holds. The claims come from these same constraints, so the fixture need not
	// satisfy the relation.
	fn double_shifted_and_constraints(n_private: usize) -> Vec<AndConstraint> {
		let pairs = irreducible_pairs();
		assert!(pairs.len() <= n_private, "the fixture needs one private word per pair");

		let term =
			|index: usize| ShiftedValueIndex::new(ValueIndex::private(index as u32), pairs[index]);
		let split = pairs.len() / 2;
		vec![AndConstraint([
			(0..split).map(term).collect(),
			// Mixing a pair with a lone shift and an unshifted term keeps every term class in one
			// operand, so the outer phase has to handle a degenerate slot alongside a working one.
			(split..pairs.len())
				.map(term)
				.chain([
					ShiftedValueIndex::rotr(ValueIndex::private(0), 9),
					ShiftedValueIndex::plain(ValueIndex::private(1)),
				])
				.collect(),
			Vec::new(),
		])]
	}

	// The batched prove round-trips with the single-instance shift verifier: the two agree on the
	// reduced challenges and witness evaluation, and that evaluation equals the direct evaluation
	// of the instance-folded witness. The prover feeds the verifier's own subroutines, so the
	// transcript is exactly what the single-instance verifier expects.
	#[test]
	fn prove_and_verify_round_trip() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		let log_instances = 6;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		let key_collection = build_key_collection(&cs, InoutSegment::Hidden);

		// The univariate bit challenge, the constraint challenge, and the instance challenge.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, cs.log_and_constraints().unwrap_or(0));
		// The Zero claim closes at its own constraint point, as wide as the ZERO set. Its value is
		// zero at any point: a satisfied ZERO constraint array vanishes identically, so its
		// multilinear extension is the zero polynomial.
		let r_x_zero = random_scalars::<B128>(&mut rng, cs.log_zero_constraints().unwrap_or(0));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		// The hidden witness folded over instances (one FoldedWord per committed word), and the
		// public constants.
		let folded_witness =
			FoldedWitness::<B128, _>::fold_instances(&table, &r_rho, &GlobalAllocator);
		let public_words = &cs.constants;

		// The bitand operand evals at (r_z, r_x, r_rho); the circuit has no IMUL constraints, so
		// the intmul claim is the zero claim over an empty point.
		let bitand_evals = evaluate_and_witness::<P>(
			&table,
			public_words,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);
		let intmul_evals = [B128::ZERO; 4];

		// Prove.
		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let prover_output = prove::<B128, P, _, _>(
			&key_collection,
			public_words,
			&folded_witness,
			OperatorClaims {
				zero: OperatorData {
					evals: [B128::ZERO],
					r_zhat_prime: r_z,
					r_x_prime: r_x_zero.clone(),
				},
				bitand: OperatorData {
					evals: bitand_evals,
					r_zhat_prime: r_z,
					r_x_prime: r_x.clone(),
				},
				intmul: OperatorData::zero_claim(r_z),
				binmul: OperatorData::zero_claim(r_z),
			},
			&domain_subspace,
			&mut prover_transcript,
			&GlobalAllocator,
		);

		// Verify against the single-instance shift verifier.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_zero = VerifierOperatorData::new(r_x_zero, [B128::ZERO]);
		let verifier_bitand = VerifierOperatorData::new(r_x, bitand_evals);
		let verifier_intmul = VerifierOperatorData::new(Vec::new(), intmul_evals);
		let verifier_bmul = VerifierOperatorData::new(Vec::new(), [B128::ZERO; 6]);
		let verifier_output = verify(
			&cs,
			InoutSegment::Hidden,
			&verifier_zero,
			&verifier_bitand,
			&verifier_intmul,
			&verifier_bmul,
			&mut verifier_transcript,
		)
		.unwrap();
		check_eval(
			&cs,
			InoutSegment::Hidden,
			public_words,
			&verifier_zero,
			&verifier_bitand,
			&verifier_intmul,
			&verifier_bmul,
			&domain_subspace,
			r_z,
			&verifier_output,
			&mut verifier_transcript,
		)
		.unwrap();
		verifier_transcript.finalize().unwrap();

		// The witness evaluation equals the instance-folded witness evaluated at the point, with
		// the segment's zero-padding contributing the (1 - r) factors above the folded length.
		let r_y = verifier_output.r_y();
		let log_folded = folded_witness.log_padded_words();
		let base = folded_witness.evaluate(verifier_output.r_j(), &r_y[..log_folded]);
		let expected_eval = r_y[log_folded..]
			.iter()
			.fold(base, |acc, &r_y_i| acc * (B128::ONE - r_y_i));
		assert_eq!(expected_eval, verifier_output.witness_eval);

		// Prover and verifier agree on the reduced challenges and the witness evaluation.
		let eval_point = [
			verifier_output.r_j(),
			r_y,
			std::slice::from_ref(&verifier_output.r_segment),
		]
		.concat();
		assert_eq!(prover_output.challenges, eval_point);
		assert_eq!(prover_output.eval, verifier_output.witness_eval);
	}

	// The phase-1 identity: summing the g·h inner products over the shift variants reconstructs the
	// lambda-batched operand evaluation claim.
	//
	// The g multilinear comes from the batched build_g on the full folded witness; h comes
	// from the single-instance prover's build_h at the same univariate challenge r_z. Their
	// inner product must equal the lambda-powers scaling of the batched AND-check operand evals
	// (the intmul claim is empty, contributing nothing).
	#[test]
	fn phase_1_g_h_inner_product_matches_batched_evals() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		let log_instances = 6;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);
		let constants = &c.circuit.constraint_system().constants;

		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		let key_collection = build_key_collection(&cs, InoutSegment::Hidden);

		// The univariate bit challenge, the constraint challenge, and the instance challenge.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, cs.log_and_constraints().unwrap_or(0));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		// The batched AND-check operand evals at (r_z, r_x, r_rho), and the full folded witness at
		// the same r_rho, so g and the claim agree on the instance point.
		let bitand_evals = evaluate_and_witness::<P>(
			&table,
			constants,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);
		// The hidden segment spans value indices `[offset_inout, combined_len)`.
		let offset = table.layout().offset_inout();
		let combined = table.layout().combined_len();
		assert_eq!(table.n_hidden_words(), combined - offset);
		let public_words = &cs.constants;
		let hidden_folded = fold_words_over_instances(&table, constants, &r_rho, offset..combined);

		// Prepare the operator data: lambda batches the three operand claims. The circuit has no
		// IMUL or BMUL constraints, so those two are zero claims at an empty point.
		// The ZERO set has its own constraint point, as wide as the set itself.
		let claims = OperatorClaims {
			zero: OperatorData {
				evals: [B128::ZERO],
				r_zhat_prime: r_z,
				r_x_prime: random_scalars::<B128>(&mut rng, cs.log_zero_constraints().unwrap_or(0)),
			},
			bitand: OperatorData {
				evals: bitand_evals,
				r_zhat_prime: r_z,
				r_x_prime: r_x,
			},
			intmul: OperatorData::zero_claim(r_z),
			binmul: OperatorData::zero_claim(r_z),
		};
		let prepared = claims.prepare(|| B128::random(&mut rng));

		// The g multilinear: the public segment folds from raw constant words via the
		// single-instance builder, the hidden segment from the instance-folded words. Add them.
		// The h multilinear comes from the single-instance prover.
		let mut g = build_slot_g::<B128, B128, _>(
			&GlobalAllocator,
			public_words,
			&key_collection.public,
			&prepared,
			ShiftSlot::Inner,
			None,
		);
		let hidden_g = build_g_from_folded_words(
			&GlobalAllocator,
			&hidden_folded,
			&key_collection.hidden,
			&prepared,
			None,
		);
		for (slot, add) in iter::zip(g.as_mut(), hidden_g.as_ref()) {
			*slot += *add;
		}
		let h = build_h::<B128, B128, _>(&GlobalAllocator, &domain_subspace, r_z);
		let inner_product = inner_product_buffers(&g, &h);

		// The lambda-powers scaling of the batched AND-check evals, plus the empty intmul claim.
		let expected = prepared.bitand.batched_eval + prepared.intmul.batched_eval;
		assert_eq!(inner_product, expected);
	}

	// The same round trip, over a constraint system whose AND operands carry genuine shift pairs.
	//
	// This is the case the batched three-phase reduction exists for: the outer phase runs against
	// the intermediate rows, formed by gathering each folded row through its inner shift.
	#[test]
	fn prove_and_verify_round_trip_with_double_shifts() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		let log_instances = 4;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		// The circuit's own constraints, with the AND set replaced by one carrying shift pairs. The
		// value counts are untouched, so the table still matches the system.
		let mut cs = c.circuit.constraint_system().clone();
		cs.and_constraints = double_shifted_and_constraints(cs.n_private);
		cs.validate().unwrap();
		assert!(
			binius_prover::protocols::shift::has_double_shift(&build_key_collection(
				&cs,
				InoutSegment::Hidden
			)),
			"the fixture must reach the outer phase"
		);
		let key_collection = build_key_collection(&cs, InoutSegment::Hidden);

		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, cs.log_and_constraints().unwrap_or(0));
		let r_x_zero = random_scalars::<B128>(&mut rng, cs.log_zero_constraints().unwrap_or(0));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		let folded_witness =
			FoldedWitness::<B128, _>::fold_instances(&table, &r_rho, &GlobalAllocator);
		let public_words = &cs.constants;

		// The operand claims, from the columns these constraints actually produce.
		let bitand_evals = evaluate_declared_and_operands::<P>(
			&table,
			public_words,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);

		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let prover_output = prove::<B128, P, _, _>(
			&key_collection,
			public_words,
			&folded_witness,
			OperatorClaims {
				zero: OperatorData {
					evals: [B128::ZERO],
					r_zhat_prime: r_z,
					r_x_prime: r_x_zero.clone(),
				},
				bitand: OperatorData {
					evals: bitand_evals,
					r_zhat_prime: r_z,
					r_x_prime: r_x.clone(),
				},
				intmul: OperatorData::zero_claim(r_z),
				binmul: OperatorData::zero_claim(r_z),
			},
			&domain_subspace,
			&mut prover_transcript,
			&GlobalAllocator,
		);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_zero = VerifierOperatorData::new(r_x_zero, [B128::ZERO]);
		let verifier_bitand = VerifierOperatorData::new(r_x, bitand_evals);
		let verifier_intmul = VerifierOperatorData::new(Vec::new(), [B128::ZERO; 4]);
		let verifier_bmul = VerifierOperatorData::new(Vec::new(), [B128::ZERO; 6]);
		let verifier_output = verify(
			&cs,
			InoutSegment::Hidden,
			&verifier_zero,
			&verifier_bitand,
			&verifier_intmul,
			&verifier_bmul,
			&mut verifier_transcript,
		)
		.unwrap();
		check_eval(
			&cs,
			InoutSegment::Hidden,
			public_words,
			&verifier_zero,
			&verifier_bitand,
			&verifier_intmul,
			&verifier_bmul,
			&domain_subspace,
			r_z,
			&verifier_output,
			&mut verifier_transcript,
		)
		.unwrap();
		verifier_transcript.finalize().unwrap();

		// The outer phase ran, which is what separates this from the two-phase round trip.
		assert!(
			verifier_output.outer.is_some(),
			"a system with genuine shift pairs runs the outer phase"
		);

		// The witness evaluation equals the instance-folded witness evaluated at the point, with
		// the segment's zero-padding contributing the (1 - r) factors above the folded length.
		let r_y = verifier_output.r_y();
		let log_folded = folded_witness.log_padded_words();
		let base = folded_witness.evaluate(verifier_output.r_j(), &r_y[..log_folded]);
		let expected_eval = r_y[log_folded..]
			.iter()
			.fold(base, |acc, &r_y_i| acc * (B128::ONE - r_y_i));
		assert_eq!(expected_eval, verifier_output.witness_eval);

		// Prover and verifier agree on the reduced challenges and the witness evaluation.
		let eval_point = [
			verifier_output.r_j(),
			r_y,
			std::slice::from_ref(&verifier_output.r_segment),
		]
		.concat();
		assert_eq!(prover_output.challenges, eval_point);
		assert_eq!(prover_output.eval, verifier_output.witness_eval);
	}

	// The outer phase's identity: summing the G·H inner products over the shift variants
	// reconstructs the lambda-batched operand evaluation claim.
	//
	// This is the batched analogue of `phase_1_g_h_inner_product_matches_batched_evals`, one phase
	// further out. `G` comes from the batched builder on the instance-folded witness, gathering
	// each row through its inner shift; `H` is the single-instance prover's `build_h`, unchanged.
	// Their inner product must equal the claim the outer sumcheck is handed — which is what pins
	// the gather against the operand columns the AND check actually produced.
	#[test]
	fn outer_g_h_inner_product_matches_batched_evals() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		let log_instances = 4;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let mut cs = c.circuit.constraint_system().clone();
		cs.and_constraints = double_shifted_and_constraints(cs.n_private);
		cs.validate().unwrap();
		let key_collection = build_key_collection(&cs, InoutSegment::Hidden);

		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, cs.log_and_constraints().unwrap_or(0));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		let bitand_evals = evaluate_declared_and_operands::<P>(
			&table,
			&cs.constants,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);

		// The hidden segment spans value indices `[offset_inout, combined_len)`.
		let offset = table.layout().offset_inout();
		let combined = table.layout().combined_len();
		let public_words = &cs.constants;
		let hidden_folded =
			fold_words_over_instances(&table, &cs.constants, &r_rho, offset..combined);

		let claims = OperatorClaims {
			zero: OperatorData {
				evals: [B128::ZERO],
				r_zhat_prime: r_z,
				r_x_prime: random_scalars::<B128>(&mut rng, cs.log_zero_constraints().unwrap_or(0)),
			},
			bitand: OperatorData {
				evals: bitand_evals,
				r_zhat_prime: r_z,
				r_x_prime: r_x,
			},
			intmul: OperatorData::zero_claim(r_z),
			binmul: OperatorData::zero_claim(r_z),
		};
		let prepared = claims.prepare(|| B128::random(&mut rng));

		// The G multilinear: the public segment through the single-instance builder on the outer
		// slot, the hidden segment through the batched one. Add them.
		let mut g = build_slot_g::<B128, B128, _>(
			&GlobalAllocator,
			public_words,
			&key_collection.public,
			&prepared,
			ShiftSlot::Outer,
			None,
		);
		let hidden_g = build_outer_g_from_folded_words(
			&GlobalAllocator,
			&hidden_folded,
			&key_collection.hidden,
			&prepared,
			None,
		);
		for (slot, add) in iter::zip(g.as_mut(), hidden_g.as_ref()) {
			*slot += *add;
		}
		let h = build_h::<B128, B128, _>(&GlobalAllocator, &domain_subspace, r_z);

		let expected = prepared.bitand.batched_eval + prepared.intmul.batched_eval;
		assert_eq!(inner_product_buffers(&g, &h), expected);
	}
}
