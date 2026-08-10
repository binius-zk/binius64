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
		KeyCollection, KeySegment, OperatorClaims, PreparedOperatorClaims,
		monster::{build_h, build_monster_segments},
		phase_1::{PHASE_1_LOG_LEN, build_g, run_phase_1_sumcheck},
		phase_2::run_sumcheck,
	},
};

use crate::witness::{FoldedWitness, FoldedWord};

/// Proves the batched shift-reduction, collapsing every operation's claims into one.
///
/// The result is a single multilinear claim on the batched witness.
///
/// This mirrors the single-instance shift reduction, with one difference.
/// The hidden witness enters already folded over the instance axis, as a [`FoldedWitness`].
/// The public words are constants shared by every instance, so they are passed as raw words.
///
/// The two phases call the single-instance prover's own subroutines.
/// Phase 1 builds the hidden segment's g with [`build_g_from_folded_words`].
/// It builds the public segment's g with the single-instance `build_g`, then sums the two.
/// Phase 2 contracts the hidden witness along its bit axis with [`FoldedWitness::fold_bits`].
/// It folds the public segment the same way, at the same challenge `r_j`.
/// It then reuses `build_monster_segments` and `run_sumcheck` unchanged.
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

	// Phase 1: build g once per key segment, then add them. The public words are
	// constants shared by every instance, so the single-instance builder folds them directly from
	// their bits; the hidden words are already folded over instances. This scalar path drives the
	// single-instance phase-1 sumcheck.
	let mut g = build_g::<F, F, _>(alloc, public_words, &key_collection.public, &prepared);
	let hidden_g =
		build_g_from_folded_words(alloc, folded_witness.words(), &key_collection.hidden, &prepared);
	for (slot, add) in iter::zip(g.as_mut(), hidden_g.as_ref()) {
		*slot += *add;
	}
	let h = build_h::<F, F, _>(alloc, domain_subspace, prepared.bitand.r_zhat_prime);
	let phase_1_output =
		run_phase_1_sumcheck::<F, F, _, _>(g, h, prepared.batched_eval(), channel, alloc);

	// Phase 2: split the phase-1 challenges into the bit position `r_j`, the shift amount `r_s`
	// and the shift variant `r_v`, in increasing order of significance.
	let SumcheckOutput {
		challenges: mut r_j,
		eval: gamma,
	} = phase_1_output;
	let r_v = r_j.split_off(Word::LOG_BITS * 2);
	let r_s = r_j.split_off(Word::LOG_BITS);
	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

	// The witness folded at `r_j`, per segment.
	// The public fold is a raw-word fold; the hidden fold contracts the already-oblong bits.
	let public_folded = fold_words::<F, P, _>(alloc, public_words, r_j_tensor.as_ref());
	let hidden_folded = folded_witness.fold_bits::<P>(r_j_tensor.as_ref(), alloc);

	let (public_monster, hidden_monster) = build_monster_segments::<F, P, _>(
		alloc,
		key_collection,
		&prepared,
		domain_subspace,
		&r_j,
		&r_s,
		&r_v,
	);

	run_sumcheck::<F, P, _, _>(
		&public_folded,
		hidden_folded,
		&public_monster,
		hidden_monster,
		public_words,
		r_j,
		gamma,
		channel,
		alloc,
	)
}

/// Constructs the phase-1 "g" multilinear parts, one per shift variant, from instance-folded words.
///
/// This is the batched analogue of the single-instance [`build_g`].
/// It consumes a key segment's words already folded over the instance axis.
/// So each word is a [`FoldedWord`], whose bits are field elements rather than a packed `u64`.
///
/// The single-instance builder scatters an accumulator onto a word's set bits by masking.
/// This one scales the accumulator by each folded bit with a field multiplication.
/// The two coincide when the folded bit is 0 or 1.
///
/// Use this for the hidden (committed) segment, whose words are folded over instances.
/// Use [`build_g`] for the public segment, whose words are shared constants.
/// Adding the two results gives the complete g multilinear.
///
/// # Arguments
///
/// - `folded_words`: the segment's words folded over instances, in `segment.key_ranges` order.
/// - `segment`: the shift keys of this segment, grouped by the word they act on.
/// - `prepared`: the per-operation claims, indexed by the operation each key names.
///
/// Words past the segment's key ranges are power-of-two padding, and are ignored.
///
/// # Returns
///
/// One multilinear of [`PHASE_1_LOG_LEN`] variables, holding every shift variant's part.
///
/// The segment's dense shift encoding decodes a key's shift index into the variant and the shift
/// amount. The variant indexes the high variables and the amount the middle ones, so a key names
/// one run of bit slots:
///
/// ```text
/// g[(variant * Word::BITS + amount) * Word::BITS + bit] += folded_bit * acc(key)
/// ```
///
/// where `acc(key)` is the key's lambda-weighted partial evaluation tensor.
/// Every word carrying that key accumulates into the same slots.
///
/// This scalar implementation ignores the single-instance builder's packing and parallelism.
pub fn build_g_from_folded_words<F: BinaryField, A: Allocator>(
	alloc: &A,
	folded_words: &[FoldedWord<F>],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
) -> FieldVec<F, A> {
	// One zeroed multilinear covering every shift, drawn from the allocator. A key names one
	// `(variant, amount)` pair, so the scatter below accumulates straight into it.
	let mut multilinear = FieldBuffer::zeros_in(alloc, PHASE_1_LOG_LEN);

	// Each folded word carries the keys named by the segment-relative range at its position.
	for (word, range) in folded_words.iter().zip(&segment.key_ranges) {
		let keys = &segment.keys[range.start as usize..range.end as usize];
		for key in keys {
			let operator_data = &prepared[key.operation];

			// The lambda-weighted partial evaluation tensor for this shifted word.
			let acc = key.accumulate(
				&segment.constraint_indices,
				operator_data.r_x_prime_tensor.as_ref(),
				&operator_data.lambda_powers,
			);

			// The key's shift variant indexes the high variables and its shift amount the middle
			// ones, which together name one run of bit slots.
			let (variant, amount) = segment.dense_shift_enc.decode(key.dense_shift_idx as usize);
			let base = (variant as usize * Word::BITS + amount as usize) * Word::BITS;
			let slots = &mut multilinear.as_mut()[base..base + Word::BITS];

			// Scatter the accumulator across this key's bit slots, scaling each by the folded bit.
			for (slot, &folded_bit) in iter::zip(slots, word) {
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
	use binius_core::{constraint_system::AndConstraint, word::Word};
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
		let key_collection = build_key_collection(&cs);

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
		let _offset = table.layout().offset_witness();
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
			&verifier_zero,
			&verifier_bitand,
			&verifier_intmul,
			&verifier_bmul,
			&mut verifier_transcript,
		)
		.unwrap();
		check_eval(
			&cs,
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
		let key_collection = build_key_collection(&cs);

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
		// The private values span value indices `[offset_witness, combined_len)`. The committed
		// rows can run past them into the protocol's zero padding, which this fixture has none of.
		let offset = table.layout().offset_witness();
		let combined = table.layout().combined_len();
		assert_eq!(table.n_hidden_words(), combined - offset, "fixture must have no padding");
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
		let mut g = build_g::<B128, B128, _>(
			&GlobalAllocator,
			public_words,
			&key_collection.public,
			&prepared,
		);
		let hidden_g = build_g_from_folded_words(
			&GlobalAllocator,
			&hidden_folded,
			&key_collection.hidden,
			&prepared,
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
}
