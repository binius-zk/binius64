// Copyright 2026 The Binius Developers

//! The M4 constraint reduction: the AND-check followed by the shift reduction, on one transcript.
//!
//! It composes the two batched reductions into one.
//! The result takes the constraint system to a single claim about the committed witness:
//!
//! 1. The AND-check reduces `A & B == C` over all rows to operand claims at a row point.
//! 2. That point splits into a constraint part `r_x` (low) and an instance part `r_rho` (high).
//! 3. The witness is folded over the instance axis at `r_rho`.
//! 4. The shift reduction reduces the operand claims to a single evaluation of the folded witness.
//!
//! The output is that witness claim.
//! Binding it to the committed trace oracle is a later step, not done here.
//! That tie binds `r_rho` and bridges the bit witness to the packed commitment.
//!
//! Only AND constraints are handled, so the circuit must have no MUL constraints.

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, Field, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::BinarySubspace;
use binius_prover::protocols::shift::{KeyCollection, OperatorData};
use binius_utils::checked_arithmetics::checked_log_2;
use binius_verifier::{config::B128, protocols::bitand::AndCheckOutput};

use crate::{
	BatchAndCheckWitness, ValueTable,
	shift::{FoldedWord, fold_instances, prove as prove_shift},
};

/// The prover's output of the M4 constraint reduction.
pub struct ReductionProverOutput {
	/// The instance-fold challenge: the high coordinates of the AND-check row point.
	pub r_rho: Vec<B128>,
	/// The reduced claim: the instance-folded witness evaluated at the shift's final point.
	pub witness_claim: SumcheckOutput<B128>,
}

/// Runs the AND-check and shift reduction over the batch witness on one transcript.
///
/// # Arguments
///
/// - `cs`: the prepared single-instance constraint system shared by every instance.
/// - `key_collection`: the shift keys for `cs`, built once in a setup phase and reused.
/// - `table`: the populated wire-major batch witness.
/// - `channel`: the prover channel recording messages and drawing Fiat-Shamir challenges.
///
/// # Panics
///
/// Panics if the constraint system has any MUL constraints, which this reduction does not handle.
pub fn prove_reduction<P, Channel>(
	cs: &ConstraintSystem,
	key_collection: &KeyCollection,
	table: &ValueTable,
	channel: &mut Channel,
) -> ReductionProverOutput
where
	P: PackedField<Scalar = B128>,
	Channel: IPProverChannel<B128>,
{
	assert!(
		cs.mul_constraints.is_empty(),
		"the M4 reduction handles only AND constraints; the circuit must have no MUL constraints"
	);

	// One base domain shared by the AND-check and the shift, consistent by construction.
	// The AND-check's univariate-skip domain spans one dimension above the 64-bit word.
	let andcheck_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1);
	// The shift domain drops that extra dimension.
	let shift_domain = andcheck_domain
		.reduce_dim(Word::LOG_BITS)
		.isomorphic::<B128>();

	// AND-check the `A & B == C` relation over all `K * n_and` rows.
	let and_witness = BatchAndCheckWitness::build(table, &cs.constants, &cs.and_constraints);
	let AndCheckOutput {
		a_eval,
		b_eval,
		c_eval,
		z_challenge,
		eval_point,
	} = and_witness.prove::<P, _>(&andcheck_domain, channel);

	// The row point is `r_x || r_rho`: the constraint index on the low coordinates, the instance
	// index on the high coordinates.
	let log_n_and = checked_log_2(cs.and_constraints.len());
	let (r_x, r_rho) = eval_point.split_at(log_n_and);

	// Fold the committed witness over the instance axis, then reshape into one folded word per
	// committed word.
	let folded = fold_instances::<B128, P>(table, r_rho);
	let scalars: Vec<B128> = folded.iter_scalars().collect();
	let folded_witness: Vec<FoldedWord<B128>> = scalars
		.chunks_exact(Word::BITS)
		.map(|chunk| chunk.try_into().expect("chunk has Word::BITS elements"))
		.collect();

	// Reduce the operand claims to one witness evaluation.
	// No MUL constraints here, so the intmul claim is a zero claim at an empty point.
	let witness_claim = prove_shift::<B128, P, _>(
		key_collection,
		&cs.constants,
		&folded_witness,
		OperatorData {
			evals: vec![a_eval, b_eval, c_eval],
			r_zhat_prime: z_challenge,
			r_x_prime: r_x.to_vec(),
		},
		OperatorData {
			evals: vec![B128::ZERO; 4],
			r_zhat_prime: z_challenge,
			r_x_prime: Vec::new(),
		},
		&shift_domain,
		channel,
	);

	ReductionProverOutput {
		r_rho: r_rho.to_vec(),
		witness_claim,
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_field::PackedBinaryGhash1x128b;
	use binius_frontend::CircuitBuilder;
	use binius_m4_verifier::verify_reduction;
	use binius_math::{inner_product::inner_product, multilinear::eq::eq_ind_partial_eval};
	use binius_prover::protocols::shift::build_key_collection;
	use binius_transcript::{ProverTranscript, VerifierTranscript};
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use binius_verifier::config::StdChallenger;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{N_INPUT_WORDS, crc64_circuit, populate_crc64_witness};

	type P = PackedBinaryGhash1x128b;

	// Evaluates the instance-folded witness directly at the shift's final point.
	// This is the independent reference the reduced claim is checked against.
	// The word axis is `r_y`: its low coordinates index the folded words.
	// Its high coordinates are padding, contributing `(1 - r_y_i)` factors.
	fn evaluate_folded_witness(folded: &[FoldedWord<B128>], r_j: &[B128], r_y: &[B128]) -> B128 {
		let log_folded = log2_ceil_usize(folded.len());
		let r_j_tensor = eq_ind_partial_eval::<B128>(r_j);
		let per_word: Vec<B128> = folded
			.iter()
			.map(|word| inner_product(word.iter().copied(), r_j_tensor.as_ref().iter().copied()))
			.collect();
		let r_y_tensor = eq_ind_partial_eval::<B128>(&r_y[..log_folded]);
		let base = inner_product(per_word.iter().copied(), r_y_tensor.as_ref().iter().copied());
		r_y[log_folded..]
			.iter()
			.fold(base, |acc, &r_y_i| acc * (B128::ONE - r_y_i))
	}

	// The prover and verifier compose the AND-check and shift reduction on one transcript.
	// They agree on the reduced claim.
	// That claim is the true instance-folded witness evaluated at the reduction's point.
	#[test]
	fn reduction_round_trips() {
		let c = crc64_circuit();

		let log_instances = 6;
		let n_instances = 1usize << log_instances;
		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();

		// Setup phase: build the shift keys once, to be reused across proofs.
		let key_collection = build_key_collection(&cs);

		// Prove the reduction on a fresh transcript.
		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let prover_out =
			prove_reduction::<P, _>(&cs, &key_collection, &table, &mut prover_transcript);

		// Verify by replaying the same transcript.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_out = verify_reduction(&cs, log_instances, &mut verifier_transcript).unwrap();
		verifier_transcript.finalize().unwrap();

		// Both sides drew the same instance challenge and reached the same reduced claim.
		assert_eq!(prover_out.r_rho, verifier_out.r_rho);
		let eval_point = [
			verifier_out.shift.r_j(),
			verifier_out.shift.r_y(),
			std::slice::from_ref(&verifier_out.shift.r_segment),
		]
		.concat();
		assert_eq!(prover_out.witness_claim.challenges, eval_point);
		assert_eq!(prover_out.witness_claim.eval, verifier_out.shift.witness_eval);

		// The reduced claim equals the instance-folded witness evaluated at the shift point.
		let folded = fold_instances::<B128, P>(&table, &verifier_out.r_rho);
		let scalars: Vec<B128> = folded.iter_scalars().collect();
		let folded_witness: Vec<FoldedWord<B128>> = scalars
			.chunks_exact(Word::BITS)
			.map(|chunk| chunk.try_into().unwrap())
			.collect();
		let expected = evaluate_folded_witness(
			&folded_witness,
			verifier_out.shift.r_j(),
			verifier_out.shift.r_y(),
		);
		assert_eq!(expected, verifier_out.shift.witness_eval);
	}

	// A tampered transcript must be rejected somewhere in the composed reduction.
	#[test]
	fn tampered_transcript_is_rejected() {
		let c = crc64_circuit();

		let log_instances = 6;
		let n_instances = 1usize << log_instances;
		let mut rng = StdRng::seed_from_u64(1);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		let key_collection = build_key_collection(&cs);

		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let _ = prove_reduction::<P, _>(&cs, &key_collection, &table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip a bit in the AND-check's first message.
		// The verifier then redraws a diverging challenge and the reduction no longer checks out.
		proof[0] ^= 1;

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		match verify_reduction(&cs, log_instances, &mut verifier_transcript) {
			Err(_) => {}
			Ok(_) => {
				verifier_transcript
					.finalize()
					.expect_err("a tampered proof must not verify and finalize cleanly");
			}
		}
	}

	#[test]
	#[should_panic(
		expected = "the M4 reduction handles only AND constraints; the circuit must have no MUL constraints"
	)]
	fn verifier_rejects_mul_constraints_before_reading_transcript() {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let y = builder.add_witness();
		let (hi, lo) = builder.smul(x, y);
		builder.force_commit(hi);
		builder.force_commit(lo);
		let circuit = builder.build();

		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), Vec::new());
		let _ = verify_reduction(&cs, 0, &mut verifier_transcript);
	}
}
