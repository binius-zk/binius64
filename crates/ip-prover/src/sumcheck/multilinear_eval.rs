// Copyright 2026 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, FieldSlice};

use super::{
	mle_store::{ColId, EvaluationChunk, MleStore},
	round_evals::RoundEvals1,
	round_evaluator::{MleCheckRoundEvaluator, SharedMleCheckProver},
};

/// MLE-check round evaluator for the multilinear evaluation of one store column.
///
/// The composition is the identity.
/// Each round polynomial is therefore degree 1, with a single sampled evaluation.
/// That evaluation is the inner product of the column's `X = 1` half with the round's eq chunk.
/// The driving [`SharedMleCheckProver`] supplies that chunk.
/// It folds the higher eq coordinates through its own reduction step.
/// No full-width eq tensor is ever materialized, streamed, or folded.
pub struct MultilinearEvalEvaluator {
	col: ColId,
}

impl MultilinearEvalEvaluator {
	/// Creates an evaluator over the store column `col`.
	pub const fn new(col: ColId) -> Self {
		Self { col }
	}
}

impl<F, P> MleCheckRoundEvaluator<F, P> for MultilinearEvalEvaluator
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn degree(&self) -> usize {
		// Identity composition: the round polynomial is degree 1.
		// One sampled evaluation suffices.
		1
	}

	fn accumulate(
		&self,
		chunk: &EvaluationChunk<'_, P>,
		eq_ind: FieldSlice<'_, P>,
		accum: &mut [<P as WideMul>::Output],
	) {
		// The column arrives split on the round's highest variable.
		// Its high half is the specialization at `X = 1`.
		let col = chunk.col(self.col);

		// R(1) = <M(.., X = 1), eq(.., z)> over this chunk.
		// Only the eq multiply is widened.
		// The wide accumulator is reduced once at the end of the chunk.
		let mut y_1 = <P as WideMul>::Output::default();
		for (idx, &eq_i) in eq_ind.as_ref().iter().enumerate() {
			y_1 += P::wide_mul(col.hi.as_ref()[idx], eq_i);
		}
		accum[0] += y_1;
	}

	fn interpolate(
		&self,
		store: &MleStore<'_, P>,
		accum: &[P],
		claim: F,
		alpha: F,
	) -> RoundCoeffs<F> {
		// The store has not folded this round yet.
		// Its remaining-variable count is therefore this round's.
		let n_vars_remaining = store.n_vars();
		assert!(n_vars_remaining > 0);

		// `accum` is already reduced by the prover's map pass.
		// Sum its lanes, then interpolate.
		// `claim` is this round's prime evaluation.
		// `alpha` is the eq coordinate that ties it to the point.
		RoundEvals1 { y_1: accum[0] }
			.sum_scalars(n_vars_remaining)
			.interpolate_eq(claim, alpha)
	}
}

/// Builds an MLE-check prover for the multilinear extension evaluation of a single multilinear.
///
/// The claim is `M(z) = s` for a multilinear `M` over the challenge field.
/// This proves the equivalent MLE-check relation `s = sum_{v in B_n} M(v) * eq(v, z)`.
/// Since `M` is multilinear, that relation holds if and only if `M(z) = s`.
///
/// The reduction runs on the split-eq [`SharedMleCheckProver`] with a degree-1 evaluator.
/// Each round expands only a small low-coordinate prefix of the eq indicator.
/// The higher coordinates are folded in through the prover's reduction step.
/// The full `2^{n-1}` eq tensor is never materialized, streamed, or folded per round.
///
/// # Arguments
///
/// * `witness` - the multilinear whose extension is evaluated.
/// * `eval_point` - the point of the evaluation claim.
/// * `eval_claim` - the claimed value of the multilinear extension at that point.
///
/// # Panics
///
/// Panics if the witness length does not match the evaluation point length.
pub fn multilinear_eval_prover<F, P>(
	witness: FieldBuffer<P>,
	eval_point: &[F],
	eval_claim: F,
) -> SharedMleCheckProver<'static, F, P, MultilinearEvalEvaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert_eq!(
		witness.log_len(),
		eval_point.len(),
		"witness must have number of variables equal to the evaluation point length"
	);

	// The store owns the witness as its single column.
	// With no borrowed data, the shared prover is `'static`.
	let mut store = MleStore::new(eval_point.len());
	let col = store.push_owned(witness);
	let evaluator = MultilinearEvalEvaluator::new(col);
	SharedMleCheckProver::new(store, [(eval_claim, evaluator)], eval_point.to_vec())
}

#[cfg(test)]
mod tests {
	use binius_field::{
		FieldOps, Random,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_ip::mlecheck;
	use binius_math::{
		multilinear::evaluate::evaluate,
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{
		common::SumcheckProver, prove_single_mlecheck,
		quadratic_mle_evaluator::quadratic_mlecheck_prover,
	};

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// A quadratic MLE-check with the identity composition and a zero infinity composition is a
	// degree-1 MLE-check over a single multilinear.
	// That is exactly the single-multilinear evaluation reduction under test, with an always-zero
	// degree-2 coefficient tacked on.
	// Drive both in lockstep and compare round polynomials and final evaluations.
	#[test]
	fn test_conformance_with_quadratic_mlecheck() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 8;

		let witness = random_field_buffer::<P>(&mut rng, n_vars);
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&witness, &eval_point);

		let mut eval_prover = multilinear_eval_prover(witness.clone(), &eval_point, eval_claim);
		let mut quadratic_prover = quadratic_mlecheck_prover(
			[witness],
			|[a]: [P; 1]| a,
			|[_a]: [P; 1]| P::zero(),
			eval_point,
			eval_claim,
		);

		for _ in 0..n_vars {
			let eval_round = eval_prover.execute();
			let mut quadratic_round = quadratic_prover.execute();
			assert_eq!(eval_round.len(), 1);
			assert_eq!(quadratic_round.len(), 1);

			// The quadratic prover sizes its round polynomial for degree 2; the leading coefficient
			// is zero because the composition is multilinear.
			assert_eq!(quadratic_round[0].0.pop(), Some(F::ZERO));
			assert_eq!(eval_round[0], quadratic_round[0]);

			// `round_claim` must agree across both provers and be stable across execute.
			assert_eq!(eval_prover.round_claim(), quadratic_prover.round_claim());

			let challenge = F::random(&mut rng);
			eval_prover.fold(challenge);
			quadratic_prover.fold(challenge);
		}

		assert_eq!(eval_prover.finish(), quadratic_prover.finish());
	}

	// Full prove/verify roundtrip through the MLE-check protocol.
	#[test]
	fn test_prove_verify_roundtrip() {
		let mut rng = StdRng::seed_from_u64(1);
		let n_vars = 7;

		let witness = random_field_buffer::<P>(&mut rng, n_vars);
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&witness, &eval_point);

		let prover = multilinear_eval_prover(witness.clone(), &eval_point, eval_claim);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript);
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output =
			mlecheck::verify::<F, _>(&eval_point, 1, eval_claim, &mut verifier_transcript).unwrap();
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(1).unwrap();

		assert_eq!(output.challenges, sumcheck_output.challenges);

		// The reduced MLE-check evaluation is the witness multilinear at the challenge point.
		assert_eq!(multilinear_evals[0], sumcheck_output.eval);

		let mut reduced_point = sumcheck_output.challenges;
		reduced_point.reverse();
		assert_eq!(evaluate(&witness, &reduced_point), multilinear_evals[0]);
	}

	// `round_claim` must return the same value before and after `execute` (lerp recovery), and the
	// post-fold claim must equal the round polynomial evaluated at the challenge.
	#[test]
	fn test_round_claim_invariant() {
		let mut rng = StdRng::seed_from_u64(2);
		let n_vars = 6;

		let witness = random_field_buffer::<P>(&mut rng, n_vars);
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&witness, &eval_point);

		let mut prover = multilinear_eval_prover(witness, &eval_point, eval_claim);
		assert_eq!(prover.round_claim(), vec![eval_claim]);

		for _ in 0..n_vars {
			let before = prover.round_claim();
			let round = prover.execute();
			assert_eq!(prover.round_claim(), before);
			let challenge = F::random(&mut rng);
			let expected_next = round[0].evaluate(challenge);
			prover.fold(challenge);
			assert_eq!(prover.round_claim(), vec![expected_next]);
		}
	}
}
