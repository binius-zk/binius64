// Copyright 2026 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;

use super::{
	common::{MleCheckProver, SumcheckProver},
	gruen32::Gruen32,
	round_evals::RoundEvals1,
	round_state::RoundState,
};

/// An [`MleCheckProver`] for the multilinear extension evaluation of a single multilinear
/// polynomial over the challenge field.
///
/// Given a multilinear polynomial $M$ over the field $F$ and a claim $M(z) = s$, this proves the
/// MLE-check relation $s = \sum_{v \in B_n} M(v) \cdot \text{eq}(v, z)$ (which, since $M$ is
/// multilinear, holds iff $M(z) = s$). It handles a single multilinear over the large field $F$.
///
/// Because $M$ is multilinear, each round polynomial is degree 1. The prover folds the witness with
/// each challenge, and in each round computes the round polynomial's evaluation at 1 by taking the
/// top half of the folded witness (the partial specialization at the highest variable = 1) and
/// dotting it with the [`Gruen32`] equality-indicator expansion over the remaining variables.
#[derive(Debug, Clone)]
pub struct MultilinearEvalProver<P: PackedField> {
	witness: FieldBuffer<P>,
	gruen32: Gruen32<P>,
	last_coeffs_or_sum: RoundState<RoundCoeffs<P::Scalar>, P::Scalar>,
}

impl<F: Field, P: PackedField<Scalar = F>> MultilinearEvalProver<P> {
	/// Constructs a prover for the multilinear `witness`, given the evaluation point `eval_point`
	/// and the claimed evaluation `eval_claim` of the multilinear extension at that point.
	///
	/// Panics if the witness length does not match the evaluation point length.
	pub fn new(witness: FieldBuffer<P>, eval_point: &[F], eval_claim: F) -> Self {
		assert_eq!(
			witness.log_len(),
			eval_point.len(),
			"witness must have number of variables equal to the evaluation point length"
		);

		Self {
			witness,
			gruen32: Gruen32::new(eval_point),
			last_coeffs_or_sum: RoundState::Claim(eval_claim),
		}
	}
}

impl<F: Field, P: PackedField<Scalar = F>> SumcheckProver<F> for MultilinearEvalProver<P> {
	fn n_vars(&self) -> usize {
		self.gruen32.n_vars_remaining()
	}

	fn n_claims(&self) -> usize {
		1
	}

	fn round_claim(&self) -> Vec<F> {
		let claim = match &self.last_coeffs_or_sum {
			RoundState::Claim(sum) => *sum,
			RoundState::Coeffs(coeffs) => {
				coeffs.lerp_over_endpoints(self.gruen32.next_coordinate())
			}
		};
		vec![claim]
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let sum = *self.last_coeffs_or_sum.claim();

		let n_vars_remaining = self.n_vars();
		assert!(n_vars_remaining > 0);

		// The eq expansion is over the lower `n_vars_remaining - 1` variables; the top (X = 1) half
		// of the witness is the partial specialization of the highest variable at 1.
		let eq_expansion = self.gruen32.eq_expansion();
		let (_evals_0, evals_1) = self.witness.split_half_ref();
		debug_assert_eq!(eq_expansion.log_len(), evals_1.log_len());

		// R(1) = <M(.., X = 1), eq(.., z)>, the multilinear evaluation of the top half.
		// The products are accumulated in unreduced (wide) form and reduced once at the end.
		let wide_y_1 = (evals_1.as_ref(), eq_expansion.as_ref())
			.into_par_iter()
			.map(|(&evals_1_i, &eq_i)| P::wide_mul(evals_1_i, eq_i))
			.reduce(<P as WideMul>::Output::default, |lhs, rhs| lhs + rhs);
		let round_evals = RoundEvals1 {
			y_1: P::reduce(wide_y_1),
		};

		let alpha = self.gruen32.next_coordinate();
		let round_coeffs = round_evals
			.sum_scalars(n_vars_remaining)
			.interpolate_eq(sum, alpha);

		self.last_coeffs_or_sum = RoundState::Coeffs(round_coeffs.clone());
		vec![round_coeffs]
	}

	fn fold(&mut self, challenge: F) {
		let coeffs = self.last_coeffs_or_sum.coeffs();

		assert!(self.n_vars() > 0);

		let sum = coeffs.evaluate(challenge);

		fold_highest_var_inplace(&mut self.witness, challenge);
		self.gruen32.fold(challenge);

		self.last_coeffs_or_sum = RoundState::Claim(sum);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars(), 0, "finish called out of order; sumcheck rounds remain");

		debug_assert_eq!(self.witness.log_len(), 0);
		vec![self.witness.get(0)]
	}
}

impl<F: Field, P: PackedField<Scalar = F>> MleCheckProver<F> for MultilinearEvalProver<P> {
	fn eval_point(&self) -> &[F] {
		self.gruen32.eval_point()
	}
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
	use crate::sumcheck::{prove_single_mlecheck, quadratic_mle::QuadraticMleCheckProver};

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// A `QuadraticMleCheckProver` with the identity composition (and zero infinity composition) is
	// a degree-1 MLE-check over a single multilinear — exactly what `MultilinearEvalProver`
	// computes, just with the high (always-zero) degree-2 coefficient included. Drive both in
	// lockstep and compare round polynomials and final evaluations.
	#[test]
	fn test_conformance_with_quadratic_mlecheck() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 8;

		let witness = random_field_buffer::<P>(&mut rng, n_vars);
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&witness, &eval_point);

		let mut eval_prover = MultilinearEvalProver::new(witness.clone(), &eval_point, eval_claim);
		let mut quadratic_prover = QuadraticMleCheckProver::new(
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

		let prover = MultilinearEvalProver::new(witness.clone(), &eval_point, eval_claim);

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

		let mut prover = MultilinearEvalProver::new(witness, &eval_point, eval_claim);
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
