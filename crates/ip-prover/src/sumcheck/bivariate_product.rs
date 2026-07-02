// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;

use crate::sumcheck::{common::SumcheckProver, round_evals::WideRoundEvals2};

/// A [`SumcheckProver`] implementation for a composite defined as the product of two multilinears.
///
/// This prover binds variables in high-to-low index order.
///
/// ## Invariants
///
/// - The length of the two multilinears is always equal
#[derive(Debug)]
pub struct BivariateProductSumcheckProver<P: PackedField> {
	multilinears: [FieldBuffer<P>; 2],
	last_coeffs_or_sum: RoundCoeffsOrSum<P::Scalar>,
}

impl<F: Field, P: PackedField<Scalar = F>> BivariateProductSumcheckProver<P> {
	/// Constructs a prover, given the multilinear polynomial evaluations and the sum over the
	/// boolean hypercube of their product.
	pub fn new(multilinears: [FieldBuffer<P>; 2], sum: F) -> Self {
		assert_eq!(
			multilinears[0].log_len(),
			multilinears[1].log_len(),
			"multilinears must have equal number of variables"
		);

		Self {
			multilinears,
			last_coeffs_or_sum: RoundCoeffsOrSum::Sum(sum),
		}
	}
}

impl<F: Field, P: PackedField<Scalar = F>> SumcheckProver<F> for BivariateProductSumcheckProver<P> {
	fn n_vars(&self) -> usize {
		self.multilinears[0].log_len()
	}

	fn n_claims(&self) -> usize {
		1
	}

	fn round_claim(&self) -> Vec<F> {
		let claim = match &self.last_coeffs_or_sum {
			RoundCoeffsOrSum::Sum(sum) => *sum,
			RoundCoeffsOrSum::Coeffs(coeffs) => coeffs.sum_over_endpoints(),
		};
		vec![claim]
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let RoundCoeffsOrSum::Sum(last_sum) = &self.last_coeffs_or_sum else {
			panic!("execute called out of order; expected fold");
		};

		// Multilinear inputs are the same length by invariant
		debug_assert_eq!(self.multilinears[0].len(), self.multilinears[1].len());

		let n_vars_remaining = self.n_vars();
		assert!(n_vars_remaining > 0);

		let (evals_a_0, evals_a_1) = self.multilinears[0].split_half_ref();
		let (evals_b_0, evals_b_1) = self.multilinears[1].split_half_ref();

		// Compute F(1) and F(∞) where F = ∑_{v ∈ B} A(v || X) B(v || X).
		//
		// The per-point products are accumulated in unreduced (wide) form and reduced a single
		// time at the end, amortizing the GF(2^128) reduction over the whole sum.
		let round_evals =
			(evals_a_0.as_ref(), evals_a_1.as_ref(), evals_b_0.as_ref(), evals_b_1.as_ref())
				.into_par_iter()
				.map(|(&evals_a_0_i, &evals_a_1_i, &evals_b_0_i, &evals_b_1_i)| {
					// Evaluate M(∞) = M(0) + M(1)
					let evals_a_inf_i = evals_a_0_i + evals_a_1_i;
					let evals_b_inf_i = evals_b_0_i + evals_b_1_i;

					WideRoundEvals2 {
						y_1: P::wide_mul(evals_a_1_i, evals_b_1_i),
						y_inf: P::wide_mul(evals_a_inf_i, evals_b_inf_i),
					}
				})
				.reduce(WideRoundEvals2::default, |lhs, rhs| lhs + rhs);

		let round_coeffs = round_evals
			.reduce::<P>()
			.sum_scalars(n_vars_remaining)
			.interpolate(*last_sum);
		self.last_coeffs_or_sum = RoundCoeffsOrSum::Coeffs(round_coeffs.clone());
		vec![round_coeffs]
	}

	fn fold(&mut self, challenge: F) {
		let RoundCoeffsOrSum::Coeffs(last_coeffs) = self.last_coeffs_or_sum.clone() else {
			panic!("fold called out of order; expected execute");
		};

		for multilin in &mut self.multilinears {
			fold_highest_var_inplace(multilin, challenge);
		}

		let round_sum = last_coeffs.evaluate(challenge);
		self.last_coeffs_or_sum = RoundCoeffsOrSum::Sum(round_sum);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars(), 0, "finish called out of order; sumcheck rounds remain");

		self.multilinears
			.into_iter()
			.map(|multilinear| multilinear.get(0))
			.collect()
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrSum<F: Field> {
	Coeffs(RoundCoeffs<F>),
	Sum(F),
}

#[cfg(test)]
mod tests {
	use binius_field::arch::{OptimalB128, OptimalPackedB128};
	use binius_ip::sumcheck::verify;
	use binius_math::{
		inner_product::inner_product_par, multilinear::evaluate::evaluate,
		test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::prove::prove_single;

	#[test]
	fn test_bivariate_product_sumcheck() {
		type F = OptimalB128;
		type P = OptimalPackedB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate two random multilinear polynomials
		let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
		let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);

		// Compute the expected sum: sum_{x in {0,1}^n} A(x) * B(x)
		let expected_sum = inner_product_par(&multilinear_a, &multilinear_b);

		// Create the prover
		let prover = BivariateProductSumcheckProver::new(
			[multilinear_a.clone(), multilinear_b.clone()],
			expected_sum,
		);

		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript);

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify(
			n_vars,
			2, // degree 2 for bivariate product
			expected_sum,
			&mut verifier_transcript,
		)
		.unwrap();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

		// Check that the product of the evaluations equals the reduced evaluation
		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1],
			sumcheck_output.eval,
			"Product of multilinear evaluations should equal the reduced evaluation"
		);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point The prover binds variables from high to low, but evaluate expects them from low
		// to high
		let mut eval_point = sumcheck_output.challenges.clone();
		eval_point.reverse();
		let eval_a = evaluate(&multilinear_a, &eval_point);
		let eval_b = evaluate(&multilinear_b, &eval_point);

		assert_eq!(
			eval_a, multilinear_evals[0],
			"Multilinear A should evaluate to the first claimed evaluation"
		);
		assert_eq!(
			eval_b, multilinear_evals[1],
			"Multilinear B should evaluate to the second claimed evaluation"
		);

		// Also verify the challenges match what the prover saw
		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}
}
