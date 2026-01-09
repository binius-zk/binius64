// Copyright 2026 The Binius Developers

//! Prover for the Libra mask polynomial in ZK MLE-check protocols.
//!
//! The Libra ZK-sumcheck protocol uses a masking polynomial g(X_0, ..., X_{n-1}) of the form:
//! g = sum_{i=0}^{n-1} g_i(X_i)
//!
//! where each g_i(X) is a univariate polynomial of configurable degree. This separable structure
//! allows efficient computation of round polynomials without iterating over the full hypercube.

use std::iter;

use binius_field::Field;
use binius_math::{line::extrapolate_line_packed, univariate::evaluate_univariate};
use binius_verifier::protocols::sumcheck::RoundCoeffs;

use super::{
	Error,
	common::{MleCheckProver, SumcheckProver},
};

/// Prover for the Libra mask polynomial in ZK MLE-check.
///
/// The mask polynomial has the separable form $g(X_0, ..., X_{n-1}) = sum_{i} g_i(X_i)$,
/// where each $g_i$ is a univariate polynomial of configurable degree.
///
/// This structure allows efficient round polynomial computation in O(degree) time per round.
pub struct MleCheckMaskProver<F: Field> {
	/// Univariate polynomial coefficients for each g_i. Shape: [n_vars][degree+1]
	coefficients: Vec<Vec<F>>,
	/// The evaluation point z (in high-to-low variable order)
	eval_point: Vec<F>,
	/// Number of variables remaining to process
	n_vars_remaining: usize,
	/// Accumulated sum of g_j(r_j) for already-folded variables
	prefix_sum: F,
	/// Precomputed (1-z_j)*g_j(0) + z_j*g_j(1) for each variable
	suffix_sums: Vec<F>,
	/// State: either last round coefficients (after execute) or current claim (after fold)
	last_coeffs_or_claim: RoundCoeffsOrClaim<F>,
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrClaim<F: Field> {
	Coeffs(RoundCoeffs<F>),
	Claim(F),
}

impl<F: Field> MleCheckMaskProver<F> {
	/// Creates a new prover for the Libra mask polynomial.
	///
	/// # Arguments
	///
	/// * `coefficients` - Univariate polynomial coefficients for each variable. The outer Vec
	///   has length n_vars, and each inner Vec contains coefficients [a_0, a_1, ..., a_d] for
	///   a polynomial g_i(X) = a_0 + a_1*X + ... + a_d*X^d. All inner Vecs must have the same
	///   length (degree + 1).
	/// * `eval_point` - The evaluation point z for the MLE-check claim, in high-to-low order.
	/// * `eval_claim` - The claimed value of the MLE of g at the evaluation point.
	///
	/// # Panics
	///
	/// Panics if `coefficients.len() != eval_point.len()` or if the inner coefficient vectors
	/// don't all have the same length.
	pub fn new(coefficients: Vec<Vec<F>>, eval_point: Vec<F>, eval_claim: F) -> Self {
		assert_eq!(
			coefficients.len(),
			eval_point.len(),
			"coefficients length must match eval_point length"
		);

		let degree_plus_one = coefficients.first().map_or(0, Vec::len);
		for (i, coeffs) in coefficients.iter().enumerate() {
			assert_eq!(
				coeffs.len(),
				degree_plus_one,
				"coefficient vector {i} has length {}, expected {degree_plus_one}",
				coeffs.len()
			);
		}

		let n_vars = eval_point.len();

		// Precompute suffix_sums[j] = (1-z_j)*g_j(0) + z_j*g_j(1)
		// This equals extrapolate_line_packed(g_j(0), g_j(1), z_j)
		let suffix_sums: Vec<F> = iter::zip(&coefficients, &eval_point)
			.map(|(coeffs, &z_j)| {
				let g_at_0 = coeffs.first().copied().unwrap_or(F::ZERO);
				let g_at_1 = evaluate_univariate(coeffs, F::ONE);
				extrapolate_line_packed(g_at_0, g_at_1, z_j)
			})
			.collect();

		Self {
			coefficients,
			eval_point,
			n_vars_remaining: n_vars,
			prefix_sum: F::ZERO,
			suffix_sums,
			last_coeffs_or_claim: RoundCoeffsOrClaim::Claim(eval_claim),
		}
	}

	/// Returns the index of the current variable being processed.
	/// Processing is high-to-low, so we start at n_vars-1 and decrease.
	fn current_var_index(&self) -> usize {
		self.n_vars_remaining - 1
	}

	/// Evaluates the univariate g_i at a point.
	fn evaluate_univariate_at(&self, var_index: usize, x: F) -> F {
		evaluate_univariate(&self.coefficients[var_index], x)
	}
}

impl<F: Field> SumcheckProver<F> for MleCheckMaskProver<F> {
	fn n_vars(&self) -> usize {
		self.n_vars_remaining
	}

	fn n_claims(&self) -> usize {
		1
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let RoundCoeffsOrClaim::Claim(_claim) = &self.last_coeffs_or_claim else {
			return Err(Error::ExpectedFold);
		};

		if self.n_vars_remaining == 0 {
			return Err(Error::ExpectedFinish);
		}

		let var_idx = self.current_var_index();

		// Compute suffix sum for variables that haven't been processed yet (indices 0 to var_idx-1)
		// Since we process high-to-low (n-1, n-2, ..., 0), the suffix is the lower-indexed variables
		let suffix_sum: F = self.suffix_sums[..var_idx].iter().copied().sum();

		// Compute the constant offset: prefix_sum + suffix_sum
		let constant_offset = self.prefix_sum + suffix_sum;

		// Build the round polynomial R(X) = g_i(X) + constant_offset
		// g_i(X) = sum_{k=0}^{d} a_{i,k} * X^k
		// So coefficients are: [a_0 + offset, a_1, a_2, ..., a_d]
		let g_i_coeffs = &self.coefficients[var_idx];
		let mut round_coeffs_vec: Vec<F> = g_i_coeffs.clone();
		if round_coeffs_vec.is_empty() {
			round_coeffs_vec.push(constant_offset);
		} else {
			round_coeffs_vec[0] += constant_offset;
		}

		let round_coeffs = RoundCoeffs(round_coeffs_vec);
		self.last_coeffs_or_claim = RoundCoeffsOrClaim::Coeffs(round_coeffs.clone());
		Ok(vec![round_coeffs])
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		let RoundCoeffsOrClaim::Coeffs(coeffs) = &self.last_coeffs_or_claim else {
			return Err(Error::ExpectedExecute);
		};

		// Evaluate round polynomial at challenge to get new claim
		let new_claim = coeffs.evaluate(challenge);

		let var_idx = self.current_var_index();

		// Update prefix_sum: add g_i(r_i)
		self.prefix_sum += self.evaluate_univariate_at(var_idx, challenge);

		self.n_vars_remaining -= 1;
		self.last_coeffs_or_claim = RoundCoeffsOrClaim::Claim(new_claim);

		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		if self.n_vars_remaining > 0 {
			return match self.last_coeffs_or_claim {
				RoundCoeffsOrClaim::Coeffs(_) => Err(Error::ExpectedFold),
				RoundCoeffsOrClaim::Claim(_) => Err(Error::ExpectedExecute),
			};
		}

		// Final evaluation of g at the challenge point is prefix_sum
		// (since g(r_0, ..., r_{n-1}) = sum_i g_i(r_i))
		Ok(vec![self.prefix_sum])
	}
}

impl<F: Field> MleCheckProver<F> for MleCheckMaskProver<F> {
	fn eval_point(&self) -> &[F] {
		// Return remaining coordinates (high-to-low means we return the first n_vars_remaining elements)
		&self.eval_point[..self.n_vars_remaining]
	}
}

#[cfg(test)]
mod tests {
	use binius_field::arch::OptimalB128;
	use binius_math::test_utils::random_scalars;
	use binius_transcript::ProverTranscript;
	use binius_verifier::{config::StdChallenger, protocols::mlecheck};
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;
	use crate::protocols::sumcheck::prove_single_mlecheck;

	type B128 = OptimalB128;

	/// Evaluates the mask polynomial g(X) = sum_i g_i(X_i) at a point.
	fn evaluate_mask_polynomial<F: Field>(coefficients: &[Vec<F>], point: &[F]) -> F {
		iter::zip(coefficients, point)
			.map(|(coeffs, &x)| evaluate_univariate(coeffs, x))
			.sum()
	}

	/// Computes the MLE of the mask polynomial at a point.
	///
	/// The MLE is sum_{v in {0,1}^n} g(v) * eq(v, z).
	fn compute_mask_mle<F: Field>(coefficients: &[Vec<F>], eval_point: &[F]) -> F {
		// Due to the separable structure of g, the MLE simplifies:
		// sum_{v} g(v) * eq(v, z) = sum_i sum_{v_i} g_i(v_i) * eq_1(v_i, z_i) * prod_{j!=i}
		// sum_{v_j} eq_1(v_j, z_j)
		//
		// Since sum_{v_j in {0,1}} eq_1(v_j, z_j) = 1, this simplifies to:
		// sum_i [(1-z_i)*g_i(0) + z_i*g_i(1)]

		iter::zip(coefficients, eval_point)
			.map(|(coeffs, &z_i)| {
				let g_at_0 = coeffs.first().copied().unwrap_or(F::ZERO);
				let g_at_1 = evaluate_univariate(coeffs, F::ONE);
				extrapolate_line_packed(g_at_0, g_at_1, z_i)
			})
			.sum()
	}

	fn test_mask_prover_with_degree(degree: usize) {
		let n_vars = 6;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random mask coefficients
		let coefficients: Vec<Vec<B128>> = (0..n_vars)
			.map(|_| random_scalars(&mut rng, degree + 1))
			.collect();

		// Generate random evaluation point
		let eval_point: Vec<B128> = random_scalars(&mut rng, n_vars);

		// Compute the MLE of the mask polynomial at eval_point
		let eval_claim = compute_mask_mle(&coefficients, &eval_point);

		// Create the prover
		let prover =
			MleCheckMaskProver::new(coefficients.clone(), eval_point.clone(), eval_claim);

		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript).unwrap();

		// Write the multilinear evaluation to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = mlecheck::verify::<B128, _>(
			&eval_point,
			degree, // round polynomial degree equals the univariate g_i degree
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		// Read the mask evaluation from the transcript
		let mask_eval_out: B128 = verifier_transcript.message().read().unwrap();

		// Verify the reduced evaluation equals the composition of the evaluations
		// The mask polynomial is the single "multilinear" here, so its eval should match
		assert_eq!(mask_eval_out, sumcheck_output.eval);

		// Compute the challenge point (reverse for high-to-low order)
		let mut challenge_point = sumcheck_output.challenges.clone();
		challenge_point.reverse();

		// Check that the final evaluation matches direct computation
		let expected_eval = evaluate_mask_polynomial(&coefficients, &challenge_point);
		assert_eq!(output.multilinear_evals[0], expected_eval);
	}

	#[test]
	fn test_linear_mask() {
		test_mask_prover_with_degree(1);
	}

	#[test]
	fn test_quadratic_mask() {
		test_mask_prover_with_degree(2);
	}

	#[test]
	fn test_cubic_mask() {
		test_mask_prover_with_degree(3);
	}

	#[test]
	fn test_single_variable() {
		let mut rng = StdRng::seed_from_u64(0);

		// Single variable mask
		let coefficients: Vec<Vec<B128>> = vec![random_scalars(&mut rng, 3)];
		let eval_point: Vec<B128> = random_scalars(&mut rng, 1);
		let eval_claim = compute_mask_mle(&coefficients, &eval_point);

		let prover =
			MleCheckMaskProver::new(coefficients.clone(), eval_point.clone(), eval_claim);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript).unwrap();

		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output =
			mlecheck::verify::<B128, _>(&eval_point, 2, eval_claim, &mut verifier_transcript)
				.unwrap();

		let mut challenge_point = sumcheck_output.challenges.clone();
		challenge_point.reverse();

		let expected_eval = evaluate_mask_polynomial(&coefficients, &challenge_point);
		assert_eq!(output.multilinear_evals[0], expected_eval);
	}
}
