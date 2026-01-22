// Copyright 2026 The Binius Developers

//! Prover for the Libra mask polynomial in ZK MLE-check protocols.
//!
//! The Libra ZK-sumcheck protocol uses a masking polynomial g(X_0, ..., X_{n-1}) of the form:
//! g = sum_{i=0}^{n-1} g_i(X_i)
//!
//! where each g_i(X) is a univariate polynomial of configurable degree. This separable structure
//! allows efficient computation of round polynomials without iterating over the full hypercube.

use std::iter;

use binius_field::{Field, PackedField};
use binius_math::{
	field_buffer::FieldBuffer, line::extrapolate_line_packed, univariate::evaluate_univariate,
};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::{mlecheck, sumcheck::RoundCoeffs};
use rand::CryptoRng;

use super::{
	Error, ProveSingleOutput,
	common::{MleCheckProver, SumcheckProver},
};

/// Libra mask polynomial for ZK MLE-check protocols.
///
/// Stores coefficients for a separable polynomial $g(X) = \sum_i g_i(X_i)$
/// where each $g_i$ is a univariate polynomial of degree $d$.
///
/// The coefficients are stored in a `FieldBuffer` with `m_n + m_d` variables where:
/// - `m_n = ceil(log2(n))` - log of number of variables
/// - `m_d = ceil(log2(d + 1))` - log of degree + 1
///
/// The buffer is conceptually an `n × (d+1)` matrix padded to `2^m_n × 2^m_d`,
/// with random values in the `n × (d+1)` submatrix and zeros elsewhere.
pub struct Mask<P: PackedField> {
	/// Number of variables (n)
	n_vars: usize,
	/// Degree of each univariate polynomial (d)
	degree: usize,
	/// Coefficients stored as a FieldBuffer with log_len = m_n + m_d.
	/// Layout: row i contains [g_i(0), g_i(1), ..., g_i(d), 0, ..., 0]
	/// where row i spans indices [i * 2^m_d, (i+1) * 2^m_d).
	buffer: FieldBuffer<P>,
}

impl<F: Field, P: PackedField<Scalar = F>> Mask<P> {
	/// Creates a new random mask polynomial.
	///
	/// # Arguments
	///
	/// * `n_vars` - Number of variables (n). Must be > 0.
	/// * `degree` - Degree of each univariate polynomial (d). Must be > 0.
	/// * `rng` - Cryptographic random number generator.
	pub fn random(n_vars: usize, degree: usize, mut rng: impl CryptoRng) -> Self {
		let m_n = n_vars.next_power_of_two().ilog2() as usize;
		let m_d = (degree + 1).next_power_of_two().ilog2() as usize;
		let log_len = m_n + m_d;

		// Create buffer filled with zeros
		let mut buffer = FieldBuffer::<P>::zeros(log_len);

		// Fill the n × (d+1) submatrix with random values
		let row_stride = 1 << m_d;
		for i in 0..n_vars {
			for j in 0..(degree + 1) {
				buffer.set(i * row_stride + j, F::random(&mut rng));
			}
		}

		Self {
			n_vars,
			degree,
			buffer,
		}
	}

	/// Returns the number of variables.
	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the degree of each univariate polynomial.
	pub fn degree(&self) -> usize {
		self.degree
	}

	/// Returns m_d = ceil(log2(degree + 1)).
	fn log_degree_plus_one(&self) -> usize {
		(self.degree + 1).next_power_of_two().ilog2() as usize
	}

	/// Gets the coefficient $g_{i,j}$ (coefficient of $X^j$ in $g_i$).
	pub fn get_coeff(&self, var_index: usize, coeff_index: usize) -> F {
		debug_assert!(var_index < self.n_vars);
		debug_assert!(coeff_index <= self.degree);
		let row_stride = 1 << self.log_degree_plus_one();
		self.buffer.get(var_index * row_stride + coeff_index)
	}

	/// Returns coefficients for variable i as an iterator over [g_i(0), g_i(1), ..., g_i(d)].
	pub fn coeffs_for_var(&self, var_index: usize) -> impl Iterator<Item = F> + '_ {
		debug_assert!(var_index < self.n_vars);
		let m_d = self.log_degree_plus_one();
		let row_stride = 1 << m_d;
		let start = var_index * row_stride;
		(0..=self.degree).map(move |j| self.buffer.get(start + j))
	}

	/// Evaluates g_i(x) for a specific variable using Horner's method.
	pub fn evaluate_univariate(&self, var_index: usize, x: F) -> F {
		let coeffs: Vec<_> = self.coeffs_for_var(var_index).collect();
		evaluate_univariate(&coeffs, x)
	}

	/// Computes the MLE of the mask polynomial at a point.
	///
	/// For a mask polynomial $g(X) = \sum_i g_i(X_i)$ where each $g_i$ is univariate,
	/// the MLE at point $z$ is:
	///
	/// $$
	/// \sum_{v \in \{0,1\}^n} g(v) \cdot eq(v, z) = \sum_i [(1-z_i) g_i(0) + z_i g_i(1)]
	/// $$
	///
	/// This simplification arises because $\sum_{v_j \in \{0,1\}} eq_1(v_j, z_j) = 1$.
	pub fn evaluate_mle(&self, eval_point: &[F]) -> F {
		assert_eq!(eval_point.len(), self.n_vars);

		iter::zip(0..self.n_vars, eval_point)
			.map(|(i, &z_i)| {
				let g_at_0 = self.get_coeff(i, 0);
				let g_at_1 = self.evaluate_univariate(i, F::ONE);
				extrapolate_line_packed(g_at_0, g_at_1, z_i)
			})
			.sum()
	}
}

impl<P: PackedField> AsRef<FieldBuffer<P>> for Mask<P> {
	fn as_ref(&self) -> &FieldBuffer<P> {
		&self.buffer
	}
}

/// Prover for the Libra mask polynomial in ZK MLE-check.
///
/// The mask polynomial has the separable form $g(X_0, ..., X_{n-1}) = sum_{i} g_i(X_i)$,
/// where each $g_i$ is a univariate polynomial of configurable degree.
///
/// This structure allows efficient round polynomial computation in O(degree) time per round.
pub struct MleCheckMaskProver<'a, F: Field, P: PackedField<Scalar = F>> {
	/// Reference to the mask polynomial
	mask: &'a Mask<P>,
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

impl<'a, F: Field, P: PackedField<Scalar = F>> MleCheckMaskProver<'a, F, P> {
	/// Creates a new prover for the Libra mask polynomial.
	///
	/// # Arguments
	///
	/// * `mask` - Reference to the mask polynomial.
	/// * `eval_point` - The evaluation point z for the MLE-check claim, in high-to-low order.
	/// * `eval_claim` - The claimed value of the MLE of g at the evaluation point.
	///
	/// # Panics
	///
	/// Panics if `mask.n_vars() != eval_point.len()`.
	pub fn new(mask: &'a Mask<P>, eval_point: Vec<F>, eval_claim: F) -> Self {
		assert_eq!(mask.n_vars(), eval_point.len(), "mask n_vars must match eval_point length");

		let n_vars = eval_point.len();

		// Precompute suffix_sums[j] = (1-z_j)*g_j(0) + z_j*g_j(1)
		// This equals extrapolate_line_packed(g_j(0), g_j(1), z_j)
		let suffix_sums: Vec<F> = iter::zip(0..n_vars, &eval_point)
			.map(|(i, &z_j)| {
				let g_at_0 = mask.get_coeff(i, 0);
				let g_at_1 = mask.evaluate_univariate(i, F::ONE);
				extrapolate_line_packed(g_at_0, g_at_1, z_j)
			})
			.collect();

		Self {
			mask,
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
}

impl<'a, F: Field, P: PackedField<Scalar = F>> SumcheckProver<F> for MleCheckMaskProver<'a, F, P> {
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
		// Since we process high-to-low (n-1, n-2, ..., 0), the suffix is the lower-indexed
		// variables
		let suffix_sum: F = self.suffix_sums[..var_idx].iter().copied().sum();

		// Compute the constant offset: prefix_sum + suffix_sum
		let constant_offset = self.prefix_sum + suffix_sum;

		// Build the round polynomial R(X) = g_i(X) + constant_offset
		// g_i(X) = sum_{k=0}^{d} a_{i,k} * X^k
		// So coefficients are: [a_0 + offset, a_1, a_2, ..., a_d]
		let mut round_coeffs_vec: Vec<F> = self.mask.coeffs_for_var(var_idx).collect();
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
		self.prefix_sum += self.mask.evaluate_univariate(var_idx, challenge);

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

impl<'a, F: Field, P: PackedField<Scalar = F>> MleCheckProver<F> for MleCheckMaskProver<'a, F, P> {
	fn eval_point(&self) -> &[F] {
		// Return remaining coordinates (high-to-low means we return the first n_vars_remaining
		// elements)
		&self.eval_point[..self.n_vars_remaining]
	}
}

/// Executes the zero-knowledge MLE-check proving protocol for a single multivariate polynomial.
///
/// This function proves a single MLE-check while batching with a Libra mask polynomial
/// to achieve zero-knowledge. The mask polynomial has the separable form
/// $g(X_0, \ldots, X_{n-1}) = \sum_i g_i(X_i)$ where each $g_i$ is a univariate polynomial.
///
/// # Protocol Flow
///
/// 1. Compute and write `mask_eval` (MLE of mask polynomial at the evaluation point)
/// 2. Sample `batch_challenge` and batch evaluation claims
/// 3. For each round, batch the main and mask round polynomials
/// 4. Write `mask_eval_out` (mask evaluation at the challenge point)
///
/// # Arguments
///
/// * `main_prover` - The MLE-check prover for the main polynomial. Must have exactly one claim
///   (i.e., `n_claims() == 1`).
/// * `mask` - The mask polynomial.
/// * `transcript` - The Fiat-Shamir transcript
///
/// # Returns
///
/// Returns [`ProveSingleOutput`] containing the main polynomial's multilinear evaluations
/// and the round challenges.
///
/// # Panics
///
/// Panics if `main_prover.n_claims() != 1`.
pub fn prove<F: Field, P: PackedField<Scalar = F>, Challenger_: Challenger>(
	mut main_prover: impl MleCheckProver<F>,
	mask: &Mask<P>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<ProveSingleOutput<F>, Error> {
	assert_eq!(
		main_prover.n_claims(),
		1,
		"prove requires main_prover to have exactly 1 claim, but it has {}",
		main_prover.n_claims()
	);

	let n_vars = main_prover.n_vars();
	let eval_point = main_prover.eval_point().to_vec();

	// Compute and write mask_eval (MLE of mask polynomial at the evaluation point)
	let mask_eval = mask.evaluate_mle(&eval_point);
	transcript.message().write(&mask_eval);

	// Sample batch challenge and construct mask prover
	let batch_challenge: F = transcript.sample();
	let batched_mask_eval = batch_challenge * mask_eval;
	let mut mask_prover = MleCheckMaskProver::new(mask, eval_point, batched_mask_eval);

	let mut challenges = Vec::with_capacity(n_vars);

	for _ in 0..n_vars {
		// Execute both provers
		let mut main_round_coeffs_vec = main_prover.execute()?;
		let main_round_coeffs = main_round_coeffs_vec.pop().expect("n_claims == 1");

		let mut mask_round_coeffs_vec = mask_prover.execute()?;
		let mask_round_coeffs = mask_round_coeffs_vec
			.pop()
			.expect("mask prover has 1 claim");

		// Batch the round coefficients: batched = main + batch_challenge * mask
		let batched_round_coeffs = main_round_coeffs + &(mask_round_coeffs * batch_challenge);

		// Write truncated coefficients to transcript
		transcript
			.message()
			.write_slice(mlecheck::RoundProof::truncate(batched_round_coeffs).coeffs());

		// Sample challenge and fold both provers
		let challenge = transcript.sample();
		challenges.push(challenge);
		main_prover.fold(challenge)?;
		mask_prover.fold(challenge)?;
	}

	// Finish both provers
	let main_evals = main_prover.finish()?;
	let mask_evals = mask_prover.finish()?;
	let mask_eval_out = mask_evals[0];

	// Write final mask evaluation
	transcript.message().write(&mask_eval_out);

	Ok(ProveSingleOutput {
		multilinear_evals: main_evals,
		challenges,
	})
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

	/// Evaluates the mask polynomial g(X) = sum_i g_i(X_i) at a point using the Mask struct.
	fn evaluate_mask_polynomial<P: PackedField>(mask: &Mask<P>, point: &[P::Scalar]) -> P::Scalar {
		iter::zip(0..mask.n_vars(), point)
			.map(|(i, &x)| mask.evaluate_univariate(i, x))
			.sum()
	}

	fn test_mask_prover_with_degree(degree: usize) {
		let n_vars = 6;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random mask
		let mask = Mask::<B128>::random(n_vars, degree, &mut rng);

		// Generate random evaluation point
		let eval_point: Vec<B128> = random_scalars(&mut rng, n_vars);

		// Compute the MLE of the mask polynomial at eval_point
		let eval_claim = mask.evaluate_mle(&eval_point);

		// Create the prover
		let prover = MleCheckMaskProver::new(&mask, eval_point.clone(), eval_claim);

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
		let expected_eval = evaluate_mask_polynomial(&mask, &challenge_point);
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
		let mask = Mask::<B128>::random(1, 2, &mut rng);
		let eval_point: Vec<B128> = random_scalars(&mut rng, 1);
		let eval_claim = mask.evaluate_mle(&eval_point);

		let prover = MleCheckMaskProver::new(&mask, eval_point.clone(), eval_claim);

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

		let expected_eval = evaluate_mask_polynomial(&mask, &challenge_point);
		assert_eq!(output.multilinear_evals[0], expected_eval);
	}

	#[test]
	fn test_prove() {
		let n_vars = 6;
		let main_degree = 2;
		let mask_degree = 2;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random main mask (using Mask as a simple MleCheckProver for testing)
		let main_mask = Mask::<B128>::random(n_vars, main_degree, &mut rng);

		// Generate random ZK mask
		let zk_mask = Mask::<B128>::random(n_vars, mask_degree, &mut rng);

		// Generate random evaluation point
		let eval_point: Vec<B128> = random_scalars(&mut rng, n_vars);

		// Compute the MLE of the main polynomial at eval_point
		let main_eval_claim = main_mask.evaluate_mle(&eval_point);

		// Create the main prover (using MleCheckMaskProver as a simple MleCheckProver)
		let main_prover = MleCheckMaskProver::new(&main_mask, eval_point.clone(), main_eval_claim);

		// Run the ZK proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove(main_prover, &zk_mask, &mut prover_transcript).unwrap();

		// Write the main polynomial evaluation to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run ZK verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mlecheck::VerifyZKOutput {
			eval,
			mask_eval,
			challenges,
		} = mlecheck::verify_zk::<B128, _>(
			&eval_point,
			main_degree.max(mask_degree), // batched polynomial degree
			main_eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		// Read the main polynomial evaluation from the transcript
		let main_eval_out: B128 = verifier_transcript.message().read().unwrap();

		// Verify the reduced evaluation matches
		assert_eq!(main_eval_out, eval);

		// Compute the challenge point (reverse for high-to-low order)
		let mut challenge_point = challenges;
		challenge_point.reverse();

		// Check that the final main evaluation matches direct computation
		let expected_main_eval = evaluate_mask_polynomial(&main_mask, &challenge_point);
		assert_eq!(output.multilinear_evals[0], expected_main_eval);

		// Check that the mask evaluation matches the zk_mask evaluation at the challenge point
		let expected_mask_eval = evaluate_mask_polynomial(&zk_mask, &challenge_point);
		assert_eq!(mask_eval, expected_mask_eval);
	}
}
