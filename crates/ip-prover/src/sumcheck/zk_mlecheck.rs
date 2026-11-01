// Copyright 2026 The Binius Developers

//! Prover for the Libra mask polynomial in ZK MLE-check protocols.
//!
//! The Libra ZK-sumcheck protocol uses a masking polynomial g(X_0, ..., X_{n-1}) of the form:
//! g = sum_{i=0}^{n-1} g_i(X_i)
//!
//! where each g_i(X) is a univariate polynomial of configurable degree. This separable structure
//! allows efficient computation of round polynomials without iterating over the full hypercube.

use std::{iter, ops::Deref};

use binius_field::{Field, PackedField};
use binius_ip::{mlecheck, sumcheck::RoundCoeffs};
use binius_math::line::extrapolate_line;

use super::{common::MleCheckProver, round_state::RoundState};
use crate::{channel::IPProverChannel, libra::Mask};

/// Output of the ZK MLE-check proving protocol.
#[derive(Debug, Clone)]
pub struct ProveZKOutput<F: Field> {
	/// Evaluations of the main multilinear polynomials at the challenge point.
	pub multilinear_evals: Vec<F>,
	/// Evaluation of the mask polynomial at the challenge point.
	pub mask_eval: F,
	/// Verifier challenges for each round of the sumcheck protocol.
	pub challenges: Vec<F>,
}

/// Prover for the Libra mask polynomial in ZK MLE-check.
///
/// The mask polynomial has the separable form $g(X_0, ..., X_{n-1}) = sum_{i} g_i(X_i)$,
/// where each $g_i$ is a univariate polynomial of configurable degree.
///
/// This structure allows efficient round polynomial computation in O(degree) time per round.
pub struct MleCheckMaskProver<F: Field, P: PackedField<Scalar = F>, Data: Deref<Target = [P]>> {
	/// The mask polynomial (owned)
	mask: Mask<P, Data>,
	/// The evaluation point z (in high-to-low variable order)
	eval_point: Vec<F>,
	/// Number of variables remaining to process
	n_vars_remaining: usize,
	/// Accumulated sum of g_j(r_j) for already-folded variables
	prefix_sum: F,
	/// Precomputed (1-z_j)*g_j(0) + z_j*g_j(1) for each variable
	suffix_sums: Vec<F>,
	/// State: either last round coefficients (after execute) or current claim (after fold)
	last_coeffs_or_claim: RoundState<RoundCoeffs<F>, F>,
}

impl<F: Field, P: PackedField<Scalar = F>, Data: Deref<Target = [P]>>
	MleCheckMaskProver<F, P, Data>
{
	/// Creates a new prover for the Libra mask polynomial.
	///
	/// # Arguments
	///
	/// * `mask` - The mask polynomial (takes ownership).
	/// * `eval_point` - The evaluation point z for the MLE-check claim, in high-to-low order.
	/// * `eval_claim` - The claimed value of the MLE of g at the evaluation point.
	///
	/// # Panics
	///
	/// Panics if `mask.n_vars() != eval_point.len()`.
	pub fn new(mask: Mask<P, Data>, eval_point: Vec<F>, eval_claim: F) -> Self {
		assert_eq!(mask.n_vars(), eval_point.len(), "mask n_vars must match eval_point length");

		let n_vars = eval_point.len();

		// Precompute suffix_sums[j] = (1-z_j)*g_j(0) + z_j*g_j(1)
		// That is the line through g_j(0) and g_j(1), read at z_j.
		let suffix_sums: Vec<F> = iter::zip(0..n_vars, &eval_point)
			.map(|(i, &z_j)| {
				let g_at_0 = mask.get_coeff(i, 0);
				let g_at_1 = mask.evaluate_univariate(i, F::ONE);
				extrapolate_line(g_at_0, g_at_1, z_j)
			})
			.collect();

		Self {
			mask,
			eval_point,
			n_vars_remaining: n_vars,
			prefix_sum: F::ZERO,
			suffix_sums,
			last_coeffs_or_claim: RoundState::Claim(eval_claim),
		}
	}

	/// Returns the index of the current variable being processed.
	/// Processing is high-to-low, so we start at n_vars-1 and decrease.
	const fn current_var_index(&self) -> usize {
		self.n_vars_remaining - 1
	}
}

impl<F: Field, P: PackedField<Scalar = F>, Data: Deref<Target = [P]>> MleCheckProver<F>
	for MleCheckMaskProver<F, P, Data>
{
	fn n_vars(&self) -> usize {
		self.n_vars_remaining
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		self.last_coeffs_or_claim.claim();

		assert_ne!(self.n_vars_remaining, 0, "execute called out of order; expected finish");

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
		self.last_coeffs_or_claim = RoundState::Coeffs(round_coeffs.clone());
		vec![round_coeffs]
	}

	fn fold(&mut self, challenge: F) {
		let coeffs = self.last_coeffs_or_claim.coeffs();

		// Evaluate round polynomial at challenge to get new claim
		let new_claim = coeffs.evaluate(&challenge);

		let var_idx = self.current_var_index();

		// Update prefix_sum: add g_i(r_i)
		self.prefix_sum += self.mask.evaluate_univariate(var_idx, challenge);

		self.n_vars_remaining -= 1;
		self.last_coeffs_or_claim = RoundState::Claim(new_claim);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars_remaining, 0, "finish called out of order; sumcheck rounds remain");

		// Final evaluation of g at the challenge point is prefix_sum
		// (since g(r_0, ..., r_{n-1}) = sum_i g_i(r_i))
		vec![self.prefix_sum]
	}

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
/// * `main_prover` - The MLE-check prover for the main polynomial. Must carry exactly one claim.
/// * `mask` - The mask polynomial.
/// * `channel` - The channel for sending prover messages and sampling challenges
///
/// # Returns
///
/// Returns [`ProveZKOutput`] containing the main polynomial's multilinear evaluations
/// and the round challenges.
///
/// # Pre-conditions
///
/// * The mask's univariate degree must be at least the degree of the main prover's round
///   polynomials.
/// * The two round polynomials are added coefficient by coefficient, so a shorter mask leaves the
///   high-degree coefficients uncovered and the protocol is no longer hiding.
///
/// # Panics
///
/// Panics if the main prover emits more than one round polynomial.
/// Panics if the mask's round polynomial is shorter than the main prover's.
pub fn prove<F: Field, P: PackedField<Scalar = F>, Data: Deref<Target = [P]>>(
	mut main_prover: impl MleCheckProver<F>,
	mask: Mask<P, Data>,
	channel: &mut impl IPProverChannel<F>,
) -> ProveZKOutput<F> {
	let n_vars = main_prover.n_vars();
	let eval_point = main_prover.eval_point().to_vec();

	// Compute and write mask_eval (MLE of mask polynomial at the evaluation point)
	let mask_eval = mask.evaluate_mle(&eval_point);
	channel.send_one(mask_eval);

	// Sample batch challenge and construct mask prover
	let batch_challenge: F = channel.sample();
	let batched_mask_eval = batch_challenge * mask_eval;
	let mut mask_prover = MleCheckMaskProver::new(mask, eval_point, batched_mask_eval);

	let mut challenges = Vec::with_capacity(n_vars);

	for _ in 0..n_vars {
		// Execute both provers
		let mut main_round_coeffs_vec = main_prover.execute();
		assert_eq!(
			main_round_coeffs_vec.len(),
			1,
			"prove requires a main prover with one claim, but it emitted {}",
			main_round_coeffs_vec.len()
		);
		let main_round_coeffs = main_round_coeffs_vec.pop().expect("length checked above");

		let mut mask_round_coeffs_vec = mask_prover.execute();
		let mask_round_coeffs = mask_round_coeffs_vec
			.pop()
			.expect("mask prover has 1 claim");

		// The batching below pads the shorter polynomial with zeros, so any coefficient beyond
		// the mask's degree would reach the channel exactly as the witness produced it.
		//
		//     main   : [a_0, a_1, a_2]
		//     mask   : [b_0, b_1]      -> padded to [b_0, b_1, 0]
		//     batched: [a_0 + c*b_0, a_1 + c*b_1, a_2]
		//                                         ^^^ unmasked
		assert!(
			mask_round_coeffs.0.len() >= main_round_coeffs.0.len(),
			"the mask round polynomial has {} coefficients against the main round polynomial's \
			 {}, so the excess would be sent unmasked",
			mask_round_coeffs.0.len(),
			main_round_coeffs.0.len()
		);

		// Batch the round coefficients: batched = main + batch_challenge * mask
		let batched_round_coeffs = main_round_coeffs + &(mask_round_coeffs * batch_challenge);

		// Write truncated coefficients to channel
		channel.send_many(mlecheck::RoundProof::truncate(batched_round_coeffs).coeffs());

		// Sample challenge and fold both provers
		let challenge = channel.sample();
		challenges.push(challenge);
		main_prover.fold(challenge);
		mask_prover.fold(challenge);
	}

	// Finish both provers
	let main_evals = main_prover.finish();
	let mask_evals = mask_prover.finish();
	let mask_eval_out = mask_evals[0];

	// Write final mask evaluation
	channel.send_one(mask_eval_out);

	ProveZKOutput {
		multilinear_evals: main_evals,
		mask_eval: mask_eval_out,
		challenges,
	}
}

#[cfg(test)]
mod tests {
	use binius_field::arch::OptimalB128;
	use binius_ip::{libra::mask_buffer_dimensions, mlecheck};
	use binius_math::test_utils::{random_field_buffer, random_scalars};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::prove_single_mlecheck;

	type B128 = OptimalB128;

	/// Evaluates the mask polynomial g(X) = sum_i g_i(X_i) at a point using the Mask struct.
	fn evaluate_mask_polynomial<P: PackedField, Data: Deref<Target = [P]>>(
		mask: &Mask<P, Data>,
		point: &[P::Scalar],
	) -> P::Scalar {
		iter::zip(0..mask.n_vars(), point)
			.map(|(i, &x)| mask.evaluate_univariate(i, x))
			.sum()
	}

	fn test_mask_prover_with_degree(degree: usize) {
		let n_vars = 6;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random mask buffer
		let (m_n, m_d) = mask_buffer_dimensions(n_vars, degree, 0);
		let buffer = random_field_buffer::<B128>(&mut rng, m_n + m_d);

		// Generate random evaluation point
		let eval_point: Vec<B128> = random_scalars(&mut rng, n_vars);

		// Compute the MLE of the mask polynomial at eval_point
		let mask = Mask::new(n_vars, degree, buffer.as_view());
		let eval_claim = mask.evaluate_mle(&eval_point);

		// Create the prover (takes ownership of a borrowed mask view)
		let prover = MleCheckMaskProver::new(mask, eval_point.clone(), eval_claim);

		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript);

		// Write the multilinear evaluation to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = mlecheck::verify(
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
		let mut challenge_point = sumcheck_output.challenges;
		challenge_point.reverse();

		// Check that the final evaluation matches direct computation
		let mask = Mask::new(n_vars, degree, buffer.as_view());
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

		// Single variable mask buffer
		let (m_n, m_d) = mask_buffer_dimensions(1, 2, 0);
		let buffer = random_field_buffer::<B128>(&mut rng, m_n + m_d);

		let eval_point: Vec<B128> = random_scalars(&mut rng, 1);
		let mask = Mask::new(1, 2, buffer.as_view());
		let eval_claim = mask.evaluate_mle(&eval_point);

		let prover = MleCheckMaskProver::new(mask, eval_point.clone(), eval_claim);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript);

		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output =
			mlecheck::verify(&eval_point, 2, eval_claim, &mut verifier_transcript).unwrap();

		let mut challenge_point = sumcheck_output.challenges;
		challenge_point.reverse();

		let mask = Mask::new(1, 2, buffer.as_view());
		let expected_eval = evaluate_mask_polynomial(&mask, &challenge_point);
		assert_eq!(output.multilinear_evals[0], expected_eval);
	}

	fn test_prove_with_degrees(main_degree: usize, mask_degree: usize) {
		let n_vars = 6;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random main mask buffer (using Mask as a simple MleCheckProver for testing)
		let (m_n, m_d) = mask_buffer_dimensions(n_vars, main_degree, 0);
		let main_buffer = random_field_buffer::<B128>(&mut rng, m_n + m_d);

		// Generate random ZK mask buffer
		let (zk_m_n, zk_m_d) = mask_buffer_dimensions(n_vars, mask_degree, 0);
		let zk_buffer = random_field_buffer::<B128>(&mut rng, zk_m_n + zk_m_d);

		// Generate random evaluation point
		let eval_point: Vec<B128> = random_scalars(&mut rng, n_vars);

		// Compute the MLE of the main polynomial at eval_point
		let main_mask = Mask::new(n_vars, main_degree, main_buffer.as_view());
		let main_eval_claim = main_mask.evaluate_mle(&eval_point);

		// Create the main prover (using MleCheckMaskProver as a simple MleCheckProver)
		let main_prover = MleCheckMaskProver::new(main_mask, eval_point.clone(), main_eval_claim);

		// Run the ZK proving protocol
		let zk_mask = Mask::new(n_vars, mask_degree, zk_buffer.as_view());
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove(main_prover, zk_mask, &mut prover_transcript);

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
		} = mlecheck::verify_zk(
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
		let main_mask = Mask::new(n_vars, main_degree, main_buffer.as_view());
		let expected_main_eval = evaluate_mask_polynomial(&main_mask, &challenge_point);
		assert_eq!(output.multilinear_evals[0], expected_main_eval);

		// Check that the mask evaluation matches the zk_mask evaluation at the challenge point
		let zk_mask = Mask::new(n_vars, mask_degree, zk_buffer.as_view());
		let expected_mask_eval = evaluate_mask_polynomial(&zk_mask, &challenge_point);
		assert_eq!(mask_eval, expected_mask_eval);
	}

	#[test]
	fn test_prove() {
		// Both round polynomials carry three coefficients, so every one of them is covered.
		test_prove_with_degrees(2, 2);
	}

	#[test]
	fn test_prove_mask_above_main_degree() {
		// A mask that overshoots is still hiding.
		//
		//     main   : [a_0, a_1]
		//     mask   : [b_0, b_1, b_2]
		//     batched: [a_0 + c*b_0, a_1 + c*b_1, c*b_2]
		//
		// The verifier is told the batched degree, so the proof still checks out.
		test_prove_with_degrees(1, 2);
	}

	#[test]
	#[should_panic(expected = "would be sent unmasked")]
	fn test_prove_mask_below_main_degree() {
		// A mask one degree short leaves the top coefficient of every round in the clear.
		//
		//     main   : [a_0, a_1, a_2]
		//     mask   : [b_0, b_1]      -> zero-padded to [b_0, b_1, 0]
		//     batched: [a_0 + c*b_0, a_1 + c*b_1, a_2]
		//                                         ^^^ witness-dependent, unmasked
		//
		// The transcript truncates the constant term, so this coefficient is sent every round.
		test_prove_with_degrees(2, 1);
	}
}
