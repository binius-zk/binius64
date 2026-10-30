// Copyright 2026 The Binius Developers

//! The prover's Libra mask polynomial.
//!
//! See [`binius_ip::libra`] for the mask and its coefficient table.

use std::{iter, ops::Deref};

use binius_compute::Allocator;
use binius_field::{Field, PackedField, util::powers};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{
	FieldVec, field_buffer::FieldBuffer, line::extrapolate_line, univariate::evaluate_univariate,
};

/// Generates hypercube evaluations of the libra_eval polynomial.
///
/// For a mask polynomial with `n_vars` variables and `degree`, generates a `FieldBuffer`
/// of size `2^(m_n + m_d)` containing:
///
/// ```text
/// libra_eval_r(j, k) = r[j]^k  if j < n_vars and k ≤ degree
///                    = 0       otherwise
/// ```
///
/// where `j` and `k` are derived from the hypercube index: `idx = j * 2^m_d + k`.
///
/// # Arguments
///
/// * `alloc` - The allocator the expansion is drawn from
/// * `challenge_point` - The challenge point `r` from sumcheck (length `n_vars`)
/// * `n_vars` - Number of variables in the mask polynomial
/// * `degree` - Degree of each univariate in the mask polynomial
/// * `m_n` - Log of number of rows (must satisfy `2^m_n >= n_vars`)
/// * `m_d` - Log of row size (must satisfy `2^m_d >= degree + 1`)
///
/// # Panics
///
/// Panics (in debug mode) if `n_vars > 2^m_n` or `degree + 1 > 2^m_d`.
pub fn expand_libra_eval<A: Allocator, P: PackedField>(
	alloc: &A,
	challenge_point: &[P::Scalar],
	n_vars: usize,
	degree: usize,
	m_n: usize,
	m_d: usize,
) -> FieldVec<P, A> {
	debug_assert!(challenge_point.len() == n_vars);
	debug_assert!(n_vars <= 1 << m_n);
	debug_assert!(degree < 1 << m_d);

	let log_size = m_n + m_d;
	let mut buffer = FieldBuffer::zeros_in(alloc, log_size);
	let row_stride = 1 << m_d;

	for (j, &r_j) in challenge_point.iter().enumerate() {
		let base_idx = j * row_stride;
		for (k, power) in powers(r_j).take(degree + 1).enumerate() {
			buffer.set(base_idx + k, power);
		}
	}

	buffer
}

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
///
/// The type is generic over the buffer storage type `Data`, allowing it to work
/// with both owned buffers (`Box<[P]>`) and borrowed slices.
pub struct Mask<P: PackedField, Data: Deref<Target = [P]> = Box<[P]>> {
	/// Number of variables (n)
	n_vars: usize,
	/// Degree of each univariate polynomial (d)
	degree: usize,
	/// Coefficients stored as a FieldBuffer with log_len = m_n + m_d.
	/// Layout: row i contains the monomial coefficients [g_{i,0}, ..., g_{i,d}, 0, ..., 0].
	/// Row i spans indices [i * 2^m_d, (i+1) * 2^m_d).
	buffer: FieldBuffer<P, Data>,
}

impl<F: Field, P: PackedField<Scalar = F>, Data: Deref<Target = [P]>> Mask<P, Data> {
	/// Creates a new mask polynomial from a pre-allocated buffer.
	///
	/// # Arguments
	///
	/// * `n_vars` - Number of variables (n).
	/// * `degree` - Degree of each univariate polynomial (d).
	/// * `buffer` - Buffer with log_len = m_n + m_d.
	pub const fn new(n_vars: usize, degree: usize, buffer: FieldBuffer<P, Data>) -> Self {
		Self {
			n_vars,
			degree,
			buffer,
		}
	}

	/// Returns the number of variables.
	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the degree of each univariate polynomial.
	pub const fn degree(&self) -> usize {
		self.degree
	}

	/// Returns m_d = ceil(log2(degree + 1)).
	const fn log_degree_plus_one(&self) -> usize {
		(self.degree + 1).next_power_of_two().ilog2() as usize
	}

	/// Gets the coefficient $g_{i,j}$ (coefficient of $X^j$ in $g_i$).
	pub fn get_coeff(&self, var_index: usize, coeff_index: usize) -> F {
		debug_assert!(var_index < self.n_vars);
		debug_assert!(coeff_index <= self.degree);
		let row_stride = 1 << self.log_degree_plus_one();
		self.buffer.get(var_index * row_stride + coeff_index)
	}

	/// Returns the monomial coefficients [g_{i,0}, g_{i,1}, ..., g_{i,d}] for variable i.
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
		evaluate_univariate(&coeffs, &x)
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
				extrapolate_line(g_at_0, g_at_1, z_i)
			})
			.sum()
	}

	/// Evaluates the mask polynomial $g(r) = \sum_i g_i(r_i)$.
	pub fn evaluate(&self, point: &[F]) -> F {
		assert_eq!(point.len(), self.n_vars);
		iter::zip(0..self.n_vars, point)
			.map(|(i, &r_i)| self.evaluate_univariate(i, r_i))
			.sum()
	}

	/// The sum of $g_i$ over $\{0, \infty\}$: its constant plus its leading coefficient.
	fn sum_univariate_over_infinity(&self, var_index: usize) -> F {
		self.get_coeff(var_index, 0) + self.get_coeff(var_index, self.degree)
	}

	/// The sum $\sum_{x \in \{0, \infty\}^n} g(x)$.
	///
	/// Each $g_i$ is summed over its own coordinate, and every other coordinate contributes one.
	pub fn sum_over_infinity_cube(&self) -> F {
		(0..self.n_vars)
			.map(|i| self.sum_univariate_over_infinity(i))
			.sum()
	}

	/// The round polynomial of the sumcheck of $g$ over $\{0, \infty\}^n$, after the top variables
	/// are bound to `challenges` in binding order.
	///
	/// The round's variable is $X_c$, $c = n - 1 - i$ in round $i$. The univariates above it are
	/// evaluated at their challenges, those below are summed over $\{0, \infty\}$, and $g_c$ is
	/// the round polynomial's non-constant part.
	///
	/// ## Preconditions
	///
	/// * `challenges.len() < n_vars`
	pub fn infinity_round_coeffs(&self, challenges: &[F]) -> RoundCoeffs<F> {
		let var_index = self.n_vars - 1 - challenges.len();
		let bound = iter::zip((var_index + 1..self.n_vars).rev(), challenges)
			.map(|(i, &r_i)| self.evaluate_univariate(i, r_i))
			.sum::<F>();
		let free = (0..var_index)
			.map(|i| self.sum_univariate_over_infinity(i))
			.sum::<F>();
		let mut coeffs = self.coeffs_for_var(var_index).collect::<Vec<_>>();
		coeffs[0] += bound + free;
		RoundCoeffs(coeffs)
	}
}

impl<P: PackedField, Data: Deref<Target = [P]>> AsRef<FieldBuffer<P, Data>> for Mask<P, Data> {
	fn as_ref(&self) -> &FieldBuffer<P, Data> {
		&self.buffer
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Random, arch::OptimalB128};
	use binius_ip::libra::{libra_eval, mask_buffer_dimensions};
	use binius_math::{
		multilinear::hypercube::{InfCube, OneCube, eq_ind_partial_eval_scalars},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	type B128 = OptimalB128;

	#[test]
	fn test_libra_eval_inner_product_equals_mask_eval() {
		use binius_compute::GlobalAllocator;
		use binius_math::inner_product::inner_product_buffers;

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 6;
		let degree = 2;

		// Create random mask buffer
		let (m_n, m_d) = mask_buffer_dimensions(n_vars, degree, 0);
		let mask_buffer = random_field_buffer::<B128>(&mut rng, m_n + m_d);

		// Create random challenge point
		let challenge_point: Vec<B128> = random_scalars(&mut rng, n_vars);

		// Compute g(r) using direct computation.evaluate()
		let mask = Mask::new(n_vars, degree, mask_buffer.as_view());
		let direct_eval: B128 = (0..n_vars)
			.map(|i| mask.evaluate_univariate(i, challenge_point[i]))
			.sum();

		// Compute <g', libra_eval_r> using inner product
		let libra_eval_tensor = expand_libra_eval::<_, B128>(
			&GlobalAllocator,
			&challenge_point,
			n_vars,
			degree,
			m_n,
			m_d,
		);
		let inner_product_eval = inner_product_buffers(&mask_buffer, &libra_eval_tensor);

		assert_eq!(
			direct_eval, inner_product_eval,
			"Inner product <g', libra_eval_r> should equal g(r)"
		);
	}

	#[test]
	fn test_libra_eval_sumcheck() {
		use binius_compute::GlobalAllocator;
		use binius_ip::{libra::libra_eval, sumcheck::verify};
		use binius_math::{inner_product::inner_product_par, multilinear::evaluate::evaluate};

		use crate::sumcheck::{bivariate_product_prover, prove_single};

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 6;
		let degree = 2;
		let alloc = GlobalAllocator;

		// Create random mask buffer (g')
		let (m_n, m_d) = mask_buffer_dimensions(n_vars, degree, 0);
		let log_size = m_n + m_d;
		let mask_buffer = random_field_buffer::<B128>(&mut rng, log_size);

		// Create random challenge point r
		let challenge_point: Vec<B128> = random_scalars(&mut rng, n_vars);

		// Generate libra_eval tensor
		let libra_eval_tensor = expand_libra_eval::<_, B128>(
			&GlobalAllocator,
			&challenge_point,
			n_vars,
			degree,
			m_n,
			m_d,
		);

		// Compute the claimed sum: <g', libra_eval_r> = g(r)
		let claimed_sum = inner_product_par(&mask_buffer, &libra_eval_tensor);

		// Create the bivariate product sumcheck prover
		let prover =
			bivariate_product_prover(&alloc, [mask_buffer.clone(), libra_eval_tensor], claimed_sum);

		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript);

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Verify with sumcheck verifier
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify(
			log_size,
			2, // degree 2 for bivariate product
			claimed_sum,
			&mut verifier_transcript,
		)
		.unwrap();

		// Read the multilinear evaluations
		let multilinear_evals: Vec<B128> = verifier_transcript.message().read_vec(2).unwrap();
		let [g_prime_eval, libra_eval_out] = [multilinear_evals[0], multilinear_evals[1]];

		// Check that the product equals the reduced evaluation
		assert_eq!(g_prime_eval * libra_eval_out, sumcheck_output.eval);

		// Verify libra_eval_out using libra_eval
		// The sumcheck binds variables high-to-low, so we need to reverse to get low-to-high order
		// In the buffer layout, the low-order bits encode k (first m_d variables) and
		// high-order bits encode j (next m_n variables)
		let mut query_point = sumcheck_output.challenges;
		query_point.reverse();
		let (query_k, query_j) = query_point.split_at(m_d);

		let expected_libra_eval_out =
			libra_eval::<OneCube, B128>(&challenge_point, query_j, query_k, n_vars, degree);
		assert_eq!(
			libra_eval_out, expected_libra_eval_out,
			"libra_eval should match the sumcheck-reduced evaluation"
		);

		// Also verify g' evaluation directly
		let expected_g_prime_eval = evaluate(&mask_buffer, &query_point);
		assert_eq!(g_prime_eval, expected_g_prime_eval);
	}

	#[test]
	fn libra_eval_over_the_infinity_cube_reads_monomial_coefficients() {
		let mut rng = StdRng::seed_from_u64(0);
		let degree = 2;
		for n_vars in [0, 1, 3, 4, 5] {
			let (m_n, m_d) = mask_buffer_dimensions(n_vars, degree, 0);
			let challenge_point = random_scalars::<B128>(&mut rng, n_vars);
			let query_point = random_scalars::<B128>(&mut rng, m_n + m_d);

			let table = expand_libra_eval::<_, B128>(
				&GlobalAllocator,
				&challenge_point,
				n_vars,
				degree,
				m_n,
				m_d,
			);
			let monomials = eq_ind_partial_eval_scalars::<InfCube, B128>(&query_point);
			let expected = iter::zip(table.iter_scalars(), monomials)
				.map(|(t, m)| t * m)
				.sum::<B128>();

			let (query_k, query_j) = query_point.split_at(m_d);
			assert_eq!(
				libra_eval::<InfCube, B128>(&challenge_point, query_j, query_k, n_vars, degree),
				expected,
				"n_vars={n_vars}"
			);
		}
	}

	#[test]
	fn infinity_round_polynomials_run_the_sumcheck() {
		let mut rng = StdRng::seed_from_u64(0);
		let (n_vars, degree) = (5, 2);
		let (m_n, m_d) = mask_buffer_dimensions(n_vars, degree, 0);
		let buffer = random_field_buffer::<B128>(&mut rng, m_n + m_d);
		let mask = Mask::new(n_vars, degree, buffer.as_view());

		let mut claim = mask.sum_over_infinity_cube();
		let mut challenges = Vec::new();
		for _ in 0..n_vars {
			let coeffs = mask.infinity_round_coeffs(&challenges);
			// R(0) + R(∞) is the constant plus the leading coefficient.
			assert_eq!(coeffs[0] + coeffs[degree], claim);
			let challenge = B128::random(&mut rng);
			claim = coeffs.evaluate(&challenge);
			challenges.push(challenge);
		}
		challenges.reverse();
		assert_eq!(mask.evaluate(&challenges), claim);
	}
}
