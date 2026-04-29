// Copyright 2025 Irreducible Inc.

//! Shift indicator partial evaluation functions.
//!
//! This module provides functions for computing partial evaluations of shift indicator
//! multilinear extensions and their helper polynomials.

use binius_field::FieldOps;

/// Partial evaluation of the shift indicator helper polynomials $\sigma, \sigma'$ over all i on the
/// hypercube.
///
/// Given fixed j and s, computes sigma and sigma_prime for all possible i values.
/// Returns (sigma, sigma_prime) as Vecs of length `1 << r_j.len()`.
pub fn partial_eval_sigmas<E: FieldOps>(r_j: &[E], r_s: &[E]) -> (Vec<E>, Vec<E>) {
	assert_eq!(r_j.len(), r_s.len(), "r_j and r_s must have the same length");

	let n = r_j.len();
	let mut sigma = vec![E::zero(); 1 << n];
	let mut sigma_prime = vec![E::zero(); 1 << n];
	sigma[0] = E::one();

	// Process each bit position
	for k in 0..n {
		let j_k = r_j[k].clone();
		let s_k = r_s[k].clone();

		// Precompute boolean combinations for this bit
		let both = j_k.clone() * &s_k;
		let j_one_s = j_k.clone() - &both; // j_k * (1 - s_k)
		let one_j_s = s_k.clone() - &both; // (1 - j_k) * s_k
		let xor = j_k + s_k;
		let eq = E::one() + &xor;

		// Update arrays for this bit position
		for i in 0..(1 << k) {
			// Update upper halves first (i_k = 1)
			sigma[(1 << k) | i] = j_one_s.clone() * &sigma[i];
			sigma_prime[(1 << k) | i] = one_j_s.clone() * &sigma[i] + eq.clone() * &sigma_prime[i];

			// Update lower halves (i_k = 0)
			let sigma_i = sigma[i].clone();
			let sigma_prime_i = sigma_prime[i].clone();
			sigma[i] = eq.clone() * &sigma_i + j_one_s.clone() * &sigma_prime_i;
			sigma_prime[i] = sigma_prime_i * &one_j_s;
		}
	}

	(sigma, sigma_prime)
}

/// Partial evaluation of the shift indicator helper polynomial $\phi$ over all i on the hypercube.
///
/// Given fixed s, computes phi for all possible i values.
pub fn partial_eval_phi<E: FieldOps>(r_s: &[E]) -> Vec<E> {
	let n = r_s.len();
	let mut phi = vec![E::zero(); 1 << n];

	// Process each bit position
	for k in 0..n {
		let s_k = r_s[k].clone();

		// Update arrays for this bit position
		for i in 0..(1 << k) {
			// Update for i_k = 1
			phi[(1 << k) | i] = s_k.clone() + (E::one() + &s_k) * &phi[i];
			let temp = phi[(1 << k) | i].clone() - &s_k;
			phi[i] += &temp;
		}
	}

	phi
}

/// Partial evaluation of transposed sigma for SLL.
///
/// Since sll_ind(i, j, s) = srl_ind(j, i, s), this computes sigma with i and j swapped.
pub fn partial_eval_sigmas_transpose<E: FieldOps>(r_j: &[E], r_s: &[E]) -> Vec<E> {
	assert_eq!(r_j.len(), r_s.len(), "r_j and r_s must have the same length");

	let n = r_j.len();
	let mut sigma_transpose = vec![E::zero(); 1 << n];
	let mut sigma_transpose_prime = vec![E::zero(); 1 << n];
	sigma_transpose[0] = E::one();

	// Process each bit position
	for k in 0..n {
		let j_k = r_j[k].clone();
		let s_k = r_s[k].clone();

		// Precompute boolean combinations for this bit (with i and j swapped)
		let both = j_k.clone() * &s_k;
		let xor = j_k + s_k;
		let eq = E::one() + &xor;
		let zero = eq.clone() + &both;

		// Update arrays for this bit position
		for i in 0..(1 << k) {
			// Update for i_k = 1
			sigma_transpose[(1 << k) | i] =
				xor.clone() * &sigma_transpose[i] + zero.clone() * &sigma_transpose_prime[i];
			sigma_transpose_prime[(1 << k) | i] = both.clone() * &sigma_transpose_prime[i];

			// Update for i_k = 0
			let sigma_t = sigma_transpose[i].clone();
			sigma_transpose_prime[i] =
				both.clone() * &sigma_t + xor.clone() * &sigma_transpose_prime[i];
			sigma_transpose[i] = zero.clone() * &sigma_t;
		}
	}

	sigma_transpose
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash, Field};
	use binius_math::{
		multilinear::shift::{rotr_ind, sll_ind, sra_ind, srl_ind},
		test_utils::{index_to_hypercube_point, random_scalars},
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	fn sll_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> Vec<F> {
		partial_eval_sigmas_transpose(j, s)
	}

	fn srl_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> Vec<F> {
		let (sigma, _) = partial_eval_sigmas(j, s);
		sigma
	}

	fn sra_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> Vec<F> {
		assert_eq!(j.len(), s.len(), "j and s must have the same length");

		let (sigma, _) = partial_eval_sigmas(j, s);
		let phi = partial_eval_phi(s);
		let j_prod = j.iter().product::<F>();

		let n = j.len();
		let mut result = vec![F::ZERO; 1 << n];
		for i in 0..(1 << n) {
			result[i] = sigma[i] + j_prod * phi[i];
		}

		result
	}

	fn rotr_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> Vec<F> {
		let (sigma, sigma_prime) = partial_eval_sigmas(j, s);

		let n = j.len();
		let mut result = vec![F::ZERO; 1 << n];
		for i in 0..(1 << n) {
			result[i] = sigma[i] + sigma_prime[i];
		}

		result
	}

	type ShiftIndicatorFn<F> = fn(&[F], &[F], &[F]) -> F;

	fn test_partial_eval_helper<F: Field>(
		partial_eval_fn: fn(&[F], &[F]) -> Vec<F>,
		direct_fn: ShiftIndicatorFn<F>,
		j: &[F],
		s: &[F],
	) {
		let n = j.len();
		let partial_eval = partial_eval_fn(j, s);

		assert_eq!(partial_eval.len(), 1 << n);

		for i_idx in 0..(1 << n) {
			let i = index_to_hypercube_point::<F>(n, i_idx);
			let expected = direct_fn(&i, j, s);
			let actual = partial_eval[i_idx];
			assert_eq!(
				actual, expected,
				"Mismatch at i_idx={}, i={:?}, j={:?}, s={:?}",
				i_idx, i, j, s
			);
		}
	}

	#[test]
	fn test_sll_ind_partial_eval() {
		let mut rng = StdRng::seed_from_u64(0);
		let n = 6;

		// Test with random j and s
		let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
		let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

		test_partial_eval_helper(sll_ind_partial_eval, sll_ind, &j, &s);
	}

	#[test]
	fn test_srl_ind_partial_eval() {
		let mut rng = StdRng::seed_from_u64(0);
		let n = 6;

		// Test with random j and s
		let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
		let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

		test_partial_eval_helper(srl_ind_partial_eval, srl_ind, &j, &s);
	}

	#[test]
	fn test_sra_ind_partial_eval() {
		let mut rng = StdRng::seed_from_u64(0);
		let n = 6;

		// Test with random j and s
		let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
		let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

		test_partial_eval_helper(sra_ind_partial_eval, sra_ind, &j, &s);
	}

	#[test]
	fn test_rotr_ind_partial_eval() {
		let mut rng = StdRng::seed_from_u64(0);
		let n = 6;

		// Test with random j and s
		let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
		let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

		test_partial_eval_helper(rotr_ind_partial_eval, rotr_ind, &j, &s);
	}
}
