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
	use binius_field::{BinaryField128bGhash as B128, Field};
	use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, test_utils::random_scalars};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	// Ground truth for a shift-indicator MLE, independent of the recurrence under test.
	//
	// Fix j, s to the challenges r_j, r_s.
	//
	// Over the hypercube in i, the indicator's MLE expands over the (j, s) cube as:
	//     mle[i] = sum_{j, s in {0,1}^n : cond(i, j, s)} eq(r_j, j) * eq(r_s, s)
	fn reference_indicator(
		r_j: &[B128],
		r_s: &[B128],
		cond: impl Fn(usize, usize, usize) -> bool,
	) -> Vec<B128> {
		let n = r_j.len();
		// eq_j[j] = eq(r_j, j), eq_s[s] = eq(r_s, s).
		//
		// Both index little-endian, matching the recurrence's bit order.
		let eq_j = eq_ind_partial_eval_scalars(r_j);
		let eq_s = eq_ind_partial_eval_scalars(r_s);

		(0..1 << n)
			.map(|i| {
				let mut acc = B128::ZERO;
				for j in 0..1 << n {
					for s in 0..1 << n {
						if cond(i, j, s) {
							acc += eq_j[j] * eq_s[s];
						}
					}
				}
				acc
			})
			.collect()
	}

	// Draw a pseudo-random challenge (r_j, r_s).
	// The fixed seed keeps failures reproducible.
	fn challenges(n: usize) -> (Vec<B128>, Vec<B128>) {
		let mut rng = StdRng::seed_from_u64(0);
		(random_scalars(&mut rng, n), random_scalars(&mut rng, n))
	}

	#[test]
	fn srl_matches_reference() {
		// srl: output bit i reads input bit j = i + s.
		// Bits shifted past the top vanish, since no such j is in range.
		let (r_j, r_s) = challenges(6);
		let (sigma, _) = partial_eval_sigmas(&r_j, &r_s);
		assert_eq!(sigma, reference_indicator(&r_j, &r_s, |i, j, s| j == i + s));
	}

	#[test]
	fn sll_matches_reference() {
		// sll is the transpose of srl.
		// Output bit i = j + s reads input bit j.
		let (r_j, r_s) = challenges(6);
		let sigma_transpose = partial_eval_sigmas_transpose(&r_j, &r_s);
		assert_eq!(sigma_transpose, reference_indicator(&r_j, &r_s, |i, j, s| i == j + s));
	}

	#[test]
	fn sra_matches_reference() {
		// sra behaves like srl within range.
		// Past the shift, the sign bit j = 2^n - 1 fills every position.
		let (r_j, r_s) = challenges(6);
		let n = r_j.len();
		let (sigma, _) = partial_eval_sigmas(&r_j, &r_s);
		let phi = partial_eval_phi(&r_s);
		// prod(r_j) is the eq-indicator selecting the all-ones sign position j = 2^n - 1.
		let j_product: B128 = r_j.iter().copied().product();
		let sra: Vec<_> = (0..1 << n).map(|i| sigma[i] + j_product * phi[i]).collect();
		assert_eq!(sra, reference_indicator(&r_j, &r_s, |i, j, s| j == (i + s).min((1 << n) - 1)));
	}

	#[test]
	fn rotr_matches_reference() {
		// rotr wraps bits leaving the bottom back to the top.
		// So j = (i + s) mod 2^n.
		let (r_j, r_s) = challenges(6);
		let n = r_j.len();
		let (sigma, sigma_prime) = partial_eval_sigmas(&r_j, &r_s);
		let rotr: Vec<_> = (0..1 << n).map(|i| sigma[i] + sigma_prime[i]).collect();
		assert_eq!(rotr, reference_indicator(&r_j, &r_s, |i, j, s| j == (i + s) % (1 << n)));
	}
}
