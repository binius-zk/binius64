// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Univariate polynomials, in the two forms a protocol carries them in.
//!
//! ```text
//!     coefficients   ->   evaluate_univariate      Horner over the monomial basis
//!     evaluations    ->   BinarySubspace methods   Lagrange interpolation over a domain
//! ```
//!
//! The evaluation domain is always a [`BinarySubspace`], and that is what makes the second form
//! cheap.
//!
//! A subspace is an additive group, so every one of its points shares a single barycentric weight.
//! The usual Lagrange formula needs one weight per point, and one inversion to build each.
//! Here there is one weight and one inversion, whatever the domain size.

use std::{iter, mem, ops::Deref};

use binius_field::{BinaryField, field::FieldOps};
use itertools::izip;

use super::{BinarySubspace, FieldBuffer};

/// Evaluates a univariate polynomial given by its monomial coefficients.
///
/// Most callers reach for this to batch several claims under one challenge.
/// Reading `coeffs` as the claims and `x` as the challenge, the result is `sum_i claim_i * x^i`.
///
/// # Arguments
///
/// * `coeffs` - coefficients ordered from the low-degree term to the high-degree term
/// * `x` - the point to evaluate at
pub fn evaluate_univariate<F: FieldOps>(coeffs: &[F], x: &F) -> F {
	let Some((highest_degree, rest)) = coeffs.split_last() else {
		return F::zero();
	};

	// Horner's method, from the highest-degree coefficient down.
	rest.iter()
		.rev()
		.fold(highest_degree.clone(), |acc, coeff| acc * x + coeff)
}

/// Lagrange interpolation over a domain of `2^dim` points.
///
/// Bring this into scope to read a [`BinarySubspace`] as the domain a polynomial is given on.
///
/// Every method carries two field parameters:
///
/// ```text
///     F   the domain's own field, where the points live
///     E   the field the arithmetic runs in, which F embeds into
/// ```
///
/// A native verifier takes `E = F`.
/// A recursion verifier takes `E` to be its channel's element type, so the same code builds a
/// circuit.
pub trait EvaluationDomain<F: BinaryField> {
	/// The Lagrange basis evaluated at `z`, one value per domain point.
	///
	/// Entry `i` is `L_i(z) = w * prod_{j != i} (z - d_j)`, for the shared weight `w`.
	fn lagrange_evals<E: FieldOps + From<F>>(&self, z: &E) -> Vec<E>;

	/// The Lagrange basis at `z`, packed into a buffer instead of a vector.
	///
	/// Same values as [`Self::lagrange_evals`], for callers that feed a buffer-shaped consumer.
	fn lagrange_evals_buffer(&self, z: F) -> FieldBuffer<F>;

	/// Evaluates at `z` the polynomial that takes `values` on this domain.
	///
	/// This is the inner product of `values` with [`Self::lagrange_evals`], without building that
	/// vector:
	///
	/// ```text
	///     f(z) = w * sum_i values_i * prod_{j != i} (z - d_j)
	/// ```
	///
	/// # Panics
	///
	/// Panics unless `values` holds one entry per domain point.
	fn extrapolate<E: FieldOps + From<F>>(&self, values: &[E], z: &E) -> E;
}

impl<F: BinaryField, Data: Deref<Target = [F]>> EvaluationDomain<F> for BinarySubspace<F, Data> {
	/// Two sweeps build every entry without ever dividing:
	///
	/// ```text
	///     backward:   r_i <- w * prod_{j > i} (z - d_j)
	///     forward:    r_i <- r_i * prod_{j < i} (z - d_j)
	/// ```
	///
	/// That is about `4n` multiplications and the single inversion the weight costs.
	fn lagrange_evals<E: FieldOps + From<F>>(&self, z: &E) -> Vec<E> {
		// Seed the output with the linear terms t_i = z - d_i.
		let mut result: Vec<E> = self.iter().map(|d| z.clone() - E::from(d)).collect();

		// Backward sweep: replace t_i with w * prod_{j > i} t_j.
		// Seeding the accumulator with the weight absorbs the multiply-by-w pass.
		let mut suffix = barycentric_weight::<F, E, Data>(self);
		for r_i in result.iter_mut().rev() {
			let t_i = mem::replace(r_i, suffix.clone());
			suffix *= t_i;
		}

		// Forward sweep: multiply in prefix_i = prod_{j < i} t_j, completing
		//
		//     L_i(z) = w * prod_{j > i} t_j * prod_{j < i} t_j = w * prod_{j != i} (z - d_j).
		//
		// The terms are recomputed on the fly; iterating the subspace is a cheap XOR walk.
		let mut prefix = E::one();
		for (r_i, d) in iter::zip(&mut result, self.iter()) {
			*r_i *= prefix.clone();
			prefix *= z.clone() - E::from(d);
		}

		result
	}

	fn lagrange_evals_buffer(&self, z: F) -> FieldBuffer<F> {
		FieldBuffer::new(self.dim(), self.lagrange_evals(&z))
	}

	/// One prefix-product accumulator carries the whole sum in a single pass, so the extra space
	/// is constant rather than `O(n)`.
	fn extrapolate<E: FieldOps + From<F>>(&self, values: &[E], z: &E) -> E {
		assert_eq!(
			values.len(),
			1 << self.dim(),
			"precondition: values must hold one entry per domain point"
		);

		// Fold sum_i values_i * prod_{j != i} (z - d_j), carrying the running prefix product.
		let (acc, _) = izip!(values, self.iter()).fold(
			(E::zero(), E::one()),
			|(acc, prod), (value, point)| {
				let term = z.clone() - E::from(point);
				let next_acc = acc * &term + prod.clone() * value;
				(next_acc, prod * term)
			},
		);

		acc * barycentric_weight::<F, E, Data>(self)
	}
}

/// The barycentric weight shared by every point of a binary subspace.
///
/// The usual weight at point `d_i` is `prod_{j != i} (d_i - d_j)^{-1}`.
///
/// Subtracting `d_i` permutes the subspace, so that product runs over the non-zero elements
/// whichever `i` it started from.
///
/// One weight therefore serves every point:
///
/// ```text
///     w = (prod_{j >= 1} d_j)^{-1}
/// ```
fn barycentric_weight<F, E, Data>(subspace: &BinarySubspace<F, Data>) -> E
where
	F: BinaryField,
	E: FieldOps + From<F>,
	Data: Deref<Target = [F]>,
{
	let product = subspace
		.iter()
		.skip(1)
		.map(E::from)
		.fold(E::one(), |acc, d| acc * d);

	// SAFETY: the product runs over the subspace's non-zero elements — `skip(1)` drops index 0,
	// which is the zero element — so it is non-zero by construction, whatever the caller passes.
	// Inverting without the zero case spares the wrapper channels a constraint that could never
	// fire.
	unsafe { product.invert() }
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Ghash128b, Random, util::powers};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::{
		BinarySubspace,
		inner_product::inner_product,
		test_utils::{B128, random_scalars},
	};

	type F = Ghash128b;

	/// The definition of [`evaluate_univariate`], written out as a sum over powers.
	fn evaluate_univariate_with_powers<F: Field>(coeffs: &[F], x: F) -> F {
		inner_product(coeffs.iter().copied(), powers(x).take(coeffs.len()))
	}

	/// The textbook Lagrange basis: one weight per point, each built with its own inversion.
	///
	/// This is what the subspace methods collapse to a single shared weight, so it is the
	/// reference the collapse is pinned against.
	fn lagrange_evals_reference<F: Field>(domain: &[F], z: F) -> Vec<F> {
		domain
			.iter()
			.map(|&d_i| {
				let denominator: F = domain
					.iter()
					.filter(|&&d_j| d_j != d_i)
					.map(|&d_j| d_i - d_j)
					.product();
				let numerator: F = domain
					.iter()
					.filter(|&&d_j| d_j != d_i)
					.map(|&d_j| z - d_j)
					.product();
				numerator * denominator.invert_or_zero()
			})
			.collect()
	}

	#[test]
	fn evaluate_univariate_matches_the_sum_over_powers() {
		let mut rng = StdRng::seed_from_u64(0);

		// An empty coefficient slice is the zero polynomial, which the fold has to special-case.
		for n_coeffs in [0, 1, 2, 5, 10] {
			let coeffs = random_scalars(&mut rng, n_coeffs);
			let x = F::random(&mut rng);
			assert_eq!(
				evaluate_univariate(&coeffs, &x),
				evaluate_univariate_with_powers(&coeffs, x)
			);
		}
	}

	#[test]
	fn lagrange_evals_is_the_dual_basis_of_the_domain() {
		let mut rng = StdRng::seed_from_u64(0);

		// A one-point domain is the boundary: the basis is the constant 1, with no other point to
		// divide against.
		for dim in 0..=4 {
			let subspace = BinarySubspace::<F>::with_dim(dim);
			let domain: Vec<F> = subspace.iter().collect();

			// The basis sums to one everywhere, since it interpolates the constant polynomial 1.
			let evals = subspace.lagrange_evals(&F::random(&mut rng));
			assert_eq!(evals.iter().copied().sum::<F>(), F::ONE, "partition of unity at dim={dim}");

			// L_i(d_j) is one when i == j and zero otherwise.
			for (j, &d_j) in domain.iter().enumerate() {
				let at_domain = subspace.lagrange_evals(&d_j);
				for (i, &value) in at_domain.iter().enumerate() {
					let expected = if i == j { F::ONE } else { F::ZERO };
					assert_eq!(value, expected, "L_{i}({j}) at dim={dim}");
				}
			}
		}
	}

	#[test]
	fn lagrange_evals_buffer_holds_what_lagrange_evals_returns() {
		let mut rng = StdRng::seed_from_u64(0);

		// The buffer variant is a repack, so it must agree entry for entry and carry the dimension.
		for dim in 0..=4 {
			let subspace = BinarySubspace::<F>::with_dim(dim);
			let z = F::random(&mut rng);

			let buffer = subspace.lagrange_evals_buffer(z);
			assert_eq!(buffer.log_len(), dim);
			assert_eq!(buffer.iter_scalars().collect::<Vec<_>>(), subspace.lagrange_evals(&z));
		}
	}

	#[test]
	#[should_panic(expected = "precondition: values must hold one entry per domain point")]
	fn extrapolate_rejects_a_mismatched_value_count() {
		let subspace = BinarySubspace::<F>::with_dim(3);
		subspace.extrapolate(&[F::ONE; 4], &F::ONE);
	}

	proptest! {
		/// The shared-weight collapse must agree with one weight per point, built the long way.
		#[test]
		fn lagrange_evals_matches_the_per_point_weights(dim in 0usize..=5, seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let subspace = BinarySubspace::<F>::with_dim(dim);
			let domain: Vec<F> = subspace.iter().collect();
			let z = F::random(&mut rng);

			prop_assert_eq!(subspace.lagrange_evals(&z), lagrange_evals_reference(&domain, z));
		}

		/// Interpolating a polynomial's own evaluations must reproduce the polynomial.
		#[test]
		fn extrapolate_matches_direct_evaluation(dim in 0usize..=5, seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let subspace = BinarySubspace::<F>::with_dim(dim);

			// Degree below the domain size, so the interpolant is the polynomial itself.
			let coeffs: Vec<F> = random_scalars(&mut rng, 1 << dim);
			let values: Vec<F> = subspace
				.iter()
				.map(|point| evaluate_univariate(&coeffs, &point))
				.collect();

			let z = F::random(&mut rng);
			prop_assert_eq!(
				subspace.extrapolate(&values, &z),
				evaluate_univariate(&coeffs, &z)
			);
		}

		/// On arbitrary values, extrapolation must equal the inner product with the basis.
		#[test]
		fn extrapolate_matches_the_inner_product_with_the_basis(dim in 0usize..=5, seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let subspace = BinarySubspace::<B128>::with_dim(dim);

			// Values off any low-degree polynomial, so only the basis identity can hold.
			let values: Vec<B128> = random_scalars(&mut rng, 1 << dim);
			let z = B128::random(&mut rng);

			let expected = inner_product(values.iter().copied(), subspace.lagrange_evals(&z));
			prop_assert_eq!(subspace.extrapolate(&values, &z), expected);
		}
	}
}
