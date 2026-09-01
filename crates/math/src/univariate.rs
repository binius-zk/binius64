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
///     w = (prod_{d != 0} d)^{-1}
/// ```
///
/// # Algorithm
///
/// That product is the linear coefficient of the subspace polynomial, and the subspace polynomial
/// has a recurrence:
///
/// ```text
///     W_0(X)     = X
///     W_{i+1}(X) = W_i(X) * (W_i(X) + W_i(b_i))
/// ```
///
/// Squaring a linearized polynomial doubles every exponent, so it contributes no linear term.
/// Each step therefore multiplies the linear coefficient by one number, leaving
///
/// ```text
///     prod_{d != 0} d = prod_i W_i(b_i)
/// ```
///
/// which costs a square of the dimension rather than one multiplication per point of the domain.
///
/// The weight depends on the subspace alone, so all of it runs in the domain's own field and
/// crosses into the arithmetic field once.
/// That is what allows the checked inversion below: one wrapper channel's element type offers the
/// unchecked inverse alone.
///
/// # Panics
///
/// Panics if the basis is linearly dependent, which makes the product vanish.
fn barycentric_weight<F, E, Data>(subspace: &BinarySubspace<F, Data>) -> E
where
	F: BinaryField,
	E: FieldOps + From<F>,
	Data: Deref<Target = [F]>,
{
	// Seed the recurrence at the polynomial `X`, whose values on the basis are the basis itself.
	let mut evals = subspace.basis().to_vec();

	let mut product = F::ONE;
	for i in 0..evals.len() {
		// Entry `i` has reached the polynomial vanishing on everything below it, so it is the
		// factor this step contributes.
		let normalizer = evals[i];
		product *= normalizer;

		// Advance the entries still to come one polynomial along.
		for eval in &mut evals[i + 1..] {
			*eval *= *eval + normalizer;
		}
	}

	// Invariant: each factor is a subspace polynomial evaluated off the subspace it vanishes on,
	// which is nonzero exactly when the basis is independent. The subspace type leaves that to its
	// caller, so it is checked here rather than assumed.
	assert_ne!(product, F::ZERO, "precondition: the subspace basis must be independent");

	E::from(product.invert_or_zero())
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Field, Ghash128b, Random, Rijndael8b, arithmetic_traits::InvertOrZero, util::powers,
	};
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
	fn the_weight_recurrence_matches_the_product_over_the_subspace() {
		// Invariant: the weight is the inverse of the product of every nonzero point of the
		// domain. That product is what the weight is defined as, so it is the reference here.
		//
		// Fixture state: dim 0 is the one-point domain, whose empty product is one.
		// Dim 8 is 255 factors, enough that a wrong recurrence cannot coincide.
		for dim in 0..=8 {
			let subspace = BinarySubspace::<F>::with_dim(dim);

			let product: F = subspace.iter().skip(1).product();
			let expected = product.invert_or_zero();

			assert_eq!(barycentric_weight::<F, F, _>(&subspace), expected, "dim={dim}");
		}
	}

	#[test]
	fn the_weight_crosses_fields_once_and_lands_on_the_same_value() {
		// Invariant: which field the arithmetic runs in cannot change the weight.
		//
		// The recurrence runs entirely in the domain's own field and embeds its result, so this
		// pins that the embedding lands on the finished weight rather than partway through.
		for dim in 0..=6 {
			let subspace = BinarySubspace::<Rijndael8b>::with_dim(dim);

			let native = barycentric_weight::<Rijndael8b, Rijndael8b, _>(&subspace);
			let embedded = barycentric_weight::<Rijndael8b, B128, _>(&subspace);

			assert_eq!(embedded, B128::from(native), "dim={dim}");
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn a_dependent_basis_is_rejected() {
		// A repeated basis element spans fewer dimensions than the basis claims, so some index
		// past zero also maps to the zero point and the product vanishes.
		let subspace = BinarySubspace::<F>::new_unchecked(vec![F::ONE, F::ONE]);
		let _ = barycentric_weight::<F, F, _>(&subspace);
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
