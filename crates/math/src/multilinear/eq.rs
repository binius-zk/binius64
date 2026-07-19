// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::DerefMut;

use binius_field::{PackedField, field::FieldOps};

use super::hypercube::{self, Hypercube, OneCube};
use crate::FieldBuffer;

/// Left tensor of values with the eq indicator evaluated at extra_query_coordinates.
///
/// This is [`hypercube::tensor_prod_eq_ind_prepend`] over the Boolean hypercube.
///
/// ## Preconditions
///
/// * `values` must have enough capacity: `values.log_cap() >= values.log_len() +
///   extra_query_coordinates.len()`
pub fn tensor_prod_eq_ind_prepend<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	extra_query_coordinates: &[P::Scalar],
) {
	hypercube::tensor_prod_eq_ind_prepend::<OneCube, _, _>(values, extra_query_coordinates)
}

/// Computes the partial evaluation of the equality indicator polynomial.
///
/// Given an $n$-coordinate point $r_0, ..., r_n$, this computes the partial evaluation of the
/// equality indicator polynomial $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ and
/// returns its values over the $n$-dimensional hypercube.
///
/// The returned values are equal to the tensor product
///
/// $$
/// (1 - r_0, r_0) \otimes ... \otimes (1 - r_{n-1}, r_{n-1}).
/// $$
///
/// See [DP23], Section 2.1 for more information about the equality indicator polynomial.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
pub fn eq_ind_partial_eval<P: PackedField>(point: &[P::Scalar]) -> FieldBuffer<P> {
	hypercube::eq_ind_partial_eval::<OneCube, P>(point)
}

/// Computes the partial evaluation of the equality indicator polynomial, scaled by a constant.
///
/// Every hypercube value of the equality indicator is multiplied by `scale`:
///
/// $$
/// scale \cdot (1 - r_0, r_0) \otimes ... \otimes (1 - r_{n-1}, r_{n-1}).
/// $$
///
/// # Arguments
///
/// * `point` - The evaluation point whose length is the number of variables.
/// * `scale` - The constant every returned value is multiplied by.
///
/// A scale of one reproduces the unscaled equality indicator.
pub fn scaled_eq_ind_partial_eval<P: PackedField>(
	point: &[P::Scalar],
	scale: P::Scalar,
) -> FieldBuffer<P> {
	hypercube::scaled_eq_ind_partial_eval::<OneCube, P>(point, scale)
}

/// Writes the scaled equality indicator expansion of `point` into a caller-supplied buffer.
///
/// This is the in-place form of [`scaled_eq_ind_partial_eval`]: the caller owns the output buffer,
/// so its allocation can be hoisted out of a hot or parallel region. On return `buffer` has
/// `log_len == point.len()`; any prior contents are overwritten. A scale of one reproduces the
/// unscaled equality indicator.
///
/// ## Preconditions
///
/// * `buffer.log_cap()` must be at least `point.len()`.
pub fn scaled_eq_ind_partial_eval_into<P: PackedField, Data: DerefMut<Target = [P]>>(
	buffer: &mut FieldBuffer<P, Data>,
	point: &[P::Scalar],
	scale: P::Scalar,
) {
	hypercube::scaled_eq_ind_partial_eval_into::<OneCube, _, _>(buffer, point, scale)
}

/// Truncate the equality indicator expansion to the low indexed variables.
///
/// This routine computes $\widetilde{eq}(X_0, ..., X_{n'-1}, r_0, ..., r_{n'-1})$ from
/// $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ where $n' \le n$ by repeatedly summing
/// field buffer "halves" inplace. The equality indicator expansion occupies a prefix of
/// the field buffer; scalars after the truncated length are zeroed out.
///
/// ## Preconditions
///
/// * `truncated_log_len` must be at most `values.log_len()`
pub fn eq_ind_truncate_low_inplace<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	truncated_log_len: usize,
) {
	hypercube::eq_ind_truncate_low_inplace::<OneCube, _, _>(values, truncated_log_len)
}

/// Evaluates the 2-variate multilinear which indicates the equality condition over the hypercube.
///
/// This evaluates the bivariate polynomial
///
/// $$
/// \widetilde{eq}(X, Y) = X Y + (1 - X) (1 - Y)
/// $$
///
/// In the special case of binary fields, the evaluation can be simplified to
///
/// $$
/// \widetilde{eq}(X, Y) = X + Y + 1
/// $$
#[inline(always)]
pub fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
	OneCube::eq_one_var(x, y)
}

/// Evaluates the equality indicator multilinear at a pair of coordinates.
///
/// This evaluates the 2n-variate multilinear polynomial
///
/// $$
/// \widetilde{eq}(X_0, \ldots, X_{n-1}, Y_0, \ldots, Y_{n-1}) = \prod_{i=0}^{n-1} X_i Y_i + (1 -
/// X_i) (1 - Y_i) $$
///
/// In the special case of binary fields, the evaluation can be simplified to
///
/// See [DP23], Section 2.1 for more information about the equality indicator polynomial.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
pub fn eq_ind<F: FieldOps>(x: &[F], y: &[F]) -> F {
	hypercube::eq_ind::<OneCube, F>(x, y)
}

/// Evaluates the equality indicator multilinear with one operand fixed to all zeros.
///
/// This is `eq_ind(0^n, point)`, which simplifies to
///
/// $$
/// \widetilde{eq}(0^n, Y_0, \ldots, Y_{n-1}) = \prod_{i=0}^{n-1} (1 - Y_i).
/// $$
pub fn eq_ind_zero<F: FieldOps>(point: &[F]) -> F {
	hypercube::eq_ind_zero::<OneCube, F>(point)
}

/// Computes the partial evaluation of the equality indicator polynomial, returning scalars.
///
/// This is a scalar-only variant of [`eq_ind_partial_eval`] that returns a `Vec<F>` instead of
/// a [`FieldBuffer`]. It computes the tensor product
///
/// $$
/// (1 - r_0, r_0) \otimes ... \otimes (1 - r_{n-1}, r_{n-1}).
/// $$
pub fn eq_ind_partial_eval_scalars<F: FieldOps>(point: &[F]) -> Vec<F> {
	hypercube::eq_ind_partial_eval_scalars::<OneCube, F>(point)
}

/// Computes the partial evaluation of the equality indicator polynomial scaled by a constant,
/// returning scalars.
///
/// This is a scalar-only variant of [`scaled_eq_ind_partial_eval`] that returns a `Vec<F>` instead
/// of a [`FieldBuffer`]. Every hypercube value of the tensor product
///
/// $$
/// (1 - r_0, r_0) \otimes ... \otimes (1 - r_{n-1}, r_{n-1})
/// $$
///
/// is multiplied by `scale`. A scale of one reproduces [`eq_ind_partial_eval_scalars`].
pub fn scaled_eq_ind_partial_eval_scalars<F: FieldOps>(point: &[F], scale: F) -> Vec<F> {
	hypercube::scaled_eq_ind_partial_eval_scalars::<OneCube, F>(point, scale)
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Random};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::{
		multilinear::hypercube::tensor_prod_eq_ind,
		test_utils::{B128, Packed128b, index_to_hypercube_point, random_scalars},
	};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn test_tensor_prod_eq_ind() {
		let v0 = F::from(1);
		let v1 = F::from(2);
		let query = vec![v0, v1];
		// log_len = 0, query.len() = 2, so total log_cap = 2
		let mut result = FieldBuffer::zeros_truncated(0, query.len());
		result.set(0, F::ONE);
		tensor_prod_eq_ind::<OneCube, P, _>(&mut result, &query);
		let result_vec: Vec<F> = P::iter_slice(result.as_ref()).collect();
		assert_eq!(
			result_vec,
			vec![
				(F::ONE - v0) * (F::ONE - v1),
				v0 * (F::ONE - v1),
				(F::ONE - v0) * v1,
				v0 * v1
			]
		);
	}

	#[test]
	fn test_tensor_prod_eq_ind_inplace_expansion() {
		let mut rng = StdRng::seed_from_u64(0);

		let exps = 4;
		let max_n_vars = exps * (exps + 1) / 2;
		let mut coords = Vec::with_capacity(max_n_vars);
		let mut eq_expansion = FieldBuffer::zeros_truncated(0, max_n_vars);
		eq_expansion.set(0, F::ONE);

		for extra_count in 1..=exps {
			let extra = random_scalars(&mut rng, extra_count);

			tensor_prod_eq_ind::<OneCube, P, _>(&mut eq_expansion, &extra);
			coords.extend(&extra);

			assert_eq!(eq_expansion.log_len(), coords.len());
			for i in 0..eq_expansion.len() {
				let v = eq_expansion.get(i);
				let hypercube_point = index_to_hypercube_point(coords.len(), i);
				assert_eq!(v, eq_ind(&hypercube_point, &coords));
			}
		}
	}

	#[test]
	fn test_eq_ind_zero() {
		let mut rng = StdRng::seed_from_u64(0);
		for n in 0..5 {
			let point = random_scalars::<F>(&mut rng, n);
			let expected: F = point.iter().map(|&r| F::ONE - r).product();
			assert_eq!(eq_ind_zero(&point), expected);
			assert_eq!(eq_ind_zero(&point), eq_ind(&vec![F::ZERO; n], &point));
		}
	}

	#[test]
	fn test_eq_ind_partial_eval_empty() {
		let result = eq_ind_partial_eval::<P>(&[]);
		// For P with LOG_WIDTH = 2, the minimum buffer size is 4 elements
		assert_eq!(result.log_len(), 0);
		assert_eq!(result.len(), 1);
		let result_mut = result;
		assert_eq!(result_mut.get(0), F::ONE);
	}

	#[test]
	fn test_eq_ind_partial_eval_single_var() {
		// Only one query coordinate
		let r0 = F::new(2);
		let result = eq_ind_partial_eval::<P>(&[r0]);
		assert_eq!(result.log_len(), 1);
		assert_eq!(result.len(), 2);
		let result_mut = result;
		assert_eq!(result_mut.get(0), F::ONE - r0);
		assert_eq!(result_mut.get(1), r0);
	}

	#[test]
	fn test_eq_ind_partial_eval_two_vars() {
		// Two query coordinates
		let r0 = F::new(2);
		let r1 = F::new(3);
		let result = eq_ind_partial_eval::<P>(&[r0, r1]);
		assert_eq!(result.log_len(), 2);
		assert_eq!(result.len(), 4);
		let result_vec: Vec<F> = P::iter_slice(result.as_ref()).collect();
		let expected = vec![
			(F::ONE - r0) * (F::ONE - r1),
			r0 * (F::ONE - r1),
			(F::ONE - r0) * r1,
			r0 * r1,
		];
		assert_eq!(result_vec, expected);
	}

	#[test]
	fn test_eq_ind_partial_eval_three_vars() {
		// Case with three query coordinates
		let r0 = F::new(2);
		let r1 = F::new(3);
		let r2 = F::new(5);
		let result = eq_ind_partial_eval::<P>(&[r0, r1, r2]);
		assert_eq!(result.log_len(), 3);
		assert_eq!(result.len(), 8);
		let result_vec: Vec<F> = P::iter_slice(result.as_ref()).collect();

		let expected = vec![
			(F::ONE - r0) * (F::ONE - r1) * (F::ONE - r2),
			r0 * (F::ONE - r1) * (F::ONE - r2),
			(F::ONE - r0) * r1 * (F::ONE - r2),
			r0 * r1 * (F::ONE - r2),
			(F::ONE - r0) * (F::ONE - r1) * r2,
			r0 * (F::ONE - r1) * r2,
			(F::ONE - r0) * r1 * r2,
			r0 * r1 * r2,
		];
		assert_eq!(result_vec, expected);
	}

	// Property-based test that eq_ind_partial_eval is consistent with eq_ind at a random index.
	#[test]
	fn test_eq_ind_partial_eval_consistent_on_hypercube() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 5;

		let point = random_scalars(&mut rng, n_vars);
		let result = eq_ind_partial_eval::<P>(&point);
		let index = rng.random_range(..1 << n_vars);

		// Query the value at that index
		let result_mut = result;
		let partial_eval_value = result_mut.get(index);

		let index_bits = index_to_hypercube_point(n_vars, index);
		let eq_ind_value = eq_ind(&point, &index_bits);

		assert_eq!(partial_eval_value, eq_ind_value);
	}

	#[test]
	fn test_eq_ind_truncate_low_inplace() {
		let mut rng = StdRng::seed_from_u64(0);

		let reds = 4;
		let n_vars = reds * (reds + 1) / 2;
		let point = random_scalars(&mut rng, n_vars);

		let mut eq_ind = eq_ind_partial_eval::<P>(&point);
		let mut log_n_values = n_vars;

		for reduction in (0..=reds).rev() {
			let truncated_log_n_values = log_n_values - reduction;
			eq_ind_truncate_low_inplace(&mut eq_ind, truncated_log_n_values);

			let eq_ind_ref = eq_ind_partial_eval::<P>(&point[..truncated_log_n_values]);
			assert_eq!(eq_ind_ref.len(), eq_ind.len());
			for i in 0..eq_ind.len() {
				assert_eq!(eq_ind.get(i), eq_ind_ref.get(i));
			}

			log_n_values = truncated_log_n_values;
		}

		assert_eq!(log_n_values, 0);
	}

	#[test]
	fn test_eq_ind_partial_eval_scalars_consistency() {
		let mut rng = StdRng::seed_from_u64(0);

		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);

			let packed_result = eq_ind_partial_eval::<P>(&point);
			let scalar_result = eq_ind_partial_eval_scalars(&point);

			let packed_scalars: Vec<F> = packed_result.iter_scalars().collect();
			assert_eq!(packed_scalars, scalar_result, "mismatch at log_n={log_n}");
		}
	}

	#[test]
	fn test_scaled_eq_ind_partial_eval_scalars_is_unscaled_times_scale() {
		let mut rng = StdRng::seed_from_u64(0);

		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = F::random(&mut rng);

			let scaled = scaled_eq_ind_partial_eval_scalars(&point, scale);
			let expected: Vec<F> = eq_ind_partial_eval_scalars(&point)
				.into_iter()
				.map(|x| x * scale)
				.collect();
			assert_eq!(scaled, expected, "mismatch at log_n={log_n}");
		}
	}

	#[test]
	fn test_tensor_prod_eq_prepend_conforms_to_append() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 10;
		let base_vars = 4;

		let point = random_scalars::<F>(&mut rng, n_vars);

		let append = eq_ind_partial_eval(&point);

		let mut prepend = FieldBuffer::<P>::zeros_truncated(0, n_vars);
		let (prefix, suffix) = point.split_at(n_vars - base_vars);
		prepend.set(0, F::ONE);
		tensor_prod_eq_ind::<OneCube, P, _>(&mut prepend, suffix);
		tensor_prod_eq_ind_prepend(&mut prepend, prefix);

		assert_eq!(append, prepend);
	}

	#[test]
	fn test_scaled_eq_ind_partial_eval_scale_one_matches_unscaled() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a scale of one is the identity on the expansion.
		// So the scaled and unscaled indicators must be the identical buffer.
		//
		// Sizes span the empty point (0 variables) up to a 256-value cube (8 variables).
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);

			// Equality is checked packed-word for packed-word, not just value by value.
			assert_eq!(
				scaled_eq_ind_partial_eval::<P>(&point, F::ONE),
				eq_ind_partial_eval::<P>(&point),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn scaled_eq_ind_partial_eval_into_matches_allocating() {
		let mut rng = StdRng::seed_from_u64(2);

		// Invariant: filling a caller-allocated buffer must reproduce the allocating variant
		// exactly, including for a buffer whose capacity exceeds the point length and whose
		// prior contents are stale.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// Seed the reusable buffer at the maximum capacity and fill it with unrelated data to
			// confirm `_into` overwrites rather than accumulates.
			let mut buffer = FieldBuffer::<P>::zeros_truncated(8, 8);
			for i in 0..buffer.len() {
				buffer.set(i, scale + F::ONE);
			}

			scaled_eq_ind_partial_eval_into(&mut buffer, &point, scale);

			assert_eq!(buffer.log_len(), log_n, "wrong length at log_n={log_n}");
			assert_eq!(
				buffer,
				scaled_eq_ind_partial_eval::<P>(&point, scale),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn test_scaled_eq_ind_partial_eval_scale_zero_is_zero() {
		let mut rng = StdRng::seed_from_u64(1);

		// Invariant: the expansion is linear in its starting value.
		// So a starting value of zero yields the all-zero polynomial.
		for log_n in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, log_n);

			// Every one of the 2^log_n hypercube values must be zero.
			let scaled = scaled_eq_ind_partial_eval::<P>(&point, F::ZERO);
			assert!(scaled.iter_scalars().all(|v| v == F::ZERO), "nonzero at log_n={log_n}");
		}
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(16))]

		// Invariant: scaling commutes with the expansion, value by value.
		//
		//     scaled_eq(point, s)[i] == s * eq(point)[i]   for every hypercube index i
		#[test]
		fn scaled_eq_ind_partial_eval_matches_scaled_reference(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			// Draw the point and an independent scale from one seeded stream.
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// Reference: the unscaled expansion, to be compared against the scaled one.
			let scaled = scaled_eq_ind_partial_eval::<P>(&point, scale);
			let reference = eq_ind_partial_eval::<P>(&point);

			// The scaled value at each index must equal the scale times the reference value.
			for (got, base) in scaled.iter_scalars().zip(reference.iter_scalars()) {
				prop_assert_eq!(got, scale * base);
			}
		}
	}
}
