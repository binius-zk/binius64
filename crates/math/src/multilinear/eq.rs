// Copyright 2024-2025 Irreducible Inc.

use std::{iter, ops::DerefMut};

use binius_field::{
	Field, PackedField,
	field::FieldOps,
	packed::{get_packed_slice, set_packed_slice},
};
use binius_utils::rayon::prelude::*;

use crate::FieldBuffer;

/// Tensor of values with the eq indicator evaluated at extra_query_coordinates.
///
/// Let $n$ be log_n_values, $p$, $k$ be the lengths of `packed_values` and
/// `extra_query_coordinates`. Requires
///     * $n \geq k$
///     * p = max(1, 2^{n+k} / P::WIDTH)
/// Let $v$ be a vector corresponding to the first $2^n$ scalar values of `values`.
/// Let $r = (r_0, \ldots, r_{k-1})$ be the vector of `extra_query_coordinates`.
///
/// ## Preconditions
///
/// * `values` must have enough capacity: `values.log_cap() >= values.log_len() +
///   extra_query_coordinates.len()`
/// * `values` must be zero-extended to the new log length before calling this function. This
///   condition is necessary to get the best performance.
///
/// # Formal Definition
/// `values` is updated to contain the result of:
/// $v \otimes (1, r_0) \otimes \ldots \otimes (1, r_{k-1})$
/// (the monomial-basis eq indicator factors)
/// which is now a vector of length $2^{n+k}$. If 2^{n+k} < P::WIDTH, then
/// the result is packed into a single element of `values` where only the first
/// 2^{n+k} elements have meaning.
///
/// # Interpretation
/// Let $f$ be an $n$ variate multilinear polynomial that has evaluations over
/// the $n$ dimensional hypercube corresponding to $v$.
/// Then `values` is updated to contain the evaluations of $g$ over the $n+k$-dimensional
/// hypercube where
/// * $g(x_0, \ldots, x_{n+k-1}) = f(x_0, \ldots, x_{n-1}) * eq(x_n, \ldots, x_{n+k-1}, r)$
fn tensor_prod_eq_ind<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	extra_query_coordinates: &[P::Scalar],
) {
	let new_log_len = values.log_len() + extra_query_coordinates.len();

	assert!(
		values.log_cap() >= new_log_len,
		"precondition: values capacity must be sufficient for expansion"
	);

	for &r_i in extra_query_coordinates {
		let packed_r_i = P::broadcast(r_i);

		values.resize(values.log_len() + 1);
		let mut split = values.split_half_mut();
		let (mut lo, mut hi) = split.halves();

		(lo.as_mut(), hi.as_mut())
			.into_par_iter()
			.for_each(|(lo_i, hi_i)| {
				// Monomial basis: the tensor factor is `(1, r_i)`, so the low half (the constant
				// monomial coefficient) is unchanged and the high half is `lo * r_i`.
				*hi_i = (*lo_i) * packed_r_i;
			});
	}
}

/// Left tensor of values with the eq indicator evaluated at extra_query_coordinates.
///
/// # Formal definition
/// This differs from `tensor_prod_eq_ind` in tensor product being applied on the left
/// and in reversed order:
/// $(1, r_{k-1}) \otimes \ldots \otimes (1, r_0) \otimes v$
///
/// # Implementation
/// This operation is inplace, singlethreaded, and not very optimized. Main intent is to
/// use it on small tensors out of the hot paths.
///
/// ## Preconditions
///
/// * `values` must have enough capacity: `values.log_cap() >= values.log_len() +
///   extra_query_coordinates.len()`
pub fn tensor_prod_eq_ind_prepend<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	extra_query_coordinates: &[P::Scalar],
) {
	let new_log_len = values.log_len() + extra_query_coordinates.len();

	assert!(
		values.log_cap() >= new_log_len,
		"precondition: values capacity must be sufficient for expansion"
	);

	for &r_i in extra_query_coordinates.iter().rev() {
		values.zero_extend(values.log_len() + 1);
		for i in (0..values.len() / 2).rev() {
			let eval = get_packed_slice(values.as_ref(), i);
			// Monomial basis: factor `(1, r_i)` — low half unchanged, high half is `eval * r_i`.
			set_packed_slice(values.as_mut(), 2 * i, eval);
			set_packed_slice(values.as_mut(), 2 * i + 1, eval * r_i);
		}
	}
}

/// Computes the partial evaluation of the equality indicator polynomial.
///
/// Given an $n$-coordinate point $r_0, ..., r_n$, this computes the partial evaluation of the
/// equality indicator polynomial $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ and
/// returns its monomial coefficients over the $n$-dimensional hypercube.
///
/// The returned values are equal to the tensor product
///
/// $$
/// (1, r_0) \otimes ... \otimes (1, r_{n-1}),
/// $$
///
/// i.e. the monomial coefficients of $\widetilde{eq}(X, r) = \prod_i (1 + X_i r_i)$.
///
/// See [2026/762], Section 4.2 for the monomial-basis equality indicator.
///
/// [2026/762]: <https://eprint.iacr.org/2026/762>
pub fn eq_ind_partial_eval<P: PackedField>(point: &[P::Scalar]) -> FieldBuffer<P> {
	// The buffer needs to have the correct size: 2^max(point.len(), P::LOG_WIDTH) elements
	// but since tensor_prod_eq_ind starts with log_n_values=0, we need the final size
	let log_size = point.len();
	let mut buffer = FieldBuffer::zeros_truncated(0, log_size);
	buffer.set(0, P::Scalar::ONE);
	tensor_prod_eq_ind(&mut buffer, point);
	buffer
}

/// Truncate the equality indicator expansion to the low indexed variables.
///
/// This routine computes $\widetilde{eq}(X_0, ..., X_{n'-1}, r_0, ..., r_{n'-1})$ from
/// $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ where $n' \le n$. In the monomial basis
/// the expansion is the tensor $\bigotimes_i (1, r_i)$, whose low-variable restriction is exactly
/// the prefix of monomial coefficients (those whose high-variable monomials are constant), so the
/// truncation is a plain length truncation.
///
/// ## Preconditions
///
/// * `truncated_log_len` must be at most `values.log_len()`
pub fn eq_ind_truncate_low_inplace<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	truncated_log_len: usize,
) {
	assert!(
		truncated_log_len <= values.log_len(),
		"precondition: truncated_log_len must be at most values.log_len()"
	);

	values.truncate(truncated_log_len);
}

/// Evaluates the 2-variate multilinear which indicates the equality condition over the
/// infinity hypercube.
///
/// This evaluates the monomial-basis bivariate polynomial
///
/// $$
/// \widetilde{eq}(X, Y) = 1 + X Y
/// $$
///
/// the interpolant of the equality function on the infinity hypercube. See [2026/762], Section 4.2.
///
/// [2026/762]: <https://eprint.iacr.org/2026/762>
#[inline(always)]
pub fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
	F::one() + x * y
}

/// Evaluates the equality indicator multilinear at a pair of coordinates.
///
/// This evaluates the 2n-variate multilinear polynomial
///
/// $$
/// \widetilde{eq}(X_0, \ldots, X_{n-1}, Y_0, \ldots, Y_{n-1}) = \prod_{i=0}^{n-1} (1 + X_i Y_i)
/// $$
///
/// See [2026/762], Section 4.2 for the monomial-basis equality indicator.
///
/// [2026/762]: <https://eprint.iacr.org/2026/762>
pub fn eq_ind<F: FieldOps>(x: &[F], y: &[F]) -> F {
	assert_eq!(x.len(), y.len(), "pre-condition: x and y must be the same length");
	iter::zip(x, y)
		.map(|(x, y)| eq_one_var(x.clone(), y.clone()))
		.product()
}

/// Evaluates the equality indicator multilinear with one operand fixed to all zeros.
///
/// This is `eq_ind(0^n, point)`. In the monomial basis $\widetilde{eq}(0, Y) = 1 + 0 \cdot Y = 1$,
/// so this is identically `1`:
///
/// $$
/// \widetilde{eq}(0^n, Y_0, \ldots, Y_{n-1}) = \prod_{i=0}^{n-1} (1 + 0 \cdot Y_i) = 1.
/// $$
pub fn eq_ind_zero<F: FieldOps>(point: &[F]) -> F {
	let _ = point;
	F::one()
}

/// Computes the partial evaluation of the equality indicator polynomial, returning scalars.
///
/// This is a scalar-only variant of [`eq_ind_partial_eval`] that returns a `Vec<F>` instead of
/// a [`FieldBuffer`]. It computes the tensor product
///
/// $$
/// (1, r_0) \otimes ... \otimes (1, r_{n-1}).
/// $$
pub fn eq_ind_partial_eval_scalars<F: FieldOps>(point: &[F]) -> Vec<F> {
	let mut result = Vec::with_capacity(1 << point.len());
	result.push(F::one());

	for r_i in point {
		// Double the buffer size. Monomial-basis factor `(1, r_i)`: the lo half (constant monomial
		// coefficient) is unchanged and the hi half gets `val * r_i`.
		let len = result.len();
		for j in 0..len {
			let prod = result[j].clone() * r_i.clone();
			result.push(prod);
		}
	}
	result
}

#[cfg(test)]
mod tests {
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b, random_scalars};

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
		tensor_prod_eq_ind(&mut result, &query);
		let result_vec: Vec<F> = P::iter_slice(result.as_ref()).collect();
		// Monomial-basis tensor `(1, v0) ⊗ (1, v1)`.
		assert_eq!(result_vec, vec![F::ONE, v0, v1, v0 * v1]);
	}

	/// Monomial coefficient of `eq_ind_partial_eval(point)` at hypercube `index`:
	/// the product of `point[j]` over coordinates `j` whose bit is set in `index`.
	fn monomial_coeff(point: &[F], index: usize) -> F {
		(0..point.len())
			.filter(|j| (index >> j) & 1 == 1)
			.map(|j| point[j])
			.product()
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

			tensor_prod_eq_ind::<P, _>(&mut eq_expansion, &extra);
			coords.extend(&extra);

			assert_eq!(eq_expansion.log_len(), coords.len());
			for i in 0..eq_expansion.len() {
				let v = eq_expansion.get(i);
				assert_eq!(v, monomial_coeff(&coords, i));
			}
		}
	}

	#[test]
	fn test_eq_ind_zero() {
		let mut rng = StdRng::seed_from_u64(0);
		for n in 0..5 {
			let point = random_scalars::<F>(&mut rng, n);
			// Monomial basis: eq(0; ·) is identically 1.
			assert_eq!(eq_ind_zero(&point), F::ONE);
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
		assert_eq!(result_mut.get(0), F::ONE);
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
		let expected = vec![F::ONE, r0, r1, r0 * r1];
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

		let expected = vec![F::ONE, r0, r1, r0 * r1, r2, r0 * r2, r1 * r2, r0 * r1 * r2];
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

		// In the monomial basis the entry is the monomial coefficient at `index`.
		assert_eq!(partial_eval_value, monomial_coeff(&point, index));
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
	fn test_tensor_prod_eq_prepend_conforms_to_append() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 10;
		let base_vars = 4;

		let point = random_scalars::<F>(&mut rng, n_vars);

		let append = eq_ind_partial_eval(&point);

		let mut prepend = FieldBuffer::<P>::zeros_truncated(0, n_vars);
		let (prefix, suffix) = point.split_at(n_vars - base_vars);
		prepend.set(0, F::ONE);
		tensor_prod_eq_ind(&mut prepend, suffix);
		tensor_prod_eq_ind_prepend(&mut prepend, prefix);

		assert_eq!(append, prepend);
	}
}
