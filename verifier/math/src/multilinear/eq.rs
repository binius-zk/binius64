// Copyright 2024-2025 Irreducible Inc.

use std::{iter, ops::DerefMut};

use binius_field::{
	Field, PackedField,
	packed::{get_packed_slice, set_packed_slice},
};
use binius_utils::rayon::prelude::*;

use crate::{Error, FieldBuffer};

/// Tensor of values with the eq indicator evaluated at extra_query_coordinates.
///
/// Let $n$ be log_n_values, $p$, $k$ be the lengths of `packed_values` and
/// `extra_query_coordinates`. Requires
///     * $n \geq k$
///     * p = max(1, 2^{n+k} / P::WIDTH)
/// Let $v$ be a vector corresponding to the first $2^n$ scalar values of `values`.
/// Let $r = (r_0, \ldots, r_{k-1})$ be the vector of `extra_query_coordinates`.
///
/// # Precondition:
/// * `values` must be zero-extended to the new log length before calling this function. This
///   condition is necessary to get the best performance.
///
/// # Formal Definition
/// `values` is updated to contain the result of:
/// $v \otimes (1 - r_0, r_0) \otimes \ldots \otimes (1 - r_{k-1}, r_{k-1})$
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
) -> Result<(), Error> {
	let new_log_len = values.log_len() + extra_query_coordinates.len();

	if values.log_cap() < new_log_len {
		return Err(Error::IncorrectArgumentLength {
			arg: "values.log_cap()".to_string(),
			expected: new_log_len,
		});
	}

	for &r_i in extra_query_coordinates {
		let packed_r_i = P::broadcast(r_i);

		values.resize(values.log_len() + 1)?;
		let mut split = values.split_half_mut().expect("doubled by zero_extend()");
		let (mut lo, mut hi) = split.halves();

		(lo.as_mut(), hi.as_mut())
			.into_par_iter()
			.for_each(|(lo_i, hi_i)| {
				let prod = (*lo_i) * packed_r_i;
				*lo_i -= prod;
				*hi_i = prod;
			});
	}

	Ok(())
}

/// Left tensor of values with the eq indicator evaluated at extra_query_coordinates.
///
/// # Formal definition
/// This differs from `tensor_prod_eq_ind` in tensor product being applied on the left
/// and in reversed order:
/// $(1 - r_{k-1}, r_{k-1}) \otimes \ldots \otimes (1 - r_0, r_0) \otimes v$
///
/// # Implementation
/// This operation is inplace, singlethreaded, and not very optimized. Main intent is to
/// use it on small tensors out of the hot paths.
pub fn tensor_prod_eq_ind_prepend<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	extra_query_coordinates: &[P::Scalar],
) -> Result<(), Error> {
	let new_log_len = values.log_len() + extra_query_coordinates.len();

	if values.log_cap() < new_log_len {
		return Err(Error::IncorrectArgumentLength {
			arg: "values.log_cap()".to_string(),
			expected: new_log_len,
		});
	}

	for &r_i in extra_query_coordinates.iter().rev() {
		values.zero_extend(values.log_len() + 1)?;
		for i in (0..values.len() / 2).rev() {
			let eval = get_packed_slice(values.as_ref(), i);
			set_packed_slice(values.as_mut(), 2 * i, eval * (P::Scalar::ONE - r_i));
			set_packed_slice(values.as_mut(), 2 * i + 1, eval * r_i);
		}
	}

	Ok(())
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
	// The buffer needs to have the correct size: 2^max(point.len(), P::LOG_WIDTH) elements
	// but since tensor_prod_eq_ind starts with log_n_values=0, we need the final size
	let log_size = point.len();
	let mut buffer = FieldBuffer::zeros_truncated(0, log_size).expect("log_size >= 0");
	buffer.set(0, P::Scalar::ONE);
	tensor_prod_eq_ind(&mut buffer, point).expect("buffer is allocated with the correct length");
	buffer
}

/// Truncate the equality indicator expansion to the low indexed variables.
///
/// This routine computes $\widetilde{eq}(X_0, ..., X_{n'-1}, r_0, ..., r_{n'-1})$ from
/// $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ where $n' \le n$ by repeatedly summing
/// field buffer "halves" inplace. The equality indicator expansion occupies a prefix of
/// the field buffer; scalars after the truncated length are zeroed out.
pub fn eq_ind_truncate_low_inplace<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	truncated_log_len: usize,
) -> Result<(), Error> {
	if truncated_log_len > values.log_len() {
		return Err(Error::ArgumentRangeError {
			arg: "truncated_log_len".to_string(),
			range: 0..values.log_len() + 1,
		});
	}

	for log_len in (truncated_log_len..values.log_len()).rev() {
		{
			let mut split = values.split_half_mut()?;
			let (mut lo, hi) = split.halves();
			(lo.as_mut(), hi.as_ref())
				.into_par_iter()
				.for_each(|(zero, one)| {
					*zero += *one;
				});
		}

		values.truncate(log_len);
	}

	Ok(())
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
pub fn eq_one_var<F: Field>(x: F, y: F) -> F {
	if F::CHARACTERISTIC == 2 {
		// Optimize away the multiplication for binary fields
		x + y + F::ONE
	} else {
		x * y + (F::ONE - x) * (F::ONE - y)
	}
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
pub fn eq_ind<F: Field>(x: &[F], y: &[F]) -> F {
	assert_eq!(x.len(), y.len(), "pre-condition: x and y must be the same length");
	iter::zip(x, y).map(|(&x, &y)| eq_one_var(x, y)).product()
}

#[cfg(test)]
mod tests {
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b, index_to_hypercube_point, random_scalars};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn test_tensor_prod_eq_ind() {
		let v0 = F::from(1);
		let v1 = F::from(2);
		let query = vec![v0, v1];
		// log_len = 0, query.len() = 2, so total log_cap = 2
		let mut result = FieldBuffer::zeros_truncated(0, query.len()).unwrap();
		result.set_checked(0, F::ONE).unwrap();
		tensor_prod_eq_ind(&mut result, &query).unwrap();
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
		let mut eq_expansion = FieldBuffer::zeros_truncated(0, max_n_vars).unwrap();
		eq_expansion.set_checked(0, F::ONE).unwrap();

		for extra_count in 1..=exps {
			let extra = random_scalars(&mut rng, extra_count);

			tensor_prod_eq_ind::<P, _>(&mut eq_expansion, &extra).unwrap();
			coords.extend(&extra);

			assert_eq!(eq_expansion.log_len(), coords.len());
			for i in 0..eq_expansion.len() {
				let v = eq_expansion.get_checked(i).unwrap();
				let hypercube_point = index_to_hypercube_point(coords.len(), i);
				assert_eq!(v, eq_ind(&hypercube_point, &coords));
			}
		}
	}

	#[test]
	fn test_eq_ind_partial_eval_empty() {
		let result = eq_ind_partial_eval::<P>(&[]);
		// For P with LOG_WIDTH = 2, the minimum buffer size is 4 elements
		assert_eq!(result.log_len(), 0);
		assert_eq!(result.len(), 1);
		let result_mut = result;
		assert_eq!(result_mut.get_checked(0).unwrap(), F::ONE);
	}

	#[test]
	fn test_eq_ind_partial_eval_single_var() {
		// Only one query coordinate
		let r0 = F::new(2);
		let result = eq_ind_partial_eval::<P>(&[r0]);
		assert_eq!(result.log_len(), 1);
		assert_eq!(result.len(), 2);
		let result_mut = result;
		assert_eq!(result_mut.get_checked(0).unwrap(), F::ONE - r0);
		assert_eq!(result_mut.get_checked(1).unwrap(), r0);
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
		let partial_eval_value = result_mut.get_checked(index).unwrap();

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
			eq_ind_truncate_low_inplace(&mut eq_ind, truncated_log_n_values).unwrap();

			let eq_ind_ref = eq_ind_partial_eval::<P>(&point[..truncated_log_n_values]);
			assert_eq!(eq_ind_ref.len(), eq_ind.len());
			for i in 0..eq_ind.len() {
				assert_eq!(eq_ind.get_checked(i).unwrap(), eq_ind_ref.get_checked(i).unwrap());
			}

			log_n_values = truncated_log_n_values;
		}

		assert_eq!(log_n_values, 0);
	}

	#[test]
	fn test_tensor_prod_eq_prepend_conforms_to_append() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 10;
		let base_vars = 4;

		let point = random_scalars::<F>(&mut rng, n_vars);

		let append = eq_ind_partial_eval(&point);

		let mut prepend = FieldBuffer::<P>::zeros_truncated(0, n_vars).unwrap();
		let (prefix, suffix) = point.split_at(n_vars - base_vars);
		prepend.set_checked(0, F::ONE).unwrap();
		tensor_prod_eq_ind(&mut prepend, suffix).unwrap();
		tensor_prod_eq_ind_prepend(&mut prepend, prefix).unwrap();

		assert_eq!(append, prepend);
	}
}
