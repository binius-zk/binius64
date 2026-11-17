// Copyright 2026 The Binius Developers

//! The prefix equality indicator over the Boolean hypercube.
//!
//! For `m <= 2^n`, `eq_{<m}(X, Y)` is the `2n`-variate multilinear that agrees with
//! `eq(X, y)` at the hypercube vertices `y < m` and vanishes at the vertices `y >= m`, reading
//! a vertex as the integer whose bit `k` is coordinate `k`. Its partial evaluation
//! `eq_{<m}(r, ·)` is the length-`m` prefix of the eq tensor of `r`, zero-extended.
//!
//! The evaluators run the recursion from the lowest coordinate up, with `m = Σ m_k 2^k`:
//!
//! ```text
//! P_0 = 0,  E_0 = 1
//! P_{k+1} = (1 - X_k)(1 - Y_k) · (m_k ? E_k : P_k) + X_k Y_k · (m_k ? P_k : 0)
//! E_{k+1} = eq(X_k, Y_k) · E_k
//! ```
//!
//! Factoring each term as a lerp at `X_k` leaves only two products with `Y_k` per step,
//! `E_k Y_k` and `P_k Y_k`, which matters in the tensor algebra, where those are the costly
//! horizontal scalings:
//!
//! ```text
//! P_{k+1} = m_k ? lerp(X_k, E_k - E_k Y_k, P_k Y_k) : lerp(X_k, P_k - P_k Y_k, 0)
//! E_{k+1} = lerp(X_k, E_k - E_k Y_k, E_k Y_k)
//! ```
//!
//! where `lerp(t, a, b) = a + t (b - a)`, computed by [`extrapolate_line`] with arguments
//! `(a, b, t)`.
//!
//! `P_k` is the indicator of the low `k` coordinates against `m mod 2^k`, so the result is
//! `P_n`, except that `m = 2^n` selects `E_n = eq(X, Y)`.

use std::iter;

use binius_compute::{Allocator, CollectIntoAllocVec};
use binius_field::{ExtensionField, Field, PackedField, field::FieldOps};
use binius_utils::rayon::{iter::repeat_n, prelude::*, task_size::min_len_for_bytes};

use super::eq::{eq_ind_partial_eval_in, eq_ind_partial_eval_scalars};
use crate::{FieldBuffer, FieldVec, line::extrapolate_line, tensor_algebra::TensorAlgebra};

/// Evaluates the prefix equality indicator `eq_{<m}(x, y)` in `O(n)` field operations.
///
/// ## Preconditions
///
/// * `x` and `y` must have the same length `n`
/// * `m` must be at most `2^n`
pub fn eq_ind_prefix_eval<F: FieldOps>(m: usize, x: &[F], y: &[F]) -> F {
	let n = check_prefix_args(m, x.len(), y.len());

	let mut prefix = F::zero();
	let mut eq = F::one();
	for (k, (x_k, y_k)) in iter::zip(x, y).enumerate() {
		let eq_hi = eq.clone() * y_k;
		let eq_lo = eq - &eq_hi;
		let prefix_hi = prefix.clone() * y_k;
		prefix = if (m >> k) & 1 == 1 {
			extrapolate_line(eq_lo.clone(), prefix_hi, x_k.clone())
		} else {
			extrapolate_line(prefix - prefix_hi, F::zero(), x_k.clone())
		};
		eq = extrapolate_line(eq_lo, eq_hi, x_k.clone());
	}
	if m >> n == 1 { eq } else { prefix }
}

/// Evaluates `eq_{<m}(φ0(x), φ1(y))` in the tensor algebra.
///
/// `x` is embedded through the vertical subring and `y` through the horizontal subring. Each step
/// of the recursion costs two horizontal and two vertical scalings.
///
/// ## Preconditions
///
/// * `x` and `y` must have the same length `n`
/// * `m` must be at most `2^n`
pub fn eq_ind_prefix_eval_tensor_algebra<F, FE>(
	m: usize,
	x: &[FE],
	y: &[FE],
) -> TensorAlgebra<F, FE>
where
	F: Field,
	FE: FieldOps<Scalar: ExtensionField<F>>,
{
	let n = check_prefix_args(m, x.len(), y.len());

	// The lerp parameter is vertical, so it costs a vertical scaling only.
	let lerp =
		|t: &FE, a: TensorAlgebra<F, FE>, b: TensorAlgebra<F, FE>| (b - &a).scale_vertical(t) + &a;

	let mut prefix = TensorAlgebra::default();
	let mut eq = TensorAlgebra::one();
	for (k, (x_k, y_k)) in iter::zip(x, y).enumerate() {
		let eq_hi = eq.clone().scale_horizontal(y_k);
		let eq_lo = eq - &eq_hi;
		let prefix_hi = prefix.clone().scale_horizontal(y_k);
		prefix = if (m >> k) & 1 == 1 {
			lerp(x_k, eq_lo.clone(), prefix_hi)
		} else {
			lerp(x_k, prefix - &prefix_hi, TensorAlgebra::default())
		};
		eq = lerp(x_k, eq_lo, eq_hi);
	}
	if m >> n == 1 { eq } else { prefix }
}

/// Computes the partial evaluation `eq_{<m}(r, ·)` into a buffer drawn from an allocator: the eq
/// tensor of `r` with every entry at index `>= m` zeroed.
///
/// ## Preconditions
///
/// * `m` must be at most `2^n`, where `n` is the length of `point`
pub fn eq_ind_prefix_partial_eval_in<A: Allocator, P: PackedField>(
	alloc: &A,
	m: usize,
	point: &[P::Scalar],
) -> FieldVec<P, A> {
	let n = point.len();
	assert!(n < usize::BITS as usize && m <= 1 << n, "precondition: m must be at most 2^n");

	// Cut the point as the eq expansion does, so that
	//
	//     index = high * 2^low_len + low
	//     coeff = high_expansion[high] * low_expansion[low]
	//
	// Only the words holding scalars below m are multiplied out, so the cost is about m
	// multiplications rather than 2^n.
	let low_len = (n / 2).max(P::LOG_WIDTH).min(n);
	let (low_coords, high_coords) = point.split_at(low_len);
	let low = eq_ind_partial_eval_in::<A, P>(alloc, low_coords);
	let low_words = low.as_ref();
	let high = eq_ind_partial_eval_scalars(high_coords);

	// Each word is written once, straight into the allocator's buffer, with no zero-fill pass.
	let log_block_words = low_len.saturating_sub(P::LOG_WIDTH);
	let block_mask = (1 << log_block_words) - 1;
	let kept_words = m.div_ceil(P::WIDTH);
	let total_words = 1 << n.saturating_sub(P::LOG_WIDTH);
	let words = (0..kept_words)
		.into_par_iter()
		// A task walks consecutive words, so it broadcasts each block's coefficient once.
		.map_init(
			|| (usize::MAX, P::zero()),
			|(block, coeff), word| {
				if word >> log_block_words != *block {
					*block = word >> log_block_words;
					*coeff = P::broadcast(high[*block]);
				}
				low_words[word & block_mask] * *coeff
			},
		)
		.chain(repeat_n(P::zero(), total_words - kept_words))
		.with_min_len(min_len_for_bytes::<P>())
		.collect_into_alloc_vec(alloc);
	let mut values = FieldBuffer::new(n, words);

	// The last word multiplied out can run past m.
	for i in m..m.next_multiple_of(P::WIDTH).min(values.len()) {
		values.set(i, P::Scalar::ZERO);
	}
	values
}

fn check_prefix_args(m: usize, x_len: usize, y_len: usize) -> usize {
	assert_eq!(x_len, y_len, "precondition: x and y must be the same length");
	assert!(x_len < usize::BITS as usize && m <= 1 << x_len, "precondition: m must be at most 2^n");
	x_len
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{BinaryField1b as B1, Field};
	use rand::prelude::*;

	use super::*;
	use crate::{
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind,
		test_utils::{
			B128, Packed128b, index_to_hypercube_point, random_field_buffer, random_scalars,
		},
	};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn scalar_eval_matches_naive_sum() {
		let mut rng = StdRng::seed_from_u64(0);
		for n in 0..=4 {
			let x = random_scalars::<F>(&mut rng, n);
			let y = random_scalars::<F>(&mut rng, n);
			for m in 0..=1 << n {
				let expected = (0..m)
					.map(|v| {
						let vertex = index_to_hypercube_point::<F>(n, v);
						eq_ind(&x, &vertex) * eq_ind(&vertex, &y)
					})
					.sum::<F>();
				assert_eq!(eq_ind_prefix_eval(m, &x, &y), expected, "n={n}, m={m}");
			}
		}
	}

	#[test]
	fn scalar_eval_endpoints() {
		let mut rng = StdRng::seed_from_u64(0);
		let n = 5;
		let x = random_scalars::<F>(&mut rng, n);
		let y = random_scalars::<F>(&mut rng, n);
		assert_eq!(eq_ind_prefix_eval(1 << n, &x, &y), eq_ind(&x, &y));
		assert_eq!(eq_ind_prefix_eval(0, &x, &y), F::ZERO);
	}

	#[test]
	fn tensor_algebra_eval_matches_naive_sum() {
		let mut rng = StdRng::seed_from_u64(0);
		for n in 0..=3 {
			let x = random_scalars::<F>(&mut rng, n);
			let y = random_scalars::<F>(&mut rng, n);
			for m in 0..=1 << n {
				let expected = (0..m)
					.map(|v| {
						let vertex = index_to_hypercube_point::<F>(n, v);
						TensorAlgebra::<B1, F>::tensor(eq_ind(&x, &vertex), eq_ind(&vertex, &y))
					})
					.fold(TensorAlgebra::default(), |sum, term| sum + &term);
				assert_eq!(eq_ind_prefix_eval_tensor_algebra(m, &x, &y), expected, "n={n}, m={m}");
			}
		}
	}

	#[test]
	fn tensor_algebra_eval_with_subfield_horizontal_matches_scalar() {
		// A horizontal operand in the subfield embeds identically through both subrings, so the
		// result is the scalar indicator in the vertical subring.
		let mut rng = StdRng::seed_from_u64(0);
		let n = 4;
		let x = random_scalars::<F>(&mut rng, n);
		for y_index in 0..1 << n {
			let y = index_to_hypercube_point::<F>(n, y_index);
			for m in 0..=1 << n {
				let elem = eq_ind_prefix_eval_tensor_algebra::<B1, F>(m, &x, &y);
				assert_eq!(elem.try_extract_vertical(), Some(eq_ind_prefix_eval(m, &x, &y)));
			}
		}
	}

	#[test]
	fn partial_eval_inner_product_matches_scalar_eval() {
		let mut rng = StdRng::seed_from_u64(0);
		for n in 0..=7 {
			let r = random_scalars::<F>(&mut rng, n);
			let table = random_field_buffer::<P>(&mut rng, n);
			for m in 0..=1 << n {
				let tensor = eq_ind_prefix_partial_eval_in::<_, P>(&GlobalAllocator, m, &r);
				// Σ_y eq_{<m}(r, y) t(y), with the scalar indicator at each vertex y.
				let expected = (0..1 << n)
					.map(|v| {
						let vertex = index_to_hypercube_point::<F>(n, v);
						eq_ind_prefix_eval(m, &r, &vertex) * table.get(v)
					})
					.sum::<F>();
				assert_eq!(inner_product_buffers(&tensor, &table), expected, "n={n}, m={m}");
			}
		}
	}
}
