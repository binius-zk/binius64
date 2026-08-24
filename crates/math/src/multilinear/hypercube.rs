// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Multilinear tensor expansions, generic over the hypercube the coefficients are indexed by.
//!
//! A cube fixes the per-variable basis through three required methods.
//! Every expansion routine is a provided method, so both cubes share one implementation.

use std::{iter, slice};

use binius_compute::Allocator;
use binius_field::{Field, PackedField, field::FieldOps};
use binius_utils::{
	buffer::{BufferData, VecLike},
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};

use crate::{FieldBuffer, FieldVec};

/// A hypercube of coefficients for multilinear polynomials.
///
/// An $n$-variate multilinear is represented by $2^n$ coefficients against a polynomial basis that
/// factors as a tensor product over the variables. Each variable contributes the same linear basis
/// $(b_0, b_1)$, which determines the cube completely.
///
/// * [`OneCube`] is the Boolean hypercube $\\{0, 1\\}^n$, with basis $(1 - X, X)$. The coefficients
///   of a multilinear are its evaluations over the cube.
/// * [`InfCube`] is the infinity hypercube $\\{0, \infty\\}^n$, with basis $(1, X)$. The
///   coefficients of a multilinear are its monomial coefficients.
pub trait Hypercube: Sized {
	/// Evaluates the linear basis of one variable at a coordinate.
	///
	/// Returns $(b_0(r), b_1(r))$ for the coordinate $r$.
	fn basis<F: FieldOps>(coord: &F) -> [F; 2];

	/// Scales the linear basis of one variable by a value.
	///
	/// Returns $(v \cdot b_0(r), v \cdot b_1(r))$ for the value $v$ and coordinate $r$. This is the
	/// inner loop of a tensor expansion, so implementations do it in fewer multiplications than the
	/// two that scaling [`Hypercube::basis`] would take.
	fn expand_var<F: FieldOps>(value: &F, coord: &F) -> [F; 2];

	/// Contracts the two halves of a tensor expansion, stripping one variable's basis factor.
	///
	/// The halves hold $v \cdot b_0(r)$ and $v \cdot b_1(r)$ for the stripped variable's coordinate
	/// $r$; `lo` is overwritten with $v$. This is the sum $\sum_i w_i \cdot v \cdot b_i(r)$ for the
	/// unique weights $w$ with $\sum_i w_i b_i(X) = 1$, which recover $v$ whatever $r$ is.
	fn contract_var<F: FieldOps>(lo: &mut F, hi: &F);

	/// Evaluates the equality indicator of one variable, $\sum_i b_i(X) b_i(Y)$.
	fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
		let [x_0, x_1] = Self::basis(&x);
		let [y_0, y_1] = Self::basis(&y);
		x_0 * y_0 + x_1 * y_1
	}

	/// Tensor of values with the equality indicator evaluated at `extra_query_coordinates`.
	///
	/// Let $n$ be the log length of `values` and $k$ the number of extra coordinates.
	/// The returned buffer grows its backing `Vec` by one variable per coordinate.
	///
	/// # Formal Definition
	///
	/// Write $r = (r_0, \ldots, r_{k-1})$ for the extra coordinates.
	/// The result is the tensor product of the $2^n$ values with the cube's bases at those $r_i$:
	///
	/// $$
	/// v \otimes b(r_0) \otimes \ldots \otimes b(r_{k-1}),
	/// $$
	///
	/// a vector of length $2^{n+k}$.
	///
	/// # Interpretation
	///
	/// Let $f$ be the $n$-variate multilinear with coefficients $v$ over the cube.
	/// Then the result holds the coefficients of the $(n+k)$-variate multilinear
	///
	/// $$
	/// g(X_0, \ldots, X_{n+k-1}) = f(X_0, \ldots, X_{n-1}) \cdot
	///     \widetilde{eq}(X_n, \ldots, X_{n+k-1}, r).
	/// $$
	fn tensor_prod_eq_ind<P: PackedField>(
		values: FieldBuffer<P, Vec<P>>,
		extra_query_coordinates: &[P::Scalar],
	) -> FieldBuffer<P, Vec<P>> {
		let start_log_len = values.log_len();
		let final_log_len = start_log_len + extra_query_coordinates.len();
		let mut data = values.take_data();

		// Reserve the full final capacity once, so no growth step reallocates. Each packed step
		// then grows the length in place through the reserved spare capacity, writing the new
		// coefficients directly rather than zero-initializing a region the expansion overwrites.
		let final_packed_len = 1usize << final_log_len.saturating_sub(P::LOG_WIDTH);
		data.reserve_exact(final_packed_len.saturating_sub(data.len()));

		tensor_prod_eq_ind_reserved::<Self, P, _>(
			FieldBuffer::new(start_log_len, data),
			extra_query_coordinates,
		)
	}

	/// Computes the partial evaluation of the equality indicator polynomial.
	///
	/// Take an $n$-coordinate point $r = (r_0, \ldots, r_{n-1})$.
	/// The coefficients of $\widetilde{eq}(X_0, \ldots, X_{n-1}, r)$ over the cube are
	///
	/// $$
	/// b(r_0) \otimes \ldots \otimes b(r_{n-1}).
	/// $$
	fn eq_ind_partial_eval<P: PackedField>(point: &[P::Scalar]) -> FieldBuffer<P> {
		// The unscaled indicator is the scaled indicator with a scale of one.
		Self::scaled_eq_ind_partial_eval::<P>(point, P::Scalar::ONE)
	}

	/// Builds the equality indicator expansion of `point` into a buffer drawn from `alloc`.
	///
	/// This is the allocator-aware counterpart to the plain expansion.
	/// Under a `BufferPool` the result is a recyclable pooled buffer rather than a fresh `Vec`.
	fn eq_ind_partial_eval_in<A: Allocator, P: PackedField>(
		alloc: &A,
		point: &[P::Scalar],
	) -> FieldVec<P, A> {
		let packed_len = 1 << point.len().saturating_sub(P::LOG_WIDTH);
		Self::scaled_eq_ind_partial_eval_into::<P, _>(
			point,
			P::Scalar::ONE,
			alloc.alloc::<P>(packed_len),
		)
	}

	/// Computes the partial evaluation of the equality indicator polynomial, scaled by a constant.
	///
	/// Every coefficient of the equality indicator is multiplied by `scale`.
	/// A scale of one reproduces the unscaled expansion.
	///
	/// # Arguments
	///
	/// * `point` - The evaluation point whose length is the number of variables.
	/// * `scale` - The constant every returned value is multiplied by.
	fn scaled_eq_ind_partial_eval<P: PackedField>(
		point: &[P::Scalar],
		scale: P::Scalar,
	) -> FieldBuffer<P> {
		// Reserve the final packed length up front so the per-variable growth never reallocates.
		let packed_len = 1 << point.len().saturating_sub(P::LOG_WIDTH);
		Self::scaled_eq_ind_partial_eval_into::<P, _>(point, scale, Vec::with_capacity(packed_len))
	}

	/// Builds the scaled equality indicator expansion of `point` in a caller-supplied buffer.
	///
	/// This is the allocation-hoisting form of the scaled expansion.
	/// The caller owns the backing buffer, so its allocation can be drawn from a pool.
	/// It can equally be reserved on a different thread than the one that fills it.
	///
	/// The buffer is cleared and seeded with `scale`.
	/// Each coordinate then multiplies its basis in, doubling the length.
	/// The returned buffer has `log_len == point.len()`.
	///
	/// # Preconditions
	///
	/// * `buffer.capacity()` must be at least `1 << point.len().saturating_sub(P::LOG_WIDTH)`, so
	///   the growth never reallocates and the final wrap keeps the same allocation.
	fn scaled_eq_ind_partial_eval_into<P: PackedField, Data: VecLike<P>>(
		point: &[P::Scalar],
		scale: P::Scalar,
		mut buffer: Data,
	) -> FieldBuffer<P, Data> {
		let packed_len = 1usize << point.len().saturating_sub(P::LOG_WIDTH);
		assert!(
			buffer.capacity() >= packed_len,
			"precondition: buffer capacity must cover the packed expansion length"
		);

		// Seed a single-scalar buffer with the scale; the expansion multiplies it through, so every
		// coefficient ends up scaled.
		buffer.clear();
		buffer.push(P::from_scalars(iter::once(scale)));
		let values = FieldBuffer::new(0, buffer);
		tensor_prod_eq_ind_reserved::<Self, P, Data>(values, point)
	}

	/// Truncate the equality indicator expansion to the low indexed variables.
	///
	/// Each step contracts the two halves of the buffer, stripping the highest variable.
	/// Truncating to $n'$ variables therefore leaves the indicator over $r_0, \ldots, r_{n'-1}$.
	///
	/// The expansion occupies a prefix of the field buffer.
	/// Scalars after the truncated length are zeroed out.
	///
	/// ## Preconditions
	///
	/// * `truncated_log_len` must be at most `values.log_len()`
	fn eq_ind_truncate_low_inplace<P: PackedField, Data: BufferData<P>>(
		values: &mut FieldBuffer<P, Data>,
		truncated_log_len: usize,
	) {
		assert!(
			truncated_log_len <= values.log_len(),
			"precondition: truncated_log_len must be at most values.log_len()"
		);

		for log_len in (truncated_log_len..values.log_len()).rev() {
			{
				let mut split = values.split_half_mut();
				let (mut lo, hi) = split.halves();
				// Contracting a variable costs additions only.
				// So the cost of one step is the two words it reads, not its arithmetic.
				(lo.as_mut(), hi.as_ref())
					.into_par_iter()
					.with_min_task_bytes::<[P; 2]>()
					.for_each(|(zero, one)| {
						Self::contract_var(zero, one);
					});
			}

			values.truncate(log_len);
		}
	}

	/// Evaluates the equality indicator multilinear at a pair of points.
	///
	/// This evaluates the $2n$-variate multilinear polynomial
	///
	/// $$
	/// \widetilde{eq}(X_0, \ldots, X_{n-1}, Y_0, \ldots, Y_{n-1}) =
	///     \prod_{i=0}^{n-1} \sum_j b_j(X_i) b_j(Y_i).
	/// $$
	fn eq_ind<F: FieldOps>(x: &[F], y: &[F]) -> F {
		assert_eq!(x.len(), y.len(), "pre-condition: x and y must be the same length");
		iter::zip(x, y)
			.map(|(x, y)| Self::eq_one_var(x.clone(), y.clone()))
			.product()
	}

	/// Evaluates the equality indicator multilinear with one operand fixed to all zeros.
	///
	/// This is the indicator at $(0^n, point)$.
	/// It is the product of the cube's constant basis polynomial over the coordinates:
	///
	/// $$
	/// \widetilde{eq}(0^n, Y_0, \ldots, Y_{n-1}) = \prod_{i=0}^{n-1} b_0(Y_i).
	/// $$
	fn eq_ind_zero<F: FieldOps>(point: &[F]) -> F {
		point
			.iter()
			.map(|y| {
				let [y_0, _] = Self::basis(y);
				y_0
			})
			.product()
	}

	/// Computes the partial evaluation of the equality indicator polynomial, returning scalars.
	///
	/// This is the scalar-only variant of the expansion.
	/// It returns a `Vec<F>` rather than a packed field buffer.
	fn eq_ind_partial_eval_scalars<F: FieldOps>(point: &[F]) -> Vec<F> {
		// The unscaled indicator is the scaled indicator with a scale of one.
		Self::scaled_eq_ind_partial_eval_scalars::<F>(point, F::one())
	}

	/// Computes the scaled partial evaluation of the equality indicator, returning scalars.
	///
	/// This is the scalar-only variant of the scaled expansion.
	/// It returns a `Vec<F>` rather than a packed field buffer.
	/// A scale of one reproduces the unscaled scalars.
	fn scaled_eq_ind_partial_eval_scalars<F: FieldOps>(point: &[F], scale: F) -> Vec<F> {
		let mut result = Vec::with_capacity(1 << point.len());
		// Seed with the scale; the expansion multiplies it through every coefficient.
		result.push(scale);

		for r_i in point {
			// Double the buffer size. For each existing value in 0..size, the lo half gets the
			// value scaled by the constant basis polynomial and the hi half by the linear one.
			// Process in reverse so that writes to hi don't overwrite values we need.
			let len = result.len();
			for j in 0..len {
				let [lo, hi] = Self::expand_var(&result[j], r_i);
				result[j] = lo;
				result.push(hi);
			}
		}
		result
	}
}

/// The Boolean hypercube $\\{0, 1\\}^n$, whose linear basis is $(1 - X, X)$.
#[derive(Debug)]
pub struct OneCube;

impl Hypercube for OneCube {
	#[inline(always)]
	fn basis<F: FieldOps>(coord: &F) -> [F; 2] {
		[F::one() - coord, coord.clone()]
	}

	#[inline(always)]
	fn expand_var<F: FieldOps>(value: &F, coord: &F) -> [F; 2] {
		// Both basis polynomials share the product `value * coord`, so one multiplication suffices.
		let prod = value.clone() * coord;
		[value.clone() - &prod, prod]
	}

	#[inline(always)]
	fn contract_var<F: FieldOps>(lo: &mut F, hi: &F) {
		// The basis polynomials sum to one, so the weights are both one.
		*lo += hi;
	}

	#[inline(always)]
	fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
		// Over characteristic 2, `X·Y + (1−X)(1−Y)` simplifies to `X + Y + 1` (the `2·X·Y` term
		// vanishes). The condition is a compile-time constant, so only one arm is generated.
		if F::Scalar::CHARACTERISTIC == 2 {
			x + y + F::one()
		} else {
			let one = F::one();
			x.clone() * y.clone() + (one.clone() - x) * (one - y)
		}
	}
}

/// The infinity hypercube $\\{0, \infty\\}^n$, whose linear basis is $(1, X)$.
///
/// The vertex $\infty$ selects a multilinear's leading coefficient in that variable, so a
/// coefficient indexed by $v \in \\{0, \infty\\}^n$ is the monomial coefficient of $\prod_{i : v_i
/// = \infty} X_i$.
#[derive(Debug)]
pub struct InfCube;

impl Hypercube for InfCube {
	#[inline(always)]
	fn basis<F: FieldOps>(coord: &F) -> [F; 2] {
		[F::one(), coord.clone()]
	}

	#[inline(always)]
	fn expand_var<F: FieldOps>(value: &F, coord: &F) -> [F; 2] {
		// The constant basis polynomial leaves the value alone.
		[value.clone(), value.clone() * coord]
	}

	#[inline(always)]
	fn contract_var<F: FieldOps>(_lo: &mut F, _hi: &F) {
		// The constant basis polynomial is already one, so the low half is the value and the
		// weights are one and zero.
	}

	#[inline(always)]
	fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
		F::one() + x * y
	}
}

/// A tensor expansion over a backing store that already has room for the result.
///
/// The public entry point reserves the final capacity on its `Vec` before calling here.
/// The allocator-backed callers pass a buffer drawn at the final size, which cannot grow.
/// For them the reservation is a precondition rather than a step of the expansion.
///
/// This is a free function because it is private, while every method of a public trait is public.
///
/// # Preconditions
///
/// * `values.capacity()` must be at least `1 << (values.log_len() +
///   extra_query_coordinates.len()).saturating_sub(P::LOG_WIDTH)`.
fn tensor_prod_eq_ind_reserved<Cube: Hypercube, P: PackedField, Data: VecLike<P>>(
	values: FieldBuffer<P, Data>,
	extra_query_coordinates: &[P::Scalar],
) -> FieldBuffer<P, Data> {
	let start_log_len = values.log_len();
	let final_log_len = start_log_len + extra_query_coordinates.len();
	let mut data = values.take_data();

	// precondition
	debug_assert!(data.capacity() >= 1usize << final_log_len.saturating_sub(P::LOG_WIDTH));

	// The coordinates split cleanly: while the expansion is narrower than one packed word it lives
	// entirely in `data[0]`, and once it fills a word every step doubles the packed length.
	let sub_width_count = extra_query_coordinates
		.len()
		.min(P::LOG_WIDTH.saturating_sub(start_log_len));
	let (sub_width_coords, packed_coords) = extra_query_coordinates.split_at(sub_width_count);

	// Sub-packing-width: the whole buffer is the single word `data[0]`. Split it into the two
	// `log_len`-variable halves, expand, and interleave the halves back together — the backing
	// `Vec` stays one element.
	for (i, &r_i) in sub_width_coords.iter().enumerate() {
		let log_len = start_log_len + i;
		let packed_r_i = P::broadcast(r_i);
		let (lo, _) = data[0].interleave(P::zero(), log_len);
		let [lo, hi] = Cube::expand_var(&lo, &packed_r_i);
		data[0] = lo.interleave(hi, log_len).0;
	}

	// Packed: the `old_packed` initialized words are the low half of the result. Expand each into
	// its low word (in place) and its high word (written once into the reserved spare capacity),
	// then bump the length to cover the newly written high half.
	for &r_i in packed_coords {
		let packed_r_i = P::broadcast(r_i);
		let old_packed = data.len();

		// The high half is written once through the reserved spare capacity; the low half is the
		// initialized prefix, expanded in place. `spare_capacity_mut` gives the high half directly;
		// the low half needs `from_raw_parts_mut` because the safe two-slice split
		// (`Vec::split_at_spare_mut`) is still unstable (rust-lang/rust#81944).
		let low_ptr = data.as_mut_ptr();
		let high = &mut data.spare_capacity_mut()[..old_packed];
		// SAFETY: `[0, old_packed)` is the initialized low half, disjoint from the spare `high`
		// half `[old_packed, 2 * old_packed)`; the two slices never overlap.
		let low = unsafe { slice::from_raw_parts_mut(low_ptr, old_packed) };
		// Each coordinate doubles the expansion, starting from a single word.
		// The first iterations are therefore far too small to be worth splitting.
		(low, high)
			.into_par_iter()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(low_i, high_i)| {
				let [new_low, new_high] = Cube::expand_var(low_i, &packed_r_i);
				*low_i = new_low;
				high_i.write(new_high);
			});
		// SAFETY: the loop above initialized every one of the `old_packed` spare words.
		unsafe { data.set_len(2 * old_packed) };
	}

	FieldBuffer::new(final_log_len, data)
}

#[cfg(test)]
mod tests {
	use binius_utils::rayon::task_size::{min_len_for_bytes, min_len_for_work};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b, index_to_hypercube_point, random_scalars};

	type P = Packed128b;
	type F = B128;

	/// The coefficients of the equality indicator over the infinity cube, computed directly from
	/// the definition of the tensor product of the bases $(1, r_i)$.
	fn inf_cube_reference(point: &[F]) -> Vec<F> {
		(0..1 << point.len())
			.map(|index| {
				point
					.iter()
					.enumerate()
					.filter(|(i, _)| index >> i & 1 == 1)
					.map(|(_, r_i)| *r_i)
					.product()
			})
			.collect()
	}

	#[test]
	fn test_inf_cube_eq_ind_partial_eval_matches_definition() {
		let mut rng = StdRng::seed_from_u64(0);

		for n_vars in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let expansion = InfCube::eq_ind_partial_eval::<P>(&point);
			let expansion_scalars = expansion.iter_scalars().collect::<Vec<_>>();
			assert_eq!(expansion_scalars, inf_cube_reference(&point), "mismatch at {n_vars} vars");
		}
	}

	/// The multilinear with the given infinity cube coefficients, evaluated at a point.
	///
	/// The coefficient at index $v$ belongs to the monomial $\prod_{i : v_i = 1} X_i$.
	fn eval_monomial_basis(coeffs: &[F], point: &[F]) -> F {
		coeffs
			.iter()
			.enumerate()
			.map(|(index, coeff)| {
				*coeff
					* point
						.iter()
						.enumerate()
						.filter(|(i, _)| index >> i & 1 == 1)
						.map(|(_, x_i)| *x_i)
						.product::<F>()
			})
			.sum()
	}

	/// The infinity cube expansion of a point holds the monomial coefficients of the infinity
	/// cube's equality indicator partially evaluated at that point.
	#[test]
	fn test_inf_cube_expansion_holds_eq_ind_coefficients() {
		let mut rng = StdRng::seed_from_u64(0);

		for n_vars in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let coeffs = InfCube::eq_ind_partial_eval_scalars::<F>(&point);

			let x = random_scalars::<F>(&mut rng, n_vars);
			assert_eq!(eval_monomial_basis(&coeffs, &x), InfCube::eq_ind::<F>(&x, &point));
		}
	}

	/// The infinity cube expansion of a point is the functional that evaluates a multilinear,
	/// given by its monomial coefficients, at that point.
	#[test]
	fn test_inf_cube_expansion_evaluates_monomial_coefficients() {
		let mut rng = StdRng::seed_from_u64(0);

		for n_vars in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let coeffs = random_scalars::<F>(&mut rng, 1 << n_vars);

			let expansion = InfCube::eq_ind_partial_eval_scalars::<F>(&point);
			let inner_product = iter::zip(&coeffs, &expansion)
				.map(|(c, e)| *c * e)
				.sum::<F>();
			assert_eq!(inner_product, eval_monomial_basis(&coeffs, &point));
		}
	}

	#[test]
	fn test_eq_one_var_matches_basis_definition() {
		let mut rng = StdRng::seed_from_u64(0);

		// `eq_one_var` is specialized in both impls, so check it against the generic definition.
		let [x, y] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);
		let eq_from_basis = |[x_0, x_1]: [F; 2], [y_0, y_1]: [F; 2]| x_0 * y_0 + x_1 * y_1;
		assert_eq!(
			OneCube::eq_one_var(x, y),
			eq_from_basis(OneCube::basis(&x), OneCube::basis(&y))
		);
		assert_eq!(
			InfCube::eq_one_var(x, y),
			eq_from_basis(InfCube::basis(&x), InfCube::basis(&y))
		);
	}

	#[test]
	fn test_expand_var_matches_scaled_basis() {
		let mut rng = StdRng::seed_from_u64(0);

		// `expand_var` saves multiplications over scaling the basis; check the two agree.
		let [value, coord] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);
		assert_eq!(
			OneCube::expand_var(&value, &coord),
			OneCube::basis(&coord).map(|b_i| b_i * value)
		);
		assert_eq!(
			InfCube::expand_var(&value, &coord),
			InfCube::basis(&coord).map(|b_i| b_i * value)
		);
	}

	/// Contraction inverts the expansion of one variable, for either cube.
	#[test]
	fn test_contract_var_inverts_expand_var() {
		let mut rng = StdRng::seed_from_u64(0);

		let [value, coord] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);

		let [mut lo, hi] = OneCube::expand_var(&value, &coord);
		OneCube::contract_var(&mut lo, &hi);
		assert_eq!(lo, value);

		let [mut lo, hi] = InfCube::expand_var(&value, &coord);
		InfCube::contract_var(&mut lo, &hi);
		assert_eq!(lo, value);
	}

	#[test]
	fn test_inf_cube_eq_ind_zero_is_one() {
		let mut rng = StdRng::seed_from_u64(0);

		// Every monomial with a positive degree vanishes at zero, leaving the constant one.
		for n_vars in [0, 1, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			assert_eq!(InfCube::eq_ind_zero::<F>(&point), F::ONE);
			assert_eq!(
				InfCube::eq_ind_zero::<F>(&point),
				InfCube::eq_ind::<F>(&vec![F::ZERO; n_vars], &point)
			);
		}
	}

	#[test]
	fn test_one_cube_eq_ind_zero_is_product_of_complements() {
		let mut rng = StdRng::seed_from_u64(0);

		// Over the Boolean cube the constant basis polynomial is `1 - Y`.
		for n_vars in 0..5 {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let expected: F = point.iter().map(|&r| F::ONE - r).product();
			assert_eq!(OneCube::eq_ind_zero(&point), expected);
			// The same value as evaluating the full indicator against an all-zero operand.
			assert_eq!(
				OneCube::eq_ind_zero(&point),
				OneCube::eq_ind(&vec![F::ZERO; n_vars], &point)
			);
		}
	}

	#[test]
	fn test_one_cube_eq_ind_partial_eval_consistent_on_hypercube() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 5;
		let point = random_scalars(&mut rng, n_vars);
		let expansion = OneCube::eq_ind_partial_eval::<P>(&point);

		for index in 0..1 << n_vars {
			let vertex = index_to_hypercube_point(n_vars, index);
			assert_eq!(expansion.get(index), OneCube::eq_ind::<F>(&point, &vertex));
		}
	}

	#[test]
	fn test_one_cube_eq_ind_partial_eval_empty() {
		// The empty point expands to the single coefficient one.
		let result = OneCube::eq_ind_partial_eval::<P>(&[]);
		assert_eq!(result.log_len(), 0);
		assert_eq!(result.len(), 1);
		assert_eq!(result.get(0), F::ONE);
	}

	#[test]
	fn test_one_cube_eq_ind_partial_eval_single_var() {
		// One coordinate expands to the basis `(1 - r_0, r_0)` itself.
		let r0 = F::new(2);
		let result = OneCube::eq_ind_partial_eval::<P>(&[r0]);
		assert_eq!(result.log_len(), 1);
		assert_eq!(result.len(), 2);
		assert_eq!(result.get(0), F::ONE - r0);
		assert_eq!(result.get(1), r0);
	}

	#[test]
	fn test_one_cube_eq_ind_partial_eval_two_vars() {
		// Two coordinates: the four products of one factor drawn from each basis.
		let r0 = F::new(2);
		let r1 = F::new(3);
		let result = OneCube::eq_ind_partial_eval::<P>(&[r0, r1]);
		assert_eq!(result.log_len(), 2);
		assert_eq!(result.len(), 4);
		let result_vec: Vec<F> = P::iter_slice(result.as_ref()).collect();
		// The variable index is the bit position, so `r_0` varies fastest.
		let expected = vec![
			(F::ONE - r0) * (F::ONE - r1),
			r0 * (F::ONE - r1),
			(F::ONE - r0) * r1,
			r0 * r1,
		];
		assert_eq!(result_vec, expected);
	}

	#[test]
	fn test_one_cube_eq_ind_partial_eval_three_vars() {
		// Three coordinates, spanning one full packed word for this packing width.
		let r0 = F::new(2);
		let r1 = F::new(3);
		let r2 = F::new(5);
		let result = OneCube::eq_ind_partial_eval::<P>(&[r0, r1, r2]);
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

	#[test]
	fn test_one_cube_tensor_prod_eq_ind() {
		// Appending two coordinates to the one-coefficient buffer yields the plain expansion.
		let v0 = F::from(1);
		let v1 = F::from(2);
		let query = vec![v0, v1];
		let result = FieldBuffer::<P, _>::scalar_with_capacity(F::ONE, query.len());
		let result = OneCube::tensor_prod_eq_ind::<P>(result, &query);
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
	fn test_one_cube_tensor_prod_eq_ind_inplace_expansion() {
		let mut rng = StdRng::seed_from_u64(0);

		// Append coordinates in batches of growing size, reusing one reserved backing buffer.
		let exps = 4;
		let max_n_vars = exps * (exps + 1) / 2;
		let mut coords = Vec::with_capacity(max_n_vars);
		let mut eq_expansion = FieldBuffer::<P, _>::scalar_with_capacity(F::ONE, max_n_vars);

		for extra_count in 1..=exps {
			let extra = random_scalars(&mut rng, extra_count);

			eq_expansion = OneCube::tensor_prod_eq_ind::<P>(eq_expansion, &extra);
			coords.extend(&extra);

			// Every batch must leave the buffer equal to the indicator over all coordinates so far.
			assert_eq!(eq_expansion.log_len(), coords.len());
			for i in 0..eq_expansion.len() {
				let v = eq_expansion.get(i);
				let hypercube_point = index_to_hypercube_point(coords.len(), i);
				assert_eq!(v, OneCube::eq_ind(&hypercube_point, &coords));
			}
		}
	}

	#[test]
	fn test_one_cube_tensor_prod_eq_prepend_via_bit_reverse() {
		// `BinarySwitchover` prepends one variable per round as bit-reverse + append + bit-reverse.
		// Check that this composition, iterated over all coordinates (including the
		// sub-packing-width early rounds), matches a full eq expansion.
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 10;
		let point = random_scalars::<F>(&mut rng, n_vars);

		let mut tensor = FieldBuffer::<P>::from_values(&[F::ONE]);
		for &r in point.iter().rev() {
			tensor.to_mut().bit_reverse();
			tensor = OneCube::tensor_prod_eq_ind::<P>(tensor, &[r]);
			tensor.to_mut().bit_reverse();
		}

		assert_eq!(tensor, OneCube::eq_ind_partial_eval(&point));
	}

	#[test]
	fn test_one_cube_eq_ind_truncate_low_inplace_iterated() {
		let mut rng = StdRng::seed_from_u64(0);

		// Truncate the same buffer repeatedly, by a shrinking number of variables each time.
		let reds = 4;
		let n_vars = reds * (reds + 1) / 2;
		let point = random_scalars(&mut rng, n_vars);

		let mut eq_ind = OneCube::eq_ind_partial_eval::<P>(&point);
		let mut log_n_values = n_vars;

		for reduction in (0..=reds).rev() {
			let truncated_log_n_values = log_n_values - reduction;
			OneCube::eq_ind_truncate_low_inplace(&mut eq_ind, truncated_log_n_values);

			// Each step must match a direct expansion of the surviving prefix of the point.
			let eq_ind_ref = OneCube::eq_ind_partial_eval::<P>(&point[..truncated_log_n_values]);
			assert_eq!(eq_ind_ref.len(), eq_ind.len());
			for i in 0..eq_ind.len() {
				assert_eq!(eq_ind.get(i), eq_ind_ref.get(i));
			}

			log_n_values = truncated_log_n_values;
		}

		// The last reduction is by zero variables, so the sequence ends at the empty point.
		assert_eq!(log_n_values, 0);
	}

	#[test]
	fn test_one_cube_scaled_eq_ind_partial_eval_scalars_is_unscaled_times_scale() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the scalar expansion is linear in its seed, coefficient by coefficient.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			let scaled = OneCube::scaled_eq_ind_partial_eval_scalars(&point, scale);
			let expected: Vec<F> = OneCube::eq_ind_partial_eval_scalars(&point)
				.into_iter()
				.map(|x| x * scale)
				.collect();
			assert_eq!(scaled, expected, "mismatch at log_n={log_n}");
		}
	}

	#[test]
	fn test_one_cube_scaled_eq_ind_partial_eval_scale_one_matches_unscaled() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a scale of one is the identity on the expansion.
		// So the scaled and unscaled indicators must be the identical buffer.
		//
		// Sizes span the empty point (0 variables) up to a 256-value cube (8 variables).
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);

			// Equality is checked packed-word for packed-word, not just value by value.
			assert_eq!(
				OneCube::scaled_eq_ind_partial_eval::<P>(&point, F::ONE),
				OneCube::eq_ind_partial_eval::<P>(&point),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn test_one_cube_scaled_eq_ind_partial_eval_into_matches_allocating() {
		let mut rng = StdRng::seed_from_u64(2);

		// Invariant: filling a caller-reserved backing Vec reproduces the allocating variant
		// exactly.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// Reserve the exact packed capacity the routine requires.
			let packed_len = 1 << log_n.saturating_sub(P::LOG_WIDTH);
			let result = OneCube::scaled_eq_ind_partial_eval_into(
				&point,
				scale,
				Vec::with_capacity(packed_len),
			);

			assert_eq!(result.log_len(), log_n, "wrong length at log_n={log_n}");
			assert_eq!(
				result,
				OneCube::scaled_eq_ind_partial_eval::<P>(&point, scale),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn test_one_cube_scaled_eq_ind_partial_eval_scale_zero_is_zero() {
		let mut rng = StdRng::seed_from_u64(1);

		// Invariant: the expansion is linear in its starting value.
		// So a starting value of zero yields the all-zero polynomial.
		for log_n in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, log_n);

			// Every one of the 2^log_n hypercube values must be zero.
			let scaled = OneCube::scaled_eq_ind_partial_eval::<P>(&point, F::ZERO);
			assert!(scaled.iter_scalars().all(|v| v == F::ZERO), "nonzero at log_n={log_n}");
		}
	}

	#[test]
	fn test_expansion_and_truncation_above_split_threshold() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a round splits only once it exceeds the minimum task size.
		// Below that it runs inline, leaving the parallel path unexercised.
		//
		// Expansion and contraction carry different minimums, so the larger governs:
		//
		//     words in the last round = 2^(n_vars - 1) / scalars per word
		//     smallest n_vars with words in the last round >= the larger minimum
		//
		// Every other test here is smaller, so this one covers the split.
		let min_len = min_len_for_work(WorkPerItem::FieldMuls).max(min_len_for_bytes::<[P; 2]>());
		let n_vars = (2 * min_len * P::WIDTH).next_power_of_two().ilog2() as usize;
		let point = random_scalars::<F>(&mut rng, n_vars);

		let packed = OneCube::eq_ind_partial_eval::<P>(&point);
		let reference = OneCube::eq_ind_partial_eval_scalars::<F>(&point);
		assert!(packed.iter_scalars().eq(reference.iter().copied()));

		// Contraction above the threshold: strip the top variable and compare against a
		// direct expansion of the prefix, whose rounds also run through the split path.
		let mut truncated = packed;
		OneCube::eq_ind_truncate_low_inplace(&mut truncated, n_vars - 1);
		assert_eq!(truncated, OneCube::eq_ind_partial_eval::<P>(&point[..n_vars - 1]));
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(16))]

		/// The scalar and packed expansions agree, for either cube.
		#[test]
		fn eq_ind_partial_eval_scalars_matches_packed(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);

			prop_assert_eq!(
				OneCube::eq_ind_partial_eval::<P>(&point).iter_scalars().collect::<Vec<_>>(),
				OneCube::eq_ind_partial_eval_scalars::<F>(&point)
			);
			prop_assert_eq!(
				InfCube::eq_ind_partial_eval::<P>(&point).iter_scalars().collect::<Vec<_>>(),
				InfCube::eq_ind_partial_eval_scalars::<F>(&point)
			);
		}

		/// Truncation strips the trailing variables of an expansion, for either cube.
		#[test]
		fn eq_ind_truncate_low_inplace_strips_trailing_vars(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);

			for truncated_log_len in 0..=log_n {
				let mut one_cube = OneCube::eq_ind_partial_eval::<P>(&point);
				OneCube::eq_ind_truncate_low_inplace(&mut one_cube, truncated_log_len);
				prop_assert_eq!(
					one_cube,
					OneCube::eq_ind_partial_eval::<P>(&point[..truncated_log_len])
				);

				let mut inf_cube = InfCube::eq_ind_partial_eval::<P>(&point);
				InfCube::eq_ind_truncate_low_inplace(&mut inf_cube, truncated_log_len);
				prop_assert_eq!(
					inf_cube,
					InfCube::eq_ind_partial_eval::<P>(&point[..truncated_log_len])
				);
			}
		}

		/// Scaling commutes with the expansion, coefficient by coefficient, for either cube.
		#[test]
		fn scaled_eq_ind_partial_eval_matches_scaled_reference(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			for (scaled, reference) in [
				(
					OneCube::scaled_eq_ind_partial_eval::<P>(&point, scale),
					OneCube::eq_ind_partial_eval::<P>(&point),
				),
				(
					InfCube::scaled_eq_ind_partial_eval::<P>(&point, scale),
					InfCube::eq_ind_partial_eval::<P>(&point),
				),
			] {
				for (got, base) in scaled.iter_scalars().zip(reference.iter_scalars()) {
					prop_assert_eq!(got, scale * base);
				}
			}
		}
	}
}
