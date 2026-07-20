// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Multilinear tensor expansions, generic over the hypercube the coefficients are indexed by.
//!
//! See [`Hypercube`] for the abstraction and [`OneCube`]/[`InfCube`] for the two instances. The
//! routines here mirror those in [`super::eq`], which specialize them to [`OneCube`].

use std::{iter, slice};

use binius_field::{Field, PackedField, field::FieldOps};
use binius_utils::rayon::prelude::*;

use crate::{FieldBuffer, field_buffer::BufferData};

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
pub trait Hypercube {
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

/// Tensor of values with the equality indicator evaluated at `extra_query_coordinates`.
///
/// Let $n$ be `values.log_len()` and $k$ be the length of `extra_query_coordinates`. The returned
/// buffer grows its backing `Vec` by one variable per coordinate.
///
/// # Formal Definition
///
/// The result is the tensor product of `values`' $2^n$ values with the linear bases of `Cube`
/// evaluated at $r = (r_0, \ldots, r_{k-1})$:
///
/// $$
/// v \otimes b(r_0) \otimes \ldots \otimes b(r_{k-1}),
/// $$
///
/// a vector of length $2^{n+k}$.
///
/// # Interpretation
///
/// Let $f$ be the $n$-variate multilinear with coefficients $v$ over `Cube`. Then the result holds
/// the coefficients of the $(n+k)$-variate multilinear
///
/// $$
/// g(X_0, \ldots, X_{n+k-1}) = f(X_0, \ldots, X_{n-1}) \cdot
///     \widetilde{eq}(X_n, \ldots, X_{n+k-1}, r).
/// $$
pub fn tensor_prod_eq_ind<Cube: Hypercube, P: PackedField>(
	values: FieldBuffer<P, Vec<P>>,
	extra_query_coordinates: &[P::Scalar],
) -> FieldBuffer<P, Vec<P>> {
	let start_log_len = values.log_len();
	let final_log_len = start_log_len + extra_query_coordinates.len();
	let mut data = values.take_data();

	// Reserve the full final capacity once, so no growth step reallocates. Each packed step then
	// grows the length in place through the reserved spare capacity, writing the new coefficients
	// directly rather than zero-initializing a region the expansion immediately overwrites.
	let final_packed_len = 1usize << final_log_len.saturating_sub(P::LOG_WIDTH);
	data.reserve_exact(final_packed_len.saturating_sub(data.len()));

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
		(low, high).into_par_iter().for_each(|(low_i, high_i)| {
			let [new_low, new_high] = Cube::expand_var(low_i, &packed_r_i);
			*low_i = new_low;
			high_i.write(new_high);
		});
		// SAFETY: the loop above initialized every one of the `old_packed` spare words.
		unsafe { data.set_len(2 * old_packed) };
	}

	FieldBuffer::new(final_log_len, data)
}

/// Computes the partial evaluation of the equality indicator polynomial.
///
/// Given an $n$-coordinate point $r_0, \ldots, r_{n-1}$, this computes the partial evaluation of
/// the equality indicator polynomial $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ and
/// returns its coefficients over `Cube`, which are the tensor product
///
/// $$
/// b(r_0) \otimes \ldots \otimes b(r_{n-1}).
/// $$
pub fn eq_ind_partial_eval<Cube: Hypercube, P: PackedField>(point: &[P::Scalar]) -> FieldBuffer<P> {
	// The unscaled indicator is the scaled indicator with a scale of one.
	scaled_eq_ind_partial_eval::<Cube, P>(point, P::Scalar::ONE)
}

/// Computes the partial evaluation of the equality indicator polynomial, scaled by a constant.
///
/// Every coefficient of the equality indicator is multiplied by `scale`. A scale of one reproduces
/// [`eq_ind_partial_eval`].
///
/// # Arguments
///
/// * `point` - The evaluation point whose length is the number of variables.
/// * `scale` - The constant every returned value is multiplied by.
pub fn scaled_eq_ind_partial_eval<Cube: Hypercube, P: PackedField>(
	point: &[P::Scalar],
	scale: P::Scalar,
) -> FieldBuffer<P> {
	// Reserve the final packed length up front so the per-variable growth never reallocates.
	let packed_len = 1 << point.len().saturating_sub(P::LOG_WIDTH);
	scaled_eq_ind_partial_eval_into::<Cube, P>(point, scale, Vec::with_capacity(packed_len))
}

/// Builds the scaled equality indicator expansion of `point` in a caller-supplied backing `Vec`.
///
/// This is the allocation-hoisting form of [`scaled_eq_ind_partial_eval`]: the caller owns the
/// backing `Vec`, so its allocation can be reserved on a different thread than the one that fills
/// it. The `Vec` is cleared, seeded with `scale`, grown one variable at a time into the tensor
/// product $scale \cdot b(r_0) \otimes \ldots \otimes b(r_{n-1})$, and wrapped into the returned
/// buffer with `log_len == point.len()`.
///
/// # Preconditions
///
/// * `buffer.capacity()` must equal `1 << point.len().saturating_sub(P::LOG_WIDTH)`, so the growth
///   never reallocates and the final wrap keeps the same allocation.
pub fn scaled_eq_ind_partial_eval_into<Cube: Hypercube, P: PackedField>(
	point: &[P::Scalar],
	scale: P::Scalar,
	mut buffer: Vec<P>,
) -> FieldBuffer<P> {
	let packed_len = 1 << point.len().saturating_sub(P::LOG_WIDTH);
	assert_eq!(
		buffer.capacity(),
		packed_len,
		"precondition: buffer capacity must match the packed expansion length"
	);

	// Seed a single-scalar buffer with the scale; the expansion multiplies it through, so every
	// coefficient ends up scaled.
	buffer.clear();
	buffer.push(P::from_scalars(iter::once(scale)));
	let values = FieldBuffer::new(0, buffer);
	tensor_prod_eq_ind::<Cube, P>(values, point)
}

/// Truncate the equality indicator expansion to the low indexed variables.
///
/// This routine computes $\widetilde{eq}(X_0, ..., X_{n'-1}, r_0, ..., r_{n'-1})$ from
/// $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ where $n' \le n$ by repeatedly
/// contracting field buffer "halves" inplace. The equality indicator expansion occupies a prefix of
/// the field buffer; scalars after the truncated length are zeroed out.
///
/// ## Preconditions
///
/// * `truncated_log_len` must be at most `values.log_len()`
pub fn eq_ind_truncate_low_inplace<Cube: Hypercube, P: PackedField, Data: BufferData<P>>(
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
			(lo.as_mut(), hi.as_ref())
				.into_par_iter()
				.for_each(|(zero, one)| {
					Cube::contract_var(zero, one);
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
pub fn eq_ind<Cube: Hypercube, F: FieldOps>(x: &[F], y: &[F]) -> F {
	assert_eq!(x.len(), y.len(), "pre-condition: x and y must be the same length");
	iter::zip(x, y)
		.map(|(x, y)| Cube::eq_one_var(x.clone(), y.clone()))
		.product()
}

/// Evaluates the equality indicator multilinear with one operand fixed to all zeros.
///
/// This is `eq_ind(0^n, point)`, which simplifies to the product of the constant basis polynomials:
///
/// $$
/// \widetilde{eq}(0^n, Y_0, \ldots, Y_{n-1}) = \prod_{i=0}^{n-1} b_0(Y_i).
/// $$
pub fn eq_ind_zero<Cube: Hypercube, F: FieldOps>(point: &[F]) -> F {
	point
		.iter()
		.map(|y| {
			let [y_0, _] = Cube::basis(y);
			y_0
		})
		.product()
}

/// Computes the partial evaluation of the equality indicator polynomial, returning scalars.
///
/// This is a scalar-only variant of [`eq_ind_partial_eval`] that returns a `Vec<F>` instead of a
/// [`FieldBuffer`].
pub fn eq_ind_partial_eval_scalars<Cube: Hypercube, F: FieldOps>(point: &[F]) -> Vec<F> {
	// The unscaled indicator is the scaled indicator with a scale of one.
	scaled_eq_ind_partial_eval_scalars::<Cube, F>(point, F::one())
}

/// Computes the partial evaluation of the equality indicator polynomial scaled by a constant,
/// returning scalars.
///
/// This is a scalar-only variant of [`scaled_eq_ind_partial_eval`] that returns a `Vec<F>` instead
/// of a [`FieldBuffer`]. A scale of one reproduces [`eq_ind_partial_eval_scalars`].
pub fn scaled_eq_ind_partial_eval_scalars<Cube: Hypercube, F: FieldOps>(
	point: &[F],
	scale: F,
) -> Vec<F> {
	let mut result = Vec::with_capacity(1 << point.len());
	// Seed with the scale; the expansion multiplies it through every coefficient.
	result.push(scale);

	for r_i in point {
		// Double the buffer size. For each existing value in 0..size, the lo half gets the value
		// scaled by the constant basis polynomial and the hi half by the linear one. Process in
		// reverse so that writes to hi don't overwrite values we need.
		let len = result.len();
		for j in 0..len {
			let [lo, hi] = Cube::expand_var(&result[j], r_i);
			result[j] = lo;
			result.push(hi);
		}
	}
	result
}

#[cfg(test)]
mod tests {
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
			let expansion = eq_ind_partial_eval::<InfCube, P>(&point);
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
			let coeffs = eq_ind_partial_eval_scalars::<InfCube, F>(&point);

			let x = random_scalars::<F>(&mut rng, n_vars);
			assert_eq!(eval_monomial_basis(&coeffs, &x), eq_ind::<InfCube, F>(&x, &point));
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

			let expansion = eq_ind_partial_eval_scalars::<InfCube, F>(&point);
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
			assert_eq!(eq_ind_zero::<InfCube, F>(&point), F::ONE);
			assert_eq!(
				eq_ind_zero::<InfCube, F>(&point),
				eq_ind::<InfCube, F>(&vec![F::ZERO; n_vars], &point)
			);
		}
	}

	#[test]
	fn test_one_cube_eq_ind_partial_eval_consistent_on_hypercube() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 5;
		let point = random_scalars(&mut rng, n_vars);
		let expansion = eq_ind_partial_eval::<OneCube, P>(&point);

		for index in 0..1 << n_vars {
			let vertex = index_to_hypercube_point(n_vars, index);
			assert_eq!(expansion.get(index), eq_ind::<OneCube, F>(&point, &vertex));
		}
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
				eq_ind_partial_eval::<OneCube, P>(&point).iter_scalars().collect::<Vec<_>>(),
				eq_ind_partial_eval_scalars::<OneCube, F>(&point)
			);
			prop_assert_eq!(
				eq_ind_partial_eval::<InfCube, P>(&point).iter_scalars().collect::<Vec<_>>(),
				eq_ind_partial_eval_scalars::<InfCube, F>(&point)
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
				let mut one_cube = eq_ind_partial_eval::<OneCube, P>(&point);
				eq_ind_truncate_low_inplace::<OneCube, _, _>(&mut one_cube, truncated_log_len);
				prop_assert_eq!(
					one_cube,
					eq_ind_partial_eval::<OneCube, P>(&point[..truncated_log_len])
				);

				let mut inf_cube = eq_ind_partial_eval::<InfCube, P>(&point);
				eq_ind_truncate_low_inplace::<InfCube, _, _>(&mut inf_cube, truncated_log_len);
				prop_assert_eq!(
					inf_cube,
					eq_ind_partial_eval::<InfCube, P>(&point[..truncated_log_len])
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

			let scaled = scaled_eq_ind_partial_eval::<InfCube, P>(&point, scale);
			let reference = eq_ind_partial_eval::<InfCube, P>(&point);
			for (got, base) in scaled.iter_scalars().zip(reference.iter_scalars()) {
				prop_assert_eq!(got, scale * base);
			}
		}
	}
}
