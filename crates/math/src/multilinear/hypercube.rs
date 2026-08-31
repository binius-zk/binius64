// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Multilinear tensor expansions, generic over the hypercube the coefficients are indexed by.
//!
//! An `n`-variate multilinear is stored as `2^n` coefficients.
//! The basis those coefficients are taken against factors as a tensor product over the variables.
//! Every variable contributes the same two-element basis `(b_0, b_1)` of linear polynomials.
//! That single choice fixes the cube, and with it what each coefficient means:
//!
//! ```text
//! basis (1 - X, X)    vertices {0, 1}      coefficients are evaluations
//! basis (1, X)        vertices {0, inf}    coefficients are monomial coefficients
//! ```
//!
//! The object built over a cube again and again is the equality indicator.
//! Written `eq(X, Y)`, it extends the predicate `X == Y` multilinearly over the cube.
//! Fixing one operand to a point leaves `2^n` coefficients, called the expansion of that point.
//!
//! Every routine here is generic over the cube.
//! The Boolean-cube specializations live beside them, in the sibling equality indicator module.

use std::{iter, slice};

use binius_compute::{Allocator, BufferData, VecLike};
use binius_field::{Field, PackedField, field::FieldOps};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem, min_len_for_bytes},
};

use crate::{FieldBuffer, FieldVec};

/// A hypercube of coefficients for multilinear polynomials.
///
/// A cube is fixed by the two-element basis `(b_0, b_1)` that each of its variables contributes.
/// That basis is a pair of linear polynomials, so a cube is a choice between two of them.
/// Everything else is derived from that choice, and shared by every implementor.
pub trait Hypercube {
	/// Evaluates the basis of one variable at a coordinate.
	///
	/// Returns `(b_0(r), b_1(r))` for the coordinate `r`.
	fn basis<F: FieldOps>(coord: &F) -> [F; 2];

	/// Scales the basis of one variable by a value.
	///
	/// Returns `(v * b_0(r), v * b_1(r))` for the value `v` and the coordinate `r`.
	/// This is the inner loop of every expansion.
	/// So an implementor beats the two multiplications that scaling the basis directly costs.
	fn expand_var<F: FieldOps>(value: &F, coord: &F) -> [F; 2];

	/// Strips one variable's basis factor from the two halves of an expansion.
	///
	/// The halves hold `v * b_0(r)` and `v * b_1(r)` for the stripped variable's coordinate `r`.
	/// The low half is overwritten with `v`.
	///
	/// Recovering `v` is one fixed linear combination of the two halves:
	///
	/// ```text
	/// sum_i w_i * v * b_i(r) = v    where    sum_i w_i * b_i(X) = 1
	/// ```
	///
	/// Those weights are unique and free of `r`, so the same combination works at any coordinate.
	fn contract_var<F: FieldOps>(lo: &mut F, hi: &F);

	/// Evaluates the equality indicator of one variable.
	///
	/// ```text
	/// eq(X, Y) = sum_i b_i(X) * b_i(Y)
	/// ```
	///
	/// An implementor overrides this with a cheaper closed form.
	fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
		// The generic definition: pair the two bases at the two coordinates and sum.
		let [x_0, x_1] = Self::basis(&x);
		let [y_0, y_1] = Self::basis(&y);
		x_0 * y_0 + x_1 * y_1
	}
}

/// The Boolean cube `{0, 1}^n`, whose per-variable basis is `(1 - X, X)`.
///
/// That basis is the pair of Lagrange polynomials on the two vertices `0` and `1`.
/// So the coefficient indexed by a vertex is the multilinear's evaluation at that vertex.
#[derive(Debug)]
pub struct OneCube;

impl Hypercube for OneCube {
	#[inline(always)]
	fn basis<F: FieldOps>(coord: &F) -> [F; 2] {
		[F::one() - coord, coord.clone()]
	}

	#[inline(always)]
	fn expand_var<F: FieldOps>(value: &F, coord: &F) -> [F; 2] {
		// Both halves share the product `value * coord`, so one multiplication covers both.
		let prod = value.clone() * coord;
		[value.clone() - &prod, prod]
	}

	#[inline(always)]
	fn contract_var<F: FieldOps>(lo: &mut F, hi: &F) {
		// The two basis polynomials sum to one, so both recovery weights are one.
		*lo += hi;
	}

	#[inline(always)]
	fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
		// Over characteristic two the `2 * X * Y` term vanishes, so
		//
		//     X * Y + (1 - X) * (1 - Y)  =  X + Y + 1
		//
		// The condition is a compile-time constant, so only one arm is ever generated.
		if F::Scalar::CHARACTERISTIC == 2 {
			x + y + F::one()
		} else {
			let one = F::one();
			x.clone() * y.clone() + (one.clone() - x) * (one - y)
		}
	}
}

/// The infinity cube `{0, inf}^n`, whose per-variable basis is `(1, X)`.
///
/// The vertex `inf` selects a multilinear's leading coefficient in that variable.
/// So the coefficient indexed by a vertex `v` belongs to the monomial
///
/// ```text
/// prod_{i : v_i = inf} X_i
/// ```
#[derive(Debug)]
pub struct InfCube;

impl Hypercube for InfCube {
	#[inline(always)]
	fn basis<F: FieldOps>(coord: &F) -> [F; 2] {
		[F::one(), coord.clone()]
	}

	#[inline(always)]
	fn expand_var<F: FieldOps>(value: &F, coord: &F) -> [F; 2] {
		// The constant basis polynomial is one, so the low half is the value untouched.
		[value.clone(), value.clone() * coord]
	}

	#[inline(always)]
	fn contract_var<F: FieldOps>(_lo: &mut F, _hi: &F) {
		// The low half already holds the value, so the weights are one and zero.
		// Contracting a variable is therefore free for this cube.
	}

	#[inline(always)]
	fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
		// The constant basis polynomial contributes one, and the linear one the product.
		F::one() + x * y
	}
}

/// Tensor of values with the equality indicator evaluated at extra coordinates.
///
/// Take `n` values and the `k` coordinates `r = (r_0, ..., r_{k-1})`.
/// The result is the tensor product of those values with the basis at every coordinate:
///
/// ```text
/// v (x) b(r_0) (x) ... (x) b(r_{k-1})
/// ```
///
/// It holds `2^(n + k)` coefficients, one variable added per coordinate.
///
/// Read as polynomials, the input holds an `n`-variate multilinear `f`.
/// The output then holds the `(n + k)`-variate multilinear
///
/// ```text
/// g(X_0, ..., X_{n+k-1}) = f(X_0, ..., X_{n-1}) * eq(X_n, ..., X_{n+k-1}, r)
/// ```
///
/// Appending the coordinate `r` doubles the length, turning every value `v` into a pair:
///
/// ```text
/// before   [ v_0            v_1            ]
/// after    [ v_0 * b_0(r)   v_1 * b_0(r)   |   v_0 * b_1(r)   v_1 * b_1(r) ]
/// ```
///
/// The two halves sit one after the other, so the appended variable is the highest indexed one.
pub fn tensor_prod_eq_ind<Cube: Hypercube, P: PackedField>(
	values: FieldBuffer<P, Vec<P>>,
	extra_query_coordinates: &[P::Scalar],
) -> FieldBuffer<P, Vec<P>> {
	let start_log_len = values.log_len();
	let final_log_len = start_log_len + extra_query_coordinates.len();
	let mut data = values.into_inner();

	// Reserve the whole final capacity once, so no round reallocates.
	// Each round then writes its new coefficients straight into the reserved spare capacity,
	// instead of zero-initializing a region the expansion immediately overwrites.
	let final_packed_len = packed_words::<P>(final_log_len);
	data.reserve_exact(final_packed_len.saturating_sub(data.len()));

	tensor_prod_eq_ind_reserved::<Cube, P, _>(
		FieldBuffer::new(start_log_len, data),
		extra_query_coordinates,
	)
}

/// The number of packed words an expansion of that many variables occupies.
///
/// Below one packed word the count is one, since a single word backs any shorter length.
const fn packed_words<P: PackedField>(log_len: usize) -> usize {
	1usize << log_len.saturating_sub(P::LOG_WIDTH)
}

/// Appends one variable per coordinate to a store that already has room for the result.
///
/// The public entry point reserves the final capacity before calling here.
/// The allocator-backed callers pass a buffer drawn at the final size, which cannot grow.
/// So for them the reservation is a precondition rather than a step of the expansion.
///
/// # Preconditions
///
/// * The store's capacity must cover the packed length of the final expansion.
fn tensor_prod_eq_ind_reserved<Cube: Hypercube, P: PackedField, Data: VecLike<P>>(
	values: FieldBuffer<P, Data>,
	extra_query_coordinates: &[P::Scalar],
) -> FieldBuffer<P, Data> {
	let start_log_len = values.log_len();
	let final_log_len = start_log_len + extra_query_coordinates.len();
	let mut data = values.into_inner();

	// precondition
	debug_assert!(data.capacity() >= packed_words::<P>(final_log_len));

	// The coordinates split cleanly in two at the packing width:
	//
	//     narrower than one word   the whole expansion lives in data[0]
	//     one word or wider        every round doubles the packed length
	let sub_width_count = extra_query_coordinates
		.len()
		.min(P::LOG_WIDTH.saturating_sub(start_log_len));
	let (sub_width_coords, packed_coords) = extra_query_coordinates.split_at(sub_width_count);

	// Sub-packing-width rounds: both halves of the result share the single word data[0].
	// Split that word into its two halves, expand them, and interleave them back together.
	// The backing store stays one element long throughout.
	for (i, &r_i) in sub_width_coords.iter().enumerate() {
		let log_len = start_log_len + i;
		let packed_r_i = P::broadcast(r_i);
		let (lo, _) = data[0].interleave(P::zero(), log_len);
		let [lo, hi] = Cube::expand_var(&lo, &packed_r_i);
		data[0] = lo.interleave(hi, log_len).0;
	}

	// Packed rounds: the initialized words are exactly the low half of the result.
	//
	//     low half    the initialized prefix, expanded in place
	//     high half   reserved spare capacity, written once
	for &r_i in packed_coords {
		let packed_r_i = P::broadcast(r_i);
		let old_packed = data.len();

		// The safe two-slice split of a Vec into its initialized prefix and its spare capacity
		// is still unstable (rust-lang/rust#81944).
		// So the spare half comes from the safe accessor and the initialized half from a raw part.
		let low_ptr = data.as_mut_ptr();
		let high = &mut data.spare_capacity_mut()[..old_packed];
		// SAFETY: `[0, old_packed)` is the initialized low half, disjoint from the spare `high`
		// half `[old_packed, 2 * old_packed)`; the two slices never overlap.
		let low = unsafe { slice::from_raw_parts_mut(low_ptr, old_packed) };
		// Each round doubles the expansion, starting from a single word.
		// So the first rounds are far too small to be worth splitting across threads.
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

/// Computes the partial evaluation of the equality indicator polynomial.
///
/// For the point `r = (r_0, ..., r_{n-1})` the result holds the `2^n` coefficients
///
/// ```text
/// b(r_0) (x) ... (x) b(r_{n-1})
/// ```
///
/// which are the coefficients of the equality indicator `eq(X_0, ..., X_{n-1}, r)` over the cube.
pub fn eq_ind_partial_eval<Cube: Hypercube, P: PackedField>(point: &[P::Scalar]) -> FieldBuffer<P> {
	// The unscaled indicator is the scaled indicator with a scale of one.
	scaled_eq_ind_partial_eval::<Cube, P>(point, P::Scalar::ONE)
}

/// Builds the equality indicator expansion of a point into a buffer drawn from an allocator.
///
/// Backed by a pool, the result is a recyclable buffer rather than a fresh allocation.
pub fn eq_ind_partial_eval_in<Cube: Hypercube, A: Allocator, P: PackedField>(
	alloc: &A,
	point: &[P::Scalar],
) -> FieldVec<P, A> {
	// The allocator hands out the final packed length, which the expansion never outgrows.
	let packed_len = packed_words::<P>(point.len());
	scaled_eq_ind_partial_eval_into::<Cube, P, _>(
		point,
		P::Scalar::ONE,
		alloc.alloc::<P>(packed_len),
	)
}

/// Computes the partial evaluation of the equality indicator polynomial, scaled by a constant.
///
/// Every coefficient of the equality indicator is multiplied by the scale.
/// A scale of one is the identity, since the expansion is linear in it.
///
/// # Arguments
///
/// * `point` - The evaluation point whose length is the number of variables.
/// * `scale` - The constant every returned value is multiplied by.
pub fn scaled_eq_ind_partial_eval<Cube: Hypercube, P: PackedField>(
	point: &[P::Scalar],
	scale: P::Scalar,
) -> FieldBuffer<P> {
	// Reserving the final packed length keeps the per-variable growth reallocation free.
	let packed_len = packed_words::<P>(point.len());
	scaled_eq_ind_partial_eval_into::<Cube, P, _>(point, scale, Vec::with_capacity(packed_len))
}

/// Builds the scaled equality indicator expansion of a point in a caller-supplied store.
///
/// This is the allocation-hoisting form.
/// The caller owns the store, so it can be drawn from a pool.
/// It can equally be reserved on a different thread than the one that fills it.
///
/// # Preconditions
///
/// * The store's capacity must cover the packed length of the expansion.
pub fn scaled_eq_ind_partial_eval_into<Cube: Hypercube, P: PackedField, Data: VecLike<P>>(
	point: &[P::Scalar],
	scale: P::Scalar,
	mut buffer: Data,
) -> FieldBuffer<P, Data> {
	assert!(
		buffer.capacity() >= packed_words::<P>(point.len()),
		"precondition: buffer capacity must cover the packed expansion length"
	);

	// Seed a one-coefficient expansion with the scale.
	// Appending the coordinates multiplies it through, so every coefficient ends up scaled.
	buffer.clear();
	buffer.push(P::from_scalars(iter::once(scale)));
	let seed = FieldBuffer::new(0, buffer);

	match split_low_len::<P>(point.len()) {
		Some(low_len) => split_expand::<Cube, P, Data>(seed, point, low_len),
		None => tensor_prod_eq_ind_reserved::<Cube, P, Data>(seed, point),
	}
}

/// Bytes one block of a split expansion spans.
///
/// # Why this value
///
/// A block is read once per block of the result and written once in total.
/// Sizing it for the second cache level keeps those repeated reads off memory, while leaving room
/// beside them for the writes they feed.
///
/// Halving it measurably loses at the largest sizes, and quadrupling it gains nothing.
const BLOCK_BYTES: usize = 1 << 16;

/// Blocks the result must span before the split earns its keep, as a base-2 logarithm.
///
/// # Why this value
///
/// The split trades the rounds' repeated rewrites for two sub-expansions and one extra pass.
/// That trade only pays once the result stops fitting in the last cache level, since below it
/// the traffic the split saves was never memory traffic.
///
/// Measured over 128-bit scalars, the two paths cross where the result reaches a few megabytes,
/// which this puts at 64 blocks:
///
/// ```text
///     coefficients   2^15   2^16   2^17   2^18   2^20   2^22   2^24
///     split / rounds 0.89x  0.94x  0.98x  1.64x  1.83x  1.65x  1.39x
/// ```
const LOG_MIN_BLOCKS: usize = 6;

/// Coordinates the low half of a split takes, or nothing when the point is too short to split.
///
/// A block spans whole packed words, so it never falls below one.
fn split_low_len<P: PackedField>(n_vars: usize) -> Option<usize> {
	let log_scalar_bytes = size_of::<P::Scalar>().next_power_of_two().ilog2() as usize;
	let low_len = BLOCK_BYTES.ilog2().saturating_sub(log_scalar_bytes as u32) as usize;
	let low_len = low_len.max(P::LOG_WIDTH);

	(n_vars >= low_len + LOG_MIN_BLOCKS).then_some(low_len)
}

/// Expands a point by cutting it in two and taking the outer product of the two expansions.
///
/// A coefficient is a product over the coordinates, so cutting the index cuts the product:
///
/// ```text
///     index = high * 2^k + low
///     coeff = high_expansion[high] * low_expansion[low]
/// ```
///
/// The low expansion is therefore one block of the result, and every other block is that block
/// scaled by one coefficient of the high expansion.
/// Each coefficient is written once, where appending a coordinate rewrites everything already
/// expanded.
///
/// # Preconditions
///
/// * the seed holds one coefficient, and its store has room for the whole expansion
/// * the cut leaves at least one packed word below it, and at least one block above it
fn split_expand<Cube: Hypercube, P: PackedField, Data: VecLike<P>>(
	seed: FieldBuffer<P, Data>,
	point: &[P::Scalar],
	low_len: usize,
) -> FieldBuffer<P, Data> {
	let (low_coords, high_coords) = point.split_at(low_len);

	// The low expansion is built straight into the store's first block, which is where the result
	// wants it anyway. Nothing else is expanded over the result's own length.
	let low = tensor_prod_eq_ind_reserved::<Cube, P, Data>(seed, low_coords);
	let block = low.as_ref().len();
	let mut data = low.into_inner();

	// One scalar per block of the result, so the high expansion stays far smaller than it.
	let high = eq_ind_partial_eval_scalars::<Cube, P::Scalar>(high_coords);
	let total = block * high.len();
	debug_assert_eq!(total, packed_words::<P>(point.len()));

	// The safe two-slice split of a Vec into its initialized prefix and its spare capacity is
	// still unstable (rust-lang/rust#81944).
	// So the spare blocks come from the safe accessor and the first block from a raw part.
	let first_ptr = data.as_mut_ptr();
	let spare = &mut data.spare_capacity_mut()[..total - block];
	// SAFETY: `[0, block)` is the initialized first block, disjoint from the spare blocks past
	// it; the two slices never overlap.
	let first = unsafe { slice::from_raw_parts(first_ptr, block) };

	// One item here is a whole block, so the byte floor is divided down by what a block holds.
	let min_len = (min_len_for_bytes::<P>() / block).max(1);
	spare
		.par_chunks_mut(block)
		.zip(high[1..].par_iter())
		.with_min_len(min_len)
		.for_each(|(dst, &coeff)| {
			let coeff = P::broadcast(coeff);
			for (dst_i, &src_i) in iter::zip(dst, first) {
				dst_i.write(src_i * coeff);
			}
		});
	// SAFETY: the loop above initialized every spare word up to the total.
	unsafe { data.set_len(total) };

	// The first block still holds the low expansion unscaled, since every other block read it.
	// Its own coefficient therefore lands last.
	let coeff = P::broadcast(high[0]);
	for word in &mut data[..block] {
		*word *= coeff;
	}

	FieldBuffer::new(point.len(), data)
}

/// Truncates a built equality indicator expansion to its low indexed variables.
///
/// Each step contracts the two halves of the buffer, stripping the highest variable.
/// That removes the highest variable's basis factor, whatever its coordinate was.
/// Truncating to `n'` variables therefore leaves the indicator over `r_0, ..., r_{n'-1}`.
///
/// The expansion occupies a prefix of the buffer.
/// Scalars after the truncated length are dropped.
///
/// # Preconditions
///
/// * the truncated length must be at most the buffer's current length
pub fn eq_ind_truncate_low_inplace<Cube: Hypercube, P: PackedField, Data: BufferData<P>>(
	values: &mut FieldBuffer<P, Data>,
	truncated_log_len: usize,
) {
	assert!(
		truncated_log_len <= values.log_len(),
		"precondition: truncated_log_len must be at most values.log_len()"
	);

	// One round per variable stripped, highest first, so the survivors stay in a prefix.
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
					Cube::contract_var(zero, one);
				});
		}

		values.truncate(log_len);
	}
}

/// Evaluates the equality indicator multilinear at a pair of points.
///
/// This is the `2n`-variate multilinear
///
/// ```text
/// eq(X_0, ..., X_{n-1}, Y_0, ..., Y_{n-1}) = prod_i sum_j b_j(X_i) * b_j(Y_i)
/// ```
pub fn eq_ind<Cube: Hypercube, F: FieldOps>(x: &[F], y: &[F]) -> F {
	assert_eq!(x.len(), y.len(), "pre-condition: x and y must be the same length");
	// The indicator factors over the variables, so one per-variable product suffices.
	iter::zip(x, y)
		.map(|(x, y)| Cube::eq_one_var(x.clone(), y.clone()))
		.product()
}

/// Evaluates the equality indicator multilinear with one operand fixed to all zeros.
///
/// Only the constant basis polynomial survives at a zero coordinate:
///
/// ```text
/// eq(0^n, Y_0, ..., Y_{n-1}) = prod_i b_0(Y_i)
/// ```
pub fn eq_ind_zero<Cube: Hypercube, F: FieldOps>(point: &[F]) -> F {
	// The linear basis polynomial is multiplied by a zero coordinate, so it drops out.
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
/// This is the scalar-only engine, which never touches a packed store.
pub fn eq_ind_partial_eval_scalars<Cube: Hypercube, F: FieldOps>(point: &[F]) -> Vec<F> {
	// The unscaled indicator is the scaled indicator with a scale of one.
	scaled_eq_ind_partial_eval_scalars::<Cube, F>(point, F::one())
}

/// Computes the scaled partial evaluation of the equality indicator, returning scalars.
///
/// This is the scalar-only engine, which never touches a packed store.
/// A scale of one is the identity, since the expansion is linear in it.
pub fn scaled_eq_ind_partial_eval_scalars<Cube: Hypercube, F: FieldOps>(
	point: &[F],
	scale: F,
) -> Vec<F> {
	// One coefficient per cube vertex, allocated once.
	let mut result = Vec::with_capacity(1 << point.len());
	// Seed with the scale, which every later multiplication carries through.
	result.push(scale);

	for r_i in point {
		// Each coordinate doubles the length.
		// The low half takes the constant basis factor, the appended high half the linear one.
		//
		//     read index j  ->  overwrite result[j], push its partner past the end
		//
		// Walking the low half front to back is safe, since pushing only appends past it.
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
	use binius_utils::rayon::task_size::{min_len_for_bytes, min_len_for_work};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b, random_scalars};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn expand_var_matches_scaled_basis() {
		let mut rng = StdRng::seed_from_u64(0);

		// Each implementor saves a multiplication over scaling the basis the plain way.
		// So both must land on the same pair.
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

	#[test]
	fn contract_var_inverts_expand_var() {
		let mut rng = StdRng::seed_from_u64(0);

		// Expanding a value by a coordinate and contracting it back must be the identity.
		let [value, coord] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);

		let [mut lo, hi] = OneCube::expand_var(&value, &coord);
		OneCube::contract_var(&mut lo, &hi);
		assert_eq!(lo, value);

		let [mut lo, hi] = InfCube::expand_var(&value, &coord);
		InfCube::contract_var(&mut lo, &hi);
		assert_eq!(lo, value);
	}

	#[test]
	fn eq_one_var_matches_basis_definition() {
		let mut rng = StdRng::seed_from_u64(0);

		// Each implementor overrides this with a closed form that skips the basis evaluations.
		// So pin both against the generic pairing of the bases.
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
	fn inf_cube_eq_ind_zero_is_one() {
		let mut rng = StdRng::seed_from_u64(0);

		// Every monomial of positive degree vanishes at zero, leaving the constant one.
		for n_vars in [0, 1, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			assert_eq!(eq_ind_zero::<InfCube, F>(&point), F::ONE);

			// The same value as evaluating the full indicator against an all-zero operand.
			assert_eq!(
				eq_ind_zero::<InfCube, F>(&point),
				eq_ind::<InfCube, F>(&vec![F::ZERO; n_vars], &point)
			);
		}
	}

	/// The expansion of a point, straight from the definition of the tensor of bases `(1, r_i)`.
	fn inf_cube_reference(point: &[F]) -> Vec<F> {
		// The coefficient at an index is the product of the coordinates its set bits select.
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

	/// Evaluates the multilinear whose monomial coefficients are given, at a point.
	fn eval_monomial_basis(coeffs: &[F], point: &[F]) -> F {
		// The coefficient at an index belongs to the monomial its set bits select.
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

	#[test]
	fn inf_cube_expansion_matches_the_tensor_of_bases() {
		let mut rng = StdRng::seed_from_u64(0);

		// Sizes span the empty point up to a 256-coefficient cube.
		for n_vars in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let expansion = eq_ind_partial_eval::<InfCube, P>(&point);
			let expansion_scalars = expansion.iter_scalars().collect::<Vec<_>>();
			assert_eq!(expansion_scalars, inf_cube_reference(&point), "mismatch at {n_vars} vars");
		}
	}

	#[test]
	fn inf_cube_expansion_holds_the_monomial_coefficients_of_the_indicator() {
		let mut rng = StdRng::seed_from_u64(0);

		// The defining property of this cube: coefficients are monomial coefficients.
		//
		//     expansion of r  ->  the monomial coefficients of eq(X, r)
		//
		// So reading the expansion in the monomial basis at any x must give the indicator there.
		for n_vars in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let coeffs = eq_ind_partial_eval_scalars::<InfCube, F>(&point);

			let x = random_scalars::<F>(&mut rng, n_vars);
			assert_eq!(eval_monomial_basis(&coeffs, &x), eq_ind::<InfCube, F>(&x, &point));
		}
	}

	#[test]
	fn inf_cube_expansion_is_the_evaluation_functional() {
		let mut rng = StdRng::seed_from_u64(0);

		// Read the other way round, the expansion of a point is the functional that evaluates
		// any multilinear at that point, given the multilinear's monomial coefficients.
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
	fn growth_above_the_split_threshold_matches_the_inline_path() {
		let mut rng = StdRng::seed_from_u64(9);

		// Invariant: a round splits across threads only once it exceeds the minimum task size.
		// Below that it runs inline, leaving the parallel path unexercised.
		//
		//     words in the widest round = 2^(n_vars - 1) / scalars per word
		//     pick the smallest n_vars whose widest round reaches the minimum
		//
		// Every other test here is smaller, so this one covers the split.
		let min_len = min_len_for_work(WorkPerItem::FieldMuls);
		let n_vars = (2 * min_len * P::WIDTH).next_power_of_two().ilog2() as usize;
		let point = random_scalars::<F>(&mut rng, n_vars);

		let packed = eq_ind_partial_eval::<OneCube, P>(&point);
		let reference = eq_ind_partial_eval_scalars::<OneCube, F>(&point);
		assert!(packed.iter_scalars().eq(reference.iter().copied()));
	}

	#[test]
	fn truncation_above_the_split_threshold_matches_the_inline_path() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a round splits across threads only once it exceeds the minimum task size.
		// Below that it runs inline, leaving the parallel path unexercised.
		//
		//     words read in the first round = 2^(n_vars - 1) / scalars per word
		//     pick the smallest n_vars whose first round reaches the minimum
		//
		// Every other truncation test here is smaller, so this one covers the split.
		let min_len = min_len_for_bytes::<[P; 2]>();
		let n_vars = (2 * min_len * P::WIDTH).next_power_of_two().ilog2() as usize;
		let point = random_scalars::<F>(&mut rng, n_vars);

		// Strip the top variable and compare against a direct expansion of the prefix.
		let mut truncated = eq_ind_partial_eval::<OneCube, P>(&point);
		eq_ind_truncate_low_inplace::<OneCube, _, _>(&mut truncated, n_vars - 1);
		assert_eq!(truncated, eq_ind_partial_eval::<OneCube, P>(&point[..n_vars - 1]));
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(16))]

		#[test]
		fn the_two_engines_agree(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// Both bases, both the plain and the scaled form: the packed engine and the scalar
			// engine must land on the same coefficients.
			prop_assert_eq!(
				eq_ind_partial_eval::<OneCube, P>(&point).iter_scalars().collect::<Vec<_>>(),
				eq_ind_partial_eval_scalars::<OneCube, F>(&point)
			);
			prop_assert_eq!(
				eq_ind_partial_eval::<InfCube, P>(&point).iter_scalars().collect::<Vec<_>>(),
				eq_ind_partial_eval_scalars::<InfCube, F>(&point)
			);
			prop_assert_eq!(
				scaled_eq_ind_partial_eval::<OneCube, P>(&point, scale)
					.iter_scalars()
					.collect::<Vec<_>>(),
				scaled_eq_ind_partial_eval_scalars::<OneCube, F>(&point, scale)
			);
			prop_assert_eq!(
				scaled_eq_ind_partial_eval::<InfCube, P>(&point, scale)
					.iter_scalars()
					.collect::<Vec<_>>(),
				scaled_eq_ind_partial_eval_scalars::<InfCube, F>(&point, scale)
			);

			// A scalar field is a packed field of one lane, so the packed engine also runs at
			// a packing width of one. That width is its own path through the growth loop:
			//
			//     4 lanes per word    two rounds live inside one word, then rounds double
			//     1 lane per word     no round fits inside a word, so every round doubles
			//
			// The one-lane store is exactly the scalars, so the two must agree there too.
			prop_assert_eq!(
				eq_ind_partial_eval::<OneCube, F>(&point).into_inner(),
				eq_ind_partial_eval_scalars::<OneCube, F>(&point)
			);
			prop_assert_eq!(
				eq_ind_partial_eval::<InfCube, F>(&point).into_inner(),
				eq_ind_partial_eval_scalars::<InfCube, F>(&point)
			);
		}

		#[test]
		fn scaling_commutes_with_the_expansion(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// Scaling the seed scales every coefficient, for either basis.
			let one_scaled = scaled_eq_ind_partial_eval::<OneCube, P>(&point, scale);
			let one_plain = eq_ind_partial_eval::<OneCube, P>(&point);
			for (got, base) in one_scaled.iter_scalars().zip(one_plain.iter_scalars()) {
				prop_assert_eq!(got, scale * base);
			}

			let inf_scaled = scaled_eq_ind_partial_eval::<InfCube, P>(&point, scale);
			let inf_plain = eq_ind_partial_eval::<InfCube, P>(&point);
			for (got, base) in inf_scaled.iter_scalars().zip(inf_plain.iter_scalars()) {
				prop_assert_eq!(got, scale * base);
			}
		}

		#[test]
		fn truncation_strips_trailing_variables(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);

			// Truncating to any length must equal expanding that prefix of the point directly.
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

		#[test]
		fn the_split_agrees_with_the_doubling_rounds(
			seed in any::<u64>(),
			n_vars in 3usize..=10,
			low_len in 2usize..=8,
		) {
			// Property: cutting the point and multiplying the two expansions together is the same
			// map as appending its coordinates one at a time.
			//
			// The production cut is far above these sizes, so it is passed in directly here.
			// That covers every cut shape, including the ones a tuning change could start picking.
			//
			//     n_vars = 5, low_len = 2  ->  4 blocks of 4 coefficients
			prop_assume!(low_len >= P::LOG_WIDTH);
			prop_assume!(low_len < n_vars);

			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, n_vars);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// A seed is one coefficient carrying the scale, over a store sized for the result.
			let seeded = || {
				let mut buffer = Vec::with_capacity(1 << n_vars.saturating_sub(P::LOG_WIDTH));
				buffer.push(P::from_scalars(iter::once(scale)));
				FieldBuffer::new(0, buffer)
			};

			prop_assert_eq!(
				split_expand::<OneCube, P, _>(seeded(), &point, low_len),
				tensor_prod_eq_ind_reserved::<OneCube, P, _>(seeded(), &point)
			);
			prop_assert_eq!(
				split_expand::<InfCube, P, _>(seeded(), &point, low_len),
				tensor_prod_eq_ind_reserved::<InfCube, P, _>(seeded(), &point)
			);
		}
	}

	#[test]
	fn the_split_path_agrees_with_the_scalar_engine() {
		let mut rng = StdRng::seed_from_u64(0);

		// The public entry point picks the cut itself, and only past a crossover.
		// So this runs at sizes that actually reach it, against the scalar engine, which shares
		// no code with either packed path.
		for n_vars in [18, 19] {
			assert!(split_low_len::<P>(n_vars).is_some(), "expected the split at {n_vars} vars");

			let point = random_scalars::<F>(&mut rng, n_vars);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			let packed = scaled_eq_ind_partial_eval::<OneCube, P>(&point, scale);
			let scalars = scaled_eq_ind_partial_eval_scalars::<OneCube, F>(&point, scale);
			assert!(packed.iter_scalars().eq(scalars), "one cube at {n_vars} vars");

			let packed = scaled_eq_ind_partial_eval::<InfCube, P>(&point, scale);
			let scalars = scaled_eq_ind_partial_eval_scalars::<InfCube, F>(&point, scale);
			assert!(packed.iter_scalars().eq(scalars), "inf cube at {n_vars} vars");
		}
	}

	#[test]
	fn the_crossover_leaves_a_whole_block_below_the_cut() {
		// Invariant: a block spans whole packed words, and the result spans whole blocks.
		//
		// Both are what let the outer product address a block as a run of the store.
		for n_vars in 0..24 {
			let Some(low_len) = split_low_len::<P>(n_vars) else {
				continue;
			};
			assert!(low_len >= P::LOG_WIDTH, "block below one packed word at {n_vars} vars");
			assert!(low_len < n_vars, "no block above the cut at {n_vars} vars");
		}
	}
}
