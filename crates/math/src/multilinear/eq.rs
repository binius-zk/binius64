// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The equality indicator over the Boolean hypercube.
//!
//! Every routine here specializes a generic hypercube routine to the basis `(1 - X, X)`.
//! Under that basis the coefficients of a multilinear are its evaluations over `{0, 1}^n`.
//!
//! See [DP23], Section 2.1 for more information about the equality indicator polynomial.
//!
//! [DP23]: <https://eprint.iacr.org/2023/1784>

use binius_compute::{Allocator, BufferData, VecLike};
use binius_field::{PackedField, field::FieldOps};

use super::hypercube::{self, Hypercube, OneCube};
use crate::{FieldBuffer, FieldVec};

/// Tensor of values with the equality indicator evaluated at extra coordinates.
///
/// One variable is added per coordinate, doubling the length each time.
/// The returned buffer grows its backing store rather than allocating a fresh one.
pub fn tensor_prod_eq_ind<P: PackedField>(
	values: FieldBuffer<P, Vec<P>>,
	extra_query_coordinates: &[P::Scalar],
) -> FieldBuffer<P, Vec<P>> {
	hypercube::tensor_prod_eq_ind::<OneCube, P>(values, extra_query_coordinates)
}

/// Computes the partial evaluation of the equality indicator polynomial.
///
/// For the point `r = (r_0, ..., r_{n-1})` the result holds the `2^n` values
///
/// ```text
/// (1 - r_0, r_0) (x) ... (x) (1 - r_{n-1}, r_{n-1})
/// ```
///
/// which are the values of `eq(X_0, ..., X_{n-1}, r)` over the hypercube.
pub fn eq_ind_partial_eval<P: PackedField>(point: &[P::Scalar]) -> FieldBuffer<P> {
	hypercube::eq_ind_partial_eval::<OneCube, P>(point)
}

/// Builds the equality indicator expansion of a point into a buffer drawn from an allocator.
///
/// Backed by a pool, the result is a recyclable buffer rather than a fresh allocation.
pub fn eq_ind_partial_eval_in<A: Allocator, P: PackedField>(
	alloc: &A,
	point: &[P::Scalar],
) -> FieldVec<P, A> {
	hypercube::eq_ind_partial_eval_in::<OneCube, A, P>(alloc, point)
}

/// Computes the partial evaluation of the equality indicator polynomial, scaled by a constant.
///
/// Every hypercube value of the equality indicator is multiplied by the scale.
/// A scale of one is the identity, since the expansion is linear in it.
///
/// # Arguments
///
/// * `point` - The evaluation point whose length is the number of variables.
/// * `scale` - The constant every returned value is multiplied by.
pub fn scaled_eq_ind_partial_eval<P: PackedField>(
	point: &[P::Scalar],
	scale: P::Scalar,
) -> FieldBuffer<P> {
	hypercube::scaled_eq_ind_partial_eval::<OneCube, P>(point, scale)
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
pub fn scaled_eq_ind_partial_eval_into<P: PackedField, Data: VecLike<P>>(
	point: &[P::Scalar],
	scale: P::Scalar,
	buffer: Data,
) -> FieldBuffer<P, Data> {
	hypercube::scaled_eq_ind_partial_eval_into::<OneCube, P, Data>(point, scale, buffer)
}

/// Truncates a built equality indicator expansion to its low indexed variables.
///
/// Each step sums the two halves of the buffer, stripping the highest variable.
/// Truncating to `n'` variables leaves the indicator over `r_0, ..., r_{n'-1}`.
///
/// The expansion occupies a prefix of the buffer.
/// Scalars after the truncated length are dropped.
///
/// # Preconditions
///
/// * the truncated length must be at most the buffer's current length
pub fn eq_ind_truncate_low_inplace<P: PackedField, Data: BufferData<P>>(
	values: &mut FieldBuffer<P, Data>,
	truncated_log_len: usize,
) {
	hypercube::eq_ind_truncate_low_inplace::<OneCube, _, _>(values, truncated_log_len);
}

/// Evaluates the 2-variate multilinear which indicates the equality condition.
///
/// ```text
/// eq(X, Y) = X * Y + (1 - X) * (1 - Y)
/// ```
///
/// Over a binary field the cross term vanishes, so this simplifies to
///
/// ```text
/// eq(X, Y) = X + Y + 1
/// ```
#[inline(always)]
pub fn eq_one_var<F: FieldOps>(x: F, y: F) -> F {
	OneCube::eq_one_var(x, y)
}

/// Evaluates the equality indicator multilinear at a pair of points.
///
/// This is the `2n`-variate multilinear
///
/// ```text
/// eq(X_0, ..., X_{n-1}, Y_0, ..., Y_{n-1}) = prod_i X_i * Y_i + (1 - X_i) * (1 - Y_i)
/// ```
pub fn eq_ind<F: FieldOps>(x: &[F], y: &[F]) -> F {
	hypercube::eq_ind::<OneCube, F>(x, y)
}

/// Evaluates the equality indicator multilinear with one operand fixed to all zeros.
///
/// Only the constant basis polynomial survives at a zero coordinate:
///
/// ```text
/// eq(0^n, Y_0, ..., Y_{n-1}) = prod_i (1 - Y_i)
/// ```
pub fn eq_ind_zero<F: FieldOps>(point: &[F]) -> F {
	hypercube::eq_ind_zero::<OneCube, F>(point)
}

/// Computes the partial evaluation of the equality indicator polynomial, returning scalars.
///
/// This is the scalar-only engine, which never touches a packed store.
pub fn eq_ind_partial_eval_scalars<F: FieldOps>(point: &[F]) -> Vec<F> {
	hypercube::eq_ind_partial_eval_scalars::<OneCube, F>(point)
}

/// Computes the scaled partial evaluation of the equality indicator, returning scalars.
///
/// This is the scalar-only engine, which never touches a packed store.
/// A scale of one is the identity, since the expansion is linear in it.
pub fn scaled_eq_ind_partial_eval_scalars<F: FieldOps>(point: &[F], scale: F) -> Vec<F> {
	hypercube::scaled_eq_ind_partial_eval_scalars::<OneCube, F>(point, scale)
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::Field;
	use rand::prelude::*;

	use super::*;
	use crate::{
		bit_reverse::bit_reverse_packed,
		test_utils::{B128, Packed128b, index_to_hypercube_point, random_scalars},
	};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn expansion_holds_the_indicator_at_every_vertex() {
		let mut rng = StdRng::seed_from_u64(0);

		// The defining property of this cube: coefficients are evaluations.
		// So the coefficient at an index is the indicator evaluated at that index's vertex.
		let n_vars = 5;
		let point = random_scalars(&mut rng, n_vars);
		let expansion = eq_ind_partial_eval::<P>(&point);

		for index in 0..1 << n_vars {
			let vertex = index_to_hypercube_point(n_vars, index);
			assert_eq!(expansion.get(index), eq_ind::<F>(&point, &vertex));
		}
	}

	#[test]
	fn expansion_of_the_empty_point() {
		// The empty point has no variables, so its expansion is the single value one.
		let result = eq_ind_partial_eval::<P>(&[]);
		assert_eq!(result.log_len(), 0);
		assert_eq!(result.len(), 1);
		assert_eq!(result.get(0), F::ONE);
	}

	#[test]
	fn expansion_of_one_coordinate_is_the_basis() {
		// One coordinate expands to the basis `(1 - r_0, r_0)` itself.
		let r0 = F::new(2);
		let result = eq_ind_partial_eval::<P>(&[r0]);
		assert_eq!(result.log_len(), 1);
		assert_eq!(result.len(), 2);
		assert_eq!(result.get(0), F::ONE - r0);
		assert_eq!(result.get(1), r0);
	}

	#[test]
	fn expansion_of_two_coordinates() {
		// Two coordinates: the four products of one factor drawn from each basis.
		let r0 = F::new(2);
		let r1 = F::new(3);
		let result = eq_ind_partial_eval::<P>(&[r0, r1]);
		assert_eq!(result.log_len(), 2);
		assert_eq!(result.len(), 4);

		// The variable index is the bit position, so `r_0` varies fastest.
		let expected = vec![
			(F::ONE - r0) * (F::ONE - r1),
			r0 * (F::ONE - r1),
			(F::ONE - r0) * r1,
			r0 * r1,
		];
		assert_eq!(result.iter_scalars().collect::<Vec<F>>(), expected);
	}

	#[test]
	fn expansion_of_three_coordinates_fills_one_packed_word() {
		// Three coordinates span exactly one full packed word at this packing width.
		let r0 = F::new(2);
		let r1 = F::new(3);
		let r2 = F::new(5);
		let result = eq_ind_partial_eval::<P>(&[r0, r1, r2]);
		assert_eq!(result.log_len(), 3);
		assert_eq!(result.len(), 8);

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
		assert_eq!(result.iter_scalars().collect::<Vec<F>>(), expected);
	}

	#[test]
	fn eq_ind_zero_is_the_product_of_complements() {
		let mut rng = StdRng::seed_from_u64(0);

		// The constant basis polynomial of this cube is `1 - Y`.
		for n_vars in 0..5 {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let expected: F = point.iter().map(|&r| F::ONE - r).product();
			assert_eq!(eq_ind_zero(&point), expected);

			// The same value as evaluating the full indicator against an all-zero operand.
			assert_eq!(eq_ind_zero(&point), eq_ind(&vec![F::ZERO; n_vars], &point));
		}
	}

	#[test]
	fn every_storage_form_holds_the_same_values() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the storage choice never changes what is computed.
		//
		//     fresh store | allocator | caller's store | plain scalars
		//
		// All four must agree value for value, at every size.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let reference = eq_ind_partial_eval::<P>(&point);

			let pooled = eq_ind_partial_eval_in::<_, P>(&GlobalAllocator, &point);
			assert!(pooled.iter_scalars().eq(reference.iter_scalars()), "pool at log_n={log_n}");

			let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);
			let supplied = scaled_eq_ind_partial_eval_into::<P, _>(
				&point,
				F::ONE,
				Vec::with_capacity(capacity),
			);
			assert_eq!(supplied, reference, "supplied store at log_n={log_n}");

			let scalars = eq_ind_partial_eval_scalars(&point);
			assert!(reference.iter_scalars().eq(scalars), "scalars at log_n={log_n}");
		}
	}

	#[test]
	fn the_scale_applies_to_every_storage_form_alike() {
		let mut rng = StdRng::seed_from_u64(1);

		// Invariant: the scale is independent of where the values are stored.
		// So scaling commutes with every storage form, pooled memory included.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];
			let unscaled = eq_ind_partial_eval::<P>(&point);

			let scaled = scaled_eq_ind_partial_eval::<P>(&point, scale);
			for (got, base) in scaled.iter_scalars().zip(unscaled.iter_scalars()) {
				assert_eq!(got, scale * base, "fresh store at log_n={log_n}");
			}

			let scalars = scaled_eq_ind_partial_eval_scalars(&point, scale);
			assert!(scaled.iter_scalars().eq(scalars), "scalars at log_n={log_n}");
		}
	}

	#[test]
	fn a_scale_of_one_is_the_identity() {
		let mut rng = StdRng::seed_from_u64(2);

		// Invariant: the expansion is linear in its scale, so a scale of one changes nothing.
		// Equality is checked packed word by packed word, not just value by value.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			assert_eq!(
				scaled_eq_ind_partial_eval::<P>(&point, F::ONE),
				eq_ind_partial_eval::<P>(&point),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn a_scale_of_zero_gives_all_zeros() {
		let mut rng = StdRng::seed_from_u64(3);

		// The other end of that linearity: a scale of zero yields the all-zero polynomial.
		for log_n in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scaled = scaled_eq_ind_partial_eval::<P>(&point, F::ZERO);
			assert!(scaled.iter_scalars().all(|v| v == F::ZERO), "nonzero at log_n={log_n}");
		}
	}

	#[test]
	fn a_caller_reserved_store_matches_the_allocating_form() {
		let mut rng = StdRng::seed_from_u64(5);

		// Invariant: filling a caller-reserved store reproduces the allocating variant exactly,
		// with the store reserved to the exact packed capacity the routine demands.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);
			let result = scaled_eq_ind_partial_eval_into::<P, _>(
				&point,
				scale,
				Vec::with_capacity(capacity),
			);

			assert_eq!(result.log_len(), log_n, "wrong length at log_n={log_n}");
			assert_eq!(
				result,
				scaled_eq_ind_partial_eval::<P>(&point, scale),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn appending_onto_a_one_value_store_builds_from_scratch() {
		let mut rng = StdRng::seed_from_u64(6);

		// The values already present are the seed.
		// So appending a whole point onto the single value one is the plain expansion.
		let point = random_scalars::<F>(&mut rng, 5);
		let seed = FieldBuffer::<P, _>::scalar_with_capacity(F::ONE, point.len());

		assert_eq!(tensor_prod_eq_ind::<P>(seed, &point), eq_ind_partial_eval::<P>(&point));
	}

	#[test]
	fn appending_in_batches_matches_one_full_expansion() {
		let mut rng = StdRng::seed_from_u64(7);

		// Append coordinates in batches of growing size, reusing one reserved backing store.
		//
		//     batch sizes 1, 2, 3, 4  ->  1 + 2 + 3 + 4 = 10 variables in total
		let batches = 4;
		let max_n_vars = batches * (batches + 1) / 2;
		let mut coords = Vec::with_capacity(max_n_vars);
		let mut eq_expansion = FieldBuffer::<P, _>::scalar_with_capacity(F::ONE, max_n_vars);

		for batch_len in 1..=batches {
			let extra = random_scalars(&mut rng, batch_len);

			eq_expansion = tensor_prod_eq_ind::<P>(eq_expansion, &extra);
			coords.extend(&extra);

			// Every batch must leave the indicator over all coordinates appended so far.
			assert_eq!(eq_expansion.log_len(), coords.len());
			for i in 0..eq_expansion.len() {
				let vertex = index_to_hypercube_point(coords.len(), i);
				assert_eq!(eq_expansion.get(i), eq_ind(&vertex, &coords));
			}
		}
	}

	#[test]
	fn prepending_via_bit_reverse_matches_one_full_expansion() {
		let mut rng = StdRng::seed_from_u64(8);

		// Appending is the only primitive, so prepending a variable is spelled as
		//
		//     bit reverse  ->  append  ->  bit reverse
		//
		// which is how the binary switchover prover adds one variable per round.
		// Iterating it over ten coordinates also covers the sub-packing-width early rounds.
		let n_vars = 10;
		let point = random_scalars::<F>(&mut rng, n_vars);

		let mut tensor = FieldBuffer::<P>::from_values(&[F::ONE]);
		for &r in point.iter().rev() {
			bit_reverse_packed(tensor.as_mut_view());
			tensor = tensor_prod_eq_ind::<P>(tensor, &[r]);
			bit_reverse_packed(tensor.as_mut_view());
		}

		assert_eq!(tensor, eq_ind_partial_eval::<P>(&point));
	}

	#[test]
	fn repeated_truncation_matches_expansion_of_the_prefix() {
		let mut rng = StdRng::seed_from_u64(0);

		// Truncate the same buffer over and over, by a shrinking number of variables each time.
		//
		//     reductions 4, 3, 2, 1, 0  ->  10 variables spent in total
		let reductions = 4;
		let n_vars = reductions * (reductions + 1) / 2;
		let point = random_scalars(&mut rng, n_vars);

		let mut eq_ind = eq_ind_partial_eval::<P>(&point);
		let mut log_n_values = n_vars;

		for reduction in (0..=reductions).rev() {
			let truncated_log_n_values = log_n_values - reduction;
			eq_ind_truncate_low_inplace(&mut eq_ind, truncated_log_n_values);

			// Each step must match a direct expansion of the surviving prefix of the point.
			let eq_ind_ref = eq_ind_partial_eval::<P>(&point[..truncated_log_n_values]);
			assert_eq!(eq_ind_ref.len(), eq_ind.len());
			for i in 0..eq_ind.len() {
				assert_eq!(eq_ind.get(i), eq_ind_ref.get(i));
			}

			log_n_values = truncated_log_n_values;
		}

		// The last reduction is by zero variables, so the sequence ends at the empty point.
		assert_eq!(log_n_values, 0);
	}
}
