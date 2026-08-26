// Copyright 2026 The Binius Developers

//! The polynomial vocabulary of a buffer that holds multilinear coefficients.
//!
//! A field buffer is a container.
//! It holds packed field elements and knows how many.
//! A multilinear polynomial is one of the things such a buffer can mean.
//!
//! These traits carry the polynomial operations rather than the container.
//! A buffer holding a Merkle layer or a Reed-Solomon codeword would otherwise advertise them too.
//!
//! They also disambiguate one integer:
//!
//! ```text
//! log_len   how many elements the container holds, as a power of two
//! n_vars    how many variables the polynomial takes
//! ```
//!
//! Both are the same number.
//! Which method a call site reaches for says which of the two meanings it intends.
//!
//! Reading a polynomial needs only a shared backing store.
//! Folding halves the buffer, so mutating needs a store that shrinks in place.
//! That split is what separates the two traits.
//!
//! An operation on two polynomials takes its second operand as a borrowed buffer view.
//! Every buffer converts to one by reference, so a call site writes `a.inner_product(&b)`.

use std::ops::Deref;

use binius_compute::{Allocator, BufferData, VecLike};
use binius_field::{Field, PackedField, WideMul};
use binius_utils::{
	random_access_sequence::RandomAccessSequence,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};

use crate::{
	FieldBuffer, FieldSlice, FieldVec, inner_product::inner_product_packed,
	multilinear::hypercube::Hypercube,
};

/// A buffer of coefficients read as a multilinear polynomial.
///
/// An $n$-variate multilinear has $2^n$ coefficients, one per vertex of the Boolean hypercube.
/// The coefficient at a vertex is the polynomial's value there.
/// So the coefficient count and the variable count determine each other.
pub trait Multilinear<P: PackedField>: Sized {
	/// The number of variables the polynomial takes.
	///
	/// This is the container's log length, read as an arity.
	fn n_vars(&self) -> usize;

	/// Evaluates the polynomial at a point, leaving the coefficients in place.
	///
	/// The point holds one coordinate per variable.
	/// The result is a single field element.
	/// Memory used is on the order of the square root of the coefficient count.
	///
	/// ## Preconditions
	///
	/// * `point.len()` must equal `self.n_vars()`
	fn evaluate(&self, point: &[P::Scalar]) -> P::Scalar;

	/// Sums the coefficient-by-coefficient products of two polynomials.
	///
	/// ```text
	/// result = sum_i a_i * b_i
	/// ```
	///
	/// This is not the product polynomial; it is a single field element.
	/// Pairing a polynomial with an equality indicator expansion evaluates it at that point.
	///
	/// ## Preconditions
	///
	/// * `other` must take the same number of variables as `self`
	fn inner_product<'b>(&self, other: impl Into<FieldSlice<'b, P>>) -> P::Scalar;

	/// Sums the coefficient-by-coefficient products of two polynomials, across threads.
	///
	/// The value matches the single-threaded pairing.
	/// The coefficient range is split into tasks, each summing its own partial products.
	///
	/// ## Preconditions
	///
	/// * `other` must take the same number of variables as `self`
	fn par_inner_product<'b>(&self, other: impl Into<FieldSlice<'b, P>>) -> P::Scalar;

	/// Fixes the highest variable to a value, writing into memory drawn from an allocator.
	///
	/// Fixing one variable of an $n$-variate multilinear leaves an $(n-1)$-variate one:
	///
	/// ```text
	/// g(X_0, ..., X_{n-2}) = f(X_0, ..., X_{n-2}, r)
	/// ```
	///
	/// Each output coefficient interpolates the line through one pair of input coefficients.
	/// The input is left untouched.
	///
	/// ## Preconditions
	///
	/// * `self.n_vars()` must be at least one
	fn fold_highest_var_in<A: Allocator>(&self, alloc: &A, scalar: P::Scalar) -> FieldVec<P, A>;
}

/// A buffer of coefficients rewritten in place as a multilinear polynomial.
///
/// Every operation here overwrites the coefficients.
/// Fixing one variable halves the coefficient count, so the backing store must shrink in place.
pub trait MultilinearMut<P: PackedField>: Multilinear<P> {
	/// Fixes the highest variable to a value, in place.
	///
	/// ```text
	/// g(X_0, ..., X_{n-2}) = f(X_0, ..., X_{n-2}, r)
	/// ```
	///
	/// The result occupies the first half of the buffer.
	/// The buffer then reports one variable fewer.
	///
	/// ## Preconditions
	///
	/// * `self.n_vars()` must be at least one
	fn fold_highest_var(&mut self, scalar: P::Scalar);

	/// Contracts the highest variables away, keeping the low-indexed ones.
	///
	/// One step replaces the two halves of the buffer by their contraction.
	/// That strips the highest variable's basis factor, whatever its coordinate was.
	/// Over an equality indicator expansion the result is the indicator over the coordinates kept.
	///
	/// The cube parameter fixes the per-variable basis the coefficients are taken against.
	///
	/// ## Preconditions
	///
	/// * `truncated_n_vars` must be at most `self.n_vars()`
	fn eq_ind_truncate_low(&mut self, cube: Hypercube, truncated_n_vars: usize);

	/// Overwrites the coefficients with the high fold of a bit sequence by a tensor.
	///
	/// The bits are the coefficients of a multilinear whose values are all zero or one.
	/// Each output vertex fixes that polynomial's low-indexed variables to that vertex.
	/// What remains is then paired with the tensor.
	///
	/// This runs on one thread.
	///
	/// ## Preconditions
	///
	/// * `bits.len()` must be a power of two
	/// * `bits.len()` must equal `1 << (self.n_vars() + tensor.n_vars())`
	fn binary_fold_high<'b>(
		&mut self,
		tensor: impl Into<FieldSlice<'b, P>>,
		bits: &(impl RandomAccessSequence<bool> + Sync),
	);

	/// Evaluates the polynomial at a point, consuming the coefficients.
	///
	/// One variable is fixed at a time, in place, so nothing beyond the buffer is allocated.
	/// Each fold halves the buffer, and the last one leaves the single result.
	///
	/// ## Preconditions
	///
	/// * `coords.len()` must equal `self.n_vars()`
	fn evaluate_inplace(self, coords: &[P::Scalar]) -> P::Scalar;
}

impl<P: PackedField, Data: Deref<Target = [P]>> Multilinear<P> for FieldBuffer<P, Data> {
	fn n_vars(&self) -> usize {
		self.log_len()
	}

	fn evaluate(&self, point: &[P::Scalar]) -> P::Scalar {
		assert_eq!(point.len(), self.log_len(), "precondition: point length must equal n_vars");

		// The point splits in half, and the first half gets at least one packed word's worth.
		// Expanding only that half costs memory on the order of the square root of the whole.
		let first_half_len = (point.len() / 2).max(P::LOG_WIDTH).min(point.len());
		let (first_coords, remaining_coords) = point.split_at(first_half_len);
		let eq_tensor = Hypercube::One.expand(first_coords).build::<P>();

		// With nothing left over the expansion covers every variable, so one pairing finishes.
		if remaining_coords.is_empty() {
			return self.inner_product(&eq_tensor);
		}

		// Otherwise each chunk pairs with the expansion, and the resulting scalars are the
		// residual multilinear over the coordinates not yet used.
		let scalars = self
			.par_chunks(first_half_len)
			.map(|chunk| chunk.inner_product(&eq_tensor))
			.collect::<Vec<_>>();

		FieldBuffer::<P>::from_values(&scalars).evaluate_inplace(remaining_coords)
	}

	#[inline]
	fn inner_product<'b>(&self, other: impl Into<FieldSlice<'b, P>>) -> P::Scalar {
		let other = other.into();
		inner_product_packed(
			self.log_len(),
			self.iter_packed().copied(),
			other.iter_packed().copied(),
		)
	}

	#[inline]
	fn par_inner_product<'b>(&self, other: impl Into<FieldSlice<'b, P>>) -> P::Scalar {
		let other = other.into();
		let n = self.len();

		// Accumulate the products in unreduced (wide) form and reduce a single time at the end.
		// For packed `GF(2^128)` fields this amortizes the reduction cost across all products.
		// For every other field the widening multiply is the trivial one, so this is a plain
		// product-then-sum.
		let wide_sum = self
			.as_ref()
			.par_iter()
			.zip_eq(other.as_ref().par_iter())
			.with_min_task(WorkPerItem::FieldMuls)
			.map(|(&a_i, &b_i)| P::wide_mul(a_i, b_i))
			.sum::<<P as WideMul>::Output>();
		P::reduce(wide_sum).into_iter().take(n).sum()
	}

	fn fold_highest_var_in<A: Allocator>(&self, alloc: &A, scalar: P::Scalar) -> FieldVec<P, A> {
		assert!(self.log_len() > 0, "precondition: buffer must have at least one variable");

		// The two halves are the multilinear specialized to 0 and to 1 on the highest variable.
		let broadcast_scalar = P::broadcast(scalar);
		let (lo, hi) = self.split_half();

		// Interpolate the line through each pair at the challenge directly into a fresh buffer
		// drawn from the allocator, writing the uninitialized spare capacity in parallel rather
		// than zero-filling first.
		let len = lo.as_ref().len();
		let mut data = alloc.alloc::<P>(len);
		let spare = &mut data.spare_capacity_mut()[..len];
		(spare, lo.as_ref(), hi.as_ref())
			.into_par_iter()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(out, &lo_i, &hi_i)| {
				out.write(Hypercube::One.fold_var(lo_i, hi_i, &broadcast_scalar));
			});
		// SAFETY: the parallel loop initialized all `len` slots.
		unsafe { data.set_len(len) };
		FieldBuffer::new(self.log_len() - 1, data)
	}
}

impl<P: PackedField, Data: BufferData<P>> MultilinearMut<P> for FieldBuffer<P, Data> {
	fn fold_highest_var(&mut self, scalar: P::Scalar) {
		// Each scalar of the result costs one multiplication.
		// The result occupies a prefix, so the truncation drops the scalars past it.
		let broadcast_scalar = P::broadcast(scalar);
		{
			let mut split = self.split_half_mut();
			let (mut lo, mut hi) = split.halves();
			(lo.as_mut(), hi.as_mut())
				.into_par_iter()
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|(lo_i, hi_i)| {
					*lo_i = Hypercube::One.fold_var(*lo_i, *hi_i, &broadcast_scalar);
				});
		}

		self.truncate(self.log_len() - 1);
	}

	fn eq_ind_truncate_low(&mut self, cube: Hypercube, truncated_n_vars: usize) {
		assert!(
			truncated_n_vars <= self.log_len(),
			"precondition: truncated_n_vars must be at most n_vars"
		);

		for log_len in (truncated_n_vars..self.log_len()).rev() {
			{
				let mut split = self.split_half_mut();
				let (mut lo, hi) = split.halves();
				// Contracting a variable costs additions only.
				// So the cost of one step is the two words it reads, not its arithmetic.
				(lo.as_mut(), hi.as_ref())
					.into_par_iter()
					.with_min_task_bytes::<[P; 2]>()
					.for_each(|(zero, one)| {
						cube.contract_var(zero, one);
					});
			}

			self.truncate(log_len);
		}
	}

	fn binary_fold_high<'b>(
		&mut self,
		tensor: impl Into<FieldSlice<'b, P>>,
		bits: &(impl RandomAccessSequence<bool> + Sync),
	) {
		let tensor = tensor.into();
		assert!(bits.len().is_power_of_two(), "precondition: bits length must be a power of two");

		let values_log_len = self.log_len();
		let width = P::WIDTH.min(self.len());

		assert_eq!(
			1 << (values_log_len + tensor.log_len()),
			bits.len(),
			"precondition: bits length must equal values length times tensor length"
		);

		self.iter_packed_mut().enumerate().for_each(|(i, packed)| {
			*packed = P::from_scalars((0..width).map(|j| {
				let scalar_index = i << P::LOG_WIDTH | j;
				let mut acc = P::Scalar::ZERO;

				for (k, tensor_packed) in tensor.iter_packed().enumerate() {
					for (l, tensor_scalar) in tensor_packed.iter().take(tensor.len()).enumerate() {
						let tensor_scalar_index = k << P::LOG_WIDTH | l;
						if bits.get(tensor_scalar_index << values_log_len | scalar_index) {
							acc += tensor_scalar;
						}
					}
				}

				acc
			}));
		});
	}

	fn evaluate_inplace(mut self, coords: &[P::Scalar]) -> P::Scalar {
		assert_eq!(coords.len(), self.log_len(), "precondition: coords length must equal n_vars");

		// Fixing the highest variable first keeps the survivors in a prefix, so an $n$-variate
		// polynomial costs $2^n - 1$ multiplications and no memory beyond the buffer.
		for &coord in coords.iter().rev() {
			self.fold_highest_var(coord);
		}

		assert_eq!(self.len(), 1);
		self.get(0)
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_compute::GlobalAllocator;
	use binius_utils::rayon::task_size::min_len_for_work;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{
		B128, Packed128b, index_to_hypercube_point, random_field_buffer, random_scalars,
	};

	type P = Packed128b;
	type F = B128;

	// The packing width is four scalars, so this range straddles it in both directions.
	const MAX_VARS: usize = 8;

	#[test]
	fn n_vars_reads_the_containers_log_len() {
		let mut rng = StdRng::seed_from_u64(0);

		for n_vars in 0..=MAX_VARS {
			let buffer = random_field_buffer::<P>(&mut rng, n_vars);
			assert_eq!(buffer.n_vars(), buffer.log_len(), "arity must be the log length");
			assert_eq!(buffer.n_vars(), n_vars);
		}
	}

	#[test]
	fn read_trait_covers_a_borrowed_view() {
		let mut rng = StdRng::seed_from_u64(0);

		// A shared view carries no store of its own, so this pins the read impl's bound.
		let buffer = random_field_buffer::<P>(&mut rng, 5);
		let point = random_scalars::<F>(&mut rng, 5);
		let view: FieldSlice<'_, P> = buffer.as_view();

		assert_eq!(view.n_vars(), 5);
		assert_eq!(view.evaluate(&point), buffer.evaluate(&point));
	}

	#[test]
	fn mut_trait_covers_a_mutably_borrowed_slice() {
		let mut rng = StdRng::seed_from_u64(0);

		// A slice-backed buffer shrinks by re-slicing, which is all the mut impl's bound demands.
		let original = random_field_buffer::<P>(&mut rng, 5);
		let scalar = random_scalars::<F>(&mut rng, 1)[0];

		let mut expected = original.clone();
		expected.fold_highest_var(scalar);

		let mut owned = original;
		let mut slice = owned.as_mut_view();
		slice.fold_highest_var(scalar);

		assert_eq!(slice.n_vars(), 4);
		assert_eq!(slice, expected.as_mut_view());
	}

	#[test]
	fn fold_splits_above_the_task_threshold() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a fold splits only once each half holds two minimum tasks.
		// Below that it runs inline, leaving the parallel path unexercised.
		//
		//     words in one half = 2^(n_vars - 1) / scalars per word
		//     smallest n_vars with words in one half >= 2 * minimum
		//
		// Every other fold test is smaller, so this one covers the split.
		let min_len = min_len_for_work(WorkPerItem::FieldMuls);
		let n_vars = (2 * min_len * P::WIDTH).next_power_of_two().ilog2() as usize + 1;
		let half = 1 << (n_vars - 1);
		let original = random_field_buffer::<P>(&mut rng, n_vars);
		let challenge = random_scalars::<F>(&mut rng, 1)[0];

		let mut folded = original.clone();
		folded.fold_highest_var(challenge);
		assert_eq!(folded.n_vars(), n_vars - 1);

		// Scalar reference: each output interpolates one (lo, hi) pair at the challenge.
		for i in 0..half {
			let expected =
				Hypercube::One.fold_var(original.get(i), original.get(i | half), &challenge);
			assert_eq!(folded.get(i), expected, "mismatch at index {i}");
		}
	}

	#[test]
	fn evaluate_at_a_hypercube_vertex_reads_that_coefficient() {
		let mut rng = StdRng::seed_from_u64(0);

		// Every vertex of a small cube is cheap enough to check exhaustively.
		let n_vars = 8;
		let buffer = random_field_buffer::<F>(&mut rng, n_vars);

		for index in 0..1 << n_vars {
			let point = index_to_hypercube_point::<F>(n_vars, index);

			assert_eq!(buffer.evaluate(&point), buffer.get(index), "mismatch at vertex {index}");
		}
	}

	#[test]
	fn evaluate_is_linear_in_every_coordinate() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 8;
		let buffer = random_field_buffer::<F>(&mut rng, n_vars);
		let mut point = random_scalars::<F>(&mut rng, n_vars);

		for coord_idx in 0..n_vars {
			// Three points differing only in this coordinate must have collinear evaluations.
			let coord_vals = random_scalars::<F>(&mut rng, 3);
			let evals = coord_vals
				.iter()
				.map(|&coord_val| {
					point[coord_idx] = coord_val;
					buffer.evaluate(&point)
				})
				.collect::<Vec<_>>();

			// Collinearity of the three points, cross-multiplied so nothing is divided.
			let [x0, x1, x2] = [coord_vals[0], coord_vals[1], coord_vals[2]];
			let [y0, y1, y2] = [evals[0], evals[1], evals[2]];
			assert_eq!((y2 - y0) * (x1 - x0), (y1 - y0) * (x2 - x0));
		}
	}

	proptest! {
		#[test]
		fn the_two_evaluations_agree_with_the_definition(
			n_vars in 0..=MAX_VARS,
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let buffer = random_field_buffer::<P>(&mut rng, n_vars);
			let point = random_scalars::<F>(&mut rng, n_vars);

			// Pairing with the full expansion is the definition, and the cheapest reference.
			let reference = buffer.par_inner_product(&Hypercube::One.expand(&point).build::<P>());

			prop_assert_eq!(buffer.evaluate(&point), reference);

			// The in-place form consumes the coefficients, so it goes last.
			prop_assert_eq!(buffer.evaluate_inplace(&point), reference);
		}

		#[test]
		fn the_two_inner_products_agree_with_the_scalar_reference(
			n_vars in 0..=MAX_VARS,
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let a = random_field_buffer::<P>(&mut rng, n_vars);
			let b = random_field_buffer::<P>(&mut rng, n_vars);

			// Both forms defer the field reduction, so a scalar sum is the reference.
			let reference = (0..a.len()).map(|i| a.get(i) * b.get(i)).sum::<F>();

			prop_assert_eq!(a.inner_product(&b), reference);
			prop_assert_eq!(a.par_inner_product(&b), reference);

			// A view passed straight through must reach the same computation.
			prop_assert_eq!(a.inner_product(b.as_view()), reference);
		}

		#[test]
		fn the_two_folds_agree(
			n_vars in 1..=MAX_VARS,
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let original = random_field_buffer::<P>(&mut rng, n_vars);
			let scalar = random_scalars::<F>(&mut rng, 1)[0];

			// Out of place leaves the input alone and returns a fresh half-size buffer.
			let out_of_place = original.fold_highest_var_in(&GlobalAllocator, scalar);

			let mut in_place = original;
			in_place.fold_highest_var(scalar);

			prop_assert_eq!(out_of_place.n_vars(), n_vars - 1);
			prop_assert_eq!(out_of_place, in_place);
		}

		#[test]
		fn the_binary_fold_matches_folding_the_widened_bits(
			dest_vars in 0..=6usize,
			tensor_vars in 0..=4usize,
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, tensor_vars);
			let tensor = Hypercube::One.expand(&point).build::<P>();

			// The bit count is the product of the two lengths, as the precondition demands.
			let bits = repeat_with(|| rng.random())
				.take(1 << (dest_vars + tensor_vars))
				.collect::<Vec<bool>>();

			let mut folded = FieldBuffer::<P>::zeros(dest_vars);
			folded.binary_fold_high(&tensor, &bits.as_slice());

			// Reference: widen the bits to field elements and fold the tensor's variables off.
			let scalars = bits
				.iter()
				.map(|&bit| if bit { F::ONE } else { F::ZERO })
				.collect::<Vec<F>>();
			let mut reference = FieldBuffer::<P>::from_values(&scalars);
			for &coord in point.iter().rev() {
				reference.fold_highest_var(coord);
			}

			prop_assert_eq!(folded, reference);
		}
	}
}
