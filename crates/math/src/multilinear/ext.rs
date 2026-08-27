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

use binius_compute::{Allocator, BufferData};
use binius_field::PackedField;
use binius_utils::random_access_sequence::RandomAccessSequence;

use crate::{
	FieldBuffer, FieldSlice, FieldVec,
	inner_product::{inner_product_buffers, inner_product_par},
	multilinear::{
		evaluate::{evaluate, evaluate_inplace},
		fold::{binary_fold_high, fold_highest_var, fold_highest_var_inplace},
		hypercube::{Hypercube, eq_ind_truncate_low_inplace},
	},
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
		evaluate(self, point)
	}

	#[inline]
	fn inner_product<'b>(&self, other: impl Into<FieldSlice<'b, P>>) -> P::Scalar {
		inner_product_buffers(self, &other.into())
	}

	#[inline]
	fn par_inner_product<'b>(&self, other: impl Into<FieldSlice<'b, P>>) -> P::Scalar {
		inner_product_par(self, &other.into())
	}

	fn fold_highest_var_in<A: Allocator>(&self, alloc: &A, scalar: P::Scalar) -> FieldVec<P, A> {
		fold_highest_var(alloc, self, scalar)
	}
}

impl<P: PackedField, Data: BufferData<P>> MultilinearMut<P> for FieldBuffer<P, Data> {
	fn fold_highest_var(&mut self, scalar: P::Scalar) {
		fold_highest_var_inplace(self, scalar);
	}

	fn eq_ind_truncate_low(&mut self, cube: Hypercube, truncated_n_vars: usize) {
		eq_ind_truncate_low_inplace(cube, self, truncated_n_vars);
	}

	fn binary_fold_high<'b>(
		&mut self,
		tensor: impl Into<FieldSlice<'b, P>>,
		bits: &(impl RandomAccessSequence<bool> + Sync),
	) {
		binary_fold_high(self, &tensor.into(), bits);
	}

	fn evaluate_inplace(self, coords: &[P::Scalar]) -> P::Scalar {
		evaluate_inplace(self, coords)
	}
}

#[cfg(test)]
mod tests {
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b, random_field_buffer, random_scalars};

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
}
