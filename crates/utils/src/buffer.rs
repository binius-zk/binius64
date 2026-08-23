// Copyright 2026 The Binius Developers

//! Traits abstracting over the buffers that back Binius' working memory.
//!
//! [`BufferData`] is the shrinkable-in-place surface, and [`VecLike`] is that plus growth — the
//! subset of [`Vec`]'s API that callers rely on. Both are container vocabulary: they say what a
//! buffer can do, not where its memory came from, so a plain [`Vec`], a borrowed `&mut [T]`, or a
//! buffer drawn from a recycling pool can all satisfy them.

use std::{mem, mem::MaybeUninit, ops::DerefMut};

/// A mutable buffer of `T` that can be shrunk in place.
///
/// This is the backing store a `binius_math::FieldBuffer` needs in order to support
/// `FieldBuffer::truncate`, which shrinks the store to match a smaller `log_len`.
///
/// This trait is the shrinkable-store capability alone, and [`VecLike`] is that plus growth.
/// Three backings implement it:
///
/// - `Vec<T>` and `PoolVec` both shrink and grow, so both are [`VecLike`] as well.
/// - `&mut [T]` only shrinks, by re-slicing, which is what slice-backed sumcheck halves need.
pub trait BufferData<T>: DerefMut<Target = [T]> {
	/// Shrinks the store in place to its first `len` elements.
	///
	/// `len` must be at most the current length.
	fn truncate(&mut self, len: usize);
}

impl<T> BufferData<T> for Vec<T> {
	fn truncate(&mut self, len: usize) {
		Vec::truncate(self, len);
	}
}

impl<T> BufferData<T> for &mut [T] {
	fn truncate(&mut self, len: usize) {
		// A `&'a mut [T]` cannot be re-sliced in place through `&mut self`, so move it out and
		// slice the owned value back in.
		let full = mem::take(self);
		*self = &mut full[..len];
	}
}

/// A growable, `Vec`-like buffer.
///
/// Abstracts the buffer surface the prover uses: [`BufferData`] plus a subset of [`Vec`]'s API.
/// Implemented by `Vec<T>` and `PoolVec`, with methods added as callers need them.
/// It is not meant to mirror all of [`Vec`].
pub trait VecLike<T>: BufferData<T> + Extend<T> {
	/// Returns the number of elements the buffer can hold without reallocating.
	fn capacity(&self) -> usize;

	/// Appends an element to the back of the buffer.
	fn push(&mut self, value: T);

	/// Clears the buffer, removing all elements while retaining its capacity.
	fn clear(&mut self);

	/// Resizes the buffer to `new_len`, filling any new slots with `value`.
	fn resize(&mut self, new_len: usize, value: T)
	where
		T: Clone;

	/// Appends all elements of `other` to the back of the buffer.
	fn extend_from_slice(&mut self, other: &[T])
	where
		T: Clone;

	/// Returns the spare capacity of the buffer as a slice of `MaybeUninit<T>`.
	fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>];

	/// Forces the length of the buffer to `new_len`.
	///
	/// # Safety
	///
	/// Same contract as [`Vec::set_len`]: `new_len` must be at most [`capacity`](Self::capacity)
	/// and the elements in `0..new_len` must be initialized.
	unsafe fn set_len(&mut self, new_len: usize);
}

impl<T> VecLike<T> for Vec<T> {
	fn capacity(&self) -> usize {
		Vec::capacity(self)
	}

	fn push(&mut self, value: T) {
		Vec::push(self, value);
	}

	fn clear(&mut self) {
		Vec::clear(self);
	}

	fn resize(&mut self, new_len: usize, value: T)
	where
		T: Clone,
	{
		Vec::resize(self, new_len, value);
	}

	fn extend_from_slice(&mut self, other: &[T])
	where
		T: Clone,
	{
		Vec::extend_from_slice(self, other);
	}

	fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
		Vec::spare_capacity_mut(self)
	}

	unsafe fn set_len(&mut self, new_len: usize) {
		unsafe { Vec::set_len(self, new_len) }
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn vec_truncates_through_buffer_data() {
		let mut buffer = vec![1u64, 2, 3, 4];
		BufferData::truncate(&mut buffer, 2);
		assert_eq!(&*buffer, &[1, 2]);
	}

	#[test]
	fn slice_truncates_through_buffer_data() {
		let mut owned = [1u64, 2, 3, 4];
		let mut buffer: &mut [u64] = &mut owned;
		BufferData::truncate(&mut buffer, 3);
		assert_eq!(buffer, &[1, 2, 3]);
	}

	#[test]
	fn vec_fills_through_vec_like() {
		let mut buffer: Vec<u64> = Vec::with_capacity(4);
		buffer.push(1);
		buffer.extend_from_slice(&[2, 3]);
		VecLike::resize(&mut buffer, 5, 0);
		assert!(VecLike::capacity(&buffer) >= 5);
		assert_eq!(&*buffer, &[1, 2, 3, 0, 0]);
	}
}
