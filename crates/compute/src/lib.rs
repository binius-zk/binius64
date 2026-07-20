// Copyright 2026 The Binius Developers

//! Buffer pooling for prover working memory.
//!
//! The prover allocates many large, short-lived buffers. [`BufferPool`] is the seam through which
//! those allocations flow, so they can later be served from a recycling free list instead of the
//! global allocator. Buffers are handed out as [`PoolVec`] handles that borrow the pool for
//! `'alloc` and, in a future revision, return their block to it on drop.
//!
//! The current implementation is a thin pass-through over [`Vec`]: [`BufferPool::alloc_vec`] just
//! calls [`Vec::with_capacity`]. This establishes the API and the `'alloc` lifetime threading
//! without committing to a pool layout yet.

use std::{
	fmt,
	mem::MaybeUninit,
	ops::{Deref, DerefMut},
};

use bytemuck::Pod;

/// A pool that hands out reusable buffers for prover working memory.
///
/// Allocation goes through [`alloc_vec`](Self::alloc_vec), which currently forwards to
/// [`Vec::with_capacity`]. A pool is created once, above the code that uses it, and shared by
/// borrow — every [`PoolVec`] it produces holds a `&'alloc BufferPool`.
#[derive(Default)]
pub struct BufferPool {
	// No free list yet; buffers are allocated straight from the global allocator. A recycling
	// implementation slots in behind `alloc_vec`/`PoolVec::drop` without changing their
	// signatures.
	_private: (),
}

impl BufferPool {
	/// Creates a new pool.
	pub fn new() -> Self {
		Self::default()
	}

	/// Allocates a [`PoolVec`] with room for at least `capacity` elements.
	///
	/// The returned buffer is empty; fill it through the [`PoolVec`] interface.
	pub fn alloc_vec<T: Pod>(&self, capacity: usize) -> PoolVec<'_, T> {
		PoolVec {
			pool: self,
			data: Vec::with_capacity(capacity),
		}
	}
}

impl fmt::Debug for BufferPool {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("BufferPool").finish_non_exhaustive()
	}
}

/// A `Vec`-like buffer borrowed from a [`BufferPool`] for `'alloc`.
///
/// Dereferences to `[T]`, so all slice operations are available directly. Only the growth and
/// mutation methods actually used by callers are exposed; add more as needed rather than mirroring
/// all of [`Vec`].
pub struct PoolVec<'alloc, T: Pod> {
	// Held so a future recycling pool can return the block on drop. Unused by the pass-through.
	#[allow(dead_code)]
	pool: &'alloc BufferPool,
	data: Vec<T>,
}

impl<T: Pod> PoolVec<'_, T> {
	/// Returns the number of elements the buffer can hold without reallocating.
	pub const fn capacity(&self) -> usize {
		self.data.capacity()
	}

	/// Appends an element to the back of the buffer.
	pub fn push(&mut self, value: T) {
		self.data.push(value);
	}

	/// Appends all elements of `other` to the back of the buffer.
	pub fn extend_from_slice(&mut self, other: &[T]) {
		self.data.extend_from_slice(other);
	}

	/// Clears the buffer, removing all elements while retaining its capacity.
	pub fn clear(&mut self) {
		self.data.clear();
	}

	/// Resizes the buffer to `new_len`, filling any new slots with `value`.
	pub fn resize(&mut self, new_len: usize, value: T) {
		self.data.resize(new_len, value);
	}

	/// Returns the spare capacity of the buffer as a slice of `MaybeUninit<T>`.
	///
	/// Mirrors [`Vec::spare_capacity_mut`]: used to write into a freshly allocated buffer in place
	/// (e.g. in parallel) before committing the length with [`set_len`](Self::set_len).
	pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
		self.data.spare_capacity_mut()
	}

	/// Forces the length of the buffer to `new_len`.
	///
	/// # Safety
	///
	/// Same contract as [`Vec::set_len`]: `new_len` must be at most [`capacity`](Self::capacity)
	/// and the elements in `0..new_len` must be initialized.
	pub unsafe fn set_len(&mut self, new_len: usize) {
		unsafe { self.data.set_len(new_len) }
	}
}

impl<T: Pod> Deref for PoolVec<'_, T> {
	type Target = [T];

	fn deref(&self) -> &[T] {
		&self.data
	}
}

impl<T: Pod> DerefMut for PoolVec<'_, T> {
	fn deref_mut(&mut self) -> &mut [T] {
		&mut self.data
	}
}

impl<T: Pod> Extend<T> for PoolVec<'_, T> {
	fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
		self.data.extend(iter);
	}
}

impl<T: Pod + fmt::Debug> fmt::Debug for PoolVec<'_, T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_list().entries(self.data.iter()).finish()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn alloc_vec_reserves_capacity_and_starts_empty() {
		let pool = BufferPool::new();
		let buffer = pool.alloc_vec::<u64>(16);
		assert!(buffer.is_empty());
		assert!(buffer.capacity() >= 16);
	}

	#[test]
	fn push_extend_and_deref() {
		let pool = BufferPool::new();
		let mut buffer = pool.alloc_vec::<u64>(4);
		buffer.push(1);
		buffer.extend_from_slice(&[2, 3]);
		buffer.extend([4, 5]);
		assert_eq!(&*buffer, &[1, 2, 3, 4, 5]);

		buffer[0] = 10;
		assert_eq!(buffer[0], 10);

		buffer.resize(3, 0);
		assert_eq!(&*buffer, &[10, 2, 3]);

		buffer.clear();
		assert!(buffer.is_empty());
	}
}
