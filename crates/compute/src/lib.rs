// Copyright 2026 The Binius Developers

//! Buffer pooling for prover working memory.
//!
//! The prover allocates many large, short-lived buffers. [`BufferPool`] recycles freed blocks
//! instead of returning them to the global allocator, handing out [`PoolVec`] buffers that return
//! their block to the pool on drop. See the [`buffer_pool`] module for the concrete implementation.
//!
//! [`Allocator`] and [`VecLike`] abstract over that machinery: an [`Allocator`] hands out
//! [`VecLike`] buffers, letting the prover's allocation code be written against `&impl Allocator`
//! rather than a concrete pool. `&BufferPool` is the primary [`Allocator`], producing [`PoolVec`]
//! buffers.

use std::{
	mem::MaybeUninit,
	ops::{Deref, DerefMut},
};

pub mod buffer_pool;

pub use buffer_pool::{BufferPool, PoolVec};

/// A source of [`VecLike`] buffers.
///
/// Abstracts the allocation seam so callers can be generic over how their working buffers are
/// backed. The primary implementation is `&BufferPool`, whose [`Vec`](Allocator::Vec) is
/// [`PoolVec`] — a buffer drawn from a recycling pool.
pub trait Allocator {
	/// The buffer type this allocator hands out for element type `T`.
	type Vec<T>: VecLike<T>;

	/// Allocates an empty buffer with room for at least `capacity` elements of type `T`.
	fn alloc<T>(&self, capacity: usize) -> Self::Vec<T>;
}

/// A growable, `Vec`-like buffer.
///
/// Abstracts the buffer surface the prover relies on — a subset of [`Vec`]'s API, plus dereference
/// to `[T]`. Implemented by [`PoolVec`]; add methods here (and to the implementors) as callers need
/// them rather than mirroring all of [`Vec`].
pub trait VecLike<T>: Deref<Target = [T]> + DerefMut + Extend<T> {
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

impl<T> VecLike<T> for PoolVec<'_, T> {
	fn capacity(&self) -> usize {
		PoolVec::capacity(self)
	}

	fn push(&mut self, value: T) {
		PoolVec::push(self, value);
	}

	fn clear(&mut self) {
		PoolVec::clear(self);
	}

	fn resize(&mut self, new_len: usize, value: T)
	where
		T: Clone,
	{
		PoolVec::resize(self, new_len, value);
	}

	fn extend_from_slice(&mut self, other: &[T])
	where
		T: Clone,
	{
		PoolVec::extend_from_slice(self, other);
	}

	fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
		PoolVec::spare_capacity_mut(self)
	}

	unsafe fn set_len(&mut self, new_len: usize) {
		unsafe { PoolVec::set_len(self, new_len) }
	}
}

impl<'alloc> Allocator for &'alloc BufferPool {
	type Vec<T> = PoolVec<'alloc, T>;

	fn alloc<T>(&self, capacity: usize) -> Self::Vec<T> {
		// Copy the `&'alloc BufferPool` out of `&self` so the returned `PoolVec` borrows the pool
		// for `'alloc`, not merely for this call's `&self` borrow.
		let pool: &'alloc BufferPool = self;
		pool.alloc_vec(capacity)
	}
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

/// An [`Allocator`] that hands out ordinary heap-allocated [`Vec`]s.
///
/// The non-pooling counterpart to `&BufferPool`: every [`alloc`](Allocator::alloc) is a plain
/// [`Vec::with_capacity`], and each buffer is freed to the global allocator on drop.
#[derive(Debug, Default, Clone, Copy)]
pub struct GlobalAllocator;

impl Allocator for GlobalAllocator {
	type Vec<T> = Vec<T>;

	fn alloc<T>(&self, capacity: usize) -> Self::Vec<T> {
		Vec::with_capacity(capacity)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Fills a buffer through the [`VecLike`] surface, exercising an allocator generically.
	fn build<A: Allocator>(alloc: &A) -> A::Vec<u64> {
		let mut buffer = alloc.alloc::<u64>(4);
		assert!(buffer.capacity() >= 4);
		buffer.push(1);
		buffer.extend_from_slice(&[2, 3]);
		buffer.resize(5, 0);
		buffer
	}

	#[test]
	fn global_allocator_backs_a_plain_vec() {
		let buffer = build(&GlobalAllocator);
		assert_eq!(&*buffer, &[1, 2, 3, 0, 0]);
	}

	#[test]
	fn buffer_pool_backs_a_pool_vec() {
		let pool = BufferPool::new();
		let buffer = build(&&pool);
		assert_eq!(&*buffer, &[1, 2, 3, 0, 0]);
	}
}
