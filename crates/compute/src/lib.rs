// Copyright 2026 The Binius Developers

//! Buffer pooling for prover working memory.
//!
//! The prover allocates many large, short-lived buffers. [`BufferPool`] recycles freed blocks
//! instead of returning them to the global allocator, handing out [`PoolVec`] buffers that return
//! their block to the pool on drop. See the [`buffer_pool`] module for the concrete implementation.
//!
//! [`Allocator`] abstracts over that machinery: an allocator hands out [`VecLike`] buffers, letting
//! the prover's allocation code be written against `&impl Allocator` rather than a concrete pool.
//! `&BufferPool` is the primary [`Allocator`], producing [`PoolVec`] buffers.

use std::mem::MaybeUninit;

use binius_utils::buffer::{BufferData, VecLike};

pub mod buffer_pool;

pub use buffer_pool::{BufferPool, PoolVec};

/// A source of [`VecLike`] buffers.
///
/// Abstracts the allocation seam so callers can be generic over how their working buffers are
/// backed. The primary implementation is `&BufferPool`, whose [`Vec`](Allocator::Vec) is
/// [`PoolVec`] — a buffer drawn from a recycling pool.
///
/// [`Sync`] is required because the prover shares `&impl Allocator` across rayon tasks (e.g. the
/// parallel fractional-addition GKR reduction); both `&BufferPool` and `GlobalAllocator` are
/// `Sync`.
///
/// [`Copy`] is required because a caller often hands the same allocator to several things at once
/// — a channel and the Merkle prover inside it, say. An allocator handle is a pool reference or a
/// unit struct, so both implementors are already `Copy` and the bound costs them nothing.
pub trait Allocator: Sync + Copy {
	/// The buffer type this allocator hands out for element type `T`.
	///
	/// It is a [`VecLike`] buffer, and [`VecLike`] implies [`BufferData`].
	/// It grows and shrinks in place, so it can back a `binius_math::FieldBuffer` directly.
	///
	/// It is also [`Send`] so the prover can move pooled buffers across rayon tasks (e.g. the
	/// parallel fractional-addition GKR reduction); every element type the prover pools is itself
	/// `Send`.
	type Vec<T: Send>: VecLike<T> + Send;

	/// Allocates an empty buffer with room for at least `capacity` elements of type `T`.
	fn alloc<T: Send>(&self, capacity: usize) -> Self::Vec<T>;
}

impl<T> BufferData<T> for PoolVec<'_, T> {
	fn truncate(&mut self, len: usize) {
		PoolVec::truncate(self, len);
	}
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
	type Vec<T: Send> = PoolVec<'alloc, T>;

	fn alloc<T: Send>(&self, capacity: usize) -> Self::Vec<T> {
		// Copy the `&'alloc BufferPool` out of `&self` so the returned `PoolVec` borrows the pool
		// for `'alloc`, not merely for this call's `&self` borrow.
		let pool: &'alloc BufferPool = self;
		pool.alloc_vec(capacity)
	}
}

/// An [`Allocator`] that hands out ordinary heap-allocated [`Vec`]s.
///
/// The non-pooling counterpart to `&BufferPool`: every [`alloc`](Allocator::alloc) is a plain
/// [`Vec::with_capacity`], and each buffer is freed to the global allocator on drop.
#[derive(Debug, Default, Clone, Copy)]
pub struct GlobalAllocator;

impl Allocator for GlobalAllocator {
	type Vec<T: Send> = Vec<T>;

	fn alloc<T: Send>(&self, capacity: usize) -> Self::Vec<T> {
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
