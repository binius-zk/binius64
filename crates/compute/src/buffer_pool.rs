// Copyright 2026 The Binius Developers

//! A pool that recycles the prover's large, short-lived working buffers.
//!
//! [`BufferPool`] is the seam through which those allocations flow, recycling freed blocks instead
//! of returning them to the global allocator. Buffers are handed out as [`PoolVec`] handles that
//! borrow the pool for `'alloc` and return their block to it on drop.
//!
//! Every pooled block is allocated with a fixed `BUFFER_ALIGN`-byte alignment — wide enough for any
//! element type the prover uses — and sized to a power-of-two number of bytes. Because the
//! alignment is uniform, the free list keys purely on byte size: a block freed by one element type
//! can back a [`PoolVec`] of *any* other type of the same size. Blocks are stored as owned
//! `Vec<AlignedChunk>`, so the pool frees them through ordinary `Vec` machinery; a [`PoolVec`]
//! borrows a block's memory as a `Vec<T>` for its lifetime and hands it back on drop.

use std::{
	collections::HashMap,
	fmt,
	mem::{self, MaybeUninit},
	ops::{Deref, DerefMut},
	sync::Mutex,
};

/// The alignment, in bytes, of every pooled block.
///
/// It must be at least the alignment of every element type passed to
/// [`alloc_vec`](BufferPool::alloc_vec) — 64 bytes covers the widest packed field the prover
/// allocates. Fixing the alignment is what lets the free list key on byte size alone, so a block is
/// reusable across element types.
const BUFFER_ALIGN: usize = 64;

/// A `BUFFER_ALIGN`-aligned unit of `BUFFER_ALIGN` bytes.
///
/// Pooled blocks are `Vec<AlignedChunk>`; the type exists only to force the backing allocation to
/// `BUFFER_ALIGN` alignment while remaining an owned `Vec` the pool can free directly. Its bytes
/// are never read through this type — a block is always borrowed as a `Vec<T>` while in use.
#[repr(align(64))]
struct AlignedChunk(#[allow(dead_code)] [u8; BUFFER_ALIGN]);

/// A pool that hands out reusable buffers for prover working memory.
///
/// Allocation goes through [`alloc_vec`](Self::alloc_vec). Freed buffers are kept on an internal
/// free list, keyed by block size (in `AlignedChunk`s), and reused to satisfy later allocations
/// of the same size. A pool is created once, above the code that uses it, and shared by borrow —
/// every [`PoolVec`] it produces holds a `&'alloc BufferPool`. The pool is thread-safe: the free
/// list sits behind a [`Mutex`], so allocation and reclamation may happen from any thread.
#[derive(Default)]
pub struct BufferPool {
	free_list: Mutex<HashMap<usize, Vec<Vec<AlignedChunk>>>>,
}

impl BufferPool {
	/// Creates a new, empty pool.
	pub fn new() -> Self {
		Self::default()
	}

	/// Allocates a [`PoolVec`] with room for at least `capacity` elements.
	///
	/// The block size is rounded up to a power-of-two number of bytes. If the free list holds a
	/// block of that size it is reused; otherwise a fresh block is allocated. The returned buffer
	/// is empty; fill it through the [`PoolVec`] interface.
	///
	/// # Panics
	///
	/// Panics at compile time (via a `const` assertion) if `T`'s alignment exceeds `BUFFER_ALIGN`,
	/// or if `T`'s size is not a power of two — either would break the byte-size keyed reuse.
	pub fn alloc_vec<T>(&self, capacity: usize) -> PoolVec<'_, T> {
		PoolVec {
			pool: self,
			data: self.alloc_data(capacity),
		}
	}

	fn alloc_data<T>(&self, capacity: usize) -> Vec<T> {
		const {
			assert!(
				mem::align_of::<T>() <= BUFFER_ALIGN,
				"element alignment exceeds the pool's buffer alignment"
			);
			assert!(
				mem::size_of::<T>().is_power_of_two() || mem::size_of::<T>() == 0,
				"element size must be a power of two for byte-size-keyed reuse"
			);
		}

		// A zero-sized type never allocates, and a zero-capacity request need not; in both cases
		// there is no block to pool, so hand back a plain `Vec`.
		if mem::size_of::<T>() == 0 || capacity == 0 {
			return Vec::with_capacity(capacity);
		}

		let byte_len = (capacity * mem::size_of::<T>())
			.next_power_of_two()
			.max(BUFFER_ALIGN);
		let n_chunks = byte_len / BUFFER_ALIGN;
		let elem_cap = byte_len / mem::size_of::<T>();

		let reused = self
			.free_list
			.lock()
			.expect("free list mutex poisoned")
			.get_mut(&n_chunks)
			.and_then(Vec::pop);

		let mut block = reused.unwrap_or_else(|| Vec::with_capacity(n_chunks));
		let ptr = block.as_mut_ptr().cast::<T>();
		// The block owns the allocation; forget it so its `Drop` does not free the memory we are
		// about to hand to the `Vec`.
		mem::forget(block);
		// SAFETY: `ptr` comes from a `Vec<AlignedChunk>` with capacity `n_chunks`, i.e. an
		// allocation of `byte_len` bytes aligned to `BUFFER_ALIGN >= align_of::<T>()`. `T`'s size
		// divides `byte_len` (both are powers of two and `size_of::<T>() <= byte_len`), so it
		// holds exactly `elem_cap` elements. The length is zero, so no element needs to be
		// initialized.
		unsafe { Vec::from_raw_parts(ptr, 0, elem_cap) }
	}

	fn reclaim(&self, n_chunks: usize, block: Vec<AlignedChunk>) {
		self.free_list
			.lock()
			.expect("free list mutex poisoned")
			.entry(n_chunks)
			.or_default()
			.push(block);
	}
}

impl fmt::Debug for BufferPool {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("BufferPool").finish_non_exhaustive()
	}
}

/// A `Vec`-like buffer borrowed from a [`BufferPool`] for `'alloc`.
///
/// Dereferences to `[T]`, so all slice operations are available directly. On drop the buffer's
/// block is returned to the pool for reuse. Only the growth and mutation methods actually used by
/// callers are exposed; add more as needed rather than mirroring all of [`Vec`].
pub struct PoolVec<'alloc, T> {
	pool: &'alloc BufferPool,
	data: Vec<T>,
}

impl<T> PoolVec<'_, T> {
	/// Returns the number of elements the buffer can hold without reallocating.
	pub const fn capacity(&self) -> usize {
		self.data.capacity()
	}

	/// Appends an element to the back of the buffer.
	pub fn push(&mut self, value: T) {
		self.data.push(value);
	}

	/// Clears the buffer, removing all elements while retaining its capacity.
	pub fn clear(&mut self) {
		self.data.clear();
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

impl<T: Clone> PoolVec<'_, T> {
	/// Appends all elements of `other` to the back of the buffer.
	pub fn extend_from_slice(&mut self, other: &[T]) {
		self.data.extend_from_slice(other);
	}

	/// Resizes the buffer to `new_len`, filling any new slots with `value`.
	pub fn resize(&mut self, new_len: usize, value: T) {
		self.data.resize(new_len, value);
	}
}

impl<T> Drop for PoolVec<'_, T> {
	fn drop(&mut self) {
		let mut data = mem::take(&mut self.data);
		let elem_cap = data.capacity();
		if mem::size_of::<T>() == 0 || elem_cap == 0 {
			// Nothing was allocated (zero-sized `T` or an empty buffer); let the `Vec` drop.
			return;
		}
		let byte_len = elem_cap * mem::size_of::<T>();
		// Only the blocks we hand out are `BUFFER_ALIGN`-aligned and a whole number of chunks. If
		// the `Vec` outgrew its block and reallocated into its own (element-aligned) storage, the
		// pointer or byte length no longer matches; let such a `Vec` drop normally.
		if !byte_len.is_multiple_of(BUFFER_ALIGN)
			|| !(data.as_ptr() as usize).is_multiple_of(BUFFER_ALIGN)
		{
			return;
		}
		// Run the elements' destructors while keeping the block's allocation, so a block returned
		// to the pool holds no live values.
		data.clear();
		let n_chunks = byte_len / BUFFER_ALIGN;
		let ptr = data.as_ptr() as *mut AlignedChunk;
		// Take the allocation away from the `Vec` so it is not freed, then rebuild the owning
		// block.
		mem::forget(data);
		// SAFETY: this memory was handed out from a `Vec<AlignedChunk>` of exactly `n_chunks`
		// chunks (checked above: `BUFFER_ALIGN`-aligned, `byte_len` a multiple of
		// `BUFFER_ALIGN`), so reconstructing that `Vec` restores the original owner and frees
		// with the correct layout.
		let block = unsafe { Vec::<AlignedChunk>::from_raw_parts(ptr, 0, n_chunks) };
		self.pool.reclaim(n_chunks, block);
	}
}

impl<T> Deref for PoolVec<'_, T> {
	type Target = [T];

	fn deref(&self) -> &[T] {
		&self.data
	}
}

impl<T> DerefMut for PoolVec<'_, T> {
	fn deref_mut(&mut self) -> &mut [T] {
		&mut self.data
	}
}

impl<T> Extend<T> for PoolVec<'_, T> {
	fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
		self.data.extend(iter);
	}
}

impl<T: fmt::Debug> fmt::Debug for PoolVec<'_, T> {
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

	#[test]
	fn buffers_are_aligned_and_sized_to_a_power_of_two_byte_length() {
		let pool = BufferPool::new();
		let buffer = pool.alloc_vec::<u64>(10);
		// 10 * 8 = 80 bytes rounds up to a 128-byte block: 16 `u64`s.
		assert_eq!(buffer.capacity(), 16);
		assert!((buffer.as_ptr() as usize).is_multiple_of(BUFFER_ALIGN));
		// Small requests still take a whole minimum-size block.
		assert_eq!(pool.alloc_vec::<u64>(1).capacity(), BUFFER_ALIGN / size_of::<u64>());
	}

	#[test]
	fn freed_block_is_recycled_for_a_matching_request() {
		let pool = BufferPool::new();

		let addr = {
			let buffer = pool.alloc_vec::<u64>(10);
			buffer.as_ptr() as usize
		};
		// The freed block backs the next request of the same size, and comes back empty despite
		// having been filled before.
		let mut buffer = pool.alloc_vec::<u64>(10);
		assert_eq!(buffer.as_ptr() as usize, addr);
		assert!(buffer.is_empty());
		buffer.extend_from_slice(&[1, 2, 3]);
		assert_eq!(&*buffer, &[1, 2, 3]);
	}

	#[test]
	fn a_block_freed_by_one_type_is_reused_by_another_of_the_same_byte_size() {
		let pool = BufferPool::new();
		// A `u64` buffer of 8 elements and a `u8` buffer of 64 elements are both one 64-byte block,
		// so the freed block is reused across the two element types.
		let addr = {
			let buffer = pool.alloc_vec::<u64>(8);
			buffer.as_ptr() as usize
		};
		let buffer = pool.alloc_vec::<u8>(64);
		assert_eq!(buffer.as_ptr() as usize, addr);
	}

	#[test]
	fn distinct_sizes_do_not_share_blocks() {
		let pool = BufferPool::new();
		let small = {
			let buffer = pool.alloc_vec::<u64>(4);
			buffer.as_ptr() as usize
		};
		// A larger request needs a bigger block and cannot reuse the small freed one.
		let big = pool.alloc_vec::<u64>(64);
		assert_ne!(big.as_ptr() as usize, small);
	}

	#[test]
	fn free_list_holds_multiple_blocks_of_the_same_size() {
		let pool = BufferPool::new();

		let (addr_a, addr_b) = {
			let a = pool.alloc_vec::<u64>(8);
			let b = pool.alloc_vec::<u64>(8);
			assert_ne!(a.as_ptr() as usize, b.as_ptr() as usize);
			(a.as_ptr() as usize, b.as_ptr() as usize)
		};

		// Both freed blocks are available; two fresh allocations reuse exactly them.
		let c = pool.alloc_vec::<u64>(8);
		let d = pool.alloc_vec::<u64>(8);
		let reused = [c.as_ptr() as usize, d.as_ptr() as usize];
		assert!(reused.contains(&addr_a));
		assert!(reused.contains(&addr_b));
	}
}
