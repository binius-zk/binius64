// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The output buffer both bit-axis folds fill.

use std::{marker::PhantomData, mem::MaybeUninit};

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_math::FieldBuffer;
use binius_utils::{buffer::VecLike, checked_arithmetics::log2_ceil_usize};

/// A field buffer under construction, filled in two stages.
///
/// Stage one writes one packed element per whole chunk of words, in parallel, through spare
/// capacity. Stage two pushes the short tail and zero-fills up to the power-of-two length.
///
/// Owning both stages is what keeps the length claim in one place, so the unchecked write is
/// argued once rather than at every fold.
pub struct PackedOutput<P, V> {
	/// The elements written so far, from the allocator the buffer is drawn from.
	values: V,
	/// Packed elements the finished buffer holds, which is its power-of-two length.
	capacity: usize,
	/// Base-2 log of the word count the buffer covers, rounded up.
	log_n: usize,
	/// Ties the builder to the packing its slots are typed by.
	_marker: PhantomData<P>,
}

impl<P: PackedField, V: VecLike<P>> PackedOutput<P, V> {
	/// Claims room for a buffer covering `n_words` words, rounded up to a power of two.
	///
	/// A word count that is not a power of two leaves the high words reading as zero.
	pub fn for_words<A: Allocator<Vec<P> = V>>(alloc: &A, n_words: usize) -> Self {
		let log_n = log2_ceil_usize(n_words);
		let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);

		Self {
			values: alloc.alloc::<P>(capacity),
			capacity,
			log_n,
			_marker: PhantomData,
		}
	}

	/// The uninitialized slots for the whole chunks, to be written before closing them.
	pub fn chunk_slots(&mut self, n_chunks: usize) -> &mut [MaybeUninit<P>] {
		&mut self.values.spare_capacity_mut()[..n_chunks]
	}

	/// Closes the slots handed out by the matching call above.
	///
	/// # Safety
	///
	/// Every slot of the matching `n_chunks` must have been written.
	pub unsafe fn commit_chunks(&mut self, n_chunks: usize) {
		unsafe { self.values.set_len(n_chunks) }
	}

	/// Appends the element the short tail folds to.
	pub fn push(&mut self, elem: P) {
		self.values.push(elem);
	}

	/// Zero-fills up to the power-of-two length and closes the buffer.
	pub fn finish(mut self) -> FieldBuffer<P, V> {
		self.values.resize(self.capacity, P::default());
		FieldBuffer::new(self.log_n, self.values)
	}
}
