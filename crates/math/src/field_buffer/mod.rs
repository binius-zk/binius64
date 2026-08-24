// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! A power-of-two-sized buffer of packed field elements.
//!
//! The buffer type and all of its methods live in this file.
//! Two sibling modules hold the other types this one hands out:
//!
//! ```text
//! view        the borrowed aliases, and the store that backs a shared one
//! write_back  the guards that lend out a region narrower than one packed word
//! ```
//!
//! # Why the backing store is a type parameter
//!
//! A store's length fixes the element count only once the buffer fills a whole packed word.
//! Below one word, the same single word backs any length:
//!
//! ```text
//! WIDTH = 4 lanes per word
//!
//! 1 element   ->  [ x . . . ]   one word
//! 2 elements  ->  [ x x . . ]   one word
//! 4 elements  ->  [ x x x x ]   one word
//! ```
//!
//! So the logical length is irreducible extra state, carried alongside the store.
//! That rules out a transparent wrapper around a packed slice.
//! With it goes the deref-coercion design a vector and its two slice types enjoy.

use std::{
	ops::{Deref, DerefMut},
	slice,
};

use binius_compute::Allocator;
use binius_field::{
	Field, PackedField,
	packed::{get_packed_slice_unchecked, set_packed_slice_unchecked},
};
use binius_utils::{
	buffer::{BufferData, VecLike},
	checked_arithmetics::strict_log_2,
	rayon::{iter::Either, prelude::*, slice::ParallelSlice},
};
use bytemuck::zeroed_vec;

mod chunks;
mod scalars;
mod view;
mod write_back;

use chunks::SubWordChunk;
pub use chunks::{Chunks, ChunksMut};
pub use scalars::Scalars;
pub use view::{FieldSlice, FieldSliceData, FieldSliceMut, FieldVec};
pub use write_back::FieldBufferSplitMut;

/// A power-of-two-sized buffer containing field elements, stored in packed fields.
///
/// The backing store length is fully determined by `log_len`:
///
/// ```text
/// values.len() == 1 << log_len.saturating_sub(P::LOG_WIDTH)
/// ```
///
/// A buffer shorter than one packed word still occupies a whole word.
/// The lanes at and past the logical length are then not elements of the buffer:
///
/// ```text
/// WIDTH = 4, log_len = 1
///
/// word:  [ s_0, s_1, dead, dead ]
///          ^^^^^^^^  live prefix
/// ```
///
/// Nothing reads a dead lane as an element.
/// Equality and the halving routines touch the live prefix only.
///
/// Truncation additionally zeros the dead lanes it creates.
/// Other constructors may leave whatever the caller's store held there.
#[derive(Debug, Clone, Eq)]
pub struct FieldBuffer<P: PackedField, Data: Deref<Target = [P]> = Vec<P>> {
	/// log2 the number over elements in the buffer.
	log_len: usize,
	/// The packed values.
	values: Data,
}

impl<P: PackedField, Data: Deref<Target = [P]> + Copy> Copy for FieldBuffer<P, Data> {}

impl<P: PackedField, Data: Deref<Target = [P]>> PartialEq for FieldBuffer<P, Data> {
	fn eq(&self, other: &Self) -> bool {
		// Equality compares only the live scalars, never the raw packed backing store.
		//
		// Invariant: buffers of different lengths are never equal.
		//
		// The length-below-width branch compares only the live prefix of a single packed word,
		// never the dead lanes past it: shorter and longer buffers sharing that prefix would
		// otherwise compare equal, and the dead lanes may hold unrelated data.
		if self.log_len != other.log_len {
			return false;
		}
		if self.log_len < P::LOG_WIDTH {
			let iter_1 = self
				.values
				.first()
				.expect("len >= 1")
				.iter()
				.take(1 << self.log_len);
			let iter_2 = other
				.values
				.first()
				.expect("len >= 1")
				.iter()
				.take(1 << self.log_len);
			iter_1.eq(iter_2)
		} else {
			let prefix = 1 << (self.log_len - P::LOG_WIDTH);
			self.values[..prefix] == other.values[..prefix]
		}
	}
}

impl<P: PackedField> FieldBuffer<P> {
	/// Create a new FieldBuffer from a vector of values.
	///
	/// # Preconditions
	///
	/// * `values.len()` must be a power of two.
	#[track_caller]
	pub fn from_values(values: &[P::Scalar]) -> Self {
		let log_len =
			strict_log_2(values.len()).expect("precondition: values.len() must be a power of two");

		let packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		let mut packed_values = Vec::with_capacity(packed_len);
		packed_values.extend(
			values
				.chunks(P::WIDTH)
				.map(|chunk| P::from_scalars(chunk.iter().copied())),
		);

		Self {
			log_len,
			values: packed_values,
		}
	}

	/// Builds a buffer of `2^log_len` zeros.
	pub fn zeros(log_len: usize) -> Self {
		let packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		let values = zeroed_vec(packed_len);
		Self { log_len, values }
	}

	/// Builds a one-element buffer holding `value`, with room reserved to grow.
	///
	/// The store is sized for `2^log_capacity` elements up front.
	/// Growing the buffer to that many therefore never reallocates.
	pub fn scalar_with_capacity(value: P::Scalar, log_capacity: usize) -> Self {
		let mut values = Vec::with_capacity(1 << log_capacity.saturating_sub(P::LOG_WIDTH));
		values.push(P::from_scalars([value]));
		Self { log_len: 0, values }
	}
}

impl<P: PackedField, Data: VecLike<P>> FieldBuffer<P, Data> {
	/// Builds a zeroed buffer of `2^log_len` elements, backed by memory drawn from `alloc`.
	///
	/// The allocator-aware counterpart to the plain zeroed constructor.
	/// Under a pool the result is a recyclable buffer rather than a fresh allocation.
	pub fn zeros_in<A>(alloc: &A, log_len: usize) -> Self
	where
		A: Allocator<Vec<P> = Data>,
	{
		let packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		// An allocator hands out an empty buffer, so the words are both created and zeroed here.
		let mut values = alloc.alloc::<P>(packed_len);
		values.resize(packed_len, P::default());
		FieldBuffer::new(log_len, values)
	}

	/// Copies a borrowed buffer into memory drawn from `alloc`.
	///
	/// Whole packed words are copied, dead lanes and all, so the copy is bit-identical.
	pub fn from_view_in<A>(alloc: &A, src: FieldSlice<P>) -> Self
	where
		A: Allocator<Vec<P> = Data>,
	{
		let mut values = alloc.alloc::<P>(src.as_ref().len());
		values.extend_from_slice(src.as_ref());
		FieldBuffer::new(src.log_len(), values)
	}

	/// Builds a buffer from scalar values, directly into memory drawn from `alloc`.
	///
	/// The allocator-aware counterpart to building from a scalar slice.
	/// Packing happens straight into the allocator's buffer, so no intermediate vector is copied.
	/// Under a pool the result is a recyclable buffer.
	///
	/// # Preconditions
	///
	/// * `values.len()` must be a power of two.
	#[track_caller]
	pub fn from_values_in<A>(alloc: &A, values: &[P::Scalar]) -> Self
	where
		A: Allocator<Vec<P> = Data>,
	{
		let log_len =
			strict_log_2(values.len()).expect("precondition: values.len() must be a power of two");

		let packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		let mut packed_values = alloc.alloc::<P>(packed_len);
		packed_values.extend(
			values
				.chunks(P::WIDTH)
				.map(|chunk| P::from_scalars(chunk.iter().copied())),
		);

		FieldBuffer::new(log_len, packed_values)
	}

	/// Grows the buffer to span `log_len` variables, padding the new positions with zeros.
	///
	/// Returns the buffer untouched when it already spans that many.
	/// Padding the shorter of two buffers to match the longer hits that case when both are equal.
	///
	/// New memory comes from `alloc`.
	/// A pooled buffer therefore stays pooled, instead of escaping the pool on a reallocation.
	///
	/// # Panics
	///
	/// Panics if `log_len` is shorter than what the buffer already spans.
	/// Shrinking is a separate operation.
	/// Silently returning a narrower buffer than asked for would hide the mistake.
	#[track_caller]
	pub fn zero_extend_in<A>(self, alloc: &A, log_len: usize) -> Self
	where
		A: Allocator<Vec<P> = Data>,
	{
		assert!(
			log_len >= self.log_len,
			"precondition: log_len must be at least the buffer's own log_len"
		);
		// Nothing to pad, so the caller's buffer is handed straight back with no copy.
		if log_len == self.log_len {
			return self;
		}

		// The target starts fully zeroed, so only the occupied words need writing.
		let mut extended = Self::zeros_in(alloc, log_len);

		// Whole packed words move across untouched.
		//
		// Invariant: lanes past the logical length are zero.
		// So a trailing partial word already carries zeros in its high lanes.
		// Those are exactly the zeros the padding would otherwise write.
		extended.as_mut()[..self.as_ref().len()].copy_from_slice(self.as_ref());

		extended
	}
}

#[allow(clippy::len_without_is_empty)]
impl<P: PackedField, Data: Deref<Target = [P]>> FieldBuffer<P, Data> {
	/// Create a new FieldBuffer from a slice of packed values.
	///
	/// # Preconditions
	///
	/// * `values.len()` must equal the expected packed length for `log_len`.
	#[track_caller]
	pub fn new(log_len: usize, values: Data) -> Self {
		let expected_packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		assert!(
			values.len() == expected_packed_len,
			"precondition: values.len() must equal expected packed length"
		);

		Self { log_len, values }
	}

	/// Consumes the buffer and returns its backing data store.
	pub fn into_inner(self) -> Data {
		self.values
	}

	/// Returns log2 the number of field elements.
	pub const fn log_len(&self) -> usize {
		self.log_len
	}

	/// Returns the number of field elements.
	pub const fn len(&self) -> usize {
		1 << self.log_len
	}

	/// Borrows the whole buffer as a shared view.
	pub fn as_view(&self) -> FieldSlice<'_, P> {
		FieldSlice::from_slice(self.log_len, self.as_ref())
	}

	/// Get a field element at the given index.
	///
	/// # Preconditions
	///
	/// * the index is in the range `0..self.len()`
	#[track_caller]
	pub fn get(&self, index: usize) -> P::Scalar {
		assert!(
			index < self.len(),
			"precondition: index {index} must be less than len {}",
			self.len()
		);

		// Safety: bound check on index performed above. The buffer length is at least
		// `self.len() >> P::LOG_WIDTH` by struct invariant.
		unsafe { get_packed_slice_unchecked(&self.values, index) }
	}

	/// Returns an iterator over the scalar elements in the buffer.
	pub fn iter_scalars(&self) -> impl Iterator<Item = P::Scalar> + Send + Clone + '_ {
		P::iter_slice(self.as_ref()).take(self.len())
	}

	/// Returns an iterator over the packed words the elements occupy.
	///
	/// The run covers the live words only, never a store's spare room past them.
	/// A buffer shorter than one packed word yields that one word whole, dead lanes and all.
	#[inline]
	pub fn iter_packed(&self) -> slice::Iter<'_, P> {
		self.as_ref().iter()
	}

	/// Get an aligned chunk of size `2^log_chunk_size`.
	///
	/// A chunk's start offset is a multiple of its size.
	/// So this yields the same chunk that stepping the shared iterator to that index would.
	///
	/// # Preconditions
	///
	/// * `log_chunk_size` must be at most `log_len`.
	/// * `chunk_index` must be less than the chunk count.
	#[inline]
	#[track_caller]
	pub fn chunk(&self, log_chunk_size: usize, chunk_index: usize) -> FieldSlice<'_, P> {
		assert!(
			log_chunk_size <= self.log_len,
			"precondition: log_chunk_size must be at most log_len"
		);

		let chunk_count = 1 << (self.log_len - log_chunk_size);
		assert!(
			chunk_index < chunk_count,
			"precondition: chunk_index must be less than chunk_count"
		);

		let values = if log_chunk_size >= P::LOG_WIDTH {
			// A whole run of words, borrowed straight from the store.
			let words_per_chunk = 1 << (log_chunk_size - P::LOG_WIDTH);
			FieldSliceData::Slice(&self.values[chunk_index * words_per_chunk..][..words_per_chunk])
		} else {
			// Lanes inside one word, copied out so the chunk starts at lane 0.
			FieldSliceData::Single(
				SubWordChunk::new(log_chunk_size, chunk_index).repack(&self.values),
			)
		};

		FieldBuffer {
			log_len: log_chunk_size,
			values,
		}
	}

	/// Split the buffer into chunks of size `2^log_chunk_size`.
	///
	/// Any size up to the buffer's length works, matching what the parallel iterator accepts.
	/// A chunk of at least one packed word borrows a run of the store.
	/// A smaller one arrives as a copy of its lanes, repacked to start at lane 0.
	///
	/// # Preconditions
	///
	/// * `log_chunk_size` must be at most `log_len`.
	#[track_caller]
	pub fn chunks(&self, log_chunk_size: usize) -> Chunks<'_, P> {
		assert!(
			log_chunk_size <= self.log_len,
			"precondition: log_chunk_size must be at most log_len"
		);

		let chunk_count = 1 << (self.log_len - log_chunk_size);
		Chunks::new(self.as_ref(), log_chunk_size, chunk_count)
	}

	/// Creates an iterator over chunks of size `2^log_chunk_size` in parallel.
	///
	/// # Preconditions
	///
	/// * `log_chunk_size` must be at most `log_len`.
	#[track_caller]
	pub fn par_chunks(
		&self,
		log_chunk_size: usize,
	) -> impl IndexedParallelIterator<Item = FieldSlice<'_, P>> {
		assert!(
			log_chunk_size <= self.log_len,
			"precondition: log_chunk_size must be at most log_len"
		);

		if log_chunk_size >= P::LOG_WIDTH {
			// Each chunk spans one or more packed elements
			let packed_chunk_size = 1 << (log_chunk_size - P::LOG_WIDTH);
			Either::Left(
				self.as_ref()
					.par_chunks(packed_chunk_size)
					.map(move |chunk| FieldBuffer {
						log_len: log_chunk_size,
						values: FieldSliceData::Slice(chunk),
					}),
			)
		} else {
			// Multiple chunks fit within a single packed element
			let chunk_count = 1 << (self.log_len - log_chunk_size);
			let words = self.as_ref();
			Either::Right(
				(0..chunk_count)
					.into_par_iter()
					.map(move |chunk_index| FieldBuffer {
						log_len: log_chunk_size,
						values: FieldSliceData::Single(
							SubWordChunk::new(log_chunk_size, chunk_index).repack(words),
						),
					}),
			)
		}
	}

	/// Creates a parallel iterator over the scalars of each chunk of `2^log_chunk_size` elements.
	///
	/// The scalar-yielding counterpart to the parallel chunk iterator:
	///
	/// ```text
	/// chunk i  ->  scalars [i * 2^log_chunk_size, (i+1) * 2^log_chunk_size)
	/// ```
	///
	/// A chunk takes one of two shapes, chosen once before any scalar is read:
	///
	/// ```text
	/// chunk >= one packed word  ->  a run of whole words
	/// chunk <  one packed word  ->  a lane range inside a single word
	/// ```
	///
	/// The buffer-yielding iterator instead repacks a sub-word chunk into an owned word.
	/// Prefer this method when the consumer only reads scalars, since it copies nothing.
	///
	/// # Preconditions
	///
	/// * `log_chunk_size` must be at most `log_len`.
	#[track_caller]
	pub fn par_chunk_scalars(
		&self,
		log_chunk_size: usize,
	) -> impl IndexedParallelIterator<Item: Iterator<Item = P::Scalar> + Send + Clone + '_> {
		assert!(
			log_chunk_size <= self.log_len,
			"precondition: log_chunk_size must be at most log_len"
		);

		let words = self.as_ref();
		if log_chunk_size >= P::LOG_WIDTH {
			// A chunk is a run of whole words:
			//
			//     store = 2^(log_len - LOG_WIDTH) words
			//     chunk = 2^(log_chunk_size - LOG_WIDTH) words
			//
			// Both counts are powers of two, so the runs tile the store with none left over.
			let words_per_chunk = 1 << (log_chunk_size - P::LOG_WIDTH);
			Either::Left(
				words
					.par_chunks(words_per_chunk)
					.map(|chunk| Either::Left(P::iter_slice(chunk))),
			)
		} else {
			// Several chunks share one word, so the count comes from the logical length:
			//
			//     log_len = 1, LOG_WIDTH = 2, log_chunk_size = 0
			//     word = [s_0, s_1, dead, dead]  ->  chunks [s_0], [s_1]
			//
			// A buffer narrower than one word never turns its dead lanes into a chunk.
			let chunk_count = 1 << (self.log_len - log_chunk_size);

			Either::Right((0..chunk_count).into_par_iter().map(move |chunk_index| {
				let chunk = SubWordChunk::new(log_chunk_size, chunk_index);
				Either::Right(chunk.scalars(words[chunk.word_index()]))
			}))
		}
	}

	/// Splits the buffer in half and returns a pair of borrowed slices.
	///
	/// # Preconditions
	///
	/// * `self.log_len()` must be greater than 0.
	#[track_caller]
	pub fn split_half_ref(&self) -> (FieldSlice<'_, P>, FieldSlice<'_, P>) {
		assert!(self.log_len > 0, "precondition: cannot split a buffer of length 1");

		let new_log_len = self.log_len - 1;
		if new_log_len < P::LOG_WIDTH {
			// The result will be two Single variants
			// We have exactly one packed element that needs to be split
			let packed = self.values[0];
			let zeros = P::default();

			let (first_half, second_half) = packed.interleave(zeros, new_log_len);

			let first = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Single(first_half),
			};
			let second = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Single(second_half),
			};

			(first, second)
		} else {
			// Split the packed values slice in half
			let half_len = 1 << (new_log_len - P::LOG_WIDTH);
			let (first_half, second_half) = self.values.split_at(half_len);
			let second_half = &second_half[..half_len];

			let first = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Slice(first_half),
			};
			let second = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Slice(second_half),
			};

			(first, second)
		}
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> FieldBuffer<P, Data> {
	/// Borrows the whole buffer as a mutable view.
	pub fn as_mut_view(&mut self) -> FieldSliceMut<'_, P> {
		FieldSliceMut::from_slice(self.log_len, self.as_mut())
	}

	/// Set a field element at the given index.
	///
	/// # Preconditions
	///
	/// * the index is in the range `0..self.len()`
	#[track_caller]
	pub fn set(&mut self, index: usize, value: P::Scalar) {
		assert!(
			index < self.len(),
			"precondition: index {index} must be less than len {}",
			self.len()
		);

		// Safety: bound check on index performed above. The buffer length is at least
		// `self.len() >> P::LOG_WIDTH` by struct invariant.
		unsafe { set_packed_slice_unchecked(&mut self.values, index, value) };
	}

	/// Returns a mutable iterator over the packed words the elements occupy.
	///
	/// The mutable counterpart of the shared word iterator, covering the same run of words.
	/// A final word below the packing width is lent out whole, dead lanes included.
	/// Padding the buffer out to a wider length turns those lanes live.
	/// Zeros there are what make that padding zero.
	#[inline]
	pub fn iter_packed_mut(&mut self) -> slice::IterMut<'_, P> {
		self.as_mut().iter_mut()
	}

	/// Split the buffer into mutable chunks of size `2^log_chunk_size`.
	///
	/// A chunk must span whole words, unlike the shared iterator, which takes any size.
	/// Chunks below the packing width share a word, so lending them out means lending copies.
	/// Each copy live at once would need its own write-back into that word.
	///
	/// ```text
	/// WIDTH = 4, log_chunk_size = 1  ->  word 0 = [chunk 0 | chunk 1]
	/// ```
	///
	/// Reach for the single-chunk mutable accessor at such a size, which guards one copy.
	///
	/// # Preconditions
	///
	/// * `log_chunk_size` must be at least `P::LOG_WIDTH` and at most `log_len`.
	#[track_caller]
	pub fn chunks_mut(&mut self, log_chunk_size: usize) -> ChunksMut<'_, P> {
		assert!(
			log_chunk_size >= P::LOG_WIDTH && log_chunk_size <= self.log_len,
			"precondition: log_chunk_size must be in range [P::LOG_WIDTH, log_len]"
		);

		let chunk_count = 1 << (self.log_len - log_chunk_size);
		ChunksMut::new(self.as_mut(), log_chunk_size, chunk_count)
	}

	/// Consumes the buffer and halves it, returning a guard that owns the store.
	///
	/// The guard lends out the two halves mutably.
	///
	/// When both halves live inside one packed word, the guard holds detached copies of them.
	/// Those are merged back into the original word when it drops.
	///
	/// # Preconditions
	///
	/// * `self.log_len()` must be greater than 0.
	#[track_caller]
	pub fn split_half(self) -> FieldBufferSplitMut<P, Data> {
		assert!(self.log_len > 0, "precondition: cannot split a buffer of length 1");

		let new_log_len = self.log_len - 1;
		let singles = if new_log_len < P::LOG_WIDTH {
			let packed = self.values[0];
			let zeros = P::default();
			let (lo_half, hi_half) = packed.interleave(zeros, new_log_len);
			Some([lo_half, hi_half])
		} else {
			None
		};

		FieldBufferSplitMut {
			log_len: new_log_len,
			singles,
			data: self.values,
		}
	}

	/// Halves the buffer through a mutable borrow, rather than consuming it.
	///
	/// Equivalent to borrowing the buffer as a mutable view and halving that.
	///
	/// # Preconditions
	///
	/// * `self.log_len()` must be greater than 0.
	#[track_caller]
	pub fn split_half_mut(&mut self) -> FieldBufferSplitMut<P, &'_ mut [P]> {
		self.as_mut_view().split_half()
	}
}

impl<P: PackedField, Data: BufferData<P>> FieldBuffer<P, Data> {
	/// Truncates the buffer to a shorter length, shrinking the backing store to match.
	///
	/// Asking for a length at or above the current one does nothing.
	///
	/// A result shorter than one packed word has the dead lanes of its final word zeroed.
	/// That upholds the invariant that lanes past the logical length are zero.
	pub fn truncate(&mut self, new_log_len: usize) {
		if new_log_len >= self.log_len {
			return;
		}
		self.log_len = new_log_len;

		// Zero the lanes past the new logical length in the final word, so a sub-packing-width
		// result carries no stale scalars. Wider results drop whole words and need no masking.
		if new_log_len < P::LOG_WIDTH {
			for i in 1 << new_log_len..P::WIDTH {
				self.values[0].set(i, <P::Scalar as Field>::ZERO);
			}
		}

		self.values
			.truncate(1 << new_log_len.saturating_sub(P::LOG_WIDTH));
	}
}

impl<P: PackedField, Data: Deref<Target = [P]>> AsRef<[P]> for FieldBuffer<P, Data> {
	#[inline]
	fn as_ref(&self) -> &[P] {
		&self.values[..1 << self.log_len.saturating_sub(P::LOG_WIDTH)]
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> AsMut<[P]> for FieldBuffer<P, Data> {
	#[inline]
	fn as_mut(&mut self) -> &mut [P] {
		&mut self.values[..1 << self.log_len.saturating_sub(P::LOG_WIDTH)]
	}
}

impl<P: PackedField> FromIterator<P::Scalar> for FieldBuffer<P> {
	/// Builds a buffer over a fresh vector, packing the elements as they arrive.
	///
	/// # Panics
	///
	/// Panics unless the number of elements is a power of two.
	/// An empty iterator panics too, matching what building from an empty scalar slice does.
	/// The count is known only once the iterator runs dry, so a bad one can only panic.
	#[track_caller]
	fn from_iter<I: IntoIterator<Item = P::Scalar>>(iter: I) -> Self {
		let iter = iter.into_iter();
		// The lower bound is the whole count whenever the iterator knows its own length.
		let mut values = Vec::with_capacity(iter.size_hint().0 >> P::LOG_WIDTH);

		let mut word = P::default();
		let mut len = 0usize;
		for scalar in iter {
			let lane = len % P::WIDTH;
			word.set(lane, scalar);
			len += 1;

			// A word fills up at its last lane, and a zeroed one takes over from there.
			if lane == P::WIDTH - 1 {
				values.push(word);
				word = P::default();
			}
		}

		let log_len =
			strict_log_2(len).expect("precondition: element count must be a power of two");

		// A count below the packing width never reaches a last lane.
		// So the one word it filled is still in hand, its lanes past the count still zero.
		if log_len < P::LOG_WIDTH {
			values.push(word);
		}

		Self { log_len, values }
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::{BufferPool, GlobalAllocator};
	use binius_field::packed::get_packed_slice;
	use proptest::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b};

	type P = Packed128b;
	type F = B128;

	/// Runs every allocator-backed constructor and the zero-padding grow against one allocator.
	///
	/// Both a plain heap allocator and a recycling pool must satisfy the same contract.
	fn check_alloc_constructors<A: Allocator>(alloc: &A) {
		// Fixture state: a packed word holds 4 lanes, so lengths map to backing words as
		//
		//     16 scalars -> log_len 4 -> 4 packed words, every lane live
		//      2 scalars -> log_len 1 -> 1 packed word,  2 lanes live and 2 dead
		let scalars: Vec<F> = (0..16).map(F::new).collect();
		let src = FieldBuffer::<P>::from_values(&scalars);

		// A copy drawn from the allocator carries over both the length and the scalars.
		let cloned: FieldVec<P, A> = FieldBuffer::from_view_in(alloc, src.as_view());
		assert_eq!(cloned.log_len(), 4);
		assert_eq!(cloned.as_view(), src.as_view());

		// Copying a source shorter than one packed word keeps the two live lanes, not four.
		let small = FieldBuffer::<P>::from_values(&scalars[..2]);
		let cloned_small: FieldVec<P, A> = FieldBuffer::from_view_in(alloc, small.as_view());
		assert_eq!(cloned_small.log_len(), 1);
		assert_eq!(cloned_small.as_view(), small.as_view());

		// 32 elements requested, 32 zeros readable: the buffer arrives at full length, not empty.
		let mut zeros: FieldVec<P, A> = FieldBuffer::zeros_in(alloc, 5);
		assert_eq!(zeros.log_len(), 5);
		assert!(zeros.iter_scalars().all(|scalar| scalar == F::ZERO));

		// Index 31 is the last live element, so writing it stays inside the allocated words.
		zeros.set(31, F::new(7));
		assert_eq!(zeros.get(31), F::new(7));

		// Two elements still need the one word that spans them, rounded up from half a word.
		let zeros_small: FieldVec<P, A> = FieldBuffer::zeros_in(alloc, 1);
		assert_eq!(zeros_small.as_ref().len(), 1);
		assert!(zeros_small.iter_scalars().all(|scalar| scalar == F::ZERO));

		// Padding to a wider length keeps the live scalars and zeros every position above them.
		//
		//     source (log_len 4)    [0 .. 16]
		//     result (log_len 5)    [0 .. 16] unchanged, [16 .. 32] zero
		//
		// Under a pool the target may sit on memory a prior buffer dirtied.
		// So the zeros above the copied words have to be written, never assumed.
		let src_vec: FieldVec<P, A> = FieldBuffer::from_view_in(alloc, src.as_view());
		let extended = src_vec.zero_extend_in(alloc, 5);
		assert_eq!(extended.log_len(), 5);
		for i in 0..16 {
			assert_eq!(extended.get(i), F::new(i as u128));
		}
		assert!((16..32).all(|i| extended.get(i) == F::ZERO));

		// An already-wide-enough buffer comes back unchanged, with no copy taken.
		let same: FieldVec<P, A> = FieldBuffer::from_view_in(alloc, src.as_view());
		let same = same.zero_extend_in(alloc, 4);
		assert_eq!(same.log_len(), 4);
		assert_eq!(same.as_view(), src.as_view());

		// A source narrower than one packed word pads out of that word's dead lanes.
		//
		//     source (log_len 1)    [0, 1 | dead, dead]
		//     result (log_len 3)    [0, 1 | 0, 0] [0, 0 | 0, 0]
		//
		// The two dead lanes become live, so they read back as zeros, not stale scalars.
		let small_vec: FieldVec<P, A> = FieldBuffer::from_view_in(alloc, small.as_view());
		let widened = small_vec.zero_extend_in(alloc, 3);
		assert_eq!(widened.log_len(), 3);
		assert_eq!(widened.get(0), F::new(0));
		assert_eq!(widened.get(1), F::new(1));
		assert!((2..8).all(|i| widened.get(i) == F::ZERO));
	}

	#[test]
	fn zeros() {
		// Make a buffer with `zeros()` and check that all elements are zero.
		// Test with log_len >= LOG_WIDTH
		let buffer = FieldBuffer::<P>::zeros(6); // 64 elements
		assert_eq!(buffer.log_len(), 6);
		assert_eq!(buffer.len(), 64);

		// Check all elements are zero
		for i in 0..64 {
			assert_eq!(buffer.get(i), F::ZERO);
		}

		// Test with log_len < LOG_WIDTH
		let buffer = FieldBuffer::<P>::zeros(1); // 2 elements
		assert_eq!(buffer.log_len(), 1);
		assert_eq!(buffer.len(), 2);

		// Check all elements are zero
		for i in 0..2 {
			assert_eq!(buffer.get(i), F::ZERO);
		}
	}

	#[test]
	fn alloc_constructors_global() {
		// Every allocation is an independent heap buffer, freed on drop.
		check_alloc_constructors(&GlobalAllocator);
	}

	#[test]
	fn alloc_constructors_pooled() {
		// A pool reuses freed blocks, so a buffer may start life on memory a prior buffer dirtied.
		let pool = BufferPool::new();
		check_alloc_constructors(&&pool);
	}

	#[test]
	fn from_values_below_packing_width() {
		// Make a buffer using `from_values()`, where the number of scalars is below the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		let values = vec![F::new(1), F::new(2)]; // 2 elements < 4
		let buffer = FieldBuffer::<P>::from_values(&values);

		assert_eq!(buffer.log_len(), 1); // log2(2) = 1
		assert_eq!(buffer.len(), 2);

		// Verify the values
		assert_eq!(buffer.get(0), F::new(1));
		assert_eq!(buffer.get(1), F::new(2));
	}

	#[test]
	fn from_values_above_packing_width() {
		// Make a buffer using `from_values()`, where the number of scalars is above the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		let values: Vec<F> = (0..16).map(F::new).collect(); // 16 elements > 4
		let buffer = FieldBuffer::<P>::from_values(&values);

		assert_eq!(buffer.log_len(), 4); // log2(16) = 4
		assert_eq!(buffer.len(), 16);

		// Verify all values
		for i in 0..16 {
			assert_eq!(buffer.get(i), F::new(i as u128));
		}
	}

	#[test]
	#[should_panic(expected = "power of two")]
	fn from_values_non_power_of_two() {
		let values: Vec<F> = (0..7).map(F::new).collect(); // 7 is not a power of two
		let _ = FieldBuffer::<P>::from_values(&values);
	}

	#[test]
	#[should_panic(expected = "power of two")]
	fn from_values_empty() {
		let values: Vec<F> = vec![];
		let _ = FieldBuffer::<P>::from_values(&values);
	}

	#[test]
	fn new_below_packing_width() {
		// Make a buffer using `new()`, where the number of scalars is below the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		// For log_len = 1 (2 elements), we need 1 packed value
		let mut packed_values = vec![P::default()];
		let mut buffer = FieldBuffer::new(1, packed_values.as_mut_slice());

		assert_eq!(buffer.log_len(), 1);
		assert_eq!(buffer.len(), 2);

		// Set and verify values
		buffer.set(0, F::new(10));
		buffer.set(1, F::new(20));
		assert_eq!(buffer.get(0), F::new(10));
		assert_eq!(buffer.get(1), F::new(20));
	}

	#[test]
	fn new_above_packing_width() {
		// Make a buffer using `new()`, where the number of scalars is above the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		// For log_len = 4 (16 elements), we need 4 packed values
		let mut packed_values = vec![P::default(); 4];
		let mut buffer = FieldBuffer::new(4, packed_values.as_mut_slice());

		assert_eq!(buffer.log_len(), 4);
		assert_eq!(buffer.len(), 16);

		// Set and verify values
		for i in 0..16 {
			buffer.set(i, F::new(i as u128 * 10));
		}
		for i in 0..16 {
			assert_eq!(buffer.get(i), F::new(i as u128 * 10));
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn new_wrong_packed_length() {
		let packed_values = vec![P::default(); 3]; // Wrong: should be 4 for log_len=4
		let _ = FieldBuffer::new(4, packed_values.as_slice());
	}

	#[test]
	fn get_set() {
		let mut buffer = FieldBuffer::<P>::zeros(3); // 8 elements

		// Set some values
		for i in 0..8 {
			buffer.set(i, F::new(i as u128));
		}

		// Get them back
		for i in 0..8 {
			assert_eq!(buffer.get(i), F::new(i as u128));
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn get_out_of_bounds() {
		let buffer = FieldBuffer::<P>::zeros(3); // 8 elements
		let _ = buffer.get(8);
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn set_out_of_bounds() {
		let mut buffer = FieldBuffer::<P>::zeros(3); // 8 elements
		buffer.set(8, F::new(0));
	}

	#[test]
	fn borrowed_views() {
		let mut buffer = FieldBuffer::<P>::zeros(3);

		// Test the shared view
		let slice_ref = buffer.as_view();
		assert_eq!(slice_ref.len(), buffer.len());
		assert_eq!(slice_ref.log_len(), buffer.log_len());
		assert_eq!(slice_ref.as_ref().len(), 1 << slice_ref.log_len().saturating_sub(P::LOG_WIDTH));

		// Test the mutable view
		let mut slice_mut = buffer.as_mut_view();
		slice_mut.set(0, F::new(123));
		assert_eq!(slice_mut.as_mut().len(), 1 << slice_mut.log_len().saturating_sub(P::LOG_WIDTH));
		assert_eq!(buffer.get(0), F::new(123));
	}

	#[test]
	fn iter_scalars() {
		// Test with buffer size below packing width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		let values = vec![F::new(10), F::new(20)]; // 2 elements < 4
		let buffer = FieldBuffer::<P>::from_values(&values);

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Verify it matches individual get calls
		for (i, &val) in collected.iter().enumerate() {
			assert_eq!(val, buffer.get(i));
		}

		// Test with buffer size equal to packing width
		let values = vec![F::new(1), F::new(2), F::new(3), F::new(4)]; // 4 elements = P::WIDTH
		let buffer = FieldBuffer::<P>::from_values(&values);

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Test with buffer size above packing width
		let values: Vec<F> = (0..16).map(F::new).collect(); // 16 elements > 4
		let buffer = FieldBuffer::<P>::from_values(&values);

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Verify it matches individual get calls
		for (i, &val) in collected.iter().enumerate() {
			assert_eq!(val, buffer.get(i));
		}

		// Test with single element buffer
		let values = vec![F::new(42)];
		let buffer = FieldBuffer::<P>::from_values(&values);

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Test with large buffer
		let values: Vec<F> = (0..256).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Test that iterator is cloneable and can be used multiple times
		let values: Vec<F> = (0..8).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		let iter1 = buffer.iter_scalars();
		let iter2 = iter1.clone();

		let collected1: Vec<F> = iter1.collect();
		let collected2: Vec<F> = iter2.collect();
		assert_eq!(collected1, collected2);
		assert_eq!(collected1, values);
	}

	#[test]
	fn from_iter_below_packing_width() {
		// One element in a 4-lane word: the buffer spans 1 element, not the word's 4.
		let buffer: FieldBuffer<P> = std::iter::once(F::new(9)).collect();
		assert_eq!(buffer.log_len(), 0);
		assert_eq!(buffer.len(), 1);
		assert_eq!(buffer.get(0), F::new(9));

		// The one word packed keeps zeros in the 3 lanes past the element.
		let data = buffer.into_inner();
		assert_eq!(data.len(), 1);
		assert!((1..P::WIDTH).all(|lane| get_packed_slice(&data[..], lane) == F::ZERO));
	}

	#[test]
	#[should_panic(expected = "power of two")]
	fn from_iter_non_power_of_two() {
		// 7 elements: the count is only known once the iterator runs dry, and then it panics.
		let _: FieldBuffer<P> = (0..7).map(F::new).collect();
	}

	#[test]
	#[should_panic(expected = "power of two")]
	fn from_iter_empty() {
		// Zero elements is not a power of two, so this panics rather than making a 0-length buffer.
		let _: FieldBuffer<P> = std::iter::empty::<F>().collect();
	}

	#[test]
	fn iter_packed_covers_the_live_words() {
		// Room for 32 elements but 1 live: iteration follows the live count, not the room.
		let buffer = FieldBuffer::<P>::scalar_with_capacity(F::new(5), 5);
		let live: Vec<P> = buffer.iter_packed().copied().collect();
		assert_eq!(live.len(), 1);
		assert_eq!(get_packed_slice(&live[..], 0), F::new(5));

		// 16 elements over 4-lane words: 4 words, holding the elements in order.
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);
		let words: Vec<P> = buffer.iter_packed().copied().collect();
		assert_eq!(words.len(), 4);
		assert_eq!(P::iter_slice(&words).collect::<Vec<_>>(), values);

		// A buffer truncated to 2 elements iterates the 1 word they sit in, not the original 4.
		let mut truncated = FieldBuffer::<P>::from_values(&values);
		truncated.truncate(1);
		assert_eq!(truncated.iter_packed().count(), 1);
	}

	#[test]
	fn iter_packed_mut_writes_the_live_words() {
		// 8 elements over 2 words: writing both words writes every element.
		let mut buffer = FieldBuffer::<P>::zeros(3);
		assert_eq!(buffer.iter_packed_mut().count(), 2);
		for word in buffer.iter_packed_mut() {
			*word = P::broadcast(F::new(3));
		}
		assert!(buffer.iter_scalars().all(|scalar| scalar == F::new(3)));

		// Sub-packing-width: 2 elements share one word, which is lent out whole.
		let mut small = FieldBuffer::<P>::zeros(1);
		assert_eq!(small.iter_packed_mut().count(), 1);
		for word in small.iter_packed_mut() {
			*word = P::broadcast(F::new(7));
		}
		// Only the 2 live lanes are elements, however many lanes the write touched.
		assert_eq!(small.iter_scalars().collect::<Vec<_>>(), vec![F::new(7); 2]);
	}

	#[test]
	fn truncate_vec_backing() {
		// P::LOG_WIDTH = 2, P::WIDTH = 4.
		let make = || FieldBuffer::<P>::from_values(&(0..16).map(F::new).collect::<Vec<_>>());

		// Above-width result: the backing Vec shrinks to the new packed length.
		let mut buffer = make();
		buffer.truncate(3); // 8 elements -> 2 packed words
		assert_eq!(buffer.log_len(), 3);
		assert_eq!(buffer.len(), 8);
		for i in 0..8 {
			assert_eq!(buffer.get(i), F::new(i as u128));
		}
		assert_eq!(buffer.into_inner().len(), 2);

		// Sub-packing-width result: one word retained, live prefix kept, dead lanes zeroed.
		let mut buffer = make();
		buffer.truncate(1); // 2 elements
		assert_eq!(buffer.len(), 2);
		assert_eq!(buffer.get(0), F::new(0));
		assert_eq!(buffer.get(1), F::new(1));
		let data = buffer.into_inner();
		assert_eq!(data.len(), 1);
		assert_eq!(get_packed_slice(&data[..], 2), F::new(0));
		assert_eq!(get_packed_slice(&data[..], 3), F::new(0));

		// No-op when the requested length is not smaller.
		let mut buffer = FieldBuffer::<P>::from_values(&(0..4).map(F::new).collect::<Vec<_>>());
		buffer.truncate(5);
		assert_eq!(buffer.log_len(), 2);
		assert_eq!(buffer.into_inner().len(), 1);
	}

	#[test]
	fn truncate_slice_backing() {
		// Truncating a `&mut [P]` backing reslices it and zeros the sub-width dead lanes.
		let mut storage = vec![P::default(); 4]; // 16 elements at log_len 4
		let mut buffer = FieldSliceMut::from_slice(4, storage.as_mut_slice());
		for i in 0..16 {
			buffer.set(i, F::new(i as u128));
		}

		buffer.truncate(1); // 2 elements, sub-width
		assert_eq!(buffer.len(), 2);
		assert_eq!(buffer.get(0), F::new(0));
		assert_eq!(buffer.get(1), F::new(1));

		let data = buffer.into_inner();
		assert_eq!(data.len(), 1);
		assert_eq!(get_packed_slice(&data[..], 2), F::new(0));
		assert_eq!(get_packed_slice(&data[..], 3), F::new(0));
	}

	#[test]
	fn chunk() {
		let log_len = 8;
		let values: Vec<F> = (0..1 << log_len).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		for log_chunk_size in 0..=log_len {
			let chunk_count = 1 << (log_len - log_chunk_size);

			for chunk_index in 0..chunk_count {
				let chunk = buffer.chunk(log_chunk_size, chunk_index);
				for i in 0..1 << log_chunk_size {
					assert_eq!(chunk.get(i), buffer.get(chunk_index << log_chunk_size | i));
				}
			}
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn chunk_invalid_size() {
		let log_len = 8;
		let values: Vec<F> = (0..1 << log_len).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);
		let _ = buffer.chunk(log_len + 1, 0);
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn chunk_invalid_index() {
		let log_len = 8;
		let values: Vec<F> = (0..1 << log_len).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);
		let _ = buffer.chunk(4, 1 << (log_len - 4)); // out of range
	}

	#[test]
	fn chunks() {
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		// Split into 4 chunks of size 4
		let chunks: Vec<_> = buffer.chunks(2).collect();
		assert_eq!(chunks.len(), 4);

		for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
			assert_eq!(chunk.len(), 4);
			for i in 0..4 {
				let expected = F::new((chunk_idx * 4 + i) as u128);
				assert_eq!(chunk.get(i), expected);
			}
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn chunks_invalid_size_too_large() {
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);
		let _ = buffer.chunks(5).collect::<Vec<_>>();
	}

	#[test]
	fn chunks_below_packing_width() {
		// A word holds 4 lanes, so pairs of elements share one:
		//
		//     word 0 = [s_0, s_1, s_2, s_3]  ->  chunks [s_0, s_1], [s_2, s_3]
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		let chunks: Vec<_> = buffer.chunks(1).collect();
		assert_eq!(chunks.len(), 8);

		// Each chunk owns a repacked word, so its two elements sit at lanes 0 and 1.
		for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
			assert_eq!(chunk.len(), 2);
			assert_eq!(chunk.get(0), F::new((chunk_idx * 2) as u128));
			assert_eq!(chunk.get(1), F::new((chunk_idx * 2 + 1) as u128));
		}

		// A single-element chunk is the narrowest shape, one lane copied out on its own.
		let singles: Vec<_> = buffer.chunks(0).collect();
		assert_eq!(singles.len(), 16);
		for (index, chunk) in singles.into_iter().enumerate() {
			assert_eq!(chunk.len(), 1);
			assert_eq!(chunk.get(0), F::new(index as u128));
		}
	}

	#[test]
	fn par_chunks() {
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		// Split into 4 chunks of size 4
		let chunks: Vec<_> = buffer.par_chunks(2).collect();
		assert_eq!(chunks.len(), 4);

		for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
			assert_eq!(chunk.len(), 4);
			for i in 0..4 {
				let expected = F::new((chunk_idx * 4 + i) as u128);
				assert_eq!(chunk.get(i), expected);
			}
		}

		// Test small chunk sizes (below P::LOG_WIDTH)
		// P::LOG_WIDTH = 2, so par_chunks(0) and par_chunks(1) should work
		// Split into 8 chunks of size 2 (log_chunk_size = 1)
		let chunks: Vec<_> = buffer.par_chunks(1).collect();
		assert_eq!(chunks.len(), 8);
		for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
			assert_eq!(chunk.len(), 2);
			for i in 0..2 {
				let expected = F::new((chunk_idx * 2 + i) as u128);
				assert_eq!(chunk.get(i), expected);
			}
		}

		// Split into 16 chunks of size 1 (log_chunk_size = 0)
		let chunks: Vec<_> = buffer.par_chunks(0).collect();
		assert_eq!(chunks.len(), 16);
		for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
			assert_eq!(chunk.len(), 1);
			let expected = F::new(chunk_idx as u128);
			assert_eq!(chunk.get(0), expected);
		}
	}

	#[test]
	fn par_chunk_scalars_ignores_dead_lanes() {
		// Fixture state: 2 scalars occupy one 4-lane word, leaving two lanes dead.
		//
		//     word = [s_0, s_1, dead, dead]
		let values: Vec<F> = (0..2).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		// One scalar per chunk: 2 live scalars give 2 chunks, not the word's 4 lanes.
		let chunks: Vec<Vec<F>> = buffer
			.par_chunk_scalars(0)
			.map(|chunk| chunk.collect())
			.collect();
		assert_eq!(chunks, vec![vec![values[0]], vec![values[1]]]);
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn par_chunks_invalid_size() {
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);
		let _ = buffer.par_chunks(5).collect::<Vec<_>>();
	}

	#[test]
	fn chunks_mut() {
		let mut buffer = FieldBuffer::<P>::zeros(4); // 16 elements

		// Modify via chunks
		let mut chunks: Vec<_> = buffer.chunks_mut(2).collect();
		assert_eq!(chunks.len(), 4);

		for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
			for i in 0..chunk.len() {
				chunk.set(i, F::new((chunk_idx * 10 + i) as u128));
			}
		}

		// Verify modifications
		for chunk_idx in 0..4 {
			for i in 0..4 {
				let expected = F::new((chunk_idx * 10 + i) as u128);
				assert_eq!(buffer.get(chunk_idx * 4 + i), expected);
			}
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn chunks_mut_invalid_size() {
		let mut buffer = FieldBuffer::<P>::zeros(4); // 16 elements
		let _ = buffer.chunks_mut(0).collect::<Vec<_>>();
	}

	#[test]
	fn split_half() {
		// Test with buffer size > P::WIDTH (multiple packed elements)
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		let (first, second) = buffer.split_half_ref();
		assert_eq!(first.len(), 8);
		assert_eq!(second.len(), 8);

		// Verify values
		for i in 0..8 {
			assert_eq!(first.get(i), F::new(i as u128));
			assert_eq!(second.get(i), F::new((i + 8) as u128));
		}

		// Test with buffer size = P::WIDTH (single packed element)
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		let values: Vec<F> = (0..4).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		let (first, second) = buffer.split_half_ref();
		assert_eq!(first.len(), 2);
		assert_eq!(second.len(), 2);

		// Verify we got Single variants
		match &first.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for first half"),
		}
		match &second.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for second half"),
		}

		// Verify values
		assert_eq!(first.get(0), F::new(0));
		assert_eq!(first.get(1), F::new(1));
		assert_eq!(second.get(0), F::new(2));
		assert_eq!(second.get(1), F::new(3));

		// Test with buffer size = 2 (less than P::WIDTH)
		let values: Vec<F> = vec![F::new(10), F::new(20)];
		let buffer = FieldBuffer::<P>::from_values(&values);

		let (first, second) = buffer.split_half_ref();
		assert_eq!(first.len(), 1);
		assert_eq!(second.len(), 1);

		// Verify we got Single variants
		match &first.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for first half"),
		}
		match &second.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for second half"),
		}

		assert_eq!(first.get(0), F::new(10));
		assert_eq!(second.get(0), F::new(20));
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn split_half_ref_size_one() {
		let values = vec![F::new(42)];
		let buffer = FieldBuffer::<P>::from_values(&values);
		let _ = buffer.split_half_ref();
	}

	#[test]
	fn split_half_mut_no_closure() {
		// Test with buffer size > P::WIDTH (multiple packed elements)
		let mut buffer = FieldBuffer::<P>::zeros(4); // 16 elements

		// Fill with test data
		for i in 0..16 {
			buffer.set(i, F::new(i as u128));
		}

		{
			let mut split = buffer.split_half_mut();
			let (mut first, mut second) = split.halves();

			assert_eq!(first.len(), 8);
			assert_eq!(second.len(), 8);

			// Modify through the split halves
			for i in 0..8 {
				first.set(i, F::new((i * 10) as u128));
				second.set(i, F::new((i * 20) as u128));
			}
			// split drops here and writes back the changes
		}

		// Verify changes were made to original buffer
		for i in 0..8 {
			assert_eq!(buffer.get(i), F::new((i * 10) as u128));
			assert_eq!(buffer.get(i + 8), F::new((i * 20) as u128));
		}

		// Test with buffer size = P::WIDTH (single packed element)
		// P::LOG_WIDTH = 2, so a buffer with log_len = 2 (4 elements) can now be split
		let mut buffer = FieldBuffer::<P>::zeros(2); // 4 elements

		// Fill with test data
		for i in 0..4 {
			buffer.set(i, F::new(i as u128));
		}

		{
			let mut split = buffer.split_half_mut();
			let (mut first, mut second) = split.halves();

			assert_eq!(first.len(), 2);
			assert_eq!(second.len(), 2);

			// Modify values
			first.set(0, F::new(100));
			first.set(1, F::new(101));
			second.set(0, F::new(200));
			second.set(1, F::new(201));
			// split drops here and writes back the changes using interleave
		}

		// Verify changes were written back
		assert_eq!(buffer.get(0), F::new(100));
		assert_eq!(buffer.get(1), F::new(101));
		assert_eq!(buffer.get(2), F::new(200));
		assert_eq!(buffer.get(3), F::new(201));

		// Test with buffer size = 2
		let mut buffer = FieldBuffer::<P>::zeros(1); // 2 elements

		buffer.set(0, F::new(10));
		buffer.set(1, F::new(20));

		{
			let mut split = buffer.split_half_mut();
			let (mut first, mut second) = split.halves();

			assert_eq!(first.len(), 1);
			assert_eq!(second.len(), 1);

			// Modify values
			first.set(0, F::new(30));
			second.set(0, F::new(40));
			// split drops here and writes back the changes using interleave
		}

		// Verify changes
		assert_eq!(buffer.get(0), F::new(30));
		assert_eq!(buffer.get(1), F::new(40));
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn split_half_mut_size_one() {
		let mut buffer = FieldBuffer::<P>::zeros(0); // 1 element
		let _ = buffer.split_half_mut();
	}

	proptest! {
		#[test]
		fn unequal_length_buffers_are_never_equal(
			log_len_a in 0usize..=6,
			log_len_b in 0usize..=6,
			fill in any::<u128>(),
		) {
			// Invariant: buffers are equal iff same length and same scalars.
			let value = F::new(fill);
			let buf_a = FieldBuffer::<P>::from_values(&vec![value; 1 << log_len_a]);
			let buf_b = FieldBuffer::<P>::from_values(&vec![value; 1 << log_len_b]);

			// - Same length -> equal;
			// - Different length -> unequal despite matching scalars.
			if log_len_a == log_len_b {
				prop_assert_eq!(buf_a, buf_b);
			} else {
				prop_assert_ne!(buf_a, buf_b);
			}
		}

		#[test]
		fn from_iter_matches_from_values(log_len in 0usize..=6) {
			// The sweep straddles the packing width: a sub-word store, then multi-word ones.
			let values: Vec<F> = (0..1u128 << log_len).map(F::new).collect();

			// Collecting the scalars must rebuild what packing the slice builds, length included.
			let collected: FieldBuffer<P> = values.iter().copied().collect();
			prop_assert_eq!(collected.log_len(), log_len);
			prop_assert_eq!(&collected, &FieldBuffer::<P>::from_values(&values));

			// Round trip: iterating the collected buffer hands the scalars back, in order.
			prop_assert_eq!(&collected.iter_scalars().collect::<Vec<_>>(), &values);
			prop_assert_eq!(&IntoIterator::into_iter(&collected).collect::<Vec<_>>(), &values);
		}

		#[test]
		fn par_chunk_scalars_partitions_the_buffer(
			(log_len, log_chunk_size) in (0usize..=6).prop_flat_map(|n| (Just(n), 0usize..=n)),
		) {
			// Invariant: the chunks tile the buffer exactly.
			// The sweep reaches chunk sizes on both sides of the packing width.
			let values: Vec<F> = (0..1u128 << log_len).map(F::new).collect();
			let buffer = FieldBuffer::<P>::from_values(&values);

			let chunks: Vec<Vec<F>> = buffer
				.par_chunk_scalars(log_chunk_size)
				.map(|chunk| chunk.collect())
				.collect();

			// The count follows the logical length, not the backing word count.
			prop_assert_eq!(chunks.len(), 1 << (log_len - log_chunk_size));

			// Cross-check: each chunk matches what the serial accessor returns at the same index.
			for (index, scalars) in chunks.iter().enumerate() {
				let expected: Vec<F> = buffer.chunk(log_chunk_size, index).iter_scalars().collect();
				prop_assert_eq!(scalars, &expected);
			}

			// Concatenating the chunks reproduces the buffer, in order and without gaps.
			prop_assert_eq!(chunks.concat(), values);
		}

		#[test]
		fn serial_and_parallel_chunks_agree(
			(log_len, log_chunk_size) in (0usize..=8).prop_flat_map(|n| (Just(n), 0usize..=n)),
		) {
			// Invariant: both chunk iterators yield the same chunks, scalar for scalar.
			// A packed word holds 4 lanes, so the sweep reaches sizes on both sides of it.
			let values: Vec<F> = (0..1u128 << log_len).map(F::new).collect();
			let buffer = FieldBuffer::<P>::from_values(&values);

			let serial: Vec<Vec<F>> = buffer
				.chunks(log_chunk_size)
				.map(|chunk| chunk.iter_scalars().collect())
				.collect();
			let parallel: Vec<Vec<F>> = buffer
				.par_chunks(log_chunk_size)
				.map(|chunk| chunk.iter_scalars().collect())
				.collect();

			// The count follows the logical length, not the backing word count.
			prop_assert_eq!(serial.len(), 1 << (log_len - log_chunk_size));
			prop_assert_eq!(&serial, &parallel);

			// Concatenating the chunks reproduces the buffer, in order and without gaps.
			prop_assert_eq!(serial.concat(), values);
		}
	}
}
