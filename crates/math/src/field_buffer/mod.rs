// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! A power-of-two-sized buffer of packed field elements.
//!
//! The buffer lives in this file, with its methods and the guard its halving methods hand out.
//! `SplitMut` is built in exactly one place, so it needs no visibility beyond this module.
//!
//! Two groups sit in modules of their own:
//!
//! ```text
//! chunks  the buffer's chunk methods, and the iterators and guard they hand out
//! view    the borrowed aliases, which name the buffer rather than being produced by it
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
	mem::MaybeUninit,
	ops::{Deref, DerefMut},
	slice,
};

use binius_compute::{Allocator, BufferData, VecLike};
use binius_field::{
	Field, PackedField,
	packed::{get_packed_slice_unchecked, set_packed_slice_unchecked},
};
use binius_utils::{
	checked_arithmetics::strict_log_2,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, task_chunk_len},
	},
};
use bytemuck::zeroed_vec;

mod chunks;
mod view;

pub use chunks::{ChunkMut, Chunks, ChunksMut};
pub use view::{FieldSlice, FieldSliceData, FieldSliceMut, FieldVec};

/// A power-of-two-sized buffer containing field elements, stored in packed fields.
///
/// The backing store length is fully determined by `log_len`:
///
/// ```text
/// words.len() == 1 << log_len.saturating_sub(P::LOG_WIDTH)
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
// The buffer's methods are grouped by topic, one module per group, so its inherent impls span
// more than one file.
#[allow(clippy::multiple_inherent_impl)]
#[derive(Debug, Clone, Eq)]
pub struct FieldBuffer<P: PackedField, Data: Deref<Target = [P]> = Vec<P>> {
	/// log2 the number over elements in the buffer.
	log_len: usize,
	/// The packed words.
	words: Data,
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
				.words
				.first()
				.expect("len >= 1")
				.iter()
				.take(1 << self.log_len);
			let iter_2 = other
				.words
				.first()
				.expect("len >= 1")
				.iter()
				.take(1 << self.log_len);
			iter_1.eq(iter_2)
		} else {
			let prefix = 1 << (self.log_len - P::LOG_WIDTH);
			self.words[..prefix] == other.words[..prefix]
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
		let mut words = Vec::with_capacity(packed_len);
		words.extend(
			values
				.chunks(P::WIDTH)
				.map(|chunk| P::from_scalars(chunk.iter().copied())),
		);

		Self { log_len, words }
	}

	/// Builds a buffer of `2^log_len` zeros.
	pub fn zeros(log_len: usize) -> Self {
		let packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		let words = zeroed_vec(packed_len);
		Self { log_len, words }
	}

	/// Builds a one-element buffer holding `value`, with room reserved to grow.
	///
	/// The store is sized for `2^log_capacity` elements up front.
	/// Growing the buffer to that many therefore never reallocates.
	pub fn scalar_with_capacity(value: P::Scalar, log_capacity: usize) -> Self {
		let mut words = Vec::with_capacity(1 << log_capacity.saturating_sub(P::LOG_WIDTH));
		words.push(P::from_scalars([value]));
		Self { log_len: 0, words }
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
		let mut words = alloc.alloc::<P>(packed_len);
		words.resize(packed_len, P::default());
		FieldBuffer::new(log_len, words)
	}

	/// Copies a borrowed buffer into memory drawn from `alloc`.
	///
	/// Whole packed words are copied, dead lanes and all, so the copy is bit-identical.
	/// A long source spreads across workers.
	pub fn from_view_in<A>(alloc: &A, src: FieldSlice<'_, P>) -> Self
	where
		A: Allocator<Vec<P> = Data>,
	{
		Self::from_view_with_capacity_in(alloc, src, src.log_len())
	}

	/// Copies a borrowed buffer into memory drawn from `alloc`, with room reserved to grow.
	///
	/// The buffer spans `src.log_len()` elements and its store is sized for `2^log_capacity`.
	/// Growing it to that many elements therefore never reallocates.
	/// [`Self::repeat_extend`] is the growth this reserves for.
	///
	/// Whole packed words are copied, dead lanes and all, so the copy is bit-identical.
	/// A long source spreads across workers.
	///
	/// # Panics
	///
	/// Panics if `log_capacity` is shorter than what `src` spans.
	#[track_caller]
	pub fn from_view_with_capacity_in<A>(
		alloc: &A,
		src: FieldSlice<'_, P>,
		log_capacity: usize,
	) -> Self
	where
		A: Allocator<Vec<P> = Data>,
	{
		assert!(
			log_capacity >= src.log_len(),
			"precondition: log_capacity must be at least src.log_len()"
		);

		let mut words = alloc.alloc::<P>(1 << log_capacity.saturating_sub(P::LOG_WIDTH));

		// The allocator rounds its blocks up, so the fill is bounded to the words that are live.
		let source = src.as_ref();
		let head = &mut words.spare_capacity_mut()[..source.len()];

		// The floor holds a worker's share at one task's byte budget, so a short source stays put.
		(head.par_iter_mut(), source.par_iter())
			.into_par_iter()
			.with_min_task_bytes::<P>()
			.for_each(|(dst, src)| {
				dst.write(*src);
			});

		// SAFETY: the loop above wrote every word of the source.
		unsafe { words.set_len(source.len()) };

		FieldBuffer::new(src.log_len(), words)
	}

	/// Grows the buffer to span `2^log_len` elements, repeating what it already holds.
	///
	/// ```text
	///     [x]  ->  [x | x | x | x]
	/// ```
	///
	/// This is the buffer's [`Vec::extend_from_within`], split across workers.
	/// Every copy is drawn from the live prefix, so one pass fills the whole buffer.
	///
	/// Returns the buffer untouched when it already spans that many elements.
	///
	/// # Panics
	///
	/// Panics if `log_len` is shorter than what the buffer already spans.
	/// Panics if the store has no room reserved for `2^log_len` elements.
	#[track_caller]
	pub fn repeat_extend(&mut self, log_len: usize) {
		assert!(
			log_len >= self.log_len,
			"precondition: log_len must be at least the buffer's own log_len"
		);
		if log_len == self.log_len {
			return;
		}

		let total = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		assert!(
			total <= self.words.capacity(),
			"precondition: the store must have room reserved for 2^log_len elements"
		);

		if self.log_len < P::LOG_WIDTH {
			// The live elements share one word, so repeating them cycles lanes, not words.
			let word = P::from_scalars(self.iter_scalars().cycle());
			self.words.clear();
			self.words.resize(total, word);
		} else {
			let prefix = self.words.len();

			// A run is the words one worker copies at a time.
			// It is a power of two at most the prefix, so it divides the prefix evenly.
			let run = prefix.min(task_chunk_len::<P>().next_power_of_two());

			// One borrow has to span both the prefix the copies read and the room they fill.
			// Emptying the buffer first is what puts the whole store behind that single borrow.
			//
			// SAFETY: a packed field has no destructor, so forgetting the live words is a no-op.
			unsafe { self.words.set_len(0) };
			let store = &mut self.words.spare_capacity_mut()[..total];
			let (head, tail) = store.split_at_mut(prefix);

			// SAFETY: the split point is the length the buffer just had.
			// So every word of `head` was written before this call.
			let head = unsafe { &*(head as *const [MaybeUninit<P>] as *const [P]) };

			repeat_words(head, tail, run);

			// SAFETY: the call above wrote every word between the prefix and `total`.
			unsafe { self.words.set_len(total) };
		}

		self.log_len = log_len;
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
		let mut words = alloc.alloc::<P>(packed_len);
		words.extend(
			values
				.chunks(P::WIDTH)
				.map(|chunk| P::from_scalars(chunk.iter().copied())),
		);

		FieldBuffer::new(log_len, words)
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

		if self.log_len < P::LOG_WIDTH {
			// The source's single word also spans lanes that the padding covers.
			// Nothing sitting there is an element of it, so only the live scalars carry over.
			//
			//     source   [ s_0, s_1, x, y ]
			//     result   [ s_0, s_1, 0, 0 ] [ 0, 0, 0, 0 ]
			extended.as_mut()[0] = P::from_scalars(self.iter_scalars());
		} else {
			// Every lane of every source word is live, so whole words move across untouched.
			extended.as_mut()[..self.as_ref().len()].copy_from_slice(self.as_ref());
		}

		extended
	}
}

#[allow(clippy::len_without_is_empty)]
impl<P: PackedField, Data: Deref<Target = [P]>> FieldBuffer<P, Data> {
	/// Create a new FieldBuffer from a slice of packed words.
	///
	/// # Preconditions
	///
	/// * `words.len()` must equal the expected packed length for `log_len`.
	#[track_caller]
	pub fn new(log_len: usize, words: Data) -> Self {
		let expected_packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		assert!(
			words.len() == expected_packed_len,
			"precondition: words.len() must equal expected packed length"
		);

		Self { log_len, words }
	}

	/// Consumes the buffer and returns its backing data store.
	pub fn into_inner(self) -> Data {
		self.words
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
		unsafe { get_packed_slice_unchecked(&self.words, index) }
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

	/// Splits the buffer in half and returns a pair of borrowed slices.
	///
	/// # Preconditions
	///
	/// * `self.log_len()` must be greater than 0.
	#[track_caller]
	pub fn split_half(&self) -> (FieldSlice<'_, P>, FieldSlice<'_, P>) {
		assert!(self.log_len > 0, "precondition: cannot split a buffer of length 1");

		let new_log_len = self.log_len - 1;
		if new_log_len < P::LOG_WIDTH {
			// The result will be two Single variants
			// We have exactly one packed element that needs to be split
			let packed = self.words[0];
			let zeros = P::default();

			let (first_half, second_half) = packed.interleave(zeros, new_log_len);

			let first = FieldBuffer {
				log_len: new_log_len,
				words: FieldSliceData::Single(first_half),
			};
			let second = FieldBuffer {
				log_len: new_log_len,
				words: FieldSliceData::Single(second_half),
			};

			(first, second)
		} else {
			// Split the packed word slice in half
			let half_len = 1 << (new_log_len - P::LOG_WIDTH);
			let (first_half, second_half) = self.words.split_at(half_len);
			let second_half = &second_half[..half_len];

			let first = FieldBuffer {
				log_len: new_log_len,
				words: FieldSliceData::Slice(first_half),
			};
			let second = FieldBuffer {
				log_len: new_log_len,
				words: FieldSliceData::Slice(second_half),
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
		unsafe { set_packed_slice_unchecked(&mut self.words, index, value) };
	}

	/// Returns a mutable iterator over the packed words the elements occupy.
	///
	/// The mutable counterpart of the shared word iterator, covering the same run of words.
	/// A final word below the packing width is lent out whole, dead lanes included.
	#[inline]
	pub fn iter_packed_mut(&mut self) -> slice::IterMut<'_, P> {
		self.as_mut().iter_mut()
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
	pub fn into_split_half(self) -> SplitMut<P, Data> {
		assert!(self.log_len > 0, "precondition: cannot split a buffer of length 1");

		// Each half holds one variable fewer than the buffer, and the guard decides from that
		// alone whether the halves have to be detached from the store.
		SplitMut::new(self.log_len - 1, self.words)
	}

	/// Halves the buffer through a mutable borrow, rather than consuming it.
	///
	/// Equivalent to borrowing the buffer as a mutable view and halving that.
	///
	/// # Preconditions
	///
	/// * `self.log_len()` must be greater than 0.
	#[track_caller]
	pub fn split_half_mut(&mut self) -> SplitMut<P, &'_ mut [P]> {
		self.as_mut_view().into_split_half()
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
				self.words[0].set(i, <P::Scalar as Field>::ZERO);
			}
		}

		self.words
			.truncate(1 << new_log_len.saturating_sub(P::LOG_WIDTH));
	}
}

impl<P: PackedField, Data: Deref<Target = [P]>> AsRef<[P]> for FieldBuffer<P, Data> {
	#[inline]
	fn as_ref(&self) -> &[P] {
		&self.words[..1 << self.log_len.saturating_sub(P::LOG_WIDTH)]
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> AsMut<[P]> for FieldBuffer<P, Data> {
	#[inline]
	fn as_mut(&mut self) -> &mut [P] {
		&mut self.words[..1 << self.log_len.saturating_sub(P::LOG_WIDTH)]
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
		let mut iter = iter.into_iter();
		// The lower bound is the whole count whenever the iterator knows its own length.
		let mut words = Vec::with_capacity(iter.size_hint().0.div_ceil(P::WIDTH));

		// The length check needs the total, which only the elements themselves can give.
		let mut len = 0usize;
		loop {
			// Fill one word from at most one packing width of elements.
			// That bound is a constant, so the lanes are written at constant indices rather
			// than at a running offset into the stream.
			let mut filled = 0usize;
			let word = P::from_scalars(iter.by_ref().take(P::WIDTH).inspect(|_| filled += 1));

			// A dry iterator yields an all-zero word belonging to no element.
			if filled == 0 {
				break;
			}
			words.push(word);
			len += filled;

			// A short group can only be the last, and its unwritten lanes are already zero,
			// which is what the invariant on lanes past the logical length asks for.
			if filled < P::WIDTH {
				break;
			}
		}

		let log_len =
			strict_log_2(len).expect("precondition: element count must be a power of two");

		Self { log_len, words }
	}
}

/// Fills `tail` with copies of `head`, `run` words at a time.
///
/// ```text
///     head            tail
///     [x]      ->     [x | x | x]
/// ```
///
/// Every destination run draws from one contiguous run of `head`.
/// So a worker copying one run never reads outside it.
///
/// # Preconditions
///
/// * `head.len()` is a power of two, and `run` is a power of two at most that
/// * `tail.len()` is a multiple of `head.len()`
fn repeat_words<P: PackedField>(head: &[P], tail: &mut [MaybeUninit<P>], run: usize) {
	debug_assert!(head.len().is_power_of_two());
	debug_assert!(run.is_power_of_two() && run <= head.len());
	debug_assert_eq!(tail.len() % head.len(), 0);

	tail.par_chunks_mut(run).enumerate().for_each(|(i, dst)| {
		// Run `i` of the tail sits one whole head past run `i` of the buffer.
		// The head length is a power of two, so masking that position gives its source.
		let source = (i * run) & (head.len() - 1);
		dst.write_copy_of_slice(&head[source..source + dst.len()]);
	});
}

/// Guards the two halves of a buffer that was split along its highest variable.
///
/// Holds the parent store, and lends each half out as a mutable view.
#[derive(Debug)]
pub struct SplitMut<P: PackedField, Data: DerefMut<Target = [P]>> {
	/// Element count of each half, as a base-2 logarithm.
	log_len: usize,
	/// The detached halves, present only when a half is narrower than a packed word.
	singles: Option<[P; 2]>,
	/// The store both halves come from, and the one a detached pair merges back into.
	data: Data,
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> SplitMut<P, Data> {
	/// Guards the two halves of `data`, each holding `2^log_len` elements.
	fn new(log_len: usize, data: Data) -> Self {
		// Halves this narrow share word 0, so neither can be lent out as a slice of that store.
		// Interleaving against zero lifts each half into a word of its own, starting at lane 0.
		let singles = (log_len < P::LOG_WIDTH).then(|| {
			let (lo_half, hi_half) = data[0].interleave(P::default(), log_len);
			[lo_half, hi_half]
		});

		Self {
			log_len,
			singles,
			data,
		}
	}

	/// Lends the two halves out as mutable views, the low half first.
	///
	/// A half of whole words is a view straight onto the store, so edits land at once.
	/// A narrower half is a view onto a detached word, so edits land when this guard drops.
	pub fn halves(&mut self) -> (FieldSliceMut<'_, P>, FieldSliceMut<'_, P>) {
		match &mut self.singles {
			Some([lo_half, hi_half]) => (
				FieldBuffer {
					log_len: self.log_len,
					words: slice::from_mut(lo_half),
				},
				FieldBuffer {
					log_len: self.log_len,
					words: slice::from_mut(hi_half),
				},
			),
			None => {
				let half_len = 1 << (self.log_len - P::LOG_WIDTH);
				let (lo_half, hi_half) = self.data.split_at_mut(half_len);
				(
					FieldBuffer {
						log_len: self.log_len,
						words: lo_half,
					},
					FieldBuffer {
						log_len: self.log_len,
						words: hi_half,
					},
				)
			}
		}
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> Drop for SplitMut<P, Data> {
	fn drop(&mut self) {
		// Detached halves are the only shape with anything to write back.
		if let Some([lo_half, hi_half]) = self.singles {
			(self.data[0], _) = lo_half.interleave(hi_half, self.log_len);
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::{BufferPool, GlobalAllocator};
	use binius_field::packed::get_packed_slice;
	use binius_utils::rayon::task_size::min_len_for_bytes;
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::test_utils::{B128, Packed128b, random_field_buffer};

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
		//
		// Invariant: a lane past the logical length is not an element, whatever it holds.
		//
		// Fixture state: two live scalars, and two dead lanes carrying unrelated values.
		//
		//     word   [ 0, 1 | 0xdead, 0xbeef ]
		//              ^^^^  live prefix
		//
		// Packing from scalars would zero those lanes, so the word is supplied whole instead.
		let small = FieldBuffer::<P>::new(
			1,
			vec![P::from_scalars([
				scalars[0],
				scalars[1],
				F::new(0xdead),
				F::new(0xbeef),
			])],
		);
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

		// Reserving room copies the source and leaves the buffer at the source's own length.
		let reserved: FieldVec<P, A> =
			FieldBuffer::from_view_with_capacity_in(alloc, src.as_view(), 6);
		assert_eq!(reserved.log_len(), 4);
		assert_eq!(reserved.as_view(), src.as_view());

		// Growing into that reserved room repeats the source, four copies over.
		let mut repeated = reserved;
		repeated.repeat_extend(6);
		assert_eq!(repeated.log_len(), 6);
		assert!((0..64).all(|i| repeated.get(i) == F::new((i % 16) as u128)));

		// A source narrower than one packed word repeats its live lanes, never its dead ones.
		//
		//     source (log_len 1)    [0, 1 | dead, dead]
		//     result (log_len 3)    [0, 1 | 0, 1] [0, 1 | 0, 1]
		let mut cycled: FieldVec<P, A> =
			FieldBuffer::from_view_with_capacity_in(alloc, small.as_view(), 3);
		cycled.repeat_extend(3);
		assert_eq!(cycled.log_len(), 3);
		assert!((0..8).all(|i| cycled.get(i) == F::new((i % 2) as u128)));
	}

	/// Pins `repeat_words` against the definition: word `i` of the tail is word `i mod n` of head.
	fn check_repeat_words(log_head: usize, log_copies: usize) {
		let head: Vec<P> = (0..1 << log_head)
			.map(|i| P::broadcast(F::new(i)))
			.collect();

		// Every run width the caller can pass, from one word up to the whole head.
		// A run under the head length is what splits a fill across workers.
		for log_run in 0..=log_head {
			let mut tail = vec![MaybeUninit::uninit(); head.len() * ((1 << log_copies) - 1)];
			repeat_words(&head, &mut tail, 1 << log_run);

			for (i, word) in tail.iter().enumerate() {
				// SAFETY: the call above wrote every word of the tail.
				let word = unsafe { word.assume_init() };
				assert_eq!(word, head[i % head.len()], "log_run={log_run} i={i}");
			}
		}
	}

	#[test]
	fn repeat_words_tiles_the_head_at_every_run_width() {
		// Heads of 1 to 16 words, each repeated 1 to 8 times over.
		for log_head in 0..5 {
			for log_copies in 0..4 {
				check_repeat_words(log_head, log_copies);
			}
		}
	}

	/// Pins `repeat_extend` against the definition, in scalars rather than words.
	fn check_repeat_extend(log_src: usize, log_dst: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let src = random_field_buffer::<P>(&mut rng, log_src);

		let mut buffer: FieldVec<P, GlobalAllocator> =
			FieldBuffer::from_view_with_capacity_in(&GlobalAllocator, src.as_view(), log_dst);
		buffer.repeat_extend(log_dst);

		assert_eq!(buffer.log_len(), log_dst);
		for i in 0..1 << log_dst {
			let expected = src.get(i % (1 << log_src));
			assert_eq!(buffer.get(i), expected, "log_src={log_src} log_dst={log_dst} i={i}");
		}
	}

	#[test]
	fn repeat_extend_repeats_the_live_scalars() {
		// A packed word holds 4 lanes here, so `log_src` 0 and 1 exercise the sub-word path,
		// and a `log_dst` that stays under the width exercises repeating inside one word.
		for log_src in 0..5 {
			for log_dst in log_src..7 {
				check_repeat_extend(log_src, log_dst);
			}
		}
	}

	#[test]
	#[should_panic(expected = "precondition: log_len must be at least the buffer's own log_len")]
	fn repeat_extend_rejects_shrinking() {
		let mut buffer: FieldVec<P, GlobalAllocator> = FieldBuffer::from_view_with_capacity_in(
			&GlobalAllocator,
			FieldBuffer::<P>::zeros(4).as_view(),
			5,
		);
		buffer.repeat_extend(3);
	}

	#[test]
	#[should_panic(expected = "precondition: the store must have room reserved")]
	fn repeat_extend_rejects_a_store_without_room() {
		let mut buffer: FieldVec<P, GlobalAllocator> = FieldBuffer::from_view_with_capacity_in(
			&GlobalAllocator,
			FieldBuffer::<P>::zeros(4).as_view(),
			4,
		);
		buffer.repeat_extend(5);
	}

	#[test]
	fn a_copy_reproduces_every_word_on_both_sides_of_the_task_floor() {
		// The floor is the fewest words a worker takes, set by one task's byte budget.
		//
		//     at the floor   [--------]              one task
		//     above it       [--------][--------]    split across workers
		let floor = min_len_for_bytes::<P>();
		for words in [floor, 2 * floor] {
			let log_len = words.ilog2() as usize + P::LOG_WIDTH;
			let src = random_field_buffer::<P>(&mut StdRng::seed_from_u64(0), log_len);
			let copy: FieldVec<P, GlobalAllocator> =
				FieldBuffer::from_view_in(&GlobalAllocator, src.as_view());
			assert_eq!(copy.log_len(), log_len);
			assert_eq!(copy.as_ref(), src.as_ref(), "words={words}");
		}
	}

	#[test]
	#[should_panic(expected = "precondition: log_capacity must be at least src.log_len()")]
	fn from_view_with_capacity_rejects_a_capacity_below_the_source() {
		let src = FieldBuffer::<P>::zeros(4);
		let _: FieldVec<P, GlobalAllocator> =
			FieldBuffer::from_view_with_capacity_in(&GlobalAllocator, src.as_view(), 3);
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
	fn split_half() {
		// Test with buffer size > P::WIDTH (multiple packed elements)
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		let (first, second) = buffer.split_half();
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

		let (first, second) = buffer.split_half();
		assert_eq!(first.len(), 2);
		assert_eq!(second.len(), 2);

		// Verify we got Single variants
		match &first.words {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for first half"),
		}
		match &second.words {
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

		let (first, second) = buffer.split_half();
		assert_eq!(first.len(), 1);
		assert_eq!(second.len(), 1);

		// Verify we got Single variants
		match &first.words {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for first half"),
		}
		match &second.words {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for second half"),
		}

		assert_eq!(first.get(0), F::new(10));
		assert_eq!(second.get(0), F::new(20));
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn split_half_size_one() {
		let values = vec![F::new(42)];
		let buffer = FieldBuffer::<P>::from_values(&values);
		let _ = buffer.split_half();
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
		}
	}
}
