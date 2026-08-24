// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Iterators over the aligned chunks of a field buffer.
//!
//! A chunk of `2^k` elements starts at a multiple of its own size.
//! Chunks of one size therefore tile the buffer exactly, with none left over.
//!
//! A chunk takes one of two shapes, decided by its size against the packing width:
//!
//! ```text
//! chunk >= one packed word  ->  a run of whole words, borrowed from the store
//! chunk <  one packed word  ->  some lanes of one word
//! ```
//!
//! The iterators here cover the first shape, where a chunk is a borrowed run of words.
//! The second shape gets a type of its own, since locating those lanes is shared work.

use std::{iter, marker::PhantomData, slice};

use binius_field::PackedField;

use super::{FieldBuffer, FieldSlice, FieldSliceData, FieldSliceMut};

/// The location of one chunk that is smaller than a packed word.
///
/// A chunk this small does not get a word to itself.
/// Several of them share one word, so a chunk is a run of lanes inside it:
///
/// ```text
/// WIDTH = 4, log_chunk_size = 1, so 2 chunks per word
///
/// chunk 0 -> word 0, lane 0     chunk 2 -> word 1, lane 0
/// chunk 1 -> word 0, lane 2     chunk 3 -> word 1, lane 2
/// ```
///
/// The chunk index splits in two to get there.
/// High bits pick the word, low bits pick the lanes within it.
///
/// The packing width is part of the type.
/// So a location can only ever be read against the packing it was computed for.
#[derive(Debug, Clone, Copy)]
pub(super) struct SubWordChunk<P> {
	/// Which word of the backing store holds the chunk.
	word_index: usize,
	/// Which lane of that word the chunk's first element sits at.
	lane_offset: usize,
	/// Element count of the chunk, as a base-2 logarithm.
	log_len: usize,
	/// Ties the arithmetic above to one packing width.
	packing: PhantomData<P>,
}

impl<P: PackedField> SubWordChunk<P> {
	/// Locates the chunk at `chunk_index` among chunks of `2^log_chunk_size` elements.
	///
	/// The size must be below the packing width, which is what makes several chunks share a word.
	#[inline]
	pub(super) const fn new(log_chunk_size: usize, chunk_index: usize) -> Self {
		let log_chunks_per_word = P::LOG_WIDTH - log_chunk_size;
		let chunk_subindex = chunk_index & ((1 << log_chunks_per_word) - 1);
		Self {
			word_index: chunk_index >> log_chunks_per_word,
			lane_offset: chunk_subindex << log_chunk_size,
			log_len: log_chunk_size,
			packing: PhantomData,
		}
	}

	/// Which word of the backing store holds the chunk.
	#[inline]
	pub(super) const fn word_index(self) -> usize {
		self.word_index
	}

	/// Element count of the chunk, as a base-2 logarithm.
	#[inline]
	pub(super) const fn log_len(self) -> usize {
		self.log_len
	}

	/// Reads the chunk's elements out of the word holding it.
	#[inline]
	pub(super) fn scalars(self, word: P) -> impl Iterator<Item = P::Scalar> + Send + Clone {
		(0..1 << self.log_len).map(move |i| word.get(self.lane_offset | i))
	}

	/// Copies the chunk into a word of its own, elements starting at lane 0.
	///
	/// Lanes past the chunk come out zero, since packing from scalars starts from a zeroed word.
	#[inline]
	pub(super) fn repack(self, words: &[P]) -> P {
		P::from_scalars(self.scalars(words[self.word_index]))
	}

	/// Copies an edited chunk back into the lanes it came from.
	#[inline]
	pub(super) fn merge_into(self, word: &mut P, chunk: &P) {
		for i in 0..1 << self.log_len {
			word.set(self.lane_offset | i, chunk.get(i));
		}
	}
}

/// Iterator over a buffer's chunks of a fixed size, each borrowed as a shared view.
///
/// Yielded by asking a buffer for its chunks.
/// The chunk size is at least one packed word, so every chunk is a borrowed run of words.
pub struct Chunks<'a, P: PackedField> {
	/// Runs of words, one per chunk, cut off at the buffer's logical chunk count.
	runs: iter::Take<slice::Chunks<'a, P>>,
	/// Element count of each chunk, as a base-2 logarithm.
	log_chunk_size: usize,
}

impl<'a, P: PackedField> Chunks<'a, P> {
	/// Builds the iterator over `words`, whose length must be the buffer's live word count.
	#[inline]
	pub(super) fn new(words: &'a [P], log_chunk_size: usize, chunk_count: usize) -> Self {
		let words_per_chunk = 1 << (log_chunk_size - P::LOG_WIDTH);
		Self {
			runs: words.chunks(words_per_chunk).take(chunk_count),
			log_chunk_size,
		}
	}
}

impl<'a, P: PackedField> Iterator for Chunks<'a, P> {
	type Item = FieldSlice<'a, P>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.runs.next().map(|run| FieldBuffer {
			log_len: self.log_chunk_size,
			values: FieldSliceData::Slice(run),
		})
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.runs.size_hint()
	}
}

impl<P: PackedField> ExactSizeIterator for Chunks<'_, P> {}

impl<P: PackedField> Clone for Chunks<'_, P> {
	fn clone(&self) -> Self {
		Self {
			runs: self.runs.clone(),
			log_chunk_size: self.log_chunk_size,
		}
	}
}

/// Iterator over a buffer's chunks of a fixed size, each borrowed as a mutable view.
///
/// The mutable counterpart of the shared chunk iterator.
/// The chunks are disjoint, so each is lent out for the whole iteration rather than one at a time.
pub struct ChunksMut<'a, P: PackedField> {
	/// Runs of words, one per chunk, cut off at the buffer's logical chunk count.
	runs: iter::Take<slice::ChunksMut<'a, P>>,
	/// Element count of each chunk, as a base-2 logarithm.
	log_chunk_size: usize,
}

impl<'a, P: PackedField> ChunksMut<'a, P> {
	/// Builds the iterator over `words`, whose length must be the buffer's live word count.
	#[inline]
	pub(super) fn new(words: &'a mut [P], log_chunk_size: usize, chunk_count: usize) -> Self {
		let words_per_chunk = 1 << (log_chunk_size - P::LOG_WIDTH);
		Self {
			runs: words.chunks_mut(words_per_chunk).take(chunk_count),
			log_chunk_size,
		}
	}
}

impl<'a, P: PackedField> Iterator for ChunksMut<'a, P> {
	type Item = FieldSliceMut<'a, P>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.runs.next().map(|run| FieldBuffer {
			log_len: self.log_chunk_size,
			values: run,
		})
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.runs.size_hint()
	}
}

impl<P: PackedField> ExactSizeIterator for ChunksMut<'_, P> {}
