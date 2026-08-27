// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Locating a chunk of a field buffer that is narrower than one packed word.
//!
//! The chunk iterators and the write-back guards both need this arithmetic, so it sits in a
//! module of its own rather than in either of them.

use std::marker::PhantomData;

use binius_field::PackedField;

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
pub struct SubWordChunk<P> {
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
	pub const fn new(log_chunk_size: usize, chunk_index: usize) -> Self {
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
	pub const fn word_index(self) -> usize {
		self.word_index
	}

	/// Element count of the chunk, as a base-2 logarithm.
	#[inline]
	pub const fn log_len(self) -> usize {
		self.log_len
	}

	/// Reads the chunk's elements out of the word holding it.
	#[inline]
	pub fn scalars(self, word: P) -> impl Iterator<Item = P::Scalar> + Send + Clone {
		(0..1 << self.log_len).map(move |i| word.get(self.lane_offset | i))
	}

	/// Copies the chunk into a word of its own, elements starting at lane 0.
	///
	/// Lanes past the chunk come out zero, since packing from scalars starts from a zeroed word.
	#[inline]
	pub fn repack(self, words: &[P]) -> P {
		P::from_scalars(self.scalars(words[self.word_index]))
	}

	/// Copies an edited chunk back into the lanes it came from.
	///
	/// The inverse of copying the chunk out.
	/// Lane `i` of the chunk lands back at the lane it was read from.
	///
	/// ```text
	/// WIDTH = 4, log_len = 1, lane_offset = 2
	///
	/// chunk  [ y z . . ]
	/// word   [ a b y z ]   lanes 0 and 1 keep whatever they held
	/// ```
	///
	/// Lanes of the word outside the chunk are left untouched, since neighbouring chunks own them.
	#[inline]
	pub fn merge_into(self, word: &mut P, chunk: &P) {
		// The chunk's elements sit at lanes 0..2^log_len, so the loop walks exactly those.
		for i in 0..1 << self.log_len {
			// The lane offset is a multiple of the chunk length, so the bits below it are free.
			// Setting them with an OR therefore addresses lane i of this chunk and no other.
			word.set(self.lane_offset | i, chunk.get(i));
		}
	}
}
