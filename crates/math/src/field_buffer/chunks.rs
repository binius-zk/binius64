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
//! The shared iterator here covers both shapes, so any size up to the buffer's length works.
//! The mutable one covers only the first, since a borrowed run of words is all it can lend out.

use std::{iter, ops::Range, slice};

use binius_field::PackedField;

use super::{FieldBuffer, FieldSlice, FieldSliceData, FieldSliceMut, sub_word::SubWordChunk};

/// How a shared chunk iterator walks the store, settled by the chunk size before the first step.
///
/// ```text
/// chunk >= one packed word  ->  Words, stepping runs of the store
/// chunk <  one packed word  ->  Lanes, stepping chunk indices into one word each
/// ```
#[derive(Clone)]
enum ChunkSource<'a, P: PackedField> {
	/// Runs of words, one per chunk, cut off at the buffer's logical chunk count.
	Words(iter::Take<slice::Chunks<'a, P>>),
	/// Chunk indices to locate lanes with, alongside the store those lanes are repacked from.
	Lanes {
		words: &'a [P],
		indices: Range<usize>,
	},
}

/// Iterator over a buffer's chunks of a fixed size, each borrowed as a shared view.
///
/// Yielded by asking a buffer for its chunks.
/// A chunk of at least one packed word is a borrowed run of words.
/// A smaller one is a copy of its lanes, repacked to start at lane 0.
pub struct Chunks<'a, P: PackedField> {
	/// Where the next chunk comes from, one shape or the other for the whole iteration.
	source: ChunkSource<'a, P>,
	/// Element count of each chunk, as a base-2 logarithm.
	log_chunk_size: usize,
}

impl<'a, P: PackedField> Chunks<'a, P> {
	/// Builds the iterator over `words`, whose length must be the buffer's live word count.
	#[inline]
	pub(super) fn new(words: &'a [P], log_chunk_size: usize, chunk_count: usize) -> Self {
		let source = if log_chunk_size >= P::LOG_WIDTH {
			let words_per_chunk = 1 << (log_chunk_size - P::LOG_WIDTH);
			ChunkSource::Words(words.chunks(words_per_chunk).take(chunk_count))
		} else {
			ChunkSource::Lanes {
				words,
				indices: 0..chunk_count,
			}
		};
		Self {
			source,
			log_chunk_size,
		}
	}
}

impl<'a, P: PackedField> Iterator for Chunks<'a, P> {
	type Item = FieldSlice<'a, P>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let words = match &mut self.source {
			ChunkSource::Words(runs) => FieldSliceData::Slice(runs.next()?),
			ChunkSource::Lanes { words, indices } => FieldSliceData::Single(
				SubWordChunk::<P>::new(self.log_chunk_size, indices.next()?).repack(words),
			),
		};
		Some(FieldBuffer {
			log_len: self.log_chunk_size,
			words,
		})
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		match &self.source {
			ChunkSource::Words(runs) => runs.size_hint(),
			ChunkSource::Lanes { indices, .. } => indices.size_hint(),
		}
	}
}

impl<P: PackedField> ExactSizeIterator for Chunks<'_, P> {}

impl<P: PackedField> Clone for Chunks<'_, P> {
	fn clone(&self) -> Self {
		Self {
			source: self.source.clone(),
			log_chunk_size: self.log_chunk_size,
		}
	}
}

/// Iterator over a buffer's chunks of a fixed size, each borrowed as a mutable view.
///
/// The mutable counterpart of the shared chunk iterator, restricted to chunks of whole words.
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
			words: run,
		})
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.runs.size_hint()
	}
}

impl<P: PackedField> ExactSizeIterator for ChunksMut<'_, P> {}
