// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Aligned chunks of a field buffer.
//!
//! The buffer's chunk methods live here, alongside everything they hand out:
//!
//! ```text
//! Chunks         iterator over the buffer's aligned chunks
//! SubWordChunk   locating a chunk inside the word it shares with its neighbours
//! ```
//!
//! # How a chunk is shaped
//!
//! A chunk of `2^k` elements starts at a multiple of its own size, so chunks of one size tile the
//! buffer exactly, with none left over.
//! Its size against the packing width decides which of two shapes it takes:
//!
//! ```text
//! chunk >= one packed word  ->  a run of whole words, borrowed from the store
//! chunk <  one packed word  ->  some lanes of one word, shared with its neighbours
//! ```
//!
//! The second shape is why a chunk cannot always be lent out as a slice of the store.
//! Lending one means copying its lanes into a word of their own.

use std::{
	iter,
	marker::PhantomData,
	ops::{Deref, Range},
	slice,
};

use binius_field::PackedField;
use binius_utils::rayon::{iter::Either, prelude::*, slice::ParallelSlice};

use super::{FieldBuffer, FieldSlice, FieldSliceData};

impl<P: PackedField, Data: Deref<Target = [P]>> FieldBuffer<P, Data> {
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

		let words = if log_chunk_size >= P::LOG_WIDTH {
			// A whole run of words, borrowed straight from the store.
			let words_per_chunk = 1 << (log_chunk_size - P::LOG_WIDTH);
			FieldSliceData::Slice(&self.words[chunk_index * words_per_chunk..][..words_per_chunk])
		} else {
			// Lanes inside one word, copied out so the chunk starts at lane 0.
			FieldSliceData::Single(
				SubWordChunk::new(log_chunk_size, chunk_index).repack(&self.words),
			)
		};

		FieldBuffer {
			log_len: log_chunk_size,
			words,
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
						words: FieldSliceData::Slice(chunk),
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
						words: FieldSliceData::Single(
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
}

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
struct SubWordChunk<P> {
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
	const fn new(log_chunk_size: usize, chunk_index: usize) -> Self {
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
	const fn word_index(self) -> usize {
		self.word_index
	}

	/// Reads the chunk's elements out of the word holding it.
	#[inline]
	fn scalars(self, word: P) -> impl Iterator<Item = P::Scalar> + Send + Clone {
		(0..1 << self.log_len).map(move |i| word.get(self.lane_offset | i))
	}

	/// Copies the chunk into a word of its own, elements starting at lane 0.
	///
	/// Lanes past the chunk come out zero, since packing from scalars starts from a zeroed word.
	#[inline]
	fn repack(self, words: &[P]) -> P {
		P::from_scalars(self.scalars(words[self.word_index]))
	}
}

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
	fn new(words: &'a [P], log_chunk_size: usize, chunk_count: usize) -> Self {
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

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b};

	type P = Packed128b;
	type F = B128;

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

	proptest! {
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
