// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Contracting the word axis: one field element per bit position.

use std::iter;

use binius_core::word::Word;
use binius_field::{BinaryField, PackedBinaryField64x1b};
use binius_math::{FieldBuffer, multilinear::hypercube::Hypercube};
use binius_utils::rayon::prelude::*;

use super::{CHUNK_SIZE, FoldedWord, LOG_CHUNK_SIZE};
use crate::bit_matrix::{ColumnSums, RowFoldTables, WEIGHTS_PER_TABLE};

/// Minimum chunks one parallel task folds along the word axis.
///
/// A chunk is 64 words and folds in well under a microsecond, which is about what a handoff costs.
/// Left unbounded, the split reaches one chunk per task and the handoffs dominate.
///
/// A floor also caps how far a loop can divide.
/// A list of `n` chunks splits into at most `n / floor` tasks.
/// So a floor set too high starves the cores on a short list, and one set too low drowns a long
/// list in handoffs.
///
/// Sixteen chunks is 1024 words, which is the setting that loses at neither end.
const MIN_CHUNKS_PER_TASK: usize = 16;
/// A reusable [Method of Four Russians] folder over a fixed evaluation point.
///
/// Many word-lists often share one point, so the tables are built once here and reused.
/// The batched instance fold is that case: every committed word folds against the same point.
///
/// The two tables it holds:
/// * per-byte subset-sum lookups, built from the point's prefix.
/// * one weight per chunk, built from the point's suffix.
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
#[derive(Debug)]
pub struct WordFolder<F: BinaryField> {
	/// One 256-entry subset-sum table per byte of a word, from the prefix expansion.
	///
	/// Table `s` folds the words at positions `s * WEIGHTS_PER_TABLE + t` within a chunk.
	/// Each such word is weighted by prefix-expansion entry `t` of that group.
	lookups: RowFoldTables<F, { Word::BYTES }>,
	/// One weight per chunk of `CHUNK_SIZE` words, from the suffix expansion.
	suffix_weights: FieldBuffer<F>,
	/// Base-2 log of the word axis's length, which every folded list fits in.
	///
	/// This is the point's own width, stored rather than the length it stands for.
	/// A point as wide as a `usize` would overflow that length, but never its log.
	log_n_words: usize,
}

impl<F: BinaryField> WordFolder<F> {
	/// Builds the folding tables for `point`.
	///
	/// Each later fold takes a list of at most `2^point.len()` words, folded against
	/// this point.
	pub fn new(point: &[F]) -> Self {
		// The point splits into a prefix indexing words within a chunk and a suffix indexing
		// chunks.
		let prefix_len = point.len().min(LOG_CHUNK_SIZE);
		let (prefix, suffix) = point.split_at(prefix_len);

		// One weight per word of a chunk, from the prefix.
		// A point shorter than one chunk yields fewer weights than a chunk holds, and the table
		// build reads the rest as zero.
		// Those zeros pair with the repeated words a short list is filled with, so they add
		// nothing.
		let prefix_expansion = Hypercube::One.expand(prefix).build_scalars();
		let lookups = RowFoldTables::new(&prefix_expansion);

		// One weight per chunk of CHUNK_SIZE words, from the suffix.
		let suffix_weights = Hypercube::One.expand(suffix).build::<F>();

		Self {
			lookups,
			suffix_weights,
			log_n_words: point.len(),
		}
	}

	/// Folds one word-list against the point.
	///
	/// Returns the array whose entry at bit position `b` is
	///
	/// ```text
	/// out[b] = sum_i eq(point, i) * bit_b(words[i])
	/// ```
	///
	/// with a clear bit read as zero and a set bit read as one.
	///
	/// This runs sequentially over the list's chunks, so it leaves every other core free.
	/// It is the right driver for a caller already parallel over many lists.
	/// A caller with few lists wants the parallel driver below instead.
	///
	/// A list shorter than the word axis reads its missing high rows as zero: an absent row's
	/// weight multiplies nothing, so it contributes nothing to any bit position. Chunks lying
	/// entirely past the list's end are therefore never visited at all.
	///
	/// ## Preconditions
	///
	/// * `words.len() <= 1 << point.len()`
	pub fn fold(&self, words: &[Word]) -> FoldedWord<F> {
		assert!(words.len() <= 1 << self.log_n_words, "words.len() must not exceed 2^point.len()");

		let (chunks, tail) = words.as_chunks::<CHUNK_SIZE>();
		let mut folded = [F::ZERO; Word::BITS];

		// Accumulate each chunk's contribution, scaled by that chunk's suffix weight. Weights past
		// the list's end pair with absent rows, so the zip drops them.
		for (chunk, &suffix_weight) in iter::zip(chunks, self.suffix_weights.as_ref()) {
			self.accumulate_chunk(chunk, suffix_weight, &mut folded);
		}

		self.accumulate_tail(tail, chunks.len(), &mut folded);
		folded
	}

	/// Folds one word-list against the point, parallel over that list's chunks.
	///
	/// Returns the same array the sequential fold returns, under the same contract.
	/// The two differ only in how the chunk axis is divided across workers.
	///
	/// Reach for this when few lists share the point, so the chunk axis is the only one wide enough
	/// to divide. A caller folding many lists against one point should instead parallelize across
	/// the lists and fold each one sequentially.
	///
	/// ## Preconditions
	///
	/// * `words.len() <= 1 << point.len()`
	pub fn fold_par(&self, words: &[Word]) -> FoldedWord<F> {
		assert!(words.len() <= 1 << self.log_n_words, "words.len() must not exceed 2^point.len()");

		let (chunks, tail) = words.as_chunks::<CHUNK_SIZE>();

		// Each chunk contributes to every bit position, scaled by that chunk's suffix weight.
		// Summing the per-chunk accumulators contracts the word axis.
		// Weights past the list's end pair with absent rows, so the zip drops them.
		//
		// One accumulator per worker, not one per chunk:
		//
		//     per chunk : 512 bytes of words in, a 1 KiB accumulator zeroed and merged back out
		//     per worker: 512 bytes of words in, straight into an accumulator already live
		//
		// A merge seeded with a partial that already exists never touches a buffer of zeros.
		// An identity would zero one accumulator per chunk, then add all 64 elements of it.
		let mut folded = chunks
			.par_iter()
			.zip(self.suffix_weights.as_ref().par_iter())
			// One item is one chunk of 64 words, so the floor needs no conversion.
			.with_min_len(MIN_CHUNKS_PER_TASK)
			.fold(
				|| [F::ZERO; Word::BITS],
				|mut acc, (chunk, &suffix_weight)| {
					self.accumulate_chunk(chunk, suffix_weight, &mut acc);
					acc
				},
			)
			.reduce_with(|mut lhs, rhs| {
				for (lhs_i, rhs_i) in iter::zip(&mut lhs, rhs) {
					*lhs_i += rhs_i;
				}
				lhs
			})
			// A list with no whole chunks yields no partials at all, and folds to zero.
			.unwrap_or([F::ZERO; Word::BITS]);

		self.accumulate_tail(tail, chunks.len(), &mut folded);
		folded
	}

	/// Folds one chunk of words into the accumulator, scaled by that chunk's weight.
	///
	/// Words are 64-bit rows, so a chunk is 64 of them and the columns are the 64 bit positions.
	/// A word and a 64-bit row of single-bit scalars share one underlier, so the view below is
	/// free.
	fn accumulate_chunk(&self, chunk: &[Word; CHUNK_SIZE], weight: F, acc: &mut FoldedWord<F>) {
		// Reshape the chunk into one contiguous group of eight rows per table.
		let groups = bytemuck::must_cast_ref::<
			[Word; CHUNK_SIZE],
			[[PackedBinaryField64x1b; WEIGHTS_PER_TABLE]; Word::BYTES],
		>(chunk);

		// Sum every group's contribution before scaling, so the chunk costs one multiply per
		// column.
		let mut sums = ColumnSums::zero();
		self.lookups.fold_into(groups.iter().copied(), &mut sums);
		sums.add_scaled_to(weight, acc);
	}

	/// Accumulates the chunk the list ends in, completed with its zero rows.
	///
	/// A list whose length is a whole number of chunks has no such chunk, and this does nothing.
	///
	/// # Arguments
	///
	/// * `tail` - the words after the last whole chunk, fewer than one chunk of them
	/// * `n_whole_chunks` - how many whole chunks came before, which selects the weight to use
	/// * `folded` - the accumulator the tail's contribution is added into
	fn accumulate_tail(&self, tail: &[Word], n_whole_chunks: usize, folded: &mut FoldedWord<F>) {
		if tail.is_empty() {
			return;
		}

		// Rows past the list's end read as zero, which contributes nothing to any bit position.
		let mut chunk = [Word::ZERO; CHUNK_SIZE];
		chunk[..tail.len()].copy_from_slice(tail);
		self.accumulate_chunk(&chunk, self.suffix_weights.get(n_whole_chunks), folded);
	}
}

#[cfg(test)]
mod tests {
	use binius_math::test_utils::random_scalars;
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use binius_verifier::config::B128;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	/// Contracts the word axis, leaving one element per bit position.
	///
	/// A list shorter than the axis reads its high rows as zero, which weight nothing.
	fn reference_fold_word_axis<F: BinaryField>(words: &[Word], point: &[F]) -> FoldedWord<F> {
		assert!(words.len() <= 1 << point.len());

		let eq = Hypercube::One.expand(point).build_scalars();
		let mut out = [F::ZERO; Word::BITS];
		for (word, &weight) in iter::zip(words, &eq) {
			for (bit, out_bit) in out.iter_mut().enumerate() {
				if (word.as_u64() >> bit) & 1 == 1 {
					*out_bit += weight;
				}
			}
		}
		out
	}

	/// Word counts spanning every regime both folds branch on.
	///
	/// The folds split their input into whole chunks and a short tail, so the interesting lengths
	/// sit around those boundaries rather than at round powers of two.
	fn any_n_words() -> impl Strategy<Value = usize> {
		0..=4 * CHUNK_SIZE
	}

	fn words_of(n: usize, seed: u64) -> Vec<Word> {
		let mut rng = StdRng::seed_from_u64(seed);
		(0..n).map(|_| Word::from_u64(rng.random())).collect()
	}

	proptest! {
		#[test]
		fn word_axis_fold_matches_the_definition(n_words in any_n_words(), seed: u64) {
			// The point must cover the list, so its width is the list's rounded-up log.
			let words = words_of(n_words, seed);
			let mut rng = StdRng::seed_from_u64(seed ^ 2);
			let point = random_scalars::<B128>(&mut rng, log2_ceil_usize(words.len()));

			prop_assert_eq!(
				WordFolder::new(&point).fold_par(&words),
				reference_fold_word_axis(&words, &point),
			);
		}

		#[test]
		fn word_axis_drivers_agree(n_words in any_n_words(), seed: u64) {
			// Dividing the chunk axis across workers changes the grouping of the sums, not their
			// value, because field addition is associative and exact.
			let words = words_of(n_words, seed);
			let mut rng = StdRng::seed_from_u64(seed ^ 3);
			let point = random_scalars::<B128>(&mut rng, log2_ceil_usize(words.len()));

			let folder = WordFolder::new(&point);
			prop_assert_eq!(folder.fold_par(&words), folder.fold(&words));
		}

		#[test]
		fn word_axis_fold_reads_a_short_list_as_zero_padded(
			log_rows in LOG_CHUNK_SIZE..LOG_CHUNK_SIZE + 3,
			seed: u64,
		) {
			// A list shorter than the word axis must fold as that list zero-padded up to it,
			// without ever materializing the padding.
			let n_words = (seed as usize) % (1 << log_rows);
			let words = words_of(n_words, seed);
			let mut rng = StdRng::seed_from_u64(seed ^ 5);
			let point = random_scalars::<B128>(&mut rng, log_rows);

			let mut padded = words.clone();
			padded.resize(1 << log_rows, Word::ZERO);
			let expected = reference_fold_word_axis(&padded, &point);

			prop_assert_eq!(WordFolder::new(&point).fold(&words), expected);
			prop_assert_eq!(WordFolder::new(&point).fold_par(&words), expected);
		}
	}
}
