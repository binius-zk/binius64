// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, hint::assert_unchecked, iter};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{
	BinaryField, Divisible, PackedBinaryField64x1b, PackedField, util::expand_subset_sums_array,
};
use binius_math::{
	FieldBuffer,
	multilinear::hypercube::{Hypercube, OneCube},
};
use binius_utils::{buffer::VecLike, checked_arithmetics::log2_ceil_usize, rayon::prelude::*};

use crate::bit_matrix::{ColumnSums, RowFoldTables, WEIGHTS_PER_TABLE};

/// Number of words folded together within a single chunk.
///
/// One row-fold table covers one byte of a word, so a chunk spans every row those tables reach:
/// eight tables of eight rows each. That this equals the bit width of a word is arithmetic, not
/// definition.
const CHUNK_SIZE: usize = Word::BYTES * WEIGHTS_PER_TABLE;
/// Base-2 logarithm of the number of words folded together within a single chunk.
const LOG_CHUNK_SIZE: usize = CHUNK_SIZE.ilog2() as usize;
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
/// One 64-bit word with its bit axis expanded into full field elements.
///
/// Each bit position becomes one element, so the word is carried in oblong form.
pub type FoldedWord<F> = [F; Word::BITS];

/// Minimum words one parallel task folds along the bit axis.
///
/// A packed element is the unit of work here, and it holds as few as one word.
/// Left unbounded, the split reaches one task per word, and the handoff costs more than the fold.
///
/// A floor also caps how far a loop can divide.
/// A list of `n` words splits into at most `n / floor` tasks.
/// So the floor must stay below the shortest list this fold runs at, divided by the core count.
/// Otherwise the split stops before the cores are full.
///
/// The shared task-size budgets in the utilities crate are calibrated for wider items.
/// They land high enough here to breach that cap on a mid-sized list.
const MIN_WORDS_PER_TASK: usize = 1 << 12;

/// Computes a [`FieldBuffer`] where each element is the inner product of the bits of a word and a
/// vector of field elements.
///
/// Returns a buffer where element `i` is the inner product of the bits of word `i` in `words`
/// (mapping bit 0 to [`Field::ZERO`](binius_field::Field::ZERO) and bit 1 to
/// [`Field::ONE`](binius_field::Field::ONE)) and the values in `vec`.
///
/// This implementation uses the [Method of Four Russians] to optimize the computation by
/// precomputing a small lookup table and looking up into it using bitwise chunks of the words.
///
/// The returned buffer has `log2_ceil(words.len())` variables. `words` need not have a power-of-two
/// length; the high words up to that rounded-up length are treated as zero.
///
/// ## Preconditions
/// * `vec` contains exactly [`Word::BITS`] elements
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
pub fn fold_words<F, P, A>(alloc: &A, words: &[Word], vec: &[F]) -> FieldBuffer<P, A::Vec<P>>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	A: Allocator,
{
	BitAxisFolder::new(vec).fold(alloc, words)
}

/// Subset sums of a word's bit weights, eight bit positions at a time.
///
/// One table per byte of a word, each holding all 256 subset sums of that byte's eight weights.
/// A byte of the word then indexes those eight weights' whole contribution in one load.
///
/// This is the bit-axis counterpart of the row tables the word-axis folder holds, and the two are
/// built the same way.
///
/// Storing the tables as a fixed-length array rather than a heap-allocated one is worth 9 to 18
/// percent over the generic bytewise lookup in the field crate, which is why this specialization
/// exists at all.
///
/// This uses the [Method of Four Russians]: precompute one lookup table per byte position, then
/// combine the word's bytes.
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
#[derive(Debug)]
struct BitWeightTables<F> {
	tables: [[F; 1 << WEIGHTS_PER_TABLE]; Word::BYTES],
}

impl<F: BinaryField> BitWeightTables<F> {
	/// Builds the tables from one weight per bit position of a word.
	///
	/// # Panics
	///
	/// Panics unless there is exactly one weight per bit position.
	fn new(weights: &[F]) -> Self {
		assert_eq!(weights.len(), Word::BITS);

		// Byte `i` of a word carries bit positions `8i` through `8i + 7`, so its table covers
		// exactly those eight weights.
		let tables = array::from_fn(|byte| {
			let group: [F; WEIGHTS_PER_TABLE] = weights
				[byte * WEIGHTS_PER_TABLE..(byte + 1) * WEIGHTS_PER_TABLE]
				.try_into()
				.expect("a word has Word::BYTES groups of WEIGHTS_PER_TABLE weights");
			expand_subset_sums_array(group)
		});

		Self { tables }
	}

	/// The inner product of a word's bits with the weights.
	///
	/// A clear bit reads as zero and a set bit as one, so the sum runs over the set bits only.
	#[inline]
	fn fold(&self, word: Word) -> F {
		// Each byte selects its group's whole contribution, and the eight groups partition the
		// word's bit positions, so summing them is the full inner product.
		iter::zip(Divisible::<u8>::ref_iter(&word.0), &self.tables)
			.map(|(byte, table)| table[byte as usize])
			.fold(F::ZERO, |acc, contribution| acc + contribution)
	}
}

/// A reusable folder over a fixed vector of bit-index scalars, the [`fold_words`] analogue of
/// [`WordFolder`].
///
/// [`fold_words`] rebuilds its Method of Four Russians lookup transform on every call. A caller
/// folding several word-lists against the same scalar vector can instead build the transform once
/// with [`new`](Self::new) and reuse it across [`fold`](Self::fold) calls.
#[derive(Debug)]
pub struct BitAxisFolder<F: BinaryField> {
	tables: BitWeightTables<F>,
}

impl<F: BinaryField> BitAxisFolder<F> {
	/// Builds the folding transform for `vec`.
	///
	/// ## Preconditions
	/// * `vec` contains exactly [`Word::BITS`] elements
	pub fn new(vec: &[F]) -> Self {
		Self {
			tables: BitWeightTables::new(vec),
		}
	}

	/// Folds `words` into a [`FieldBuffer`], mapping each word to the inner product of its bits
	/// with the scalar vector. See [`fold_words`] for the exact contract.
	pub fn fold<P, A>(&self, alloc: &A, words: &[Word]) -> FieldBuffer<P, A::Vec<P>>
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		// `words` need not have a power-of-two length; the high words up to the next power of two
		// are treated as zero, so the slots after the last real word are zero-filled by resize.
		let log_n = log2_ceil_usize(words.len());
		let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);

		let mut values = alloc.alloc::<P>(capacity);

		let n_chunks = words.len() / P::WIDTH;
		let (words_aligned, words_remaining) = words.split_at(n_chunks * P::WIDTH);

		let values_aligned = &mut values.spare_capacity_mut()[..n_chunks];
		let word_chunks = words_aligned.par_chunks_exact(P::WIDTH);
		assert_eq!(values_aligned.len(), word_chunks.len());

		(values_aligned, word_chunks)
			.into_par_iter()
			// One item is one packed element, so the floor converts from words to items.
			.with_min_len(MIN_WORDS_PER_TASK.div_ceil(P::WIDTH))
			.for_each(|(out, word_chunk)| {
				// Safety:
				// - words_aligned has length that is a multiple of P::WIDTH
				// - words_aligned is split into P::WIDTH chunks
				unsafe { assert_unchecked(word_chunk.len() == P::WIDTH) };
				out.write(P::from_scalars(word_chunk.iter().map(|&word| self.tables.fold(word))));
			});

		// Safety: every one of the n_chunks slots is initialized above.
		unsafe { values.set_len(n_chunks) };

		if !words_remaining.is_empty() {
			values
				.push(P::from_scalars(words_remaining.iter().map(|&word| self.tables.fold(word))));
		}

		values.resize(capacity, P::default());

		FieldBuffer::new(log_n, values)
	}

	/// Folds the two stored BitAnd operand columns and their derived AND column in one pass.
	///
	/// # Overview
	///
	/// The BitAnd zerocheck folds three columns of the constraint `A & B = C`.
	/// On a satisfying witness the third column equals the AND of the first two.
	/// So this fold reads only the two stored columns and derives the third in registers:
	///
	/// ```text
	///     stream A ──┬──> fold ──> folded A
	///     stream B ──┼──> fold ──> folded B
	///                └──> A & B ──> fold ──> folded C   (no third input stream)
	/// ```
	///
	/// # Returns
	///
	/// Three folded buffers, in order:
	/// - the first operand column, folded as by [`fold`](Self::fold).
	/// - the second operand column, folded the same way.
	/// - the word-by-word AND of the two columns, folded the same way.
	///
	/// The AND column is derived in registers and never written to memory.
	///
	/// # Performance
	///
	/// - Two input streams instead of three.
	/// - Two register ANDs per word pair replace one memory stream.
	/// - The bytewise lookup tables stay hot across all three outputs.
	///
	/// # Preconditions
	///
	/// * The two word-lists have equal length.
	pub fn fold_bitand_operands<P, A>(
		&self,
		alloc: &A,
		a_words: &[Word],
		b_words: &[Word],
	) -> [FieldBuffer<P, A::Vec<P>>; 3]
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		assert_eq!(a_words.len(), b_words.len());

		// Padding contract, mirrored from the single-column fold:
		// the high words up to the next power of two read as zero.
		// `0 & 0 = 0`, so the derived column stays consistent over that padding.
		let log_n = log2_ceil_usize(a_words.len());
		let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);

		// One output buffer per folded column, filled through spare capacity.
		let mut a_values = alloc.alloc::<P>(capacity);
		let mut b_values = alloc.alloc::<P>(capacity);
		let mut c_values = alloc.alloc::<P>(capacity);

		// Phase 1: partition the inputs into full packed-width chunks and a short tail.
		//
		//     words:  [ chunk 0 | chunk 1 | ... | chunk n-1 | tail (< P::WIDTH) ]
		let n_chunks = a_words.len() / P::WIDTH;
		let (a_aligned, a_remaining) = a_words.split_at(n_chunks * P::WIDTH);
		let (b_aligned, b_remaining) = b_words.split_at(n_chunks * P::WIDTH);

		let a_out = &mut a_values.spare_capacity_mut()[..n_chunks];
		let b_out = &mut b_values.spare_capacity_mut()[..n_chunks];
		let c_out = &mut c_values.spare_capacity_mut()[..n_chunks];

		// Phase 2: fold the aligned chunks in parallel.
		// Each task owns one chunk of both inputs and writes one packed element per output.
		(
			a_out,
			b_out,
			c_out,
			a_aligned.par_chunks_exact(P::WIDTH),
			b_aligned.par_chunks_exact(P::WIDTH),
		)
			.into_par_iter()
			// One item is one packed element of each output, so the floor converts from words.
			.with_min_len(MIN_WORDS_PER_TASK.div_ceil(P::WIDTH))
			.for_each(|(a_i, b_i, c_i, a_chunk, b_chunk)| {
				// Safety:
				// - both aligned slices have length n_chunks * P::WIDTH
				// - both are split into P::WIDTH chunks
				unsafe {
					assert_unchecked(a_chunk.len() == P::WIDTH);
					assert_unchecked(b_chunk.len() == P::WIDTH);
				}
				// Fold each stored column by bytewise table lookup.
				a_i.write(P::from_scalars(a_chunk.iter().map(|&word| self.tables.fold(word))));
				b_i.write(P::from_scalars(b_chunk.iter().map(|&word| self.tables.fold(word))));
				// Derive the third column in registers, then fold it the same way.
				c_i.write(P::from_scalars(
					iter::zip(a_chunk, b_chunk).map(|(&a, &b)| self.tables.fold(a & b)),
				));
			});

		// Safety: every one of the n_chunks slots of each vector is initialized above.
		unsafe {
			a_values.set_len(n_chunks);
			b_values.set_len(n_chunks);
			c_values.set_len(n_chunks);
		}

		// Phase 3: fold the short tail into one final packed element per output.
		if !a_remaining.is_empty() {
			a_values.push(P::from_scalars(a_remaining.iter().map(|&word| self.tables.fold(word))));
			b_values.push(P::from_scalars(b_remaining.iter().map(|&word| self.tables.fold(word))));
			c_values.push(P::from_scalars(
				iter::zip(a_remaining, b_remaining).map(|(&a, &b)| self.tables.fold(a & b)),
			));
		}

		// Phase 4: zero-pad each output up to the power-of-two capacity.
		[a_values, b_values, c_values].map(|mut values| {
			values.resize(capacity, P::default());
			FieldBuffer::new(log_n, values)
		})
	}
}

/// Folds one chunk of words into the accumulator, scaled by that chunk's weight.
///
/// Words are 64-bit rows, so a chunk is 64 of them and the columns are the 64 bit positions.
/// A word and a 64-bit row of single-bit scalars share one underlier, so the view below is free.
fn accumulate_word_chunk<F: BinaryField>(
	chunk: &[Word; CHUNK_SIZE],
	tables: &RowFoldTables<F, { Word::BYTES }>,
	weight: F,
	acc: &mut FoldedWord<F>,
) {
	// Reshape the chunk into one contiguous group of eight rows per table.
	let groups = bytemuck::must_cast_ref::<
		[Word; CHUNK_SIZE],
		[[PackedBinaryField64x1b; WEIGHTS_PER_TABLE]; Word::BYTES],
	>(chunk);

	// Sum every group's contribution before scaling, so the chunk costs one multiply per column.
	let mut sums = ColumnSums::zero();
	tables.fold_into(groups.iter().copied(), &mut sums);
	sums.add_scaled_to(weight, acc);
}

/// Computes the bitwise fold of the word vector with a tensor product, by bit position.
///
/// This computes a binary matrix multiplication of the word matrix by the tensor expansion of the
/// point, but transposed from the order of [`fold_words`]. For $n$ challenges, and $2^n$ words,
/// this computes a vector of `F` elements, where the entry at index $i$ is the inner product of the
/// tensor expansion of the point and the bits at position $i$ across the words.
///
/// This builds the folding tables and then folds one list, running parallel over that list's
/// chunks. A caller folding several lists against one point should build the folder once and reuse
/// it, rather than calling this repeatedly.
///
/// A list shorter than the word axis reads its missing high rows as zero.
///
/// ## Preconditions
///
/// * `words.len() <= 1 << point.len()`
pub fn fold_across_words<F>(words: &[Word], point: &[F]) -> FoldedWord<F>
where
	F: BinaryField,
{
	WordFolder::new(point).fold_par(words)
}

/// A reusable [Method of Four Russians] folder over a fixed evaluation point.
///
/// [`fold_across_words`] folds one word-list per call and rebuilds its point tables each time.
/// Many word-lists often share one point, and then those tables can be built once and reused.
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
	/// Each later [`fold`](Self::fold) call folds a list of at most `2^point.len()` words against
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
		let prefix_expansion = OneCube::eq_ind_partial_eval_scalars::<F>(prefix);
		let lookups = RowFoldTables::new(&prefix_expansion);

		// One weight per chunk of CHUNK_SIZE words, from the suffix.
		let suffix_weights = OneCube::eq_ind_partial_eval::<F>(suffix);

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
			accumulate_word_chunk(chunk, &self.lookups, suffix_weight, &mut folded);
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
					accumulate_word_chunk(chunk, &self.lookups, suffix_weight, &mut acc);
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

	/// Accumulates the chunk the list ends in, completed with its zero rows.
	///
	/// A list whose length is a whole number of chunks has no such chunk, and this does nothing.
	///
	/// # Arguments
	///
	/// * `tail` - the words after the last whole chunk, fewer than one chunk of them
	/// * `n_whole_chunks` - how many whole chunks came before, which selects the weight to use
	/// * `folded` - the accumulator the tail's contribution is added into
	fn accumulate_tail(&self, tail: &[Word], n_whole_chunks: usize, folded: &mut [F; Word::BITS]) {
		if tail.is_empty() {
			return;
		}

		// Rows past the list's end read as zero, which contributes nothing to any bit position.
		let mut chunk = [Word::ZERO; CHUNK_SIZE];
		chunk[..tail.len()].copy_from_slice(tail);
		accumulate_word_chunk(
			&chunk,
			&self.lookups,
			self.suffix_weights.get(n_whole_chunks),
			folded,
		);
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, arch::OptimalPackedB128};
	use binius_math::test_utils::random_scalars;
	use binius_utils::checked_arithmetics::checked_log_2;
	use binius_verifier::config::B128;
	use rand::prelude::*;

	use super::*;

	fn naive_fold_words<F, P>(words: &[Word], vec: &[F]) -> FieldBuffer<P>
	where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		assert_eq!(vec.len(), Word::BITS);
		assert!(words.len().is_power_of_two());

		let log_n = checked_log_2(words.len());

		let values = words
			.par_chunks(P::WIDTH)
			.map(|word_chunk| {
				P::from_scalars(word_chunk.iter().map(|&word| {
					// Decompose word into bits and compute inner product
					let mut sum = F::ZERO;
					for bit_idx in 0..Word::BITS {
						if (word.as_u64() >> bit_idx) & 1 == 1 {
							sum += vec[bit_idx];
						}
					}
					sum
				}))
			})
			.collect();

		FieldBuffer::new(log_n, values)
	}

	#[test]
	fn test_fold_words_equivalence() {
		let mut rng = StdRng::seed_from_u64(0);

		let log_n = 6;
		let n_words = 1 << log_n;

		let words = (0..n_words)
			.map(|_| Word::from_u64(rng.random()))
			.collect::<Vec<_>>();

		let vec = random_scalars(&mut rng, Word::BITS);

		// Compute using both methods
		let result_optimized = fold_words::<B128, B128, _>(&GlobalAllocator, &words, &vec);
		let result_naive = naive_fold_words::<B128, B128>(&words, &vec);

		// Compare results
		assert_eq!(result_optimized, result_naive);
	}

	#[test]
	fn test_fold_bitand_operands_matches_separate_folds() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the fused three-output fold equals three independent single-column folds.
		//
		//     fused(A, B)  ==  [ fold(A), fold(B), fold(A & B) ]
		//
		// The single-column fold is itself pinned to a naive reference elsewhere in this module.
		//
		// Fixture state: word counts crossing every regime of the fused kernel.
		//
		//     0             → empty input, output is one zero element
		//     1             → tail only, no aligned chunk
		//     width         → exactly one aligned chunk, no tail
		//     width + 1     → aligned chunk plus tail
		//     4*width       → several aligned chunks
		//     4*width + 3   → several chunks plus tail
		//     40            → non-power-of-two, exercises the zero padding
		let width = OptimalPackedB128::WIDTH;
		for n_words in [0, 1, width, width + 1, 4 * width, 4 * width + 3, 40] {
			// Two random operand columns of the chosen length.
			let a_words = (0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();
			let b_words = (0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();
			// The reference third column, materialized word-by-word.
			let c_words = iter::zip(&a_words, &b_words)
				.map(|(&a, &b)| a & b)
				.collect::<Vec<_>>();

			// One random bit-weight vector shared by all folds.
			let vec = random_scalars::<B128>(&mut rng, Word::BITS);
			let folder = BitAxisFolder::new(&vec);

			// Fold the two stored columns and the derived column in one fused pass.
			let [a_fused, b_fused, c_fused] = folder.fold_bitand_operands::<OptimalPackedB128, _>(
				&GlobalAllocator,
				&a_words,
				&b_words,
			);
			// Each fused output must equal the independent single-column fold.
			assert_eq!(
				a_fused,
				folder.fold(&GlobalAllocator, &a_words),
				"a mismatch at n_words = {n_words}"
			);
			assert_eq!(
				b_fused,
				folder.fold(&GlobalAllocator, &b_words),
				"b mismatch at n_words = {n_words}"
			);
			assert_eq!(
				c_fused,
				folder.fold(&GlobalAllocator, &c_words),
				"c mismatch at n_words = {n_words}"
			);
		}
	}

	fn naive_fold_across_words<F: BinaryField>(words: &[Word], point: &[F]) -> [F; Word::BITS] {
		assert_eq!(words.len(), 1 << point.len());

		let eq = OneCube::eq_ind_partial_eval_scalars(point);
		let mut out = [F::ZERO; Word::BITS];
		for (word, &weight) in iter::zip(words, &eq) {
			for (bit_idx, out_i) in out.iter_mut().enumerate() {
				if (word.as_u64() >> bit_idx) & 1 == 1 {
					*out_i += weight;
				}
			}
		}
		out
	}

	#[test]
	fn test_fold_across_words_equivalence() {
		let mut rng = StdRng::seed_from_u64(0);

		// Cover chunks smaller than, equal to, and larger than CHUNK_SIZE.
		for log_n in [
			0,
			1,
			3,
			LOG_CHUNK_SIZE,
			LOG_CHUNK_SIZE + 1,
			LOG_CHUNK_SIZE + 4,
		] {
			let n_words = 1 << log_n;

			let words = (0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();
			let point = random_scalars::<B128>(&mut rng, log_n);

			let result_optimized = fold_across_words(&words, &point);
			let result_naive = naive_fold_across_words(&words, &point);

			assert_eq!(result_optimized, result_naive, "mismatch at log_n = {log_n}");
		}
	}

	// A word list shorter than the word axis folds as the same list zero-padded up to it, through
	// both the sequential folder and the parallel `fold_across_words`.
	//
	// The naive reference is only defined on a full axis, so it is the padded side here; the point
	// of the test is that the short side reaches the same value without materializing the padding.
	#[test]
	fn word_folder_folds_a_short_list_as_if_zero_padded() {
		let mut rng = StdRng::seed_from_u64(0);

		// (log_rows, n_words) covering: a sub-chunk list in a one-chunk axis, a non-power-of-two
		// list straddling the chunk boundary, a list filling whole chunks of a wider axis, a list
		// short of a whole chunk in a wider axis, and the empty list.
		for (log_rows, n_words) in [
			(LOG_CHUNK_SIZE, 1),
			(LOG_CHUNK_SIZE, 40),
			(LOG_CHUNK_SIZE + 2, 2 * CHUNK_SIZE),
			(LOG_CHUNK_SIZE + 2, 2 * CHUNK_SIZE + 5),
			(LOG_CHUNK_SIZE, 0),
		] {
			let words = (0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();
			let point = random_scalars::<B128>(&mut rng, log_rows);

			let mut padded = words.clone();
			padded.resize(1 << log_rows, Word::ZERO);

			let expected = naive_fold_across_words(&padded, &point);

			let folder = WordFolder::new(&point);
			assert_eq!(
				folder.fold(&words),
				expected,
				"short WordFolder::fold differs at log_rows = {log_rows}, n_words = {n_words}"
			);
			assert_eq!(
				fold_across_words(&words, &point),
				expected,
				"short fold_across_words differs at log_rows = {log_rows}, n_words = {n_words}"
			);
		}
	}

	#[test]
	fn parallel_driver_matches_sequential_driver() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: dividing the chunk axis across workers changes nothing about the result.
		// Field addition is associative and exact, so any grouping of the chunk sums agrees.
		//
		//     sequential: (((c_0 + c_1) + c_2) + c_3)
		//     parallel  : ((c_0 + c_1) + (c_2 + c_3))
		//     → identical, because the merge order is the only difference
		//
		// Fixture state: word counts crossing every regime of both drivers.
		//
		//     0                 → empty list, no chunks and no tail
		//     1                 → tail only, no whole chunk
		//     CHUNK_SIZE        → exactly one whole chunk, no tail
		//     CHUNK_SIZE + 1    → one whole chunk plus a tail
		//     5 * CHUNK_SIZE    → several whole chunks, no tail
		//     5 * CHUNK_SIZE+17 → several whole chunks plus a tail
		let log_rows = LOG_CHUNK_SIZE + 3;
		for n_words in [
			0,
			1,
			CHUNK_SIZE,
			CHUNK_SIZE + 1,
			5 * CHUNK_SIZE,
			5 * CHUNK_SIZE + 17,
		] {
			let words = (0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();
			let point = random_scalars::<B128>(&mut rng, log_rows);

			let folder = WordFolder::new(&point);
			assert_eq!(
				folder.fold_par(&words),
				folder.fold(&words),
				"drivers disagree at n_words = {n_words}"
			);
		}
	}

	#[test]
	fn test_word_folder_fold_matches_naive() {
		let mut rng = StdRng::seed_from_u64(0);

		// The sequential fold driver differs from the parallel one, so pin it to the naive
		// reference. Cover every chunk regime: sub-chunk (log_n < 6), one chunk (log_n = 6), many
		// chunks (> 6).
		for log_n in [
			0,
			1,
			3,
			LOG_CHUNK_SIZE,
			LOG_CHUNK_SIZE + 1,
			LOG_CHUNK_SIZE + 4,
		] {
			let n_words = 1 << log_n;

			let words = (0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();
			let point = random_scalars::<B128>(&mut rng, log_n);

			let result_folder = WordFolder::new(&point).fold(&words);
			let result_naive = naive_fold_across_words(&words, &point);

			assert_eq!(result_folder, result_naive, "mismatch at log_n = {log_n}");
		}
	}
}
