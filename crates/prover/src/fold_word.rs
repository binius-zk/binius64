// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, hint::assert_unchecked, iter, marker::PhantomData, mem::MaybeUninit};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{
	BinaryField, Divisible, PackedBinaryField64x1b, PackedField, util::expand_subset_sums_array,
};
use binius_math::{FieldBuffer, multilinear::hypercube::Hypercube};
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

/// A field buffer under construction, filled in two stages.
///
/// Stage one writes one packed element per whole chunk of words, in parallel, through spare
/// capacity. Stage two pushes the short tail and zero-fills up to the power-of-two length.
///
/// Owning both stages is what keeps the length claim in one place, so the unchecked write is
/// argued once rather than at every fold.
struct PackedOutput<P, V> {
	values: V,
	capacity: usize,
	log_n: usize,
	_marker: PhantomData<P>,
}

impl<P: PackedField, V: VecLike<P>> PackedOutput<P, V> {
	/// Claims room for a buffer covering `n_words` words, rounded up to a power of two.
	///
	/// A word count that is not a power of two leaves the high words reading as zero.
	fn for_words<A: Allocator<Vec<P> = V>>(alloc: &A, n_words: usize) -> Self {
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
	fn chunk_slots(&mut self, n_chunks: usize) -> &mut [MaybeUninit<P>] {
		&mut self.values.spare_capacity_mut()[..n_chunks]
	}

	/// Closes the slots handed out by the matching call above.
	///
	/// # Safety
	///
	/// Every slot of the matching `n_chunks` must have been written.
	unsafe fn commit_chunks(&mut self, n_chunks: usize) {
		unsafe { self.values.set_len(n_chunks) }
	}

	/// Appends the element the short tail folds to.
	fn push(&mut self, elem: P) {
		self.values.push(elem);
	}

	/// Zero-fills up to the power-of-two length and closes the buffer.
	fn finish(mut self) -> FieldBuffer<P, V> {
		self.values.resize(self.capacity, P::default());
		FieldBuffer::new(self.log_n, self.values)
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
		let mut out = PackedOutput::for_words(alloc, words.len());

		// Partition the words into whole packed-width chunks and a short tail.
		//
		//     words:  [ chunk 0 | chunk 1 | ... | chunk n-1 | tail (< P::WIDTH) ]
		let n_chunks = words.len() / P::WIDTH;
		let (words_aligned, words_remaining) = words.split_at(n_chunks * P::WIDTH);

		let slots = out.chunk_slots(n_chunks);
		let word_chunks = words_aligned.par_chunks_exact(P::WIDTH);
		assert_eq!(slots.len(), word_chunks.len());

		(slots, word_chunks)
			.into_par_iter()
			// One item is one packed element, so the floor converts from words to items.
			.with_min_len(MIN_WORDS_PER_TASK.div_ceil(P::WIDTH))
			.for_each(|(slot, word_chunk)| {
				// Safety:
				// - words_aligned has length that is a multiple of P::WIDTH
				// - words_aligned is split into P::WIDTH chunks
				unsafe { assert_unchecked(word_chunk.len() == P::WIDTH) };
				slot.write(P::from_scalars(word_chunk.iter().map(|&word| self.tables.fold(word))));
			});

		// Safety: the loop above writes every one of the n_chunks slots exactly once.
		unsafe { out.commit_chunks(n_chunks) };

		if !words_remaining.is_empty() {
			out.push(P::from_scalars(words_remaining.iter().map(|&word| self.tables.fold(word))));
		}

		out.finish()
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
		//
		// One output buffer per folded column, filled through spare capacity.
		let [mut a_out, mut b_out, mut c_out] =
			array::from_fn(|_| PackedOutput::for_words(alloc, a_words.len()));

		// Phase 1: partition the inputs into full packed-width chunks and a short tail.
		//
		//     words:  [ chunk 0 | chunk 1 | ... | chunk n-1 | tail (< P::WIDTH) ]
		let n_chunks = a_words.len() / P::WIDTH;
		let (a_aligned, a_remaining) = a_words.split_at(n_chunks * P::WIDTH);
		let (b_aligned, b_remaining) = b_words.split_at(n_chunks * P::WIDTH);

		let a_slots = a_out.chunk_slots(n_chunks);
		let b_slots = b_out.chunk_slots(n_chunks);
		let c_slots = c_out.chunk_slots(n_chunks);

		// Phase 2: fold the aligned chunks in parallel.
		// Each task owns one chunk of both inputs and writes one packed element per output.
		(
			a_slots,
			b_slots,
			c_slots,
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

		// Safety: the loop above writes every one of the n_chunks slots of each output exactly
		// once.
		unsafe {
			a_out.commit_chunks(n_chunks);
			b_out.commit_chunks(n_chunks);
			c_out.commit_chunks(n_chunks);
		}

		// Phase 3: fold the short tail into one final packed element per output.
		if !a_remaining.is_empty() {
			a_out.push(P::from_scalars(a_remaining.iter().map(|&word| self.tables.fold(word))));
			b_out.push(P::from_scalars(b_remaining.iter().map(|&word| self.tables.fold(word))));
			c_out.push(P::from_scalars(
				iter::zip(a_remaining, b_remaining).map(|(&a, &b)| self.tables.fold(a & b)),
			));
		}

		// Phase 4: each output zero-pads itself up to the power-of-two capacity.
		[a_out, b_out, c_out].map(PackedOutput::finish)
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
	use binius_field::{Field, PackedBinaryGhash2x128b, arch::OptimalPackedB128};
	use binius_math::test_utils::random_scalars;
	use binius_verifier::config::B128;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	/// The two folds, written straight from their definitions.
	///
	/// A word list is a matrix over GF(2): row `i` is word `i`, column `b` is bit position `b`.
	/// Each fold contracts one axis of it, and each is one nested loop over set bits.
	mod reference {
		use super::*;

		/// Contracts the bit axis, leaving one element per word.
		///
		/// A list shorter than a power of two reads its high words as zero, so the padded slots
		/// fold to zero and the buffer is the next power of two long.
		pub fn fold_bit_axis<F: Field, P: PackedField<Scalar = F>>(
			words: &[Word],
			weights: &[F],
		) -> FieldBuffer<P> {
			assert_eq!(weights.len(), Word::BITS);

			let log_n = log2_ceil_usize(words.len());
			let scalars = (0..1 << log_n)
				.map(|i| {
					// Absent high words are zero, and a zero word has no set bits to weight.
					words.get(i).map_or(F::ZERO, |word| {
						(0..Word::BITS)
							.filter(|bit| (word.as_u64() >> bit) & 1 == 1)
							.map(|bit| weights[bit])
							.sum()
					})
				})
				.collect::<Vec<_>>();

			FieldBuffer::from_values(&scalars)
		}

		/// Contracts the word axis, leaving one element per bit position.
		///
		/// A list shorter than the axis reads its high rows as zero, which weight nothing.
		pub fn fold_word_axis<F: BinaryField>(words: &[Word], point: &[F]) -> FoldedWord<F> {
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

	// The bit-axis folds split their input at a multiple of the packing width, so the width decides
	// which branches run at all:
	//
	//     width 1 : every word is its own chunk, so the short tail is never taken
	//     width 2 : an odd word count leaves a tail, and the buffer above it is zero-padded
	//
	// The optimal packing is one scalar wide on some targets, so pinning only that would leave the
	// tail and padding paths untested there. Every bit-axis property runs at both widths.
	fn check_bit_axis_fold<P: PackedField<Scalar = B128>>(words: &[Word], weights: &[B128]) {
		assert_eq!(
			fold_words::<_, P, _>(&GlobalAllocator, words, weights),
			reference::fold_bit_axis(words, weights),
			"bit-axis fold differs at P::WIDTH = {}, {} words",
			P::WIDTH,
			words.len()
		);
	}

	fn check_fused_bitand_fold<P: PackedField<Scalar = B128>>(
		a_words: &[Word],
		b_words: &[Word],
		weights: &[B128],
	) {
		let c_words = iter::zip(a_words, b_words)
			.map(|(&a, &b)| a & b)
			.collect::<Vec<_>>();
		let folder = BitAxisFolder::new(weights);

		let [a, b, c] = folder.fold_bitand_operands::<P, _>(&GlobalAllocator, a_words, b_words);
		let width = P::WIDTH;

		assert_eq!(a, folder.fold(&GlobalAllocator, a_words), "a differs at width {width}");
		assert_eq!(b, folder.fold(&GlobalAllocator, b_words), "b differs at width {width}");
		assert_eq!(c, folder.fold(&GlobalAllocator, &c_words), "c differs at width {width}");
	}

	proptest! {
		#[test]
		fn bit_axis_fold_matches_the_definition(n_words in any_n_words(), seed: u64) {
			// Every length, not just the powers of two the old fixture used. A non-power-of-two
			// list exercises the tail element and the zero padding above it, which is the path
			// the fixture never reached.
			let words = words_of(n_words, seed);
			let mut rng = StdRng::seed_from_u64(seed ^ 1);
			let weights = random_scalars::<B128>(&mut rng, Word::BITS);

			check_bit_axis_fold::<OptimalPackedB128>(&words, &weights);
			check_bit_axis_fold::<PackedBinaryGhash2x128b>(&words, &weights);
		}

		#[test]
		fn word_axis_fold_matches_the_definition(n_words in any_n_words(), seed: u64) {
			// The point must cover the list, so its width is the list's rounded-up log.
			let words = words_of(n_words, seed);
			let mut rng = StdRng::seed_from_u64(seed ^ 2);
			let point = random_scalars::<B128>(&mut rng, log2_ceil_usize(words.len()));

			prop_assert_eq!(
				fold_across_words(&words, &point),
				reference::fold_word_axis(&words, &point),
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
		fn fused_bitand_fold_matches_three_separate_folds(n_words in any_n_words(), seed: u64) {
			// The AND-reduction fold derives its third column in registers rather than reading it.
			// That must equal folding a materialized third column.
			//
			//     fused(A, B)  ==  [ fold(A), fold(B), fold(A & B) ]
			let a_words = words_of(n_words, seed);
			let b_words = words_of(n_words, seed ^ 0xff);

			let mut rng = StdRng::seed_from_u64(seed ^ 4);
			let weights = random_scalars::<B128>(&mut rng, Word::BITS);

			check_fused_bitand_fold::<OptimalPackedB128>(&a_words, &b_words, &weights);
			check_fused_bitand_fold::<PackedBinaryGhash2x128b>(&a_words, &b_words, &weights);
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
			let expected = reference::fold_word_axis(&padded, &point);

			prop_assert_eq!(WordFolder::new(&point).fold(&words), expected);
			prop_assert_eq!(fold_across_words(&words, &point), expected);
		}
	}
}
