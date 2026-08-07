// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, borrow::Cow, hint::assert_unchecked, iter, ops::BitXor};

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::{
	BinaryField, Divisible, PackedBinaryField64x1b, PackedField, UnderlierType, WideMul,
	WithUnderlier,
	linear_transformation::{
		BytewiseLookupTransformationFactory, LinearTransformationFactory,
		OutputWrappingTransformation, OutputWrappingTransformationFactory, Transformation,
	},
	util::{expand_subset_sums_array, expand_subset_xors},
};
use binius_math::{
	FieldBuffer, FieldSlice,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
};
use binius_utils::{
	checked_arithmetics::{checked_log_2, log2_ceil_usize},
	rayon::prelude::*,
};
use binius_verifier::config::B1;

/// Base-2 logarithm of the number of words folded together within a single chunk.
const LOG_CHUNK_SIZE: usize = Word::LOG_BITS;
/// Number of words folded together within a single chunk.
const CHUNK_SIZE: usize = 1 << LOG_CHUNK_SIZE;
/// Number of bits in a byte; [`fold_across_words`] processes each chunk in groups of this many
/// words, one per byte of the words.
const BITS_PER_BYTE: usize = Word::BITS / Word::BYTES;
/// Base-2 log of the row group a single subset-sum table covers.
///
/// Eight rows is the widest group whose lookup index still fits one byte.
/// One table load then replaces eight conditional additions.
const LOG_BITS_PER_BYTE: usize = BITS_PER_BYTE.ilog2() as usize;

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

/// A [`u64`]-specialized bytewise lookup transformation, folding a word's bits against a fixed
/// vector of field-element underliers.
///
/// Fixing the input to [`u64`] lets the per-byte lookup tables live in a fixed-length array rather
/// than a heap-allocated [`Vec`], holding one table per byte of the word.
///
/// This uses the [Method of Four Russians] to optimize the computation by precomputing a lookup
/// table for each byte position and combining bitwise chunks of the word.
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
#[derive(Debug)]
pub struct WordBytewiseLookupTransformation<UOut> {
	lookup: [[UOut; 1 << BITS_PER_BYTE]; Word::BYTES],
}

impl<UOut: UnderlierType> WordBytewiseLookupTransformation<UOut> {
	pub fn new(cols: &[UOut]) -> Self {
		assert_eq!(cols.len(), Word::BITS);

		let lookup = array::from_fn(|byte| {
			let group: [UOut; BITS_PER_BYTE] = cols
				[byte * BITS_PER_BYTE..(byte + 1) * BITS_PER_BYTE]
				.try_into()
				.expect("cols has Word::BITS = Word::BYTES * BITS_PER_BYTE entries");
			expand_subset_xors(group)
		});

		Self { lookup }
	}
}

impl<UOut: UnderlierType> Transformation<u64, UOut> for WordBytewiseLookupTransformation<UOut> {
	#[inline]
	fn transform(&self, data: &u64) -> UOut {
		iter::zip(Divisible::<u8>::ref_iter(data), &self.lookup)
			.map(|(byte, table)| table[byte as usize])
			.reduce(BitXor::bitxor)
			.unwrap_or(UOut::ZERO)
	}
}

/// Factory for creating [`WordBytewiseLookupTransformation`]s.
#[derive(Debug)]
pub struct WordBytewiseLookupTransformationFactory;

impl<UOut: UnderlierType> LinearTransformationFactory<u64, UOut>
	for WordBytewiseLookupTransformationFactory
{
	type Transform = WordBytewiseLookupTransformation<UOut>;

	fn create(&self, cols: &[UOut]) -> Self::Transform {
		WordBytewiseLookupTransformation::new(cols)
	}
}

/// The concrete transform [`BitAxisFolder`] folds each word through: the [`u64`]-specialized
/// bytewise lookup, wrapped to output field elements of `F`.
type BitAxisTransform<F> = OutputWrappingTransformation<
	WordBytewiseLookupTransformation<<F as WithUnderlier>::Underlier>,
	u64,
	F,
>;

/// A reusable folder over a fixed vector of bit-index scalars, the [`fold_words`] analogue of
/// [`WordFolder`].
///
/// [`fold_words`] rebuilds its Method of Four Russians lookup transform on every call. A caller
/// folding several word-lists against the same scalar vector can instead build the transform once
/// with [`new`](Self::new) and reuse it across [`fold`](Self::fold) calls.
pub struct BitAxisFolder<F: BinaryField> {
	transform: BitAxisTransform<F>,
}

impl<F: BinaryField> BitAxisFolder<F> {
	/// Builds the folding transform for `vec`.
	///
	/// ## Preconditions
	/// * `vec` contains exactly [`Word::BITS`] elements
	pub fn new(vec: &[F]) -> Self {
		let transform =
			OutputWrappingTransformationFactory::new(WordBytewiseLookupTransformationFactory)
				.create(vec);
		Self { transform }
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
			.for_each(|(out, word_chunk)| {
				// Safety:
				// - words_aligned has length that is a multiple of P::WIDTH
				// - words_aligned is split into P::WIDTH chunks
				unsafe { assert_unchecked(word_chunk.len() == P::WIDTH) };
				out.write(P::from_scalars(
					word_chunk
						.iter()
						.map(|&word| self.transform.transform(&word.0)),
				));
			});

		// Safety: every one of the n_chunks slots is initialized above.
		unsafe { values.set_len(n_chunks) };

		if !words_remaining.is_empty() {
			values.push(P::from_scalars(
				words_remaining
					.iter()
					.map(|&word| self.transform.transform(&word.0)),
			));
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
			.for_each(|(a_i, b_i, c_i, a_chunk, b_chunk)| {
				// Safety:
				// - both aligned slices have length n_chunks * P::WIDTH
				// - both are split into P::WIDTH chunks
				unsafe {
					assert_unchecked(a_chunk.len() == P::WIDTH);
					assert_unchecked(b_chunk.len() == P::WIDTH);
				}
				// Fold each stored column by bytewise table lookup.
				a_i.write(P::from_scalars(
					a_chunk
						.iter()
						.map(|&word| self.transform.transform(&word.0)),
				));
				b_i.write(P::from_scalars(
					b_chunk
						.iter()
						.map(|&word| self.transform.transform(&word.0)),
				));
				// Derive the third column in registers, then fold it the same way.
				c_i.write(P::from_scalars(
					iter::zip(a_chunk, b_chunk)
						.map(|(&a, &b)| self.transform.transform(&(a & b).0)),
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
			a_values.push(P::from_scalars(
				a_remaining
					.iter()
					.map(|&word| self.transform.transform(&word.0)),
			));
			b_values.push(P::from_scalars(
				b_remaining
					.iter()
					.map(|&word| self.transform.transform(&word.0)),
			));
			c_values.push(P::from_scalars(
				iter::zip(a_remaining, b_remaining)
					.map(|(&a, &b)| self.transform.transform(&(a & b).0)),
			));
		}

		// Phase 4: zero-pad each output up to the power-of-two capacity.
		[a_values, b_values, c_values].map(|mut values| {
			values.resize(capacity, P::default());
			FieldBuffer::new(log_n, values)
		})
	}
}

/// Folds a slice of words along both axes at once, contracting the matrix to a single scalar.
///
/// The words form a matrix over GF(2): row `i` is `words[i]`, column `b` is bit position `b`.
/// The result is the bilinear form
///
/// ```text
/// out = sum_i sum_b bit_b(words[i]) * index_scalars[b] * row_scalars[i]
/// ```
///
/// reading a clear bit as zero and a set bit as one.
///
/// - [`fold_words`] contracts only the bit-index axis, giving one scalar per word.
/// - [`fold_across_words`] contracts only the word axis, giving one scalar per bit position.
/// - This contracts both axes, giving a single scalar.
///
/// A `words` slice shorter than `row_scalars` reads the missing high rows as zero.
///
/// ## Preconditions
///
/// * `index_scalars.len()` is exactly [`Word::BITS`]
/// * `words.len()` is less than or equal to `row_scalars.len()`
pub fn fold_words_both_axes<F, P>(
	words: &[Word],
	index_scalars: &[F],
	row_scalars: FieldSlice<P>,
) -> F
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	assert_eq!(index_scalars.len(), Word::BITS);
	assert!(words.len() <= row_scalars.len());

	// Build the Method of Four Russians transform from the bit-index scalars, as `fold_words` does.
	// Each word then folds to one scalar by bytewise table lookup.
	let transform = OutputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory)
		.create(index_scalars);

	// Fold each chunk to a packed element, then wide-multiply against the matching row element.
	// Alignment: chunk `c` spans words `c*WIDTH .. (c+1)*WIDTH`, which is exactly packed row
	// element `c`.
	//
	// The zip stops at the shorter word side.
	// Dropped trailing row scalars pair with zero words, so they add nothing.
	let wide = words
		.par_chunks(P::WIDTH)
		.zip(row_scalars.as_ref().par_iter())
		.map(|(word_chunk, &row_i)| {
			let folded =
				P::from_scalars(word_chunk.iter().map(|&word| transform.transform(&word.0)));
			P::wide_mul(folded, row_i)
		})
		.reduce(<P as WideMul>::Output::default, |lhs, rhs| lhs + rhs);

	// One reduction closes the deferred products.
	// Summing the lanes collapses the packed inner product to the scalar `out` above.
	P::reduce(wide).iter().sum()
}

/// Folds one chunk of words into the accumulator, scaled by that chunk's weight.
///
/// Words are 64-bit rows, so a chunk is 64 of them and the columns are the 64 bit positions.
/// A word and a 64-bit row of single-bit scalars share one underlier, so the view below is free.
fn accumulate_word_chunk<F: BinaryField>(
	chunk: &[Word; CHUNK_SIZE],
	tables: &[[F; 1 << BITS_PER_BYTE]; Word::BYTES],
	weight: F,
	acc: &mut [F; Word::BITS],
) {
	// Reshape the chunk into one contiguous group of eight rows per table.
	let groups = bytemuck::must_cast_ref::<
		[Word; CHUNK_SIZE],
		[[PackedBinaryField64x1b; BITS_PER_BYTE]; Word::BYTES],
	>(chunk);

	// Accumulate every group into one column accumulator before scaling it.
	let mut columns = [[F::ZERO; BITS_PER_BYTE]; Word::BYTES];
	for (group, table) in iter::zip(groups, tables) {
		fold_row_group(group, table, &mut columns);
	}

	// Scale once per column and merge, unpacking the nesting into bit-position order.
	for (i, group) in columns.iter().enumerate() {
		for (j, &column) in group.iter().enumerate() {
			acc[(i << LOG_BITS_PER_BYTE) | j] += column * weight;
		}
	}
}

/// Computes the bitwise fold of the word vector with a tensor product, by bit position.
///
/// This computes a binary matrix multiplication of the word matrix by the tensor expansion of the
/// point, but transposed from the order of [`fold_words`]. For $n$ challenges, and $2^n$ words,
/// this computes a vector of `F` elements, where the entry at index $i$ is the inner product of the
/// tensor expansion of the point and the bits at position $i$ across the words.
///
/// Like [`fold_words`], this uses the [Method of Four Russians] to fold groups of words via
/// precomputed lookup tables. The point is split into a `LOG_CHUNK_SIZE`-coordinate prefix and a
/// suffix: the prefix tensor expansion is folded into each chunk of `CHUNK_SIZE` words by lookup,
/// and the suffix tensor expansion scales each chunk's contribution before the chunks are summed.
///
/// A list shorter than the word axis reads its missing high rows as zero, exactly as
/// [`WordFolder::fold`] does.
///
/// ## Preconditions
///
/// * `words.len() <= 1 << point.len()`
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
pub fn fold_across_words<F, P>(words: &[Word], point: &[F]) -> [F; Word::BITS]
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	assert!(words.len() <= 1 << point.len());

	// Build the point tables, then fold the one word-list over its chunks in parallel.
	// A single list can span many chunks (up to 2^20 words in the benchmark).
	// Parallelizing over the chunk axis is what keeps a lone fold fast.
	let folder = WordFolder::<F>::new(point);
	let (chunks, tail) = words.as_chunks::<CHUNK_SIZE>();

	// Each chunk contributes to every bit position, scaled by that chunk's suffix weight. Summing
	// the per-chunk accumulators contracts the word axis. Weights past the list's end pair with
	// absent rows, so the zip drops them.
	let folded_chunks = chunks
		.par_iter()
		.zip(folder.suffix_weights.as_ref().par_iter())
		.map(|(chunk, &suffix_weight)| {
			let mut acc = [F::ZERO; Word::BITS];
			accumulate_word_chunk(chunk, &folder.lookups, suffix_weight, &mut acc);
			acc
		})
		.reduce(
			|| [F::ZERO; Word::BITS],
			|mut lhs, rhs| {
				for (lhs_i, rhs_i) in iter::zip(&mut lhs, rhs) {
					*lhs_i += rhs_i;
				}
				lhs
			},
		);

	// The chunk the list ends in, completed with its zero rows.
	if tail.is_empty() {
		return folded_chunks;
	}

	let mut chunk = [Word::ZERO; CHUNK_SIZE];
	chunk[..tail.len()].copy_from_slice(tail);

	let mut folded = folded_chunks;
	accumulate_word_chunk(
		&chunk,
		&folder.lookups,
		folder.suffix_weights.get(chunks.len()),
		&mut folded,
	);
	folded
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
pub struct WordFolder<F: BinaryField> {
	/// One 256-entry subset-sum table per byte of a word, from the prefix expansion.
	///
	/// Table `s` folds the words at positions `s * BITS_PER_BYTE + t` within a chunk.
	/// Each such word is weighted by prefix-expansion entry `t` of that group.
	lookups: [[F; 1 << BITS_PER_BYTE]; Word::BYTES],
	/// One weight per chunk of `CHUNK_SIZE` words, from the suffix expansion.
	suffix_weights: FieldBuffer<F>,
	/// The word axis's length, which each [`fold`](Self::fold) call's list fits in:
	/// `2^point.len()`.
	n_words: usize,
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
		let prefix_expansion = eq_ind_partial_eval_scalars::<F>(prefix);
		let lookups = row_fold_tables::<F, { Word::BYTES }>(&prefix_expansion);

		// One weight per chunk of CHUNK_SIZE words, from the suffix.
		let suffix_weights = eq_ind_partial_eval::<F>(suffix);

		Self {
			lookups,
			suffix_weights,
			n_words: 1 << point.len(),
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
	/// This runs sequentially over the list's chunks.
	/// A caller folding many lists against one point should parallelize across the lists instead.
	///
	/// A list shorter than the word axis reads its missing high rows as zero: an absent row's
	/// weight multiplies nothing, so it contributes nothing to any bit position. Chunks lying
	/// entirely past the list's end are therefore never visited at all.
	///
	/// ## Preconditions
	///
	/// * `words.len() <= 1 << point.len()`
	pub fn fold(&self, words: &[Word]) -> [F; Word::BITS] {
		assert!(words.len() <= self.n_words, "words.len() must not exceed 2^point.len()");

		let (chunks, tail) = words.as_chunks::<CHUNK_SIZE>();
		let mut folded = [F::ZERO; Word::BITS];

		// Accumulate each chunk's contribution, scaled by that chunk's suffix weight. Weights past
		// the list's end pair with absent rows, so the zip drops them.
		for (chunk, &suffix_weight) in iter::zip(chunks, self.suffix_weights.as_ref()) {
			accumulate_word_chunk(chunk, &self.lookups, suffix_weight, &mut folded);
		}

		// The chunk the list ends in, completed with its zero rows.
		if !tail.is_empty() {
			let mut chunk = [Word::ZERO; CHUNK_SIZE];
			chunk[..tail.len()].copy_from_slice(tail);
			accumulate_word_chunk(
				&chunk,
				&self.lookups,
				self.suffix_weights.get(chunks.len()),
				&mut folded,
			);
		}

		folded
	}
}

/// Builds the subset-sum tables of a bitwise row fold.
///
/// # Overview
///
/// A row fold contracts a matrix over GF(2) against one weight per row:
///
/// ```text
///     out[b] = sum_r weight[r] * bit_b(row[r])
/// ```
///
/// Taking eight rows at a time turns that inner sum into a single table lookup.
/// Table `g` covers rows `8g .. 8g+8` and holds every subset sum of their eight weights.
/// A byte carrying those eight rows' bits at one column then indexes their contribution directly.
///
/// # Arguments
///
/// * `weights` - one weight per row, from the first row onwards
///
/// # Why short input is allowed
///
/// Weights past the end of the slice are read as zero.
/// They would weight rows past the end of a chunk, which are themselves read as zero.
/// So a zero weight and its absent row contribute nothing either way.
/// This is what lets one table layout serve a chunk that the row list does not fill.
pub(crate) fn row_fold_tables<F: BinaryField, const N_TABLES: usize>(
	weights: &[F],
) -> [[F; 1 << BITS_PER_BYTE]; N_TABLES] {
	array::from_fn(|group| {
		// Weights of the eight rows this table covers, zero where the slice has run out.
		// A group beyond the end of the slice starts at its end, so it copies nothing.
		let mut group_weights = [F::ZERO; BITS_PER_BYTE];
		let start = (group * BITS_PER_BYTE).min(weights.len());
		let available = (weights.len() - start).min(BITS_PER_BYTE);
		group_weights[..available].copy_from_slice(&weights[start..start + available]);

		// Enumerate all 256 subset sums, so any byte of set bits indexes its sum in one load.
		expand_subset_sums_array(group_weights)
	})
}

/// Folds one group of eight rows into a column accumulator.
///
/// # Overview
///
/// Rows arrive one bit per scalar, so a row is one packed element and a column is a scalar index.
/// One table covers one group, holding every subset sum of that group's eight row weights.
///
/// # Algorithm
///
/// Transposing the group exchanges its row axis with the low three bits of the column index:
///
/// ```text
///     before:  element r, bit 8i + j  =  row r, column 8i + j
///     after:   element j, bit 8i + t  =  row t, column 8i + j
/// ```
///
/// So byte `i` of element `j` then carries the eight rows' bits at column `8i + j`.
/// One lookup of that byte yields those rows' whole contribution to that column.
///
/// The accumulator is nested to match, as `[byte of column index][low three bits]`.
/// Reading that nesting in order walks the columns in order.
///
/// # Preconditions
///
/// * The row width in bits must equal eight times the accumulator's outer length.
/// * Rows past the end of the matrix must be passed as zero, which contributes nothing.
#[inline]
pub(crate) fn fold_row_group<F, PB, const N_TABLES: usize>(
	rows: &[PB; BITS_PER_BYTE],
	table: &[F; 1 << BITS_PER_BYTE],
	acc: &mut [[F; BITS_PER_BYTE]; N_TABLES],
) where
	F: BinaryField,
	PB: PackedField<Scalar = B1> + WithUnderlier,
	PB::Underlier: Divisible<u8>,
{
	// One byte of a row per table is what makes the nesting below line up with the columns.
	const {
		assert!(PB::WIDTH == BITS_PER_BYTE * N_TABLES, "the row width must be one byte per table");
	}

	// The transpose consumes its input, so work on a copy and leave the caller's rows intact.
	let mut group = *rows;
	square_transpose_const_size::<PB, LOG_BITS_PER_BYTE, BITS_PER_BYTE>(&mut group);

	for (j, row) in group.iter().enumerate() {
		// Byte `i` holds this group's bits at column `8i + j`, so it indexes that column's sum.
		for (i, byte) in Divisible::<u8>::value_iter(row.to_underlier()).enumerate() {
			acc[i][j] += table[byte as usize];
		}
	}
}

/// Transposes square blocks of scalars across an array of packed elements, in place.
///
/// # Overview
///
/// View the array as a matrix of elements by scalar positions.
/// This exchanges the element axis with the low `LOG_N` bits of the scalar position.
///
/// Const generic sizes let the compiler unroll the butterfly and keep the whole array in registers.
///
/// # Algorithm
///
/// A butterfly network over `LOG_N` rounds, as in Hacker's Delight, Section 7-3.
/// Round `i` interleaves element pairs `2^(log_w + i)` apart at block granularity `2^i`.
///
/// # Preconditions
///
/// All three are checked at compile time, so a violating instantiation fails to build:
///
/// * The array length must be a power of two.
/// * `LOG_N` must not exceed the base-2 log of the array length.
/// * `LOG_N` must not exceed the base-2 log of the packed width.
pub(crate) fn square_transpose_const_size<P: PackedField, const LOG_N: usize, const S: usize>(
	elems: &mut [P; S],
) {
	const {
		assert!(LOG_N <= P::LOG_WIDTH, "LOG_N must not exceed the packed width");
		assert!(LOG_N <= checked_log_2(S), "LOG_N must not exceed the array length");
	}

	let log_size = checked_log_2(S);

	// Elements per block that stays contiguous through the butterfly.
	let log_w = log_size - LOG_N;

	for i in 0..LOG_N {
		for j in 0..1 << (LOG_N - i - 1) {
			for k in 0..1 << (log_w + i) {
				// Partner elements for this round, one stride apart.
				let idx0 = (j << (log_w + i + 1)) | k;
				let idx1 = idx0 | (1 << (log_w + i));

				// Interleaving at block granularity 2^i swaps the axes one bit at a time.
				let (v0, v1) = elems[idx0].interleave(elems[idx1], i);
				elems[idx0] = v0;
				elems[idx1] = v1;
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, PackedBinaryField128x1b, arch::OptimalPackedB128};
	use binius_math::test_utils::{random_field_buffer, random_scalars};
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
			.map(|_| Word::from_u64(rng.random::<u64>()))
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
				.map(|_| Word::from_u64(rng.random::<u64>()))
				.collect::<Vec<_>>();
			let b_words = (0..n_words)
				.map(|_| Word::from_u64(rng.random::<u64>()))
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

	fn naive_fold_words_both_axes<F, P>(
		words: &[Word],
		index_scalars: &[F],
		row_scalars: &FieldBuffer<P>,
	) -> F
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
	{
		assert_eq!(index_scalars.len(), Word::BITS);
		assert!(words.len() <= row_scalars.len());

		// Contract row by row: fold each word's set bits against `index_scalars`, then weight the
		// per-word scalar by its row scalar and sum. Words beyond `words.len()` are absent (zero).
		let mut out = F::ZERO;
		for (i, &word) in words.iter().enumerate() {
			let mut per_word = F::ZERO;
			for bit_idx in 0..Word::BITS {
				if (word.as_u64() >> bit_idx) & 1 == 1 {
					per_word += index_scalars[bit_idx];
				}
			}
			out += per_word * row_scalars.get(i);
		}
		out
	}

	#[test]
	fn test_fold_words_both_axes_equivalence() {
		let mut rng = StdRng::seed_from_u64(0);

		// (log_rows, n_words) covering: single element, full chunk, shorter power-of-two list,
		// non-power-of-two list with a partial trailing chunk, a multi-chunk partial list, and the
		// empty list.
		for (log_rows, n_words) in [
			(0, 1),
			(LOG_CHUNK_SIZE, 1 << LOG_CHUNK_SIZE),
			(LOG_CHUNK_SIZE, 1 << 3),
			(LOG_CHUNK_SIZE, 40),
			(LOG_CHUNK_SIZE + 2, (1 << (LOG_CHUNK_SIZE + 2)) - 3),
			(3, 0),
		] {
			let words = (0..n_words)
				.map(|_| Word::from_u64(rng.random::<u64>()))
				.collect::<Vec<_>>();
			let index_scalars = random_scalars::<B128>(&mut rng, Word::BITS);
			let row_scalars = random_field_buffer::<OptimalPackedB128>(&mut rng, log_rows);

			let result_optimized = fold_words_both_axes::<_, OptimalPackedB128>(
				&words,
				&index_scalars,
				row_scalars.to_ref(),
			);
			let result_naive = naive_fold_words_both_axes(&words, &index_scalars, &row_scalars);

			assert_eq!(
				result_optimized, result_naive,
				"mismatch at log_rows = {log_rows}, n_words = {n_words}"
			);
		}
	}

	fn naive_fold_across_words<F: BinaryField>(words: &[Word], point: &[F]) -> [F; Word::BITS] {
		assert_eq!(words.len(), 1 << point.len());

		let eq = eq_ind_partial_eval_scalars(point);
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

	// Both row folds read a transposed group of eight rows the same way: byte `i` of element `j`
	// carries those rows' bits at column `8i + j`. That reading is what makes one byte a whole
	// lookup index, so pin the permutation directly.
	//
	//     input :  element r, bit 8i + j  =  row r, column 8i + j
	//     output:  element j, bit 8i + t  =  row t, column 8i + j
	fn check_group_transpose<PB>(seed: u64)
	where
		PB: PackedField<Scalar = B1> + WithUnderlier,
	{
		let mut rng = StdRng::seed_from_u64(seed);

		// Random bits over many trials cover every one of the 8 * WIDTH positions.
		for _ in 0..100 {
			let input: [PB; BITS_PER_BYTE] = array::from_fn(|_| PB::random(&mut rng));
			let mut output = input;
			square_transpose_const_size::<PB, LOG_BITS_PER_BYTE, BITS_PER_BYTE>(&mut output);

			// Bytes of a row, so the outer index walks the high bits of the column index.
			for i in 0..PB::WIDTH / BITS_PER_BYTE {
				// Element of the transposed group, which is the low three bits of the column.
				for j in 0..BITS_PER_BYTE {
					// Row within the group, which becomes the bit position inside the byte.
					for t in 0..BITS_PER_BYTE {
						let got = output[j].get(i * BITS_PER_BYTE + t);
						let want = input[t].get(i * BITS_PER_BYTE + j);
						assert_eq!(got, want, "i={i}, j={j}, t={t}");
					}
				}
			}
		}
	}

	#[test]
	fn transpose_exchanges_row_axis_with_low_column_bits() {
		// The two row widths the folds run at: 64-bit words and 128-bit field elements.
		check_group_transpose::<PackedBinaryField64x1b>(0);
		check_group_transpose::<PackedBinaryField128x1b>(1);
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
				.map(|_| Word::from_u64(rng.random::<u64>()))
				.collect::<Vec<_>>();
			let point = random_scalars::<B128>(&mut rng, log_n);

			let result_optimized = fold_across_words::<_, OptimalPackedB128>(&words, &point);
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
				.map(|_| Word::from_u64(rng.random::<u64>()))
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
				fold_across_words::<_, OptimalPackedB128>(&words, &point),
				expected,
				"short fold_across_words differs at log_rows = {log_rows}, n_words = {n_words}"
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
				.map(|_| Word::from_u64(rng.random::<u64>()))
				.collect::<Vec<_>>();
			let point = random_scalars::<B128>(&mut rng, log_n);

			let result_folder = WordFolder::new(&point).fold(&words);
			let result_naive = naive_fold_across_words(&words, &point);

			assert_eq!(result_folder, result_naive, "mismatch at log_n = {log_n}");
		}
	}
}
