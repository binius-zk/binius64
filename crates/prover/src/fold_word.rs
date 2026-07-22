// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, borrow::Cow, hint::assert_unchecked, iter, ops::BitXor};

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::{
	BinaryField, Divisible, Field, PackedField, UnderlierType, WideMul, WithUnderlier,
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
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};

/// Base-2 logarithm of the number of words folded together within a single chunk.
const LOG_CHUNK_SIZE: usize = Word::LOG_BITS;
/// Number of words folded together within a single chunk.
const CHUNK_SIZE: usize = 1 << LOG_CHUNK_SIZE;
/// Number of bits in a byte; [`fold_across_words`] processes each chunk in groups of this many
/// words, one per byte of the words.
const BITS_PER_BYTE: usize = Word::BITS / Word::BYTES;

/// Computes a [`FieldBuffer`] where each element is the inner product of the bits of a word and a
/// vector of field elements.
///
/// Returns a buffer where element `i` is the inner product of the bits of word `i` in `words`
/// (mapping bit 0 to [`Field::ZERO`] and bit 1 to [`Field::ONE`]) and the values in `vec`.
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

fn fold_words_with_transform<F, P, T, A>(
	alloc: &A,
	transform: &T,
	words: &[Word],
) -> FieldBuffer<P, A::Vec<P>>
where
	F: Field,
	P: PackedField<Scalar = F>,
	T: Transformation<u64, F>,
	A: Allocator,
{
	// `words` need not have a power-of-two length; the high words up to the next power of two are
	// treated as zero, so the remaining slots after the last real word are zero-filled by resize.
	let log_n = log2_ceil_usize(words.len());
	let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);

	let mut values = alloc.alloc::<P>(capacity);

	let chunk_size = P::WIDTH;
	let n_chunks = words.len() / chunk_size;
	let (words_aligned, words_remaining) = words.split_at(n_chunks * chunk_size);

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
			out.write(P::from_scalars(word_chunk.iter().map(|&word| transform.transform(&word.0))));
		});

	unsafe { values.set_len(n_chunks) };

	if !words_remaining.is_empty() {
		values.push(P::from_scalars(
			words_remaining
				.iter()
				.map(|&word| transform.transform(&word.0)),
		));
	}

	values.resize(capacity, P::default());

	FieldBuffer::new(log_n, values)
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

/// Folds two operand word-lists and their word-by-word bitwise AND in one pass.
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
/// Each output equals [`fold_words_with_transform`] on the corresponding word-list.
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
fn fold_bitand_operands_with_transform<F, P, T, A>(
	alloc: &A,
	transform: &T,
	a_words: &[Word],
	b_words: &[Word],
) -> [FieldBuffer<P, A::Vec<P>>; 3]
where
	F: Field,
	P: PackedField<Scalar = F>,
	T: Transformation<u64, F>,
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
			a_i.write(P::from_scalars(a_chunk.iter().map(|&word| transform.transform(&word.0))));
			b_i.write(P::from_scalars(b_chunk.iter().map(|&word| transform.transform(&word.0))));
			// Derive the third column in registers, then fold it the same way.
			c_i.write(P::from_scalars(
				iter::zip(a_chunk, b_chunk).map(|(&a, &b)| transform.transform(&(a & b).0)),
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
		a_values
			.push(P::from_scalars(a_remaining.iter().map(|&word| transform.transform(&word.0))));
		b_values
			.push(P::from_scalars(b_remaining.iter().map(|&word| transform.transform(&word.0))));
		c_values.push(P::from_scalars(
			iter::zip(a_remaining, b_remaining).map(|(&a, &b)| transform.transform(&(a & b).0)),
		));
	}

	// Phase 4: zero-pad each output up to the power-of-two capacity.
	[a_values, b_values, c_values].map(|mut values| {
		values.resize(capacity, P::default());
		FieldBuffer::new(log_n, values)
	})
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
		fold_words_with_transform(alloc, &self.transform, words)
	}

	/// Folds the two stored BitAnd operand columns and their derived AND column in one pass.
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
	/// # Preconditions
	///
	/// * The two word-lists have equal length.
	pub fn fold_bitand_operands<P, A>(
		&self,
		alloc: &A,
		a: &[Word],
		b: &[Word],
	) -> [FieldBuffer<P, A::Vec<P>>; 3]
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		fold_bitand_operands_with_transform(alloc, &self.transform, a, b)
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
/// ## Preconditions
///
/// * `words.len() == 1 << point.len()`
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
pub fn fold_across_words<F, P>(words: &[Word], point: &[F]) -> [F; Word::BITS]
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	assert_eq!(words.len(), 1 << point.len());

	// Build the point tables, then fold the one word-list over its chunks in parallel.
	// A single list can span many chunks (up to 2^20 words in the benchmark).
	// Parallelizing over the chunk axis is what keeps a lone fold fast.
	let folder = WordFolder::<F>::new(point);
	let chunks = duplicate_to_fixed_chunks::<CHUNK_SIZE>(words);
	debug_assert_eq!(chunks.len(), folder.suffix_weights.as_ref().len());

	// Each chunk folds into an accumulator that is transposed relative to bit position.
	// The entry at `BITS_PER_BYTE * a + j` holds the result for bit position `BITS_PER_BYTE * j +
	// a`. Scale each chunk accumulator by its suffix weight and sum them.
	// Transpose the total once at the end.
	let folded = chunks
		.as_ref()
		.par_iter()
		.zip(folder.suffix_weights.as_ref().par_iter())
		.map(|(chunk, &suffix_weight)| {
			let mut acc = fold_chunk(chunk, &folder.lookups);
			for acc_i in &mut acc {
				*acc_i *= suffix_weight;
			}
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

	transpose_accumulator(folded)
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
	/// The exact word-list length each [`fold`](Self::fold) consumes: `2^point.len()`.
	n_words: usize,
}

impl<F: BinaryField> WordFolder<F> {
	/// Builds the folding tables for `point`.
	///
	/// Each later [`fold`](Self::fold) call folds a `2^point.len()`-word list against this point.
	pub fn new(point: &[F]) -> Self {
		// Split the point into a prefix of at most LOG_CHUNK_SIZE coordinates and a suffix.
		// Zero-pad the prefix up to LOG_CHUNK_SIZE coordinates.
		// Why the padding is harmless:
		//   - a list shorter than one chunk is filled by repeating its words.
		//   - each repeated copy pairs with a zero prefix weight, so it adds nothing.
		let prefix_len = point.len().min(LOG_CHUNK_SIZE);
		let mut prefix = [F::ZERO; LOG_CHUNK_SIZE];
		prefix[..prefix_len].copy_from_slice(&point[..prefix_len]);
		let suffix = &point[prefix_len..];

		// Build one 256-entry subset-sum lookup table per byte of the words.
		// The prefix tensor expansion has CHUNK_SIZE entries, one per word in a chunk.
		// Split those entries into Word::BYTES groups of BITS_PER_BYTE.
		let prefix_expansion = eq_ind_partial_eval_scalars::<F>(&prefix);
		let lookups: [[F; 1 << BITS_PER_BYTE]; Word::BYTES] = array::from_fn(|byte| {
			let group: [F; BITS_PER_BYTE] = prefix_expansion
				[byte * BITS_PER_BYTE..(byte + 1) * BITS_PER_BYTE]
				.try_into()
				.expect("prefix_expansion has CHUNK_SIZE = Word::BYTES * BITS_PER_BYTE entries");
			expand_subset_sums_array(group)
		});

		// The suffix tensor expansion provides one weight per chunk of CHUNK_SIZE words.
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
	/// ## Preconditions
	///
	/// * `words.len() == 1 << point.len()`
	pub fn fold(&self, words: &[Word]) -> [F; Word::BITS] {
		assert_eq!(words.len(), self.n_words, "words.len() must equal 2^point.len()");

		let chunks = duplicate_to_fixed_chunks::<CHUNK_SIZE>(words);
		debug_assert_eq!(chunks.len(), self.suffix_weights.as_ref().len());

		// Accumulate each chunk's contribution, scaled by its suffix weight.
		// The accumulator stays in the chunk kernel's transposed bit layout until the end.
		let mut folded = [F::ZERO; Word::BITS];
		for (chunk, &suffix_weight) in iter::zip(chunks.as_ref(), self.suffix_weights.as_ref()) {
			let acc = fold_chunk(chunk, &self.lookups);
			for (folded_i, acc_i) in iter::zip(&mut folded, acc) {
				*folded_i += acc_i * suffix_weight;
			}
		}

		transpose_accumulator(folded)
	}
}

/// Transposes a reduced chunk accumulator back into bit-position order.
///
/// The chunk kernel stores bit `BITS_PER_BYTE * j + a`'s contribution at index `BITS_PER_BYTE * a +
/// j`. Transposing that `Word::BYTES`-by-`BITS_PER_BYTE` layout restores bit-position order.
fn transpose_accumulator<F: BinaryField>(folded: [F; Word::BITS]) -> [F; Word::BITS] {
	let mut result = [F::ZERO; Word::BITS];
	for a in 0..BITS_PER_BYTE {
		for j in 0..Word::BYTES {
			result[BITS_PER_BYTE * j + a] = folded[BITS_PER_BYTE * a + j];
		}
	}
	result
}

/// Folds a single chunk of [`CHUNK_SIZE`] words against the prefix lookup tables, accumulating into
/// an array that is transposed relative to bit position (before scaling by the suffix weight).
///
/// The entry at index `BITS_PER_BYTE * a + j` holds the contribution for bit position
/// `BITS_PER_BYTE * j + a`, matching the permutation performed by [`transpose_bits`]; the caller
/// transposes the reduced accumulator back into bit-position order.
fn fold_chunk<F: BinaryField>(
	chunk: &[Word; CHUNK_SIZE],
	lookups: &[[F; 1 << BITS_PER_BYTE]; Word::BYTES],
) -> [F; Word::BITS] {
	// Reshape the chunk into one sub-array per lookup table, each holding BITS_PER_BYTE words.
	let subchunks =
		bytemuck::must_cast_ref::<[Word; CHUNK_SIZE], [[Word; BITS_PER_BYTE]; Word::BYTES]>(chunk);

	let mut acc = [F::ZERO; Word::BITS];
	for (subchunk, lookup) in iter::zip(subchunks, lookups) {
		// After the transpose, byte `j` of output word `a` collects bit `BITS_PER_BYTE * j + a`
		// across the BITS_PER_BYTE words of this sub-chunk. Looking it up sums the prefix weights
		// of the words whose bit at that position is set.
		let mut words = *subchunk;
		transpose_bits(&mut words);
		for (a, word) in words.iter().enumerate() {
			for (j, &byte) in word.as_u64().to_le_bytes().iter().enumerate() {
				acc[BITS_PER_BYTE * a + j] += lookup[byte as usize];
			}
		}
	}
	acc
}

/// Bit-transposes the within-byte bit axis and the word axis of [`BITS_PER_BYTE`] words in place.
///
/// Viewing the input and output as `BITS_PER_BYTE`×`Word::BYTES`×`BITS_PER_BYTE` bit matrices
/// where the axes are (bit within a byte, byte within a word, word), this permutes
/// `out[i][j][k] = in[k][j][i]`, leaving the byte-within-word axis `j` unchanged. After the call,
/// byte `j` of word `i` holds, in bit `k`, the value of bit `BITS_PER_BYTE * j + i` of input word
/// `k`.
fn transpose_bits(words: &mut [Word; BITS_PER_BYTE]) {
	// Mask `b` selects, within each byte, the bit positions whose within-byte index has bit `b`
	// set.
	const MASKS: [u64; 3] = [
		0xaaaa_aaaa_aaaa_aaaa,
		0xcccc_cccc_cccc_cccc,
		0xf0f0_f0f0_f0f0_f0f0,
	];
	for (b, &mask) in MASKS.iter().enumerate() {
		let d = 1 << b;
		for k in 0..BITS_PER_BYTE {
			if k & d == 0 {
				// Swap the bit-`b`-set within-byte bits of word `k` with the bit-`b`-clear
				// within-byte bits of word `k + d`, exchanging the within-byte bit axis with
				// the word axis.
				let lo = words[k].as_u64();
				let hi = words[k + d].as_u64();
				let t = ((lo >> d) ^ hi) & (mask >> d);
				words[k] = Word::from_u64(lo ^ (t << d));
				words[k + d] = Word::from_u64(hi ^ t);
			}
		}
	}
}

/// View the words as a slice of fixed-length arrays.
///
/// If the number of words is less than N, then repeat it into an N-length array. Repeating
/// corresponds to variable padding over the boolean hypercube.
///
/// ## Preconditions
///
/// * `words` must be a power of two
/// * `N` must be a power of two
pub(crate) fn duplicate_to_fixed_chunks<const N: usize>(words: &[Word]) -> Cow<'_, [[Word; N]]> {
	assert!(words.len().is_power_of_two());
	assert!(N.is_power_of_two());

	let (chunks, leftover) = words.as_chunks::<N>();

	assert!(
		chunks.is_empty() || leftover.is_empty(),
		"words.len() and N are both powers of two; either words.len() is divisible by N or less than it"
	);

	if chunks.is_empty() {
		let mut repeated = [Word::ZERO; N];
		for chunk in repeated.chunks_mut(words.len()) {
			chunk.copy_from_slice(words);
		}
		Cow::Owned(vec![repeated])
	} else {
		Cow::Borrowed(chunks)
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::arch::OptimalPackedB128;
	use binius_math::test_utils::{random_field_buffer, random_scalars};
	use binius_utils::checked_arithmetics::log2_strict_usize;
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

		let log_n = log2_strict_usize(words.len());

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

	#[test]
	fn test_transpose_bits() {
		let mut rng = StdRng::seed_from_u64(0);
		for _ in 0..100 {
			let input: [Word; BITS_PER_BYTE] =
				array::from_fn(|_| Word::from_u64(rng.random::<u64>()));
			let mut output = input;
			transpose_bits(&mut output);

			// out[i][j][k] = in[k][j][i]: bit `BITS_PER_BYTE * j + k` of output word `i` equals bit
			// `BITS_PER_BYTE * j + i` of input word `k`.
			for i in 0..BITS_PER_BYTE {
				for j in 0..Word::BYTES {
					for k in 0..BITS_PER_BYTE {
						let out_bit = (output[i].as_u64() >> (BITS_PER_BYTE * j + k)) & 1;
						let in_bit = (input[k].as_u64() >> (BITS_PER_BYTE * j + i)) & 1;
						assert_eq!(out_bit, in_bit, "mismatch at i={i}, j={j}, k={k}");
					}
				}
			}
		}
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
