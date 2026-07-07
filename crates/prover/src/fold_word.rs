// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, borrow::Cow, iter};

use binius_core::{
	consts::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS, WORD_SIZE_BYTES},
	word::Word,
};
use binius_field::{
	BinaryField, Field, PackedField,
	linear_transformation::{
		BytewiseLookupTransformationFactory, LinearTransformationFactory,
		OutputWrappingTransformationFactory, Transformation,
	},
	util::expand_subset_sums_array,
};
use binius_math::{
	FieldBuffer,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};

/// Base-2 logarithm of the number of words folded together within a single chunk.
const LOG_CHUNK_SIZE: usize = LOG_WORD_SIZE_BITS;
/// Number of words folded together within a single chunk.
const CHUNK_SIZE: usize = 1 << LOG_CHUNK_SIZE;
/// Number of bits in a byte; [`fold_across_words`] processes each chunk in groups of this many
/// words, one per byte of the words.
const BITS_PER_BYTE: usize = WORD_SIZE_BITS / WORD_SIZE_BYTES;

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
/// * `vec` contains exactly [`binius_core::consts::WORD_SIZE_BITS`] elements
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
pub fn fold_words<F, P>(words: &[Word], vec: &[F]) -> FieldBuffer<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	fold_words_with_transform_factory(
		&OutputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory),
		words,
		vec,
	)
}

pub fn fold_words_with_transform_factory<F, P, TransformFactory>(
	transform_factory: &TransformFactory,
	words: &[Word],
	vec: &[F],
) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
	TransformFactory: LinearTransformationFactory<u64, F>,
{
	fold_words_with_transform(&transform_factory.create(vec), words)
}

pub fn fold_words_with_transform<F, P, T>(transform: &T, words: &[Word]) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
	T: Transformation<u64, F>,
{
	// `words` need not have a power-of-two length; the high words up to the next power of two are
	// treated as zero, so the remaining slots after the last real word are zero-filled by resize.
	let log_n = log2_ceil_usize(words.len());
	let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);

	let mut values = Vec::<P>::with_capacity(capacity);
	words
		.par_chunks(P::WIDTH)
		.map(|word_chunk| {
			P::from_scalars(word_chunk.iter().map(|&word| transform.transform(&word.0)))
		})
		.collect_into_vec(&mut values);
	values.resize(capacity, P::default());

	FieldBuffer::new(log_n, values.into_boxed_slice())
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
pub fn fold_across_words<F, P>(words: &[Word], point: &[F]) -> [F; WORD_SIZE_BITS]
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
			|| [F::ZERO; WORD_SIZE_BITS],
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
	lookups: [[F; 1 << BITS_PER_BYTE]; WORD_SIZE_BYTES],
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
		// Split those entries into WORD_SIZE_BYTES groups of BITS_PER_BYTE.
		let prefix_expansion = eq_ind_partial_eval_scalars::<F>(&prefix);
		let lookups: [[F; 1 << BITS_PER_BYTE]; WORD_SIZE_BYTES] = array::from_fn(|byte| {
			let group: [F; BITS_PER_BYTE] = prefix_expansion
				[byte * BITS_PER_BYTE..(byte + 1) * BITS_PER_BYTE]
				.try_into()
				.expect(
					"prefix_expansion has CHUNK_SIZE = WORD_SIZE_BYTES * BITS_PER_BYTE entries",
				);
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
	pub fn fold(&self, words: &[Word]) -> [F; WORD_SIZE_BITS] {
		assert_eq!(words.len(), self.n_words, "words.len() must equal 2^point.len()");

		let chunks = duplicate_to_fixed_chunks::<CHUNK_SIZE>(words);
		debug_assert_eq!(chunks.len(), self.suffix_weights.as_ref().len());

		// Accumulate each chunk's contribution, scaled by its suffix weight.
		// The accumulator stays in the chunk kernel's transposed bit layout until the end.
		let mut folded = [F::ZERO; WORD_SIZE_BITS];
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
/// j`. Transposing that `WORD_SIZE_BYTES`-by-`BITS_PER_BYTE` layout restores bit-position order.
fn transpose_accumulator<F: BinaryField>(folded: [F; WORD_SIZE_BITS]) -> [F; WORD_SIZE_BITS] {
	let mut result = [F::ZERO; WORD_SIZE_BITS];
	for a in 0..BITS_PER_BYTE {
		for j in 0..WORD_SIZE_BYTES {
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
	lookups: &[[F; 1 << BITS_PER_BYTE]; WORD_SIZE_BYTES],
) -> [F; WORD_SIZE_BITS] {
	// Reshape the chunk into one sub-array per lookup table, each holding BITS_PER_BYTE words.
	let subchunks = bytemuck::must_cast_ref::<
		[Word; CHUNK_SIZE],
		[[Word; BITS_PER_BYTE]; WORD_SIZE_BYTES],
	>(chunk);

	let mut acc = [F::ZERO; WORD_SIZE_BITS];
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
/// Viewing the input and output as `BITS_PER_BYTE`×`WORD_SIZE_BYTES`×`BITS_PER_BYTE` bit matrices
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
	use binius_core::consts::WORD_SIZE_BITS;
	use binius_field::arch::OptimalPackedB128;
	use binius_math::test_utils::random_scalars;
	use binius_utils::checked_arithmetics::log2_strict_usize;
	use binius_verifier::config::B128;
	use rand::prelude::*;

	use super::*;

	fn naive_fold_words<F, P>(words: &[Word], vec: &[F]) -> FieldBuffer<P>
	where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		assert_eq!(vec.len(), WORD_SIZE_BITS);
		assert!(words.len().is_power_of_two());

		let log_n = log2_strict_usize(words.len());

		let values = words
			.par_chunks(P::WIDTH)
			.map(|word_chunk| {
				P::from_scalars(word_chunk.iter().map(|&word| {
					// Decompose word into bits and compute inner product
					let mut sum = F::ZERO;
					for bit_idx in 0..WORD_SIZE_BITS {
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

		let vec = random_scalars(&mut rng, WORD_SIZE_BITS);

		// Compute using both methods
		let result_optimized = fold_words::<B128, B128>(&words, &vec);
		let result_naive = naive_fold_words::<B128, B128>(&words, &vec);

		// Compare results
		assert_eq!(result_optimized, result_naive);
	}

	fn naive_fold_across_words<F: BinaryField>(words: &[Word], point: &[F]) -> [F; WORD_SIZE_BITS] {
		assert_eq!(words.len(), 1 << point.len());

		let eq = eq_ind_partial_eval_scalars(point);
		let mut out = [F::ZERO; WORD_SIZE_BITS];
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
				for j in 0..WORD_SIZE_BYTES {
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
