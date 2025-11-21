// Copyright 2025 Irreducible Inc.

use binius_core::word::Word;
use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, span::expand_subset_sums_array};
use binius_utils::{checked_arithmetics::log2_strict_usize, rayon::prelude::*};
use binius_verifier::config::WORD_SIZE_BITS;

const CHUNK_BITS: usize = u8::BITS as usize;

/// A struct that performs a linear transformation on the bits of a word (as a bit-vector).
///
/// The transformation is computed as an inner product between the bits of the word
/// (mapping bit 0 to [`Field::ZERO`] and bit 1 to [`Field::ONE`]) and a vector of
/// field elements.
///
/// This implementation uses the [Method of Four Russians] to optimize the computation by
/// precomputing lookup tables for each byte position and using bitwise chunks of the words.
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
pub struct WordTransform<F: Field> {
	/// Lookup tables for each byte position.
	lookup_tables: Vec<[F; 1 << CHUNK_BITS]>,
}

impl<F: Field> WordTransform<F> {
	/// Creates a new `WordTransform` that performs a linear transformation defined by `vec`.
	///
	/// ## Preconditions
	/// * `vec` contains exactly [`WORD_SIZE_BITS`] elements
	pub fn new(vec: &[F]) -> Self {
		assert_eq!(vec.len(), WORD_SIZE_BITS); // precondition

		// Create a lookup table of the expanded subset sums of all 256 combinations per byte
		//
		// The lookup table size is (64 / 8) * 256 * 16 = 32 KiB when the word size is 64 bits and
		// field size is 128 bits.
		let lookup_tables = vec
			.chunks(CHUNK_BITS)
			.map(|chunk| {
				let chunk = <[F; CHUNK_BITS]>::try_from(chunk)
					.expect("vec.len() == 64; thus, chunks must be exact CHUNK_BITS in size");
				expand_subset_sums_array::<_, CHUNK_BITS, { 1 << CHUNK_BITS }>(chunk)
			})
			.collect::<Vec<_>>();

		Self { lookup_tables }
	}

	/// Transforms a word by computing the inner product of its bits with the vector
	/// used to construct this `WordTransform`.
	pub fn transform(&self, word: Word) -> F {
		// Split the word into bytes and perform one lookup per byte
		let word_bytes = word.as_u64().to_le_bytes();
		word_bytes
			.iter()
			.enumerate()
			.map(|(i_byte, &byte)| self.lookup_tables[i_byte][byte as usize])
			.sum()
	}
}

/// Computes a [`FieldBuffer`] where each element is the inner product of the bits of a word and a
/// vector of field elements.
///
/// Returns a buffer where element `i` is the inner product of the bits of word `i` in `words`
/// (mapping bit 0 to [`Field::ZERO`] and bit 1 to [`Field::ONE`]) and the values in `vec`.
///
/// This implementation uses the [Method of Four Russians] to optimize the computation by
/// precomputing a small lookup table and looking up into it using bitwise chunks of the words.
///
/// ## Preconditions
/// * `vec` contains exactly [`WORD_SIZE_BITS`] elements
/// * `words` has a power-of-two length
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
pub fn fold_words<F, P>(words: &[Word], vec: &[F]) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert!(words.len().is_power_of_two()); // precondition

	let transform = WordTransform::new(vec);

	let log_n = log2_strict_usize(words.len());

	// Collect the folded results using the WordTransform
	let values = words
		.par_chunks(P::WIDTH)
		.map(|word_chunk| P::from_scalars(word_chunk.iter().map(|&word| transform.transform(word))))
		.collect::<Vec<_>>();

	FieldBuffer::new(log_n, values.into_boxed_slice())
		.expect("log_n is calculated from words.len() and values is constructed from words")
}

#[cfg(test)]
mod tests {
	use binius_math::test_utils::random_scalars;
	use binius_verifier::config::B128;
	use rand::{Rng, SeedableRng, rngs::StdRng};

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
			.expect("log_n is calculated from words.len() and values is constructed from words")
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
}
