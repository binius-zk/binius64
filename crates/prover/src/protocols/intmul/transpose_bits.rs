// Copyright 2026 The Binius Developers

//! Bit-transposing a word list into one 64-word block per column.

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::{PackedBinaryField64x1b, transpose_square_blocks_array};
use binius_math::bit_reverse::reverse_bits;
use binius_utils::rayon::{prelude::*, task_size::IndexedParallelIteratorExt};

/// The bits of a word list, one 64-word block per column.
///
/// With `h = min(n_vars, Word::LOG_BITS)` and `C = 2^(n_vars - h)`, block `lo` is the words
/// `out[lo * 64 .. (lo + 1) * 64]`, and bit `bitrev_h(hi)` of its word `b` is bit `b` of
/// `words[hi * C + lo]`. Words at or past `words.len()` read as zero.
///
/// The bit reversal makes the bits the switchover combines in round `r` one contiguous span of
/// `2^r` bits, since the sumcheck binds the top bit of `hi` first.
///
/// ## Preconditions
///
/// * `words.len() <= 1 << n_vars`
#[cfg_attr(
	not(test),
	expect(dead_code, reason = "the switchover rewrite calls it")
)]
pub fn transpose_bits<A: Allocator>(alloc: &A, words: &[Word], n_vars: usize) -> A::Vec<Word> {
	assert!(words.len() <= 1 << n_vars, "words.len() must not exceed 2^n_vars");

	let h = n_vars.min(Word::LOG_BITS);
	let n_cols = 1 << (n_vars - h);

	let len = Word::BITS * n_cols;
	let mut out = alloc.alloc::<Word>(len);
	out.resize(len, Word::ZERO);

	out.par_chunks_exact_mut(Word::BITS)
		.enumerate()
		.with_min_task_bytes::<[Word; Word::BITS]>()
		.for_each(|(lo, block)| {
			// Gather the column at stride C into bit-reversed slots; the rest stay zero.
			for hi in 0..1 << h {
				if let Some(&word) = words.get(hi * n_cols + lo) {
					block[reverse_bits(hi, h as u32)] = word;
				}
			}

			// A word and a 64-bit row of single-bit scalars share one underlier.
			let block: &mut [Word; Word::BITS] = block
				.try_into()
				.expect("chunks are exactly Word::BITS words");
			transpose_square_blocks_array::<
				PackedBinaryField64x1b,
				{ Word::LOG_BITS },
				{ Word::BITS },
			>(bytemuck::must_cast_mut(block));
		});

	out
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	fn n_vars_and_len() -> impl Strategy<Value = (usize, usize)> {
		(0..=10usize).prop_flat_map(|n_vars| (Just(n_vars), 0..=1usize << n_vars))
	}

	proptest! {
		#[test]
		fn transpose_bits_matches_bit_extraction((n_vars, n_words) in n_vars_and_len(), seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let words = (0..n_words).map(|_| Word::from_u64(rng.random())).collect::<Vec<_>>();

			let out = transpose_bits(&GlobalAllocator, &words, n_vars);

			let h = n_vars.min(Word::LOG_BITS);
			let n_cols = 1 << (n_vars - h);
			prop_assert_eq!(out.len(), Word::BITS * (1 << n_vars.saturating_sub(Word::LOG_BITS)));

			for b in 0..Word::BITS {
				for lo in 0..n_cols {
					let row_word = out[lo * Word::BITS + b];
					for pos in 0..Word::BITS {
						// Positions with no source row read as zero, like the padding words.
						let expected = (pos < 1 << h)
							.then(|| words.get(reverse_bits(pos, h as u32) * n_cols + lo))
							.flatten()
							.is_some_and(|word| word.extract_bit(b));
						prop_assert_eq!(row_word.extract_bit(pos), expected, "b={} lo={} pos={}", b, lo, pos);
					}
				}
			}
		}
	}
}
