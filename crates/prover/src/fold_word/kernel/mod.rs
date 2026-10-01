// Copyright 2026 The Binius Developers

//! The bit-axis fold kernel, batched over a fixed number of words.
//!
//! One word's fold is the inner product of its bits with one weight per bit position:
//!
//! ```text
//!     out = sum_{i < 64} bit_i(word) * weight[i]
//! ```
//!
//! Two implementations sit behind one interface, chosen by target and by field:
//!
//! | when | how | working set |
//! |---|---|---|
//! | x86-64 with AVX-512 and GFNI, over the 128-bit GHASH field | a block matrix | 1 KiB |
//! | otherwise | the subset-sum tables | 32 KiB |
//!
//! Both sum the weights of the word's set bits, and XOR is associative, so they agree bit for bit.
//!
//! A batch is 64 words, which is one byte plane per 512-bit vector.

#[cfg(all(
	target_arch = "x86_64",
	target_feature = "avx512f",
	target_feature = "avx512vbmi",
	target_feature = "gfni"
))]
mod gfni;
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "avx512f",
	target_feature = "avx512vbmi",
	target_feature = "gfni"
))]
mod x86_64;

#[cfg(not(all(
	target_arch = "x86_64",
	target_feature = "avx512f",
	target_feature = "avx512vbmi",
	target_feature = "gfni"
)))]
mod portable;

use binius_core::word::Word;
#[cfg(not(all(
	target_arch = "x86_64",
	target_feature = "avx512f",
	target_feature = "avx512vbmi",
	target_feature = "gfni"
)))]
pub use portable::BitFold;
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "avx512f",
	target_feature = "avx512vbmi",
	target_feature = "gfni"
))]
pub use x86_64::BitFold;

/// Words one call of the batched fold covers.
///
/// A 512-bit vector holds one byte plane of 64 words, which is what fixes the batch at 64.
pub const WORDS_PER_BATCH: usize = 64;

/// Runs a full-batch fold over a short one, zero-filling the words past its end.
///
/// A zero word has no set bits to weight, so every padded entry folds to zero.
///
/// # Panics
///
/// Panics if `words` is longer than one batch.
#[inline]
fn fold_zero_padded<T>(
	words: &[Word],
	fold_batch: impl FnOnce(&[Word; WORDS_PER_BATCH]) -> [T; WORDS_PER_BATCH],
) -> [T; WORDS_PER_BATCH] {
	let mut batch = [Word::ZERO; WORDS_PER_BATCH];
	batch[..words.len()].copy_from_slice(words);
	fold_batch(&batch)
}
