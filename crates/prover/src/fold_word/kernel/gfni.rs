// Copyright 2026 The Binius Developers

//! A table-free bit-axis fold, built on the GFNI 8x8 GF(2) matrix instruction.
//!
//! The fold sums the weights of a word's set bits:
//!
//! ```text
//!     out = sum_{i < 64} bit_i(word) * weight[i]
//! ```
//!
//! Field addition is XOR, so this is GF(2)-linear: a 128-by-64 bit matrix with the weights as its
//! columns.
//!
//! Cut into 8x8 blocks that is sixteen output byte planes by eight input byte planes.
//!
//! Those 128 blocks describe the fold in 1 KiB, against 32 KiB of subset-sum tables.
//!
//! One affine product applies a block to 64 input bytes, so a call folds 64 words with no
//! data-dependent load.
//!
//! # Block layout
//!
//! The hardware reads the row producing output bit `i` from byte `7 - i` of the matrix qword:
//!
//! ```text
//!     parity( A.qword[q].byte[7 - i] & x.qword[q].byte[n] )
//! ```
//!
//! Matching that against the fold pins bit `t` of byte `7 - i` of block `(j, k)` to bit `8k + i`
//! of weight `8j + t`.
//!
//! Derived from the flock prover's table-free row fold (MIT/Apache-2.0), written independently.

use std::{
	arch::x86_64::{
		__m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_permutex2var_epi64,
		_mm512_permutexvar_epi8, _mm512_set1_epi64, _mm512_setr_epi64, _mm512_storeu_si512,
		_mm512_ternarylogic_epi64, _mm512_unpackhi_epi64, _mm512_unpacklo_epi64, _mm512_xor_si512,
	},
	array,
	mem::MaybeUninit,
};

use binius_core::word::Word;
use binius_field::{BinaryField, Ghash128b};

use super::WORDS_PER_BATCH;

/// Byte planes one 128-bit field element divides into.
const OUT_PLANES: usize = Ghash128b::N_BITS / 8;
/// Byte planes one word divides into.
const IN_PLANES: usize = Word::BYTES;

/// The 8x8 index transpose on the 64 byte positions of a vector.
///
/// It swaps the two three-bit halves of a byte index, so applying it twice is the identity.
///
/// One constant therefore serves both directions.
#[rustfmt::skip]
const BYTE_TRANSPOSE: [i8; 64] = [
	 0,  8, 16, 24, 32, 40, 48, 56,
	 1,  9, 17, 25, 33, 41, 49, 57,
	 2, 10, 18, 26, 34, 42, 50, 58,
	 3, 11, 19, 27, 35, 43, 51, 59,
	 4, 12, 20, 28, 36, 44, 52, 60,
	 5, 13, 21, 29, 37, 45, 53, 61,
	 6, 14, 22, 30, 38, 46, 54, 62,
	 7, 15, 23, 31, 39, 47, 55, 63,
];

/// The bit-axis fold as 8x8 GF(2) blocks, one per input and output byte plane pair.
#[derive(Debug, Clone)]
pub struct BitFoldMats {
	/// Block `(j, k)` — input byte plane `j`, output byte plane `k` — at index `j * 16 + k`.
	///
	/// Grouping by input plane lets one output plane's eight products walk a fixed stride.
	blocks: [u64; IN_PLANES * OUT_PLANES],
}

impl BitFoldMats {
	/// Builds the blocks from one weight per bit position of a word.
	///
	/// # Panics
	///
	/// Panics unless there is exactly one weight per bit position.
	pub fn new(weights: &[Ghash128b]) -> Self {
		assert_eq!(weights.len(), Word::BITS);

		let mut blocks = [0u64; IN_PLANES * OUT_PLANES];
		for (j, chunk) in weights.chunks_exact(8).enumerate() {
			// The eight weights of input byte plane `j`, split into the two halves a bit
			// transpose can take at once.
			let lo: [u64; 8] = array::from_fn(|t| u128::from(chunk[t]) as u64);
			let hi: [u64; 8] = array::from_fn(|t| (u128::from(chunk[t]) >> 64) as u64);

			// Column `k` of the transpose holds, in byte `i`, the byte whose bit `t` is bit `i`
			// of byte `k` of `weights[8j + t]` — which is the block's row for output bit `i`.
			// Reversing the byte order puts row `i` in byte `7 - i`, where GFNI reads it.
			for (k, column) in bit_transpose_bytes(&lo).into_iter().enumerate() {
				blocks[j * OUT_PLANES + k] = column.swap_bytes();
			}
			for (k, column) in bit_transpose_bytes(&hi).into_iter().enumerate() {
				blocks[j * OUT_PLANES + 8 + k] = column.swap_bytes();
			}
		}

		Self { blocks }
	}

	/// Folds a full batch of words, one field element per word.
	///
	/// Every element of the result is written, in the same order as the words.
	#[inline]
	pub fn fold_batch(&self, words: &[Word; WORDS_PER_BATCH]) -> [Ghash128b; WORDS_PER_BATCH] {
		let mut out = MaybeUninit::<[Ghash128b; WORDS_PER_BATCH]>::uninit();

		// SAFETY: the input is one whole batch, so its 512 bytes are readable, and the output is
		// one whole batch, so its 1024 bytes are writable.
		//
		// This module compiles only where every intrinsic below is available.
		unsafe {
			fold_batch_raw(&self.blocks, words.as_ptr(), out.as_mut_ptr().cast::<Ghash128b>());
			// SAFETY: the call above writes every element of the batch.
			out.assume_init()
		}
	}
}

/// Bit-transposes each byte column of eight 64-bit lanes.
///
/// Reading byte `c` of each lane as the rows of an 8x8 bit matrix, column `c` of the result is
/// that matrix transposed.
///
/// One shuffle and one matrix product cover all 64 bytes.
#[inline]
fn bit_transpose_bytes(lanes: &[u64; 8]) -> [u64; 8] {
	/// Each qword gathers one byte column, with the lanes in reverse order.
	///
	/// That reversal cancels the hardware's own `7 - i` row order, which is what turns the
	/// product below into a transpose.
	#[rustfmt::skip]
	const GATHER_COLUMNS: [i8; 64] = {
		let mut idx = [0i8; 64];
		let mut c = 0;
		while c < 8 {
			let mut r = 0;
			while r < 8 {
				idx[c * 8 + r] = (8 * (7 - r) + c) as i8;
				r += 1;
			}
			c += 1;
		}
		idx
	};

	let mut out = MaybeUninit::<[u64; 8]>::uninit();

	// SAFETY: the lanes, the index constant and the output are each 64 bytes, so each is one
	// whole vector, and this module compiles only where the intrinsics exist.
	unsafe {
		let gathered = _mm512_permutexvar_epi8(
			_mm512_loadu_si512(GATHER_COLUMNS.as_ptr().cast()),
			_mm512_loadu_si512(lanes.as_ptr().cast()),
		);
		// Against the identity byte, output byte `n` bit `i` reads matrix byte `7 - i`.
		// Reading that as byte `i` bit `n` is exactly the transpose.
		let identity = _mm512_set1_epi64(0x8040201008040201u64 as i64);
		_mm512_storeu_si512(
			out.as_mut_ptr().cast(),
			_mm512_gf2p8affine_epi64_epi8::<0>(identity, gathered),
		);
		// SAFETY: the store above writes all eight lanes.
		out.assume_init()
	}
}

/// Transposes eight vectors as an 8x8 matrix of qwords.
///
/// Three stages of two-source shuffles, and its own inverse.
///
/// # Safety
///
/// AVX-512F must be available.
#[inline]
unsafe fn qword_transpose(t: [__m512i; 8]) -> [__m512i; 8] {
	// SAFETY: the caller guarantees AVX-512F.
	unsafe {
		// Stage two and three select qwords across a pair of vectors; the index vectors name
		// which, with 0..8 reading the first source and 8..16 the second.
		let pair_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
		let pair_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
		let quad_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
		let quad_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

		// Stage one: interleave the even and odd qwords of each adjacent pair of vectors.
		let even = |a, b| _mm512_unpacklo_epi64(a, b);
		let odd = |a, b| _mm512_unpackhi_epi64(a, b);
		let (e01, o01) = (even(t[0], t[1]), odd(t[0], t[1]));
		let (e23, o23) = (even(t[2], t[3]), odd(t[2], t[3]));
		let (e45, o45) = (even(t[4], t[5]), odd(t[4], t[5]));
		let (e67, o67) = (even(t[6], t[7]), odd(t[6], t[7]));

		// Stage two: combine pairs into groups of four.
		let h02_a = _mm512_permutex2var_epi64(e01, pair_lo, e23);
		let h46_a = _mm512_permutex2var_epi64(e01, pair_hi, e23);
		let h13_a = _mm512_permutex2var_epi64(o01, pair_lo, o23);
		let h57_a = _mm512_permutex2var_epi64(o01, pair_hi, o23);
		let h02_b = _mm512_permutex2var_epi64(e45, pair_lo, e67);
		let h46_b = _mm512_permutex2var_epi64(e45, pair_hi, e67);
		let h13_b = _mm512_permutex2var_epi64(o45, pair_lo, o67);
		let h57_b = _mm512_permutex2var_epi64(o45, pair_hi, o67);

		// Stage three: combine the two groups of four into the eight transposed vectors.
		[
			_mm512_permutex2var_epi64(h02_a, quad_lo, h02_b),
			_mm512_permutex2var_epi64(h13_a, quad_lo, h13_b),
			_mm512_permutex2var_epi64(h02_a, quad_hi, h02_b),
			_mm512_permutex2var_epi64(h13_a, quad_hi, h13_b),
			_mm512_permutex2var_epi64(h46_a, quad_lo, h46_b),
			_mm512_permutex2var_epi64(h57_a, quad_lo, h57_b),
			_mm512_permutex2var_epi64(h46_a, quad_hi, h46_b),
			_mm512_permutex2var_epi64(h57_a, quad_hi, h57_b),
		]
	}
}

/// Folds 64 words through the block matrix, writing 64 field elements.
///
/// # Safety
///
/// * `words` must be readable for `WORDS_PER_BATCH` words.
/// * `out` must be writable for `WORDS_PER_BATCH` elements; all of them are written.
/// * AVX-512F, AVX512VBMI and GFNI must be available.
#[inline]
unsafe fn fold_batch_raw(
	blocks: &[u64; IN_PLANES * OUT_PLANES],
	words: *const Word,
	out: *mut Ghash128b,
) {
	// SAFETY: the caller guarantees the extents and the target features.
	unsafe {
		let byte_transpose = _mm512_loadu_si512(BYTE_TRANSPOSE.as_ptr().cast());

		// Build the input planes, so that byte `n` of plane `j` is byte `j` of word `n`.
		//
		//     load          vector i = words 8i..8i+8
		//     byte swap     qword j  = byte plane j of those eight words
		//     qword swap    plane j  = qword j of all eight vectors
		let loaded: [__m512i; 8] = array::from_fn(|i| _mm512_loadu_si512(words.add(8 * i).cast()));
		let planes = qword_transpose(loaded.map(|v| _mm512_permutexvar_epi8(byte_transpose, v)));

		// Byte `n` of output plane `k` is byte `k` of the fold of word `n`, so it XORs the eight
		// input planes through blocks `(0, k) .. (8, k)`.
		//
		// One product covers all 64 bytes, and three-input XORs fold the eight in four steps.
		let out_plane = |k: usize| {
			let product = |j: usize| {
				_mm512_gf2p8affine_epi64_epi8::<0>(
					planes[j],
					_mm512_set1_epi64(blocks[j * OUT_PLANES + k] as i64),
				)
			};
			let a = _mm512_ternarylogic_epi64::<0x96>(product(0), product(1), product(2));
			let b = _mm512_ternarylogic_epi64::<0x96>(product(3), product(4), product(5));
			let c = _mm512_ternarylogic_epi64::<0x96>(product(6), product(7), a);
			_mm512_xor_si512(b, c)
		};
		let low: [__m512i; 8] = array::from_fn(&out_plane);
		let high: [__m512i; 8] = array::from_fn(|k| out_plane(8 + k));

		// Back to row-major. Undoing the two transposes leaves, for each group of eight words,
		// one vector of their low halves and one of their high halves, qword `m` belonging to
		// word `8i + m`; interleaving the two rebuilds eight whole 128-bit elements.
		let low = qword_transpose(low);
		let high = qword_transpose(high);
		let interleave_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
		let interleave_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
		for i in 0..8 {
			let lo = _mm512_permutexvar_epi8(byte_transpose, low[i]);
			let hi = _mm512_permutexvar_epi8(byte_transpose, high[i]);
			let dst = out.add(8 * i).cast::<__m512i>();
			_mm512_storeu_si512(dst, _mm512_permutex2var_epi64(lo, interleave_lo, hi));
			_mm512_storeu_si512(dst.add(1), _mm512_permutex2var_epi64(lo, interleave_hi, hi));
		}
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;

	/// A bit-by-bit transpose of the eight bytes at one column, as the oracle for the kernel's
	/// vectorized transpose.
	fn bit_transpose_bytes_naive(lanes: &[u64; 8]) -> [u64; 8] {
		array::from_fn(|c| {
			let mut column = 0u64;
			for t in 0..8 {
				let byte = (lanes[t] >> (8 * c)) & 0xff;
				for i in 0..8 {
					if (byte >> i) & 1 == 1 {
						column |= 1 << (8 * i + t);
					}
				}
			}
			column
		})
	}

	proptest! {
		#[test]
		fn bit_transpose_matches_the_naive_transpose(lanes: [u64; 8]) {
			prop_assert_eq!(bit_transpose_bytes(&lanes), bit_transpose_bytes_naive(&lanes));
		}
	}

	#[test]
	fn bit_transpose_matches_the_naive_transpose_at_the_edges() {
		for lanes in [
			[0u64; 8],
			[u64::MAX; 8],
			array::from_fn(|i| 1 << i),
			array::from_fn(|i| 1 << (8 * i)),
		] {
			assert_eq!(
				bit_transpose_bytes(&lanes),
				bit_transpose_bytes_naive(&lanes),
				"lanes = {lanes:?}"
			);
		}
	}
}
