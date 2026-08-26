// Copyright 2026 The Binius Developers

//! x86-64 AVX-512 message transpose for the multi-lane Blake3 kernel.
//!
//! The lane loops hold the message as 16 words of `N` lanes, one lane per message.
//! The input arrives the other way round: `N` blocks of 16 words, one block per lane.
//! Turning one layout into the other is a 16x16 transpose of 32-bit words.
//!
//! At 16 lanes that square is exactly four 512-bit registers wide and 16 rows tall,
//! so a fixed shuffle network moves it with no memory round trip.

use std::arch::x86_64::{
	__m512i, _mm512_loadu_si512, _mm512_setzero_si512, _mm512_shuffle_i32x4, _mm512_storeu_si512,
	_mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
};

use blake3::BLOCK_LEN;

/// Lane count the shuffle network is built for.
///
/// A 64-byte block holds 16 words, so only 16 lanes make the message a square.
/// A non-square block would need a different network per shape, which the lane loops already cover.
const LANES: usize = 16;

/// Reports whether this module has a transpose for the given lane count.
#[inline(always)]
pub const fn handles_lanes(n: usize) -> bool {
	n == LANES
}

/// Transposes 16 blocks of 16 words into 16 words of 16 lanes.
///
/// Both buffers are 1024 contiguous bytes, read and written as 16 rows of one 512-bit register.
///
/// # Safety
///
/// - `src` must be readable for 16 rows of `BLOCK_LEN` bytes.
/// - `dst` must be writable for 16 rows of `LANES` words.
/// - The two regions must not overlap.
#[inline(always)]
unsafe fn transpose_16x16(src: *const u8, dst: *mut u32) {
	unsafe {
		// Row `i` of the input is lane `i`'s whole 64-byte block.
		let mut r = [_mm512_setzero_si512(); 16];
		for (i, row) in r.iter_mut().enumerate() {
			*row = _mm512_loadu_si512(src.cast::<__m512i>().add(i));
		}

		// The network walks the transpose one power of two at a time.
		// Each stage swaps blocks of size 2^k between rows 2^k apart.
		//
		//     stage 1: 32-bit words,  rows 1 apart
		//     stage 2: 64-bit pairs,  rows 2 apart
		//     stage 3: 128-bit lanes, rows 4 apart
		//     stage 4: 128-bit lanes, rows 8 apart
		let mut t = [_mm512_setzero_si512(); 16];

		// Stage 1: interleave 32-bit words between each adjacent row pair.
		for i in 0..8 {
			t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
			t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
		}

		// Stage 2: interleave 64-bit pairs between rows two apart.
		for i in 0..4 {
			let (lo, hi) = (4 * i, 4 * i + 2);
			r[4 * i] = _mm512_unpacklo_epi64(t[lo], t[hi]);
			r[4 * i + 1] = _mm512_unpackhi_epi64(t[lo], t[hi]);
			r[4 * i + 2] = _mm512_unpacklo_epi64(t[lo + 1], t[hi + 1]);
			r[4 * i + 3] = _mm512_unpackhi_epi64(t[lo + 1], t[hi + 1]);
		}

		// Stage 3: exchange whole 128-bit lanes between rows four apart.
		// Selector 0x88 takes the two low lanes of each source, 0xdd the two high lanes.
		for i in 0..2 {
			for j in 0..4 {
				let (lo, hi) = (8 * i + j, 8 * i + j + 4);
				t[8 * i + j] = _mm512_shuffle_i32x4(r[lo], r[hi], 0x88);
				t[8 * i + j + 4] = _mm512_shuffle_i32x4(r[lo], r[hi], 0xdd);
			}
		}

		// Stage 4: exchange 128-bit lanes between rows eight apart, completing the square.
		for j in 0..8 {
			r[j] = _mm512_shuffle_i32x4(t[j], t[j + 8], 0x88);
			r[j + 8] = _mm512_shuffle_i32x4(t[j], t[j + 8], 0xdd);
		}

		// Row `w` of the output is message word `w` across all 16 lanes.
		for (i, row) in r.iter().enumerate() {
			_mm512_storeu_si512(dst.cast::<__m512i>().add(i), *row);
		}
	}
}

/// Loads one 64-byte block per lane into 16 little-endian message words.
///
/// Produces the same words as the portable byte-wise loader, for every input.
/// x86-64 is little-endian, so reading a block as 32-bit words already applies the byte order.
///
/// # Panics
///
/// Panics if no transpose covers `N`.
#[inline(always)]
pub fn load_block_words<const N: usize>(block: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	assert!(handles_lanes(N), "precondition: the lane count must have a transpose");

	let mut m = [[0u32; N]; 16];

	// SAFETY:
	// - Arrays are contiguous, so the input is 16 rows of `BLOCK_LEN` bytes with no padding.
	// - The output is 16 rows of `N` words, and the check above pins `N` to 16.
	// - `m` is a fresh local, so it cannot alias `block`.
	unsafe { transpose_16x16(block.as_ptr().cast::<u8>(), m.as_mut_ptr().cast::<u32>()) };

	m
}
