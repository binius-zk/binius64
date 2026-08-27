// Copyright 2026 The Binius Developers

//! x86-64 AVX-512 SHA-256 kernel, sixteen messages at a time.
//!
//! One 512-bit register carries a single state or message word across all sixteen lanes.
//! A round is then plain 32-bit vector arithmetic, sixteen hashes wide.
//!
//! Two instructions carry it:
//!
//! ```text
//!     vprold      rotates 32-bit words: a Sigma is 3 rotates, not 6 shifts and 2 ors
//!     vpternlogd  evaluates any 3-input bit function in one instruction
//! ```
//!
//! Choice, majority, and the 3-way xor inside each Sigma are all 3-input bit functions, which
//! holds a round to seventeen vector operations for all sixteen lanes.
//!
//! This is also the only wide option on an AVX-512 machine with no SHA extension.

use std::{
	arch::x86_64::{
		__m128i, __m256i, __m512i, _mm_set_epi32, _mm256_storeu_si256, _mm512_add_epi32,
		_mm512_broadcast_i32x4, _mm512_castsi512_si256, _mm512_loadu_si512, _mm512_ror_epi32,
		_mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_shuffle_i32x4,
		_mm512_srli_epi32, _mm512_storeu_si512, _mm512_ternarylogic_epi32, _mm512_unpackhi_epi32,
		_mm512_unpackhi_epi64, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
	},
	array,
};

use super::{BLOCK_LEN, K};

/// Lane count the kernel is built for.
///
/// A 64-byte block holds 16 words, so only 16 lanes make the message a square.
/// A non-square block would need a different shuffle network per shape.
const LANES: usize = 16;

/// Reports whether this module has a kernel for the given lane count.
#[inline(always)]
pub const fn handles_lanes(n: usize) -> bool {
	n == LANES
}

/// Ternary-logic immediate for `x ^ y ^ z`.
///
/// An immediate is the function's truth table, bit `(x << 2) | (y << 1) | z`:
///
/// ```text
///     xyz : 000 001 010 011 100 101 110 111
///     out :   0   1   1   0   1   0   0   1   -> 0b1001_0110
/// ```
const XOR3: i32 = 0x96;

/// Ternary-logic immediate for `Ch(x, y, z)`, which picks `y` where `x` is set, `z` where clear.
///
/// ```text
///     xyz : 000 001 010 011 100 101 110 111
///     out :   0   1   0   1   0   0   1   1   -> 0b1100_1010
/// ```
const CH: i32 = 0xCA;

/// Ternary-logic immediate for `Maj(x, y, z)`, the bit held by at least two of the three.
///
/// ```text
///     xyz : 000 001 010 011 100 101 110 111
///     out :   0   0   0   1   0   1   1   1   -> 0b1110_1000
/// ```
const MAJ: i32 = 0xE8;

/// Byte-reverses each 32-bit word of a 512-bit register.
///
/// The shuffle indexes within each 128-bit lane, so one 16-byte pattern broadcasts to all four.
///
/// # Safety
///
/// The caller must enable `avx512f`, so the intrinsics are defined.
#[inline(always)]
unsafe fn bswap32_mask() -> __m512i {
	unsafe {
		let pattern: __m128i =
			_mm_set_epi32(0x0C0D_0E0Fu32 as i32, 0x0809_0A0Bu32 as i32, 0x0405_0607, 0x0001_0203);
		_mm512_broadcast_i32x4(pattern)
	}
}

/// Transposes a 16x16 square of 32-bit words in place.
///
/// Each stage swaps blocks of size `2^k` between rows `2^k` apart, so the square never leaves
/// the registers:
///
/// ```text
///     stage 1: 32-bit words,  rows 1 apart
///     stage 2: 64-bit pairs,  rows 2 apart
///     stage 3: 128-bit lanes, rows 4 apart
///     stage 4: 128-bit lanes, rows 8 apart
/// ```
///
/// # Safety
///
/// The caller must enable `avx512f`, so the intrinsics are defined.
#[inline(always)]
unsafe fn transpose_16x16(rows: &mut [__m512i; 16]) {
	unsafe {
		let mut t = [_mm512_setzero_si512(); 16];

		// Stage 1: interleave 32-bit words between each adjacent row pair.
		for i in 0..8 {
			t[2 * i] = _mm512_unpacklo_epi32(rows[2 * i], rows[2 * i + 1]);
			t[2 * i + 1] = _mm512_unpackhi_epi32(rows[2 * i], rows[2 * i + 1]);
		}

		// Stage 2: interleave 64-bit pairs between rows two apart.
		for i in 0..4 {
			let (lo, hi) = (4 * i, 4 * i + 2);
			rows[4 * i] = _mm512_unpacklo_epi64(t[lo], t[hi]);
			rows[4 * i + 1] = _mm512_unpackhi_epi64(t[lo], t[hi]);
			rows[4 * i + 2] = _mm512_unpacklo_epi64(t[lo + 1], t[hi + 1]);
			rows[4 * i + 3] = _mm512_unpackhi_epi64(t[lo + 1], t[hi + 1]);
		}

		// Stage 3: exchange whole 128-bit lanes between rows four apart.
		// Selector 0x88 takes the two low lanes of each source, 0xdd the two high lanes.
		for i in 0..2 {
			for j in 0..4 {
				let (lo, hi) = (8 * i + j, 8 * i + j + 4);
				t[8 * i + j] = _mm512_shuffle_i32x4(rows[lo], rows[hi], 0x88);
				t[8 * i + j + 4] = _mm512_shuffle_i32x4(rows[lo], rows[hi], 0xdd);
			}
		}

		// Stage 4: exchange 128-bit lanes between rows eight apart, completing the square.
		for j in 0..8 {
			rows[j] = _mm512_shuffle_i32x4(t[j], t[j + 8], 0x88);
			rows[j + 8] = _mm512_shuffle_i32x4(t[j], t[j + 8], 0xdd);
		}
	}
}

/// Reads 16 blocks as one transposed square of big-endian message words.
///
/// Each row is byte-reversed on the way in, then the square is transposed.
///
/// # Safety
///
/// The caller must enable `avx512f` and `avx512bw`, so the intrinsics are defined.
#[inline(always)]
unsafe fn load_message_square<const N: usize>(blocks: &[[u8; BLOCK_LEN]; N]) -> [__m512i; 16] {
	unsafe {
		let mask = bswap32_mask();
		let mut rows = [_mm512_setzero_si512(); 16];
		for (i, row) in rows.iter_mut().enumerate() {
			let raw = _mm512_loadu_si512(blocks.as_ptr().cast::<__m512i>().add(i));
			*row = _mm512_shuffle_epi8(raw, mask);
		}
		transpose_16x16(&mut rows);
		rows
	}
}

/// Loads one 64-byte block per lane into 16 big-endian message words.
///
/// Produces the same words as the portable byte-wise loader, for every input.
///
/// # Panics
///
/// Panics if no kernel covers the lane count.
#[inline(always)]
pub fn load_block_words<const N: usize>(blocks: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	assert!(handles_lanes(N), "precondition: the lane count must have a transpose");

	let mut w = [[0u32; N]; 16];

	// SAFETY:
	// - The module is compiled in only when `avx512f` and `avx512bw` are enabled.
	// - Arrays are contiguous, so the input is 16 rows of 64 bytes with no padding.
	// - The output is 16 rows of `N` words, and the check above pins `N` to 16.
	// - The output is a fresh local, so it cannot alias the input.
	unsafe {
		let rows = load_message_square(blocks);
		for (i, row) in rows.iter().enumerate() {
			_mm512_storeu_si512(w.as_mut_ptr().cast::<__m512i>().add(i), *row);
		}
	}

	w
}

/// Applies one round to the eight state words, across all 16 lanes.
///
/// The same two writes as the portable round: only the next `a` and the next `e` are computed,
/// and returning the other six already shifted is what makes that shift cost nothing.
///
/// # Safety
///
/// The caller must enable `avx512f`, so the intrinsics are defined.
#[inline(always)]
unsafe fn round(state: [__m512i; 8], w: __m512i, k: u32) -> [__m512i; 8] {
	unsafe {
		let [a, b, c, d, e, f, g, h] = state;

		// Sigma1(x) = ROTR6 ^ ROTR11 ^ ROTR25, three rotates folded by one ternary logic.
		let sigma1 = _mm512_ternarylogic_epi32::<XOR3>(
			_mm512_ror_epi32::<6>(e),
			_mm512_ror_epi32::<11>(e),
			_mm512_ror_epi32::<25>(e),
		);
		let ch = _mm512_ternarylogic_epi32::<CH>(e, f, g);
		// The round constant is the same in every lane, so it broadcasts straight from memory.
		let wk = _mm512_add_epi32(w, _mm512_set1_epi32(k as i32));
		let t1 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(h, sigma1), ch), wk);

		// Sigma0(x) = ROTR2 ^ ROTR13 ^ ROTR22.
		let sigma0 = _mm512_ternarylogic_epi32::<XOR3>(
			_mm512_ror_epi32::<2>(a),
			_mm512_ror_epi32::<13>(a),
			_mm512_ror_epi32::<22>(a),
		);
		let maj = _mm512_ternarylogic_epi32::<MAJ>(a, b, c);

		let next_e = _mm512_add_epi32(d, t1);
		let next_a = _mm512_add_epi32(t1, _mm512_add_epi32(sigma0, maj));

		[next_a, a, b, c, next_e, e, f, g]
	}
}

/// Extends the 16-word rolling message window by 16 words, in place.
///
/// The same straight sweep the portable window uses, so the reads at offsets 1, 9, and 14
/// pick up their new values exactly when they should.
///
/// # Safety
///
/// The caller must enable `avx512f`, so the intrinsics are defined.
#[inline(always)]
unsafe fn extend_window(w: &mut [__m512i; 16]) {
	unsafe {
		for j in 0..16 {
			let m15 = w[(j + 1) & 15];
			let m7 = w[(j + 9) & 15];
			let m2 = w[(j + 14) & 15];
			// sigma0(x) = ROTR7 ^ ROTR18 ^ SHR3.
			let sigma0 = _mm512_ternarylogic_epi32::<XOR3>(
				_mm512_ror_epi32::<7>(m15),
				_mm512_ror_epi32::<18>(m15),
				_mm512_srli_epi32::<3>(m15),
			);
			// sigma1(x) = ROTR17 ^ ROTR19 ^ SHR10.
			let sigma1 = _mm512_ternarylogic_epi32::<XOR3>(
				_mm512_ror_epi32::<17>(m2),
				_mm512_ror_epi32::<19>(m2),
				_mm512_srli_epi32::<10>(m2),
			);
			w[j] = _mm512_add_epi32(_mm512_add_epi32(w[j], sigma0), _mm512_add_epi32(m7, sigma1));
		}
	}
}

/// Compresses one 64-byte block into each of 16 independent SHA-256 states, in place.
///
/// The states arrive and leave one per lane, so both directions cross the transpose.
/// The eight unused rows of each square are zero, and nothing reads where they land.
///
/// # Panics
///
/// Panics if no kernel covers the lane count.
///
/// # Safety
///
/// The caller must enable `avx512f` and `avx512bw`, so the intrinsics are defined.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn compress256_multi<const N: usize>(
	states: &mut [[u32; 8]; N],
	blocks: &[[u8; BLOCK_LEN]; N],
) {
	assert!(handles_lanes(N), "precondition: the lane count must have a kernel");

	unsafe {
		let mut w = load_message_square(blocks);

		// Transpose the states: 16 rows of eight words in, eight rows of 16 lanes out.
		let mut square = [_mm512_setzero_si512(); 16];
		for (lane, state) in states.iter().enumerate() {
			// Only the low 32 bytes of a row carry a state.
			// The rest stays zero padding.
			let mut row = [0u32; 16];
			row[..8].copy_from_slice(state);
			square[lane] = _mm512_loadu_si512(row.as_ptr().cast::<__m512i>());
		}
		transpose_16x16(&mut square);

		let incoming: [__m512i; 8] = array::from_fn(|t| square[t]);
		let mut state = incoming;

		// The first 16 rounds read the message as it arrives.
		// Each later group of 16 extends the whole window, then consumes it.
		for group in 0..4 {
			if group > 0 {
				extend_window(&mut w);
			}
			for t in 16 * group..16 * group + 16 {
				state = round(state, w[t & 15], K[t]);
			}
		}

		// Davies-Meyer, then transpose the eight rows back into 16 states of eight words.
		let mut out = [_mm512_setzero_si512(); 16];
		for (i, (row, base)) in state.into_iter().zip(incoming).enumerate() {
			out[i] = _mm512_add_epi32(row, base);
		}
		transpose_16x16(&mut out);
		for (lane, state) in states.iter_mut().enumerate() {
			// A row holds that lane's 16 words, and the state is the low eight.
			let low: __m256i = _mm512_castsi512_si256(out[lane]);
			_mm256_storeu_si256(state.as_mut_ptr().cast::<__m256i>(), low);
		}
	}
}
