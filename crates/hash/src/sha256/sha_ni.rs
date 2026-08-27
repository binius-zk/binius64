// Copyright 2026 The Binius Developers

//! Multi-buffer SHA-256 over the x86-64 SHA extension.
//!
//! One instruction runs two rounds of a single chain, and the next pair waits on it.
//! A lone chain therefore pays that latency 32 times per block.
//! Interleaving independent chains over the one SHA unit spends the idle slots instead.
//!
//! Past four lanes the live values outnumber the registers, so message windows spill.
//! Those spills are L1 hits on a unit that is already the bottleneck, so they stay cheap.
//!
//! Output is bit-identical to the scalar block function, pinned by tests.

use std::arch::x86_64::{
	__m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi32,
	_mm_set_epi64x, _mm_setzero_si128, _mm_sha256msg1_epu32, _mm_sha256msg2_epu32,
	_mm_sha256rnds2_epu32, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_storeu_si128,
};

use super::{BLOCK_LEN, K};

/// Reports whether this module has a kernel for the given lane count.
///
/// Every count is covered: the lanes are independent, so each step is one loop over them.
#[inline(always)]
pub const fn handles_lanes(_n: usize) -> bool {
	true
}

/// Four round constants per register, the earliest round in the low 32-bit lane.
///
/// The two-round instruction reads its first round's `W + K` from the low lane and its
/// second from the next, so the low lane has to be the earlier round.
///
/// # Safety
///
/// The caller must enable `sse2`, so the intrinsics are defined.
#[inline(always)]
unsafe fn round_constants(group: usize) -> __m128i {
	unsafe {
		_mm_set_epi32(
			K[4 * group + 3] as i32,
			K[4 * group + 2] as i32,
			K[4 * group + 1] as i32,
			K[4 * group] as i32,
		)
	}
}

/// Runs four rounds of one chain, from its four `W + K` values.
///
/// The extension splits the state into `(a, b, e, f)` and `(c, d, g, h)` halves, advanced in
/// turn by the two instructions. The high pair of the input is moved down for the second.
///
/// # Safety
///
/// The caller must enable `sha`, `sse2`, and `ssse3`, so the intrinsics are defined.
#[inline(always)]
unsafe fn rounds4(abef: __m128i, cdgh: __m128i, wk: __m128i) -> (__m128i, __m128i) {
	unsafe {
		let cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
		let wk_hi = _mm_shuffle_epi32(wk, 0x0E);
		let abef = _mm_sha256rnds2_epu32(abef, cdgh, wk_hi);
		(abef, cdgh)
	}
}

/// Extends one chain's message window by four words, returning the rewritten slot.
///
/// The four inputs are the window in age order, oldest first.
/// The extension instructions fold in sigma0 and sigma1, and the splice supplies the four
/// words seven positions back, which straddle two slots.
///
/// # Safety
///
/// The caller must enable `sha`, `sse2`, and `ssse3`, so the intrinsics are defined.
#[inline(always)]
unsafe fn schedule4(oldest: __m128i, w1: __m128i, w2: __m128i, newest: __m128i) -> __m128i {
	unsafe {
		let folded = _mm_sha256msg1_epu32(oldest, w1);
		let carried = _mm_alignr_epi8(newest, w2, 4);
		_mm_sha256msg2_epu32(_mm_add_epi32(folded, carried), newest)
	}
}

/// Compresses one 64-byte block into each of a batch of independent SHA-256 states, in place.
///
/// The chains advance together, one interleaved group of four rounds at a time, so every
/// chain in the batch occupies the SHA pipeline at once.
///
/// # Safety
///
/// The caller must enable `sha`, `sse2`, `ssse3`, and `sse4.1`, so the intrinsics are defined.
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
pub unsafe fn compress256_multi<const N: usize>(
	states: &mut [[u32; 8]; N],
	blocks: &[[u8; BLOCK_LEN]; N],
) {
	unsafe {
		// Byte-reverses each 32-bit word, applying the big-endian order of FIPS 180-4 section 5.1.
		let bswap32 =
			_mm_set_epi64x(0x0C0D_0E0F_0809_0A0Bu64 as i64, 0x0405_0607_0001_0203u64 as i64);

		// Shuffle each lane's eight plain words into the extension's two register halves.
		let mut abef = [_mm_setzero_si128(); N];
		let mut cdgh = [_mm_setzero_si128(); N];
		for (lane, state) in states.iter().enumerate() {
			let p: *const __m128i = state.as_ptr().cast();
			let dcba = _mm_loadu_si128(p);
			let hgfe = _mm_loadu_si128(p.add(1));
			let cdab = _mm_shuffle_epi32(dcba, 0xB1);
			let efgh = _mm_shuffle_epi32(hgfe, 0x1B);
			abef[lane] = _mm_alignr_epi8(cdab, efgh, 8);
			cdgh[lane] = _mm_blend_epi16(efgh, cdab, 0xF0);
		}

		// Save the incoming state to add back as the Davies-Meyer feed-forward at the end.
		let abef_save = abef;
		let cdgh_save = cdgh;

		// Each lane's 16 message words, as a rolling window of four registers.
		let mut w = [[_mm_setzero_si128(); 4]; N];
		for (lane, block) in blocks.iter().enumerate() {
			let p: *const __m128i = block.as_ptr().cast();
			for j in 0..4 {
				w[lane][j] = _mm_shuffle_epi8(_mm_loadu_si128(p.add(j)), bswap32);
			}
		}

		// Rounds 0..16 consume the raw message words.
		for group in 0..4 {
			let kv = round_constants(group);
			for lane in 0..N {
				let wk = _mm_add_epi32(w[lane][group], kv);
				(abef[lane], cdgh[lane]) = rounds4(abef[lane], cdgh[lane], wk);
			}
		}

		// Rounds 16..64: extend the window by four words, then consume them.
		// The slot being rewritten is the oldest, at index `group mod 4`.
		for group in 4..16 {
			let kv = round_constants(group);
			let (i0, i1, i2, i3) = (group & 3, (group + 1) & 3, (group + 2) & 3, (group + 3) & 3);
			for lane in 0..N {
				w[lane][i0] = schedule4(w[lane][i0], w[lane][i1], w[lane][i2], w[lane][i3]);
				let wk = _mm_add_epi32(w[lane][i0], kv);
				(abef[lane], cdgh[lane]) = rounds4(abef[lane], cdgh[lane], wk);
			}
		}

		// Davies-Meyer, then undo the register split back to eight plain words.
		for (lane, state) in states.iter_mut().enumerate() {
			let abef_out = _mm_add_epi32(abef[lane], abef_save[lane]);
			let cdgh_out = _mm_add_epi32(cdgh[lane], cdgh_save[lane]);
			let feba = _mm_shuffle_epi32(abef_out, 0x1B);
			let dchg = _mm_shuffle_epi32(cdgh_out, 0xB1);
			let p: *mut __m128i = state.as_mut_ptr().cast();
			_mm_storeu_si128(p, _mm_blend_epi16(feba, dchg, 0xF0));
			_mm_storeu_si128(p.add(1), _mm_alignr_epi8(dchg, feba, 8));
		}
	}
}
