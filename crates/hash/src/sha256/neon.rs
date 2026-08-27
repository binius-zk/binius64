// Copyright 2026 The Binius Developers

//! Multi-buffer SHA-256 over the ARMv8 crypto extension.
//!
//! The SHA unit is pipelined, so one dependent chain leaves it under-full.
//! Interleaving four independent hashes fills it instead.
//!
//! Four lanes is the widest batch that stays in registers.
//! A lane needs two 128-bit registers for its state and four for its message window, so four
//! lanes fill 24 of the 32 available.
//!
//! Output is bit-identical to the scalar block function, pinned by tests.

use core::arch::aarch64::{
	uint8x16_t, uint32x4_t, vaddq_u32, vdupq_n_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8,
	vrev32q_u8, vsha256h2q_u32, vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
};

use super::{BLOCK_LEN, K};

/// Lane count the kernel is built for.
const LANES: usize = 4;

/// Reports whether this module has a kernel for the given lane count.
#[inline(always)]
pub const fn handles_lanes(n: usize) -> bool {
	n == LANES
}

/// Reads 16 bytes as four big-endian 32-bit words.
///
/// FIPS 180-4 section 5.1 reads a message block most significant byte first.
///
/// # Safety
///
/// The caller must enable `neon`, so the intrinsics are defined.
#[inline(always)]
unsafe fn load_be_words(bytes: uint8x16_t) -> uint32x4_t {
	unsafe { vreinterpretq_u32_u8(vrev32q_u8(bytes)) }
}

/// Runs four rounds across every lane, at the given round-constant offset.
///
/// The extension splits the state into `(a, b, c, d)` and `(e, f, g, h)` halves.
/// The two instructions advance those halves in turn.
/// The second reads the first's input, so that input is saved.
///
/// # Safety
///
/// The caller must enable `sha2`, so the crypto intrinsics are defined.
#[inline(always)]
unsafe fn rounds4<const N: usize>(
	abcd: &mut [uint32x4_t; N],
	efgh: &mut [uint32x4_t; N],
	msg: &[uint32x4_t; N],
	offset: usize,
) {
	unsafe {
		let kv = vld1q_u32(K.as_ptr().add(offset));
		for lane in 0..N {
			let wk = vaddq_u32(msg[lane], kv);
			let prev = abcd[lane];
			abcd[lane] = vsha256hq_u32(abcd[lane], efgh[lane], wk);
			efgh[lane] = vsha256h2q_u32(efgh[lane], prev, wk);
		}
	}
}

/// Extends every lane's message window by four words, rewriting the oldest slot.
///
/// The four arguments are the window in age order, oldest first.
///
/// # Safety
///
/// The caller must enable `sha2`, so the crypto intrinsics are defined.
#[inline(always)]
unsafe fn schedule4<const N: usize>(
	oldest: &mut [uint32x4_t; N],
	w1: &[uint32x4_t; N],
	w2: &[uint32x4_t; N],
	newest: &[uint32x4_t; N],
) {
	unsafe {
		for lane in 0..N {
			let folded = vsha256su0q_u32(oldest[lane], w1[lane]);
			oldest[lane] = vsha256su1q_u32(folded, w2[lane], newest[lane]);
		}
	}
}

/// Compresses one 64-byte block into each of four independent SHA-256 states, in place.
///
/// The four states advance together, one interleaved group of four rounds at a time, so four
/// independent hashes occupy the SHA pipeline at once.
///
/// The state words load and store in native word order, matching the portable block function.
/// Only the message bytes swap to big-endian.
///
/// # Panics
///
/// Panics if no kernel covers the lane count.
///
/// # Safety
///
/// The caller must enable `sha2`, so the crypto intrinsics are defined.
#[inline]
pub unsafe fn compress256_multi<const N: usize>(
	states: &mut [[u32; 8]; N],
	blocks: &[[u8; BLOCK_LEN]; N],
) {
	assert!(handles_lanes(N), "precondition: the lane count must have a kernel");

	unsafe {
		// Split each eight-word state into its two register halves.
		let mut abcd = [vdupq_n_u32(0); N];
		let mut efgh = [vdupq_n_u32(0); N];
		for (lane, state) in states.iter().enumerate() {
			abcd[lane] = vld1q_u32(state.as_ptr());
			efgh[lane] = vld1q_u32(state.as_ptr().add(4));
		}

		// Each lane's 16 message words, as a rolling window of four registers.
		let mut w0 = [vdupq_n_u32(0); N];
		let mut w1 = [vdupq_n_u32(0); N];
		let mut w2 = [vdupq_n_u32(0); N];
		let mut w3 = [vdupq_n_u32(0); N];
		for (lane, block) in blocks.iter().enumerate() {
			let p = block.as_ptr();
			w0[lane] = load_be_words(vld1q_u8(p));
			w1[lane] = load_be_words(vld1q_u8(p.add(16)));
			w2[lane] = load_be_words(vld1q_u8(p.add(32)));
			w3[lane] = load_be_words(vld1q_u8(p.add(48)));
		}

		// Save the incoming state to add back as the Davies-Meyer feed-forward at the end.
		let abcd_save = abcd;
		let efgh_save = efgh;

		// Rounds 0..16 consume the raw message words.
		rounds4(&mut abcd, &mut efgh, &w0, 0);
		rounds4(&mut abcd, &mut efgh, &w1, 4);
		rounds4(&mut abcd, &mut efgh, &w2, 8);
		rounds4(&mut abcd, &mut efgh, &w3, 12);

		// Rounds 16..64: extend the window a quarter at a time, then consume it.
		for group in 1..4 {
			schedule4(&mut w0, &w1, &w2, &w3);
			schedule4(&mut w1, &w2, &w3, &w0);
			schedule4(&mut w2, &w3, &w0, &w1);
			schedule4(&mut w3, &w0, &w1, &w2);
			rounds4(&mut abcd, &mut efgh, &w0, 16 * group);
			rounds4(&mut abcd, &mut efgh, &w1, 16 * group + 4);
			rounds4(&mut abcd, &mut efgh, &w2, 16 * group + 8);
			rounds4(&mut abcd, &mut efgh, &w3, 16 * group + 12);
		}

		// Davies-Meyer, then write the advanced states back in native word order.
		for (lane, state) in states.iter_mut().enumerate() {
			vst1q_u32(state.as_mut_ptr(), vaddq_u32(abcd[lane], abcd_save[lane]));
			vst1q_u32(state.as_mut_ptr().add(4), vaddq_u32(efgh[lane], efgh_save[lane]));
		}
	}
}
