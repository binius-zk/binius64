// Copyright 2026 The Binius Developers

//! Four-way interleaved SHA-256 over the ARMv8 crypto extension.
//!
//! The SHA unit is pipelined, so one dependent hash chain leaves it under-full.
//! Interleaving four independent hashes fills the pipeline and raises throughput.
//!
//! The target is bulk Merkle leaf hashing, where the leaves of a level are independent.
//! The `sha2` crate hashes one stream at a time, so that headroom goes unused.
//!
//! The output equals [`sha2::Sha256`] byte for byte, pinned by the proptests below.
//! Off aarch64, or without the `sha2` feature, it falls back to four `sha2` hashes.

/// A SHA-256 digest.
pub type Hash = [u8; 32];

/// Compresses one 64-byte block into each of four independent SHA-256 states, in place.
///
/// This is the four-way analogue of the portable single-stream block function.
/// Each state absorbs its own block through the full 64 rounds plus the Davies-Meyer add.
/// No padding or length suffix is involved.
/// The caller owns the block contents.
///
/// On aarch64 with the `sha2` feature the four lanes interleave over one SHA unit.
/// Elsewhere the lanes compress one at a time through the portable block function.
#[inline]
pub fn compress256_x4(states: &mut [[u32; 8]; 4], blocks: [&[u8; 64]; 4]) {
	#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
	{
		// SAFETY: the `sha2` target feature is statically enabled, so the crypto intrinsics exist.
		unsafe { neon::compress4_states(states, blocks) }
	}
	#[cfg(not(all(target_arch = "aarch64", target_feature = "sha2")))]
	{
		for (state, block) in states.iter_mut().zip(blocks) {
			sha2::block_api::compress256(state, std::slice::from_ref(block));
		}
	}
}

/// Hashes four equal-length byte inputs into four standard SHA-256 digests.
///
/// The inputs hash as four independent streams.
/// On aarch64 with the `sha2` feature those streams interleave over one SHA unit.
///
/// # Panics
///
/// Panics if the four inputs are not all the same length.
#[inline]
pub fn sha256_x4(inputs: [&[u8]; 4]) -> [Hash; 4] {
	let len = inputs[0].len();
	assert!(
		inputs.iter().all(|input| input.len() == len),
		"the four inputs must have equal length"
	);

	#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
	{
		// SAFETY: the `sha2` target feature is statically enabled, so the crypto intrinsics exist.
		unsafe { neon::hash4_equal_len(inputs) }
	}
	#[cfg(not(all(target_arch = "aarch64", target_feature = "sha2")))]
	{
		scalar::hash4_equal_len(inputs)
	}
}

/// The portable fallback: four independent [`sha2::Sha256`] hashes.
///
/// Used off aarch64, or without the `sha2` target feature, where the NEON kernel is unavailable.
#[cfg(not(all(target_arch = "aarch64", target_feature = "sha2")))]
mod scalar {
	use sha2::{Digest, Sha256};

	use super::Hash;

	/// Hashes four equal-length inputs with the portable `sha2` implementation.
	#[inline]
	pub fn hash4_equal_len(inputs: [&[u8]; 4]) -> [Hash; 4] {
		inputs.map(|input| Sha256::digest(input).into())
	}
}

/// The interleaved kernel built on the ARM SHA-256 crypto intrinsics.
#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
mod neon {
	use core::arch::aarch64::{
		uint32x4_t, vaddq_u32, vdupq_n_u32, vld1q_u8, vld1q_u32, vreinterpretq_u8_u32,
		vreinterpretq_u32_u8, vrev32q_u8, vsha256h2q_u32, vsha256hq_u32, vsha256su0q_u32,
		vsha256su1q_u32, vst1q_u8, vst1q_u32,
	};

	use super::Hash;

	/// The 64 SHA-256 round constants.
	const K: [u32; 64] = [
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
		0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
		0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
		0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
		0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
		0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
		0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
		0xc67178f2,
	];

	/// The SHA-256 initial state, split into the low half (a, b, c, d).
	const IV_ABCD: [u32; 4] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a];
	/// The SHA-256 initial state, split into the high half (e, f, g, h).
	const IV_EFGH: [u32; 4] = [0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

	/// Compresses one 64-byte block into each of four independent SHA-256 states.
	///
	/// The four states advance together, one interleaved round at a time.
	/// Four independent hashes then occupy the SHA pipeline at once.
	///
	/// # Safety
	///
	/// The caller must enable the `sha2` target feature so the crypto intrinsics are defined.
	/// Each `blocks[i]` must point to at least 64 readable bytes.
	#[inline(always)]
	unsafe fn compress4(
		abcd: &mut [uint32x4_t; 4],
		efgh: &mut [uint32x4_t; 4],
		blocks: [*const u8; 4],
	) {
		unsafe {
			// Load each block's 16 message words, byte-swapped to big-endian as SHA-256 expects.
			let mut msg0 = [vdupq_n_u32(0); 4];
			let mut msg1 = [vdupq_n_u32(0); 4];
			let mut msg2 = [vdupq_n_u32(0); 4];
			let mut msg3 = [vdupq_n_u32(0); 4];
			for i in 0..4 {
				msg0[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i])));
				msg1[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(16))));
				msg2[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(32))));
				msg3[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(48))));
			}

			// Save the incoming state to add back as the Davies-Meyer feed-forward at the end.
			let abcd_save = *abcd;
			let efgh_save = *efgh;

			// One group of four rounds across all four states, at round-constant offset `ki`.
			macro_rules! rounds4 {
				($msg:expr, $ki:expr) => {{
					let kv = vld1q_u32(K.as_ptr().add($ki));
					for i in 0..4 {
						let wk = vaddq_u32($msg[i], kv);
						let t = abcd[i];
						abcd[i] = vsha256hq_u32(abcd[i], efgh[i], wk);
						efgh[i] = vsha256h2q_u32(efgh[i], t, wk);
					}
				}};
			}

			// Extend the message schedule by four words across all four states.
			macro_rules! sched {
				($m0:expr, $m1:expr, $m2:expr, $m3:expr) => {
					for i in 0..4 {
						$m0[i] = vsha256su1q_u32(vsha256su0q_u32($m0[i], $m1[i]), $m2[i], $m3[i]);
					}
				};
			}

			// Rounds 0..16 consume the raw message words.
			rounds4!(msg0, 0);
			rounds4!(msg1, 4);
			rounds4!(msg2, 8);
			rounds4!(msg3, 12);
			// Rounds 16..64: schedule the next four words, then consume them.
			for r in 1..4 {
				sched!(msg0, msg1, msg2, msg3);
				sched!(msg1, msg2, msg3, msg0);
				sched!(msg2, msg3, msg0, msg1);
				sched!(msg3, msg0, msg1, msg2);
				rounds4!(msg0, 16 * r);
				rounds4!(msg1, 16 * r + 4);
				rounds4!(msg2, 16 * r + 8);
				rounds4!(msg3, 16 * r + 12);
			}

			// Davies-Meyer: add the saved state back into the compressed state.
			for i in 0..4 {
				abcd[i] = vaddq_u32(abcd[i], abcd_save[i]);
				efgh[i] = vaddq_u32(efgh[i], efgh_save[i]);
			}
		}
	}

	/// Compresses one 64-byte block into each of four independent word states, in place.
	///
	/// The state words load and store in native word order.
	/// No byte-swap of the state happens on either side, matching the portable block function.
	/// Only the message block bytes swap to big-endian, inside the shared round kernel.
	///
	/// # Safety
	///
	/// The caller must enable the `sha2` target feature so the crypto intrinsics are defined.
	#[inline]
	pub unsafe fn compress4_states(states: &mut [[u32; 8]; 4], blocks: [&[u8; 64]; 4]) {
		unsafe {
			// Split each 8-word state into its (a, b, c, d) and (e, f, g, h) register halves.
			let mut abcd = [vdupq_n_u32(0); 4];
			let mut efgh = [vdupq_n_u32(0); 4];
			for i in 0..4 {
				abcd[i] = vld1q_u32(states[i].as_ptr());
				efgh[i] = vld1q_u32(states[i].as_ptr().add(4));
			}

			// One interleaved 64-round compression, Davies-Meyer add included.
			compress4(&mut abcd, &mut efgh, blocks.map(|block| block.as_ptr()));

			// Write the advanced states back in native word order.
			for i in 0..4 {
				vst1q_u32(states[i].as_mut_ptr(), abcd[i]);
				vst1q_u32(states[i].as_mut_ptr().add(4), efgh[i]);
			}
		}
	}

	/// Hashes four equal-length inputs with the interleaved NEON compression.
	///
	/// # Safety
	///
	/// The caller must enable the `sha2` target feature so the crypto intrinsics are defined.
	#[inline]
	pub unsafe fn hash4_equal_len(inputs: [&[u8]; 4]) -> [Hash; 4] {
		let len = inputs[0].len();

		unsafe {
			let mut abcd = [vld1q_u32(IV_ABCD.as_ptr()); 4];
			let mut efgh = [vld1q_u32(IV_EFGH.as_ptr()); 4];

			// Absorb every full 64-byte block of the message.
			let n_full = len / 64;
			for blk in 0..n_full {
				let base = blk * 64;
				compress4(
					&mut abcd,
					&mut efgh,
					[
						inputs[0].as_ptr().add(base),
						inputs[1].as_ptr().add(base),
						inputs[2].as_ptr().add(base),
						inputs[3].as_ptr().add(base),
					],
				);
			}

			// Build the padded tail: leftover bytes, then 0x80, then zeros, then the 64-bit BE
			// length. One padding block when the leftover is <= 55 bytes, otherwise two.
			let rem = len % 64;
			let bit_len = (len as u64) * 8;
			let n_tail = if rem < 56 { 1 } else { 2 };
			let mut tails = [[0u8; 128]; 4];
			for i in 0..4 {
				tails[i][..rem].copy_from_slice(&inputs[i][len - rem..]);
				tails[i][rem] = 0x80;
				tails[i][n_tail * 64 - 8..n_tail * 64].copy_from_slice(&bit_len.to_be_bytes());
			}
			for blk in 0..n_tail {
				let base = blk * 64;
				compress4(
					&mut abcd,
					&mut efgh,
					[
						tails[0].as_ptr().add(base),
						tails[1].as_ptr().add(base),
						tails[2].as_ptr().add(base),
						tails[3].as_ptr().add(base),
					],
				);
			}

			// Serialize each state as big-endian a..h, the standard SHA-256 digest byte order.
			let mut out = [[0u8; 32]; 4];
			for i in 0..4 {
				let be_lo = vrev32q_u8(vreinterpretq_u8_u32(abcd[i]));
				let be_hi = vrev32q_u8(vreinterpretq_u8_u32(efgh[i]));
				vst1q_u8(out[i].as_mut_ptr(), be_lo);
				vst1q_u8(out[i].as_mut_ptr().add(16), be_hi);
			}
			out
		}
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;
	use sha2::{Digest, Sha256, block_api::compress256};

	use super::{compress256_x4, sha256_x4};

	proptest! {
		#[test]
		fn compress256_x4_matches_the_block_reference(
			states in prop::array::uniform4(prop::array::uniform8(any::<u32>())),
			blocks in prop::array::uniform4(any::<[u8; 64]>()),
		) {
			// Advance four independent random states through the four-way kernel.
			// Random states pin the raw-compression contract, not just the fixed IV.
			let mut got = states;
			compress256_x4(&mut got, [&blocks[0], &blocks[1], &blocks[2], &blocks[3]]);

			// Each lane must equal the portable single-stream block function on its own input.
			for i in 0..4 {
				let mut want = states[i];
				compress256(&mut want, std::slice::from_ref(&blocks[i]));
				prop_assert_eq!(got[i], want, "mismatch at lane {}", i);
			}
		}
	}

	proptest! {
		#[test]
		fn matches_sha2_reference(
			inputs in prop::array::uniform4(prop::collection::vec(any::<u8>(), 0..300)),
		) {
			// All four inputs must share a length, so pad each to the longest with zeros.
			let len = inputs.iter().map(Vec::len).max().unwrap();
			let padded: [Vec<u8>; 4] = std::array::from_fn(|i| {
				let mut v = inputs[i].clone();
				v.resize(len, 0);
				v
			});
			let refs: [&[u8]; 4] = std::array::from_fn(|i| padded[i].as_slice());

			let got = sha256_x4(refs);
			for i in 0..4 {
				let want: [u8; 32] = Sha256::digest(refs[i]).into();
				prop_assert_eq!(got[i], want, "mismatch at lane {}", i);
			}
		}
	}

	#[test]
	fn matches_sha2_at_padding_boundaries() {
		for &len in &[0usize, 1, 55, 56, 63, 64, 65, 119, 120, 128, 256] {
			// Distinct bytes per lane so the lanes cannot accidentally coincide.
			let data: [Vec<u8>; 4] = std::array::from_fn(|i| {
				(0..len)
					.map(|j| (j as u8).wrapping_add(i as u8 * 37))
					.collect()
			});
			let refs: [&[u8]; 4] = std::array::from_fn(|i| data[i].as_slice());

			let got = sha256_x4(refs);
			for i in 0..4 {
				let want: [u8; 32] = Sha256::digest(refs[i]).into();
				assert_eq!(got[i], want, "mismatch at len {len}, lane {i}");
			}
		}
	}
}
