// Copyright 2025 Irreducible Inc.

//! BLAKE3 circuit gadgets.
//!
//! This module provides circuit primitives for the BLAKE3 hash function. The primitives
//! are exposed as free functions that take input wires and return output wires — no
//! wrapping structs.
//!
//! The entry points are:
//! - [`blake3_compress`] — single-block compression primitive.
//! - [`blake3_fixed`] — single-chunk hash gadget for messages of compile-time-known
//!   length up to 1024 bytes.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

pub mod compress;

pub use compress::{blake3_compress, blake3_compress_2x};

/// BLAKE3 initial chaining value. Same as the SHA-256 IV.
pub const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Message schedule for each of the 7 rounds of the BLAKE3 compression function.
///
/// Matches the `MSG_SCHEDULE` constant in the [reference implementation].
///
/// [reference implementation]: https://github.com/BLAKE3-team/BLAKE3/blob/master/src/portable.rs
pub const MSG_SCHEDULE: [[usize; 16]; 7] = [
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
	[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
	[3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
	[10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
	[12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
	[9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
	[11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

// Domain separation flags.
pub const CHUNK_START: u32 = 1 << 0;
pub const CHUNK_END: u32 = 1 << 1;
pub const PARENT: u32 = 1 << 2;
pub const ROOT: u32 = 1 << 3;
pub const KEYED_HASH: u32 = 1 << 4;
pub const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
pub const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

/// Byte length of a BLAKE3 block.
pub const BLOCK_BYTES: usize = 64;

/// Byte length of a BLAKE3 chunk.
pub const CHUNK_BYTES: usize = 1024;

/// Computes the BLAKE3 hash of a compile-time fixed-length message.
///
/// This is the BLAKE3 analog of
/// [`sha256_fixed`](crate::sha256::sha256_fixed) — the message length is known at
/// circuit construction time, which eliminates runtime padding logic.
///
/// # Scope
///
/// Currently restricted to single-chunk inputs (`len_bytes <= 1024`). Multi-chunk
/// hashing requires BLAKE3's tree construction with parent nodes and chunk counters,
/// which is out of scope for this gadget. The underlying [`blake3_compress`] is
/// fully general and can be wrapped to support multi-chunk in a follow-up.
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `message`: Input message as 32-bit little-endian words (4 bytes per wire). The
///   high 32 bits of each wire must be zero. Length must equal
///   `len_bytes.div_ceil(4)`.
/// - `len_bytes`: The compile-time-known length of the message in bytes. Must be at
///   most [`CHUNK_BYTES`] (1024).
///
/// # Returns
///
/// The BLAKE3 digest as 8 wires, each holding a 32-bit little-endian word in its
/// low 32 bits.
pub fn blake3_fixed(builder: &CircuitBuilder, message: &[Wire], len_bytes: usize) -> [Wire; 8] {
	assert!(
		len_bytes <= CHUNK_BYTES,
		"blake3_fixed: len_bytes ({len_bytes}) exceeds single-chunk limit of {CHUNK_BYTES}",
	);
	assert_eq!(
		message.len(),
		len_bytes.div_ceil(4),
		"blake3_fixed: message.len() ({}) must equal len_bytes.div_ceil(4) ({})",
		message.len(),
		len_bytes.div_ceil(4),
	);

	let zero = builder.add_constant(Word::ZERO);

	// Build the padded message as a flat list of 32-bit LE words. BLAKE3 does not
	// append a length field; the trailing partial block is simply zero-filled and
	// its `block_len` parameter records the actual byte count.
	let n_blocks = len_bytes.div_ceil(BLOCK_BYTES).max(1);
	let n_padded_words = n_blocks * 16;

	let n_message_words = len_bytes / 4;
	let boundary_bytes = len_bytes % 4;

	let mut padded: Vec<Wire> = Vec::with_capacity(n_padded_words);
	padded.extend_from_slice(&message[..n_message_words]);
	if boundary_bytes > 0 {
		// Partial trailing word: mask the high bytes to zero (BLAKE3 words are
		// little-endian, so the valid message bytes occupy the low bytes).
		let mask_value = (1u64 << (boundary_bytes * 8)) - 1;
		let mask = builder.add_constant(Word(mask_value));
		padded.push(builder.band(message[n_message_words], mask));
	}
	padded.resize(n_padded_words, zero);

	// Constants used each block.
	let counter = zero;
	let flag_start = builder.add_constant(Word(CHUNK_START as u64));
	let flag_end_root = builder.add_constant(Word((CHUNK_END | ROOT) as u64));
	let flag_start_end_root = builder.add_constant(Word((CHUNK_START | CHUNK_END | ROOT) as u64));
	let flag_zero = zero;

	// Initial chaining value = IV.
	let mut cv: [Wire; 8] = std::array::from_fn(|i| builder.add_constant(Word(IV[i] as u64)));

	for block_idx in 0..n_blocks {
		let block: [Wire; 16] = std::array::from_fn(|i| padded[block_idx * 16 + i]);

		let block_byte_count = if block_idx + 1 == n_blocks {
			len_bytes - block_idx * BLOCK_BYTES
		} else {
			BLOCK_BYTES
		};
		let block_len = builder.add_constant(Word(block_byte_count as u64));

		let flags = match (block_idx == 0, block_idx + 1 == n_blocks) {
			(true, true) => flag_start_end_root,
			(true, false) => flag_start,
			(false, true) => flag_end_root,
			(false, false) => flag_zero,
		};

		cv = blake3_compress(
			&builder.subcircuit(format!("blake3_fixed_compress[{block_idx}]")),
			cv,
			block,
			counter,
			block_len,
			flags,
		);
	}

	cv
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Convert a byte slice into the 32-bit LE word encoding expected by [`blake3_fixed`].
	/// The last word is zero-padded in its high bytes if the length is not a multiple of 4.
	fn bytes_to_le_words(bytes: &[u8]) -> Vec<u64> {
		let n_words = bytes.len().div_ceil(4);
		(0..n_words)
			.map(|i| {
				let mut buf = [0u8; 4];
				let start = i * 4;
				let end = (start + 4).min(bytes.len());
				buf[..end - start].copy_from_slice(&bytes[start..end]);
				u32::from_le_bytes(buf) as u64
			})
			.collect()
	}

	/// Run `blake3_fixed` over `input` and assert it matches `blake3::hash(input)`.
	fn check(input: &[u8]) {
		let builder = CircuitBuilder::new();
		let message: Vec<Wire> = (0..input.len().div_ceil(4))
			.map(|_| builder.add_witness())
			.collect();
		let digest = blake3_fixed(&builder, &message, input.len());
		let digest_out: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("digest_match", digest[i], digest_out[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		let words = bytes_to_le_words(input);
		for (wire, word) in message.iter().zip(words.iter()) {
			w[*wire] = Word(*word);
		}

		let expected = blake3::hash(input);
		let expected_words: [u32; 8] = std::array::from_fn(|i| {
			u32::from_le_bytes(expected.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap())
		});
		for i in 0..8 {
			w[digest_out[i]] = Word(expected_words[i] as u64);
		}
		circuit
			.populate_wire_witness(&mut w)
			.unwrap_or_else(|e| panic!("blake3_fixed failed for len_bytes={}: {e:?}", input.len()));
	}

	#[test]
	fn empty() {
		check(b"");
	}

	#[test]
	fn one_byte() {
		check(&[0x5a]);
	}

	#[test]
	fn abc() {
		check(b"abc");
	}

	#[test]
	fn block_boundaries() {
		for &len in &[1usize, 63, 64, 65, 127, 128, 511, 512, 1023, 1024] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check(&input);
		}
	}

	#[test]
	#[should_panic(expected = "exceeds single-chunk limit")]
	fn rejects_oversize() {
		let builder = CircuitBuilder::new();
		let message: Vec<Wire> = (0..257).map(|_| builder.add_witness()).collect();
		blake3_fixed(&builder, &message, CHUNK_BYTES + 1);
	}
}
