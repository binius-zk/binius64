// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! BLAKE3 circuit gadgets.
//!
//! This module provides circuit primitives for the BLAKE3 hash function. The primitives
//! are exposed as free functions that take input wires and return output wires — no
//! wrapping structs.
//!
//! The entry points are:
//! - [`blake3_compress`] — single-block compression primitive.
//! - [`blake3_compress_2x_seq`] — two sequential compressions sharing one parallel core.
//! - [`blake3_chunk`] — single-chunk (up to 16 blocks) chaining-value gadget.
//! - [`blake3_fixed`] — full hash gadget for messages of compile-time-known length, spanning any
//!   number of chunks via BLAKE3's parent tree.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::util::clear_high_bits;

pub mod compress;

pub use compress::{blake3_compress, blake3_compress_2x, blake3_compress_2x_seq, ref_compress};

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

/// Computes the BLAKE3 chaining value of a single chunk.
///
/// A BLAKE3 chunk is up to 16 blocks (1024 bytes) compressed in a chain: the chaining value is
/// threaded block-to-block starting from the [`IV`]. The first block carries [`CHUNK_START`] and
/// the last carries [`CHUNK_END`]; every block carries the chunk's `counter` (its chunk index).
/// `last_flags_extra` is OR'd into the last block's flags — pass [`ROOT`] when this chunk is the
/// entire message (no parent tree), otherwise `0`.
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `blocks`: the chunk's message blocks (1..=16), each 16 little-endian 32-bit words.
/// - `block_lens`: the byte length (0..=64) of each block; the trailing block may be partial.
/// - `counter`: the chunk index, used as the 64-bit block counter for every block.
/// - `last_flags_extra`: extra flags OR'd into the last block (e.g. [`ROOT`] for a lone chunk).
///
/// # Returns
///
/// The chunk's 8-word chaining value, each word a 32-bit value in its low 32 bits.
pub fn blake3_chunk(
	builder: &CircuitBuilder,
	blocks: &[[Wire; 16]],
	block_lens: &[Wire],
	counter: u64,
	last_flags_extra: u32,
) -> [Wire; 8] {
	let n_blocks = blocks.len();
	assert!((1..=16).contains(&n_blocks), "blake3_chunk: n_blocks ({n_blocks}) must be in 1..=16",);
	assert_eq!(
		block_lens.len(),
		n_blocks,
		"blake3_chunk: block_lens.len() ({}) must equal blocks.len() ({n_blocks})",
		block_lens.len(),
	);

	let zero = builder.add_constant(Word::ZERO);
	let counter = builder.add_constant_64(counter);

	let mut blocks = blocks.to_vec();
	let mut block_lens = block_lens.to_vec();
	let mut flags: Vec<Wire> = (0..n_blocks)
		.map(|j| {
			let start = if j == 0 { CHUNK_START } else { 0 };
			let end = if j + 1 == n_blocks {
				CHUNK_END | last_flags_extra
			} else {
				0
			};
			builder.add_constant(Word((start | end) as u64))
		})
		.collect();

	// Pad to an even block count with one unused dummy block so the blocks pair up uniformly.
	let odd = n_blocks % 2 == 1;
	if odd {
		blocks.push([zero; 16]);
		block_lens.push(zero);
		flags.push(zero);
	}

	// Initial chaining value = IV.
	let mut cv: [Wire; 8] = std::array::from_fn(|i| builder.add_constant(Word(IV[i] as u64)));

	// Compress two blocks at a time: `blake3_compress_2x_seq` chains two sequential block
	// compressions through a single parallel core, roughly halving the per-block cost.
	let n_pairs = blocks.len() / 2;
	for pair in 0..n_pairs {
		let (lo, hi) = (2 * pair, 2 * pair + 1);
		let out = blake3_compress_2x_seq(
			&builder.subcircuit(format!("blake3_chunk_compress[{pair}]")),
			cv,
			[blocks[lo], blocks[hi]],
			counter,
			[block_lens[lo], block_lens[hi]],
			[flags[lo], flags[hi]],
		);
		// The chaining value after the pair is the second compression's output, in the low 32 bits
		// of each word. On a trailing odd block the second lane is the unused dummy, so the chunk's
		// chaining value is instead the first compression's output, in the high 32 bits.
		let last_odd = odd && pair + 1 == n_pairs;
		cv = std::array::from_fn(|i| {
			if last_odd {
				builder.shr(out[i], 32)
			} else {
				clear_high_bits(builder, out[i], 32)
			}
		});
	}

	cv
}

/// One BLAKE3 parent-node compression: combines two child chaining values into one.
///
/// The parent block is the two children concatenated (16 words); the chaining value is the
/// [`IV`], the counter is 0, the block length is [`BLOCK_BYTES`], and the flags are [`PARENT`]
/// (plus [`ROOT`] for the tree root).
fn blake3_parent(
	builder: &CircuitBuilder,
	left: [Wire; 8],
	right: [Wire; 8],
	is_root: bool,
) -> [Wire; 8] {
	let cv: [Wire; 8] = std::array::from_fn(|i| builder.add_constant(Word(IV[i] as u64)));
	let block: [Wire; 16] = std::array::from_fn(|i| if i < 8 { left[i] } else { right[i - 8] });
	let counter = builder.add_constant(Word::ZERO);
	let block_len = builder.add_constant(Word(BLOCK_BYTES as u64));
	let flags = builder.add_constant(Word((PARENT | if is_root { ROOT } else { 0 }) as u64));
	blake3_compress(builder, cv, block, counter, block_len, flags)
}

/// Two independent BLAKE3 parent-node compressions evaluated in the two lanes of
/// [`blake3_compress_2x`].
///
/// Lane 0 combines the pair `a`, lane 1 combines the pair `b`. Each child holds a 32-bit value in
/// its low bits, so a pair is packed into a 64-bit wire by placing lane 0 in bits `[0:32]` and
/// lane 1 in bits `[32:64]`. Returns the two parent chaining values, unpacked back into the
/// low-32 layout.
fn blake3_parent_2x(
	builder: &CircuitBuilder,
	a: ([Wire; 8], [Wire; 8]),
	b: ([Wire; 8], [Wire; 8]),
) -> ([Wire; 8], [Wire; 8]) {
	// lane 0 in the low 32 bits, lane 1 in the high 32 bits; both children have zero high bits,
	// so shifting lane 1 up and XOR-ing is a clean merge.
	let pack = |lo: Wire, hi: Wire| builder.bxor(lo, builder.shl(hi, 32));
	let cv: [Wire; 8] = std::array::from_fn(|i| {
		let w = IV[i] as u64;
		builder.add_constant(Word(w | (w << 32)))
	});
	let block: [Wire; 16] = std::array::from_fn(|i| {
		if i < 8 {
			pack(a.0[i], b.0[i])
		} else {
			pack(a.1[i - 8], b.1[i - 8])
		}
	});
	let zero = builder.add_constant(Word::ZERO);
	let block_len = builder.add_constant(Word((BLOCK_BYTES as u64) | ((BLOCK_BYTES as u64) << 32)));
	let flags = builder.add_constant(Word((PARENT as u64) | ((PARENT as u64) << 32)));
	let out = blake3_compress_2x(builder, cv, block, zero, zero, block_len, flags);
	let cv_a: [Wire; 8] = std::array::from_fn(|i| clear_high_bits(builder, out[i], 32));
	let cv_b: [Wire; 8] = std::array::from_fn(|i| builder.shr(out[i], 32));
	(cv_a, cv_b)
}

/// Folds chunk chaining values into the root digest through BLAKE3's binary parent tree.
///
/// The tree is built bottom-up: at each level, adjacent chaining values are paired and combined by
/// a parent compression, and a lone trailing value is promoted unchanged to the next level. This
/// bottom-up pairing reproduces BLAKE3's canonical left-full tree exactly. Parent compressions are
/// batched two at a time through [`blake3_parent_2x`]; the final root — the last level's single
/// 2->1 compression — carries [`ROOT`].
///
/// Requires at least two chunk chaining values (a single chunk needs no tree).
fn blake3_tree_root(builder: &CircuitBuilder, chunk_cvs: Vec<[Wire; 8]>) -> [Wire; 8] {
	assert!(chunk_cvs.len() >= 2, "blake3_tree_root: needs at least two chunks");

	let mut level = chunk_cvs;
	let mut depth = 0;
	loop {
		// The root is the compression that reduces the final two subtree CVs to one.
		if level.len() == 2 {
			return blake3_parent(
				&builder.subcircuit("blake3_tree_root"),
				level[0],
				level[1],
				true,
			);
		}

		let sub = builder.subcircuit(format!("blake3_tree_level[{depth}]"));
		let n = level.len();
		let n_pairs = n / 2;
		let mut next: Vec<[Wire; 8]> = Vec::with_capacity(n.div_ceil(2));

		// Combine two independent parents per `blake3_compress_2x` call.
		let mut p = 0;
		while p + 1 < n_pairs {
			let (cv_a, cv_b) = blake3_parent_2x(
				&sub,
				(level[2 * p], level[2 * p + 1]),
				(level[2 * p + 2], level[2 * p + 3]),
			);
			next.push(cv_a);
			next.push(cv_b);
			p += 2;
		}
		// A leftover unpaired parent (odd number of pairs) is done single-lane.
		if p < n_pairs {
			next.push(blake3_parent(&sub, level[2 * p], level[2 * p + 1], false));
		}
		// A lone trailing chaining value with no sibling is promoted unchanged.
		if n % 2 == 1 {
			next.push(level[n - 1]);
		}

		level = next;
		depth += 1;
	}
}

/// Computes the BLAKE3 hash of a compile-time fixed-length message.
///
/// The BLAKE3 analog of [`sha256_fixed`](crate::sha256::sha256_fixed): the message length is known
/// at circuit construction time, which fixes the chunk/tree shape and eliminates runtime padding
/// logic.
///
/// The message is split into 1024-byte chunks ([`blake3_chunk`]); each chunk's chaining value is
/// folded into the digest by BLAKE3's binary parent tree, two independent parent compressions at a
/// time via [`blake3_compress_2x`]. The single [`ROOT`] flag lands on the final compression: the
/// lone chunk when the message fits in one chunk, otherwise the tree's root parent.
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `message`: Input message as 32-bit little-endian words (4 bytes per wire). The high 32 bits of
///   each wire must be zero. Length must equal `len_bytes.div_ceil(4)`.
/// - `len_bytes`: The compile-time-known length of the message in bytes.
///
/// # Returns
///
/// The BLAKE3 digest as 8 wires, each holding a 32-bit little-endian word in its
/// low 32 bits.
pub fn blake3_fixed(builder: &CircuitBuilder, message: &[Wire], len_bytes: usize) -> [Wire; 8] {
	assert_eq!(
		message.len(),
		len_bytes.div_ceil(4),
		"blake3_fixed: message.len() ({}) must equal len_bytes.div_ceil(4) ({})",
		message.len(),
		len_bytes.div_ceil(4),
	);

	let zero = builder.add_constant(Word::ZERO);

	// Build the padded message as a flat list of 32-bit LE words. BLAKE3 does not append a length
	// field; the trailing partial block is simply zero-filled and its `block_len` parameter records
	// the actual byte count.
	let n_blocks = len_bytes.div_ceil(BLOCK_BYTES).max(1);
	let n_padded_words = n_blocks * 16;

	let n_message_words = len_bytes / 4;
	let boundary_bytes = len_bytes % 4;

	let mut padded: Vec<Wire> = Vec::with_capacity(n_padded_words);
	padded.extend_from_slice(&message[..n_message_words]);
	if boundary_bytes > 0 {
		// Partial trailing word: mask the high bytes to zero (BLAKE3 words are little-endian, so
		// the valid message bytes occupy the low bytes).
		let mask_value = (1u64 << (boundary_bytes * 8)) - 1;
		let mask = builder.add_constant(Word(mask_value));
		padded.push(builder.band(message[n_message_words], mask));
	}
	padded.resize(n_padded_words, zero);

	let block = |j: usize| -> [Wire; 16] { std::array::from_fn(|i| padded[j * 16 + i]) };
	let block_len = |j: usize| -> Wire {
		let len = if j + 1 == n_blocks {
			len_bytes - j * BLOCK_BYTES
		} else {
			BLOCK_BYTES
		};
		builder.add_constant(Word(len as u64))
	};

	// One chaining value per chunk. Every chunk but the last is a full 16 blocks (1024 bytes).
	let n_chunks = len_bytes.div_ceil(CHUNK_BYTES).max(1);
	let blocks_per_chunk = CHUNK_BYTES / BLOCK_BYTES;
	let chunk_cvs: Vec<[Wire; 8]> = (0..n_chunks)
		.map(|c| {
			let block_start = c * blocks_per_chunk;
			let block_end = ((c + 1) * blocks_per_chunk).min(n_blocks);
			let blocks: Vec<[Wire; 16]> = (block_start..block_end).map(block).collect();
			let block_lens: Vec<Wire> = (block_start..block_end).map(block_len).collect();
			// ROOT lands on the lone chunk directly; with multiple chunks it moves to the tree
			// root.
			let last_flags_extra = if n_chunks == 1 { ROOT } else { 0 };
			blake3_chunk(
				&builder.subcircuit(format!("blake3_chunk[{c}]")),
				&blocks,
				&block_lens,
				c as u64,
				last_flags_extra,
			)
		})
		.collect();

	// A single chunk is its own digest; otherwise fold the chunk chaining values through the tree.
	if n_chunks == 1 {
		chunk_cvs[0]
	} else {
		blake3_tree_root(builder, chunk_cvs)
	}
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
		// Lengths chosen to cover 1..=16 blocks, including odd block counts (3, 5, 7) that
		// exercise the trailing single-block compression after the 2x-sequential pairs.
		for &len in &[
			1usize, 63, 64, 65, 127, 128, 129, 192, 256, 257, 320, 448, 511, 512, 1023, 1024,
		] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check(&input);
		}
	}

	#[test]
	fn multi_chunk() {
		// Lengths spanning 2..=10 chunks, including odd chunk counts (3, 5, 7, 9) and a partial
		// final chunk, to exercise the parent tree: the 2x-batched parents, the single-lane
		// leftover parent, the lone-chaining-value promotion, and the ROOT node.
		for &len in &[
			1025usize, // 2 chunks: 16 blocks + 1 block
			2048,      // 2 full chunks
			2049,      // 3 chunks
			3072,      // 3 full chunks
			4096,      // 4 full chunks
			5121,      // 5 chunks (odd), partial final chunk
			7168,      // 7 full chunks
			8192,      // 8 full chunks (balanced tree)
			9217,      // 9 chunks (odd), partial final chunk
			10240,     // 10 full chunks
		] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check(&input);
		}
	}
}
