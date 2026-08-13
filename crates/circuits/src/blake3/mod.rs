// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! BLAKE3 circuit gadgets.
//!
//! This module provides circuit primitives for the BLAKE3 hash function. The primitives
//! are exposed as free functions that take input wires and return output wires — no
//! wrapping structs.
//!
//! The entry points are:
//! - [`blake3_compress`] — single-block compression primitive, its rounds split across the two
//!   32-bit lanes.
//! - [`blake3_compress_2x_seq`] — two sequential compressions sharing one parallel core.
//! - [`blake3_chunk`] — single-chunk (up to 16 blocks) chaining-value gadget.
//! - [`blake3_fixed`] — full hash gadget for messages of compile-time-known length, spanning any
//!   number of chunks via BLAKE3's parent tree.
//! - [`blake3_keyed_fixed`] — the same gadget in BLAKE3's keyed mode.
//! - [`blake3_keyed_fixed_2x`] — two keyed hashes of equal-length messages, run side by side.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::{
	fixed_byte_vec::ByteVec,
	util::{clear_high_bits, zeroed_u32_words},
};

pub mod compress;

pub use compress::{
	Blake3Compress2x, blake3_compress, blake3_compress_2x, blake3_compress_2x_seq, ref_compress,
};

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

/// Byte length of a BLAKE3 key.
pub const KEY_BYTES: usize = 32;

/// The chaining value a chunk or parent node starts from: the key in keyed mode, the [`IV`]
/// otherwise.
fn init_cv(builder: &CircuitBuilder, key: Option<[Wire; 8]>) -> [Wire; 8] {
	key.unwrap_or_else(|| std::array::from_fn(|i| builder.add_constant(Word(IV[i] as u64))))
}

/// The flag every compression carries in keyed mode, and nothing in unkeyed mode.
///
/// Keying a BLAKE3 hash is exactly two substitutions — the key replaces the [`IV`] as the starting
/// chaining value, and [`KEYED_HASH`] joins the flags — so the two travel together as one
/// `Option`.
const fn key_flag(key: Option<[Wire; 8]>) -> u32 {
	if key.is_some() { KEYED_HASH } else { 0 }
}

/// Packs `lo` into bits `[0:32]` and `hi` into bits `[32:64]` of one wire.
///
/// The shift clears `hi`'s high half, so only `lo` must arrive with a zero one.
fn pack_lanes(builder: &CircuitBuilder, lo: Wire, hi: Wire) -> Wire {
	builder.bxor(lo, builder.shl(hi, 32))
}

/// A constant holding `value` in both 32-bit lanes, the form a shared parameter takes.
fn dup32(builder: &CircuitBuilder, value: u32) -> Wire {
	let value = value as u64;
	builder.add_constant(Word(value | (value << 32)))
}

/// Computes the BLAKE3 chaining value of a single chunk.
///
/// A BLAKE3 chunk is up to 16 blocks (1024 bytes) compressed in a chain: the chaining value is
/// threaded block-to-block starting from `key`, or from the [`IV`] when unkeyed. The first block
/// carries [`CHUNK_START`] and the last carries [`CHUNK_END`]; every block carries the chunk's
/// `counter` (its chunk index). `last_flags_extra` is OR'd into the last block's flags — pass
/// [`ROOT`] when this chunk is the entire message (no parent tree), otherwise `0`.
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `key`: the 8-word key in keyed mode, `None` in unkeyed mode.
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
	key: Option<[Wire; 8]>,
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

	let counter = builder.add_constant_64(counter);

	let flags: Vec<Wire> = (0..n_blocks)
		.map(|j| {
			let start = if j == 0 { CHUNK_START } else { 0 };
			let end = if j + 1 == n_blocks {
				CHUNK_END | last_flags_extra
			} else {
				0
			};
			builder.add_constant(Word((start | end | key_flag(key)) as u64))
		})
		.collect();

	let mut cv = init_cv(builder, key);

	// Compress two blocks at a time: `blake3_compress_2x_seq` chains two sequential block
	// compressions through a single parallel core, roughly halving the per-block cost.
	//
	// The threaded chaining value carries the pair's first compression in its high half, and
	// that half is left as it is rather than masked off. Nothing downstream reads it:
	//
	// - A compression never lets a carry or a rotate cross bit 32, so the halves stay apart.
	// - The paired core takes an input chaining value's low half only, through a left shift.
	//
	// So the low half of a result depends on the low halves of its inputs alone.
	let n_pairs = n_blocks / 2;
	for pair in 0..n_pairs {
		let (lo, hi) = (2 * pair, 2 * pair + 1);
		// The chaining value after the pair is the second compression's output, in the low half.
		cv = blake3_compress_2x_seq(
			&builder.subcircuit(format!("blake3_chunk_compress[{pair}]")),
			cv,
			[blocks[lo], blocks[hi]],
			counter,
			[block_lens[lo], block_lens[hi]],
			[flags[lo], flags[hi]],
		);
	}

	// A trailing block with no partner is compressed on its own.
	//
	//     6 blocks: [0,1] [2,3] [4,5]              -> 3 paired cores
	//     7 blocks: [0,1] [2,3] [4,5] then 6 alone -> 3 paired cores + 1 lone compression
	//
	// Why not pad to an even count with a dummy block:
	// - The paired core runs all 7 rounds, where a lone compression splits its own rounds across
	//   the two lanes and runs 4.
	// - The pair also packs a second lane's counter, length and flags.
	// - All of that would be spent producing a result nothing reads.
	if n_blocks % 2 == 1 {
		let last = n_blocks - 1;
		cv = blake3_compress(
			&builder.subcircuit("blake3_chunk_compress[last]"),
			cv,
			blocks[last],
			counter,
			block_lens[last],
			flags[last],
		);
	}

	// The escaping value is the one place a clean high half is required, so mask once here
	// rather than after every pair. Callers do read those bits:
	//
	// - A parent node merges two children with a left shift and an exclusive-or.
	// - A digest is compared over all 64 bits of each word.
	std::array::from_fn(|i| clear_high_bits(builder, cv[i], 32))
}

/// One BLAKE3 parent-node compression: combines two child chaining values into one.
///
/// The parent block is the two children concatenated (16 words); the chaining value is the key (or
/// the [`IV`] when unkeyed), the counter is 0, the block length is [`BLOCK_BYTES`], and the flags
/// are [`PARENT`] (plus [`ROOT`] for the tree root).
///
/// The result's high halves are cleared, since a parent's value is read whole: the level above
/// merges it with a shift, and the root is the digest.
fn blake3_parent(
	builder: &CircuitBuilder,
	key: Option<[Wire; 8]>,
	left: [Wire; 8],
	right: [Wire; 8],
	is_root: bool,
) -> [Wire; 8] {
	let cv = init_cv(builder, key);
	let block: [Wire; 16] = std::array::from_fn(|i| if i < 8 { left[i] } else { right[i - 8] });
	let counter = builder.add_constant(Word::ZERO);
	let block_len = builder.add_constant(Word(BLOCK_BYTES as u64));
	let root_flag = if is_root { ROOT } else { 0 };
	let flags = builder.add_constant(Word((PARENT | root_flag | key_flag(key)) as u64));
	let out = blake3_compress(builder, cv, block, counter, block_len, flags);
	std::array::from_fn(|i| clear_high_bits(builder, out[i], 32))
}

/// Two independent BLAKE3 parent-node compressions of one hash, evaluated in the two lanes of
/// [`blake3_compress_2x`].
///
/// Lane 0 combines the pair `a`, lane 1 combines the pair `b`. Each child holds a 32-bit value in
/// its low bits, so a pair is packed into a 64-bit wire by placing lane 0 in bits `[0:32]` and
/// lane 1 in bits `[32:64]`. Returns the two parent chaining values, unpacked back into the
/// low-32 layout.
fn blake3_parent_pair(
	builder: &CircuitBuilder,
	key: Option<[Wire; 8]>,
	a: ([Wire; 8], [Wire; 8]),
	b: ([Wire; 8], [Wire; 8]),
) -> ([Wire; 8], [Wire; 8]) {
	// Both lanes start from the same chaining value, so each word is that word in both halves.
	// The key's words carry zero high bits, so they pack by the same shift-and-XOR as a child.
	let cv: [Wire; 8] = match key {
		Some(key) => std::array::from_fn(|i| pack_lanes(builder, key[i], key[i])),
		None => std::array::from_fn(|i| dup32(builder, IV[i])),
	};
	let block: [Wire; 16] = std::array::from_fn(|i| {
		if i < 8 {
			pack_lanes(builder, a.0[i], b.0[i])
		} else {
			pack_lanes(builder, a.1[i - 8], b.1[i - 8])
		}
	});
	let zero = builder.add_constant(Word::ZERO);
	let block_len = dup32(builder, BLOCK_BYTES as u32);
	let flags = dup32(builder, PARENT | key_flag(key));
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
/// batched two at a time through [`blake3_parent_pair`]; the final root — the last level's single
/// 2->1 compression — carries [`ROOT`].
///
/// Requires at least two chunk chaining values (a single chunk needs no tree).
fn blake3_tree_root(
	builder: &CircuitBuilder,
	key: Option<[Wire; 8]>,
	chunk_cvs: Vec<[Wire; 8]>,
) -> [Wire; 8] {
	assert!(chunk_cvs.len() >= 2, "blake3_tree_root: needs at least two chunks");

	let mut level = chunk_cvs;
	let mut depth = 0;
	loop {
		// The root is the compression that reduces the final two subtree CVs to one.
		if level.len() == 2 {
			return blake3_parent(
				&builder.subcircuit("blake3_tree_root"),
				key,
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
			let (cv_a, cv_b) = blake3_parent_pair(
				&sub,
				key,
				(level[2 * p], level[2 * p + 1]),
				(level[2 * p + 2], level[2 * p + 3]),
			);
			next.push(cv_a);
			next.push(cv_b);
			p += 2;
		}
		// A leftover unpaired parent (odd number of pairs) is done single-lane.
		if p < n_pairs {
			next.push(blake3_parent(&sub, key, level[2 * p], level[2 * p + 1], false));
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
	blake3_hash_fixed(builder, None, message, len_bytes)
}

/// Computes the keyed BLAKE3 hash of a compile-time fixed-length message.
///
/// The keyed mode of [`blake3_fixed`]: the 32-byte key replaces the [`IV`] as the chaining value
/// every chunk and parent node starts from, and every compression carries [`KEYED_HASH`]. This is
/// BLAKE3's native keying, so the digest matches [`blake3::keyed_hash`].
///
/// A key shorter than 32 bytes is zero-padded to 32 — the length of the key vector is not itself
/// hashed, so keys that agree after padding (`b"k"` and `b"k\0"`) produce the same digest.
///
/// [`blake3::keyed_hash`]: https://docs.rs/blake3/latest/blake3/fn.keyed_hash.html
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `message`: as in [`blake3_fixed`].
/// - `len_bytes`: as in [`blake3_fixed`].
/// - `key`: the key, 0 to 32 bytes packed little-endian 8 per wire. Its length must be fixed at
///   circuit construction time (a point `len_range`, as [`ByteVec::new_const_len`] builds). Bytes
///   past that length are masked to zero, so a prover cannot steer the digest through them.
///
/// # Returns
///
/// As in [`blake3_fixed`].
pub fn blake3_keyed_fixed(
	builder: &CircuitBuilder,
	message: &[Wire],
	len_bytes: usize,
	key: &ByteVec,
) -> [Wire; 8] {
	blake3_hash_fixed(builder, Some(key_words(builder, key)), message, len_bytes)
}

/// The 8 words a keyed hash seeds its chaining value with.
///
/// - A byte vector packs 8 bytes per wire and a key is 8 little-endian 32-bit words, so splitting
///   each wire in half is the whole conversion.
/// - Bytes past the key length are masked to zero, so a prover cannot steer the digest with them.
/// - Every word comes out with a zero high half, so a caller can pack two keys one per lane.
///
/// # Panics
///
/// - If the key length is not fixed at circuit construction time.
/// - If the key is longer than [`KEY_BYTES`].
fn key_words(builder: &CircuitBuilder, key: &ByteVec) -> [Wire; 8] {
	assert_eq!(
		key.len_range.start(),
		key.len_range.end(),
		"BLAKE3: the key length must be fixed at circuit construction time, but len_range is {:?}",
		key.len_range,
	);
	let key_len = *key.len_range.start();
	assert!(key_len <= KEY_BYTES, "BLAKE3: key length ({key_len}) exceeds {KEY_BYTES}");

	let words = zeroed_u32_words(builder, &key.data, key_len, 8);
	std::array::from_fn(|i| words[i])
}

/// The message zero-padded to whole blocks, as 32-bit little-endian words.
///
/// BLAKE3 appends no length field: the real byte count travels in each block's length parameter.
///
/// Masking the high halves costs a gate per message word, and is needed only for a two-lane
/// packing, where that half is the other lane rather than dead space.
///
/// # Panics
///
/// If the message is not `len_bytes.div_ceil(4)` wires long.
fn padded_message_words(
	builder: &CircuitBuilder,
	message: &[Wire],
	len_bytes: usize,
	mask_high_halves: bool,
) -> Vec<Wire> {
	assert_eq!(
		message.len(),
		len_bytes.div_ceil(4),
		"blake3: message.len() ({}) must equal len_bytes.div_ceil(4) ({})",
		message.len(),
		len_bytes.div_ceil(4),
	);

	let n_padded_words = len_bytes.div_ceil(BLOCK_BYTES).max(1) * 16;
	let n_whole_words = len_bytes / 4;
	let boundary_bytes = len_bytes % 4;

	let mut padded: Vec<Wire> = Vec::with_capacity(n_padded_words);
	padded.extend(message[..n_whole_words].iter().map(|&w| {
		if mask_high_halves {
			clear_high_bits(builder, w, 32)
		} else {
			w
		}
	}));
	if boundary_bytes > 0 {
		// Partial trailing word: a little-endian word holds its valid bytes low, so mask the rest.
		// At most three survive, so the high half comes out clean either way.
		let mask_value = (1u64 << (boundary_bytes * 8)) - 1;
		let mask = builder.add_constant(Word(mask_value));
		padded.push(builder.band(message[n_whole_words], mask));
	}
	// The padding words are the zero constant, so their high halves are already clean.
	padded.resize(n_padded_words, builder.add_constant(Word::ZERO));
	padded
}

/// The length of block `j`: a whole block, or the remainder for the final one.
fn block_len_bytes(len_bytes: usize, j: usize) -> usize {
	(len_bytes - j * BLOCK_BYTES).min(BLOCK_BYTES)
}

/// The body shared by [`blake3_fixed`] and [`blake3_keyed_fixed`]: the two differ only in the
/// chaining value chunks and parent nodes start from, and the [`KEYED_HASH`] flag that rides along
/// with it.
fn blake3_hash_fixed(
	builder: &CircuitBuilder,
	key: Option<[Wire; 8]>,
	message: &[Wire],
	len_bytes: usize,
) -> [Wire; 8] {
	let n_blocks = len_bytes.div_ceil(BLOCK_BYTES).max(1);
	// One lane, so a dirty high half never reaches the result and needs no masking.
	let padded = padded_message_words(builder, message, len_bytes, false);

	let block = |j: usize| -> [Wire; 16] { std::array::from_fn(|i| padded[j * 16 + i]) };
	let block_len =
		|j: usize| -> Wire { builder.add_constant(Word(block_len_bytes(len_bytes, j) as u64)) };

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
				key,
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
		blake3_tree_root(builder, key, chunk_cvs)
	}
}

/// Computes two keyed BLAKE3 hashes of equal-length messages side by side.
///
/// Each digest matches [`blake3_keyed_fixed`] on its own message and key.
///
/// ```text
///     bits [0:32]  = lane 0: keys[0], messages[0]  --\
///                                                     >-- one paired core per block
///     bits [32:64] = lane 1: keys[1], messages[1]  --/
/// ```
///
/// Equal lengths are what put the two hashes in lockstep.
/// - The block count, the block lengths, the flags and the tree shape all agree.
/// - So every compression of one hash has exactly one partner in the other.
///
/// The saving lands where a single hash has an odd block with no partner to pair with.
/// - That block goes through [`blake3_compress`], which fills the lanes by splitting its own rounds
///   across them and so evaluates 3 of its 7 rounds twice.
/// - Pairing two hashes puts a real second hash in that lane instead, and the duplicated rounds are
///   what it recovers: 336 AND constraints against 384, for a one-block message.
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `messages`: the two messages, each as in [`blake3_fixed`], their high halves masked here so
///   one cannot spill into the other lane.
/// - `len_bytes`: the compile-time-known length both messages share.
/// - `keys`: the two keys, each as in [`blake3_keyed_fixed`], and free to differ in length.
///
/// # Returns
///
/// The two digests, each as in [`blake3_fixed`].
///
/// # Panics
///
/// - If either message is not `len_bytes.div_ceil(4)` wires long.
/// - If either key's length is not fixed at circuit construction time.
/// - If either key is longer than [`KEY_BYTES`].
pub fn blake3_keyed_fixed_2x(
	builder: &CircuitBuilder,
	messages: [&[Wire]; 2],
	len_bytes: usize,
	keys: [&ByteVec; 2],
) -> [[Wire; 8]; 2] {
	// The key words carry zero high halves, so packing is one shift and one exclusive-or.
	let lanes = keys.map(|key| key_words(builder, key));
	let key_2x: [Wire; 8] = std::array::from_fn(|i| pack_lanes(builder, lanes[0][i], lanes[1][i]));

	// Pad each message on its own, then merge word by word into the two-lane layout.
	let padded = messages.map(|message| padded_message_words(builder, message, len_bytes, true));
	let n_blocks = len_bytes.div_ceil(BLOCK_BYTES).max(1);
	let n_content_words = len_bytes.div_ceil(4);
	let zero = builder.add_constant(Word::ZERO);
	let block = |j: usize| -> [Wire; 16] {
		std::array::from_fn(|i| {
			let k = j * 16 + i;
			// Padding is the zero constant in both lanes, so packing it would only spend shifts.
			if k >= n_content_words {
				zero
			} else {
				pack_lanes(builder, padded[0][k], padded[1][k])
			}
		})
	};
	// Every parameter but the message itself is shared, so it enters both lanes as one constant.
	let block_len = |j: usize| -> Wire { dup32(builder, block_len_bytes(len_bytes, j) as u32) };

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
			blake3_chunk_2x(
				&builder.subcircuit(format!("blake3_chunk_2x[{c}]")),
				key_2x,
				&blocks,
				&block_lens,
				c as u64,
				last_flags_extra,
			)
		})
		.collect();

	let root = if n_chunks == 1 {
		chunk_cvs[0]
	} else {
		blake3_tree_root_2x(builder, key_2x, chunk_cvs)
	};

	// Unpack the digests into the one-lane layout the single-hash gadgets return.
	[
		std::array::from_fn(|i| clear_high_bits(builder, root[i], 32)),
		std::array::from_fn(|i| builder.shr(root[i], 32)),
	]
}

/// Computes one chunk's chaining value in each of two keyed hashes at once.
///
/// The two-lane counterpart of [`blake3_chunk`], with one hash per lane.
///
/// Blocks still chain one into the next, but the lanes hold two hashes rather than two blocks, so
/// the paired core needs no hint to resolve that dependency.
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `key_2x`: the two keys packed one per lane, the chaining value both hashes start from.
/// - `blocks`: the chunk's message blocks (1..=16), each 16 words with both lanes packed.
/// - `block_lens`: each block's byte length, the same value in both lanes.
/// - `counter`: the chunk index, used as the 64-bit block counter for every block.
/// - `last_flags_extra`: extra flags OR'd into the last block, e.g. [`ROOT`] for a lone chunk.
///
/// # Returns
///
/// The chunk's chaining value for both hashes, packed one per lane.
fn blake3_chunk_2x(
	builder: &CircuitBuilder,
	key_2x: [Wire; 8],
	blocks: &[[Wire; 16]],
	block_lens: &[Wire],
	counter: u64,
	last_flags_extra: u32,
) -> [Wire; 8] {
	let n_blocks = blocks.len();
	assert!(
		(1..=16).contains(&n_blocks),
		"blake3_chunk_2x: n_blocks ({n_blocks}) must be in 1..=16",
	);
	assert_eq!(
		block_lens.len(),
		n_blocks,
		"blake3_chunk_2x: block_lens.len() ({}) must equal blocks.len() ({n_blocks})",
		block_lens.len(),
	);

	let counter_lo = dup32(builder, counter as u32);
	let counter_hi = dup32(builder, (counter >> 32) as u32);

	let mut cv = key_2x;
	for (j, block) in blocks.iter().enumerate() {
		let start = if j == 0 { CHUNK_START } else { 0 };
		let end = if j + 1 == n_blocks {
			CHUNK_END | last_flags_extra
		} else {
			0
		};
		let flags = dup32(builder, start | end | KEYED_HASH);
		cv = blake3_compress_2x(
			&builder.subcircuit(format!("blake3_chunk_2x_compress[{j}]")),
			cv,
			*block,
			counter_lo,
			counter_hi,
			block_lens[j],
			flags,
		);
	}

	// Both halves of every word are live — one hash each — so nothing is masked off here.
	cv
}

/// One parent-node compression in each of two keyed hashes at once.
///
/// The children already carry one hash per lane, so the parent block is the two concatenated.
fn blake3_parent_2x(
	builder: &CircuitBuilder,
	key_2x: [Wire; 8],
	left: [Wire; 8],
	right: [Wire; 8],
	is_root: bool,
) -> [Wire; 8] {
	let block: [Wire; 16] = std::array::from_fn(|i| if i < 8 { left[i] } else { right[i - 8] });
	let zero = builder.add_constant(Word::ZERO);
	let block_len = dup32(builder, BLOCK_BYTES as u32);
	let root_flag = if is_root { ROOT } else { 0 };
	let flags = dup32(builder, PARENT | root_flag | KEYED_HASH);
	blake3_compress_2x(builder, key_2x, block, zero, zero, block_len, flags)
}

/// Folds two hashes' chunk chaining values into their root digests, one hash per lane.
///
/// Simpler than the one-lane tree: the two shapes agree, so every parent already has a partner
/// and none is left over to batch.
///
/// - Levels are built bottom-up, and a lone trailing value is promoted unchanged.
/// - The final 2->1 compression carries [`ROOT`].
/// - Requires at least two chunk chaining values, since a single chunk needs no tree.
fn blake3_tree_root_2x(
	builder: &CircuitBuilder,
	key_2x: [Wire; 8],
	chunk_cvs: Vec<[Wire; 8]>,
) -> [Wire; 8] {
	assert!(chunk_cvs.len() >= 2, "blake3_tree_root_2x: needs at least two chunks");

	let mut level = chunk_cvs;
	let mut depth = 0;
	loop {
		// The root is the compression that reduces the final two subtree CVs to one.
		if level.len() == 2 {
			return blake3_parent_2x(
				&builder.subcircuit("blake3_tree_root_2x"),
				key_2x,
				level[0],
				level[1],
				true,
			);
		}

		let sub = builder.subcircuit(format!("blake3_tree_level_2x[{depth}]"));
		let n = level.len();
		let mut next: Vec<[Wire; 8]> = Vec::with_capacity(n.div_ceil(2));
		for p in 0..n / 2 {
			next.push(blake3_parent_2x(&sub, key_2x, level[2 * p], level[2 * p + 1], false));
		}
		// A lone trailing chaining value with no sibling is promoted unchanged.
		if n % 2 == 1 {
			next.push(level[n - 1]);
		}

		level = next;
		depth += 1;
	}
}

#[cfg(test)]
mod tests {
	use binius_frontend::CircuitStat;
	use hex_literal::hex;
	use proptest::prelude::*;

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

	/// Hashes `input` in-circuit and asserts the digest equals `expected`.
	///
	/// The digest wires are public inputs, so filling them with the expected bytes turns the
	/// in-circuit equality into the assertion under test.
	///
	/// A disagreement therefore surfaces as a failure to populate the witness.
	fn check_digest(input: &[u8], expected: [u8; 32]) {
		let builder = CircuitBuilder::new();
		// The message is private, one 32-bit little-endian word per wire.
		let message: Vec<Wire> = (0..input.len().div_ceil(4))
			.map(|_| builder.add_witness())
			.collect();
		let digest = blake3_fixed(&builder, &message, input.len());
		// The expected digest is public, and pinned word for word against the computed one.
		let digest_out: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("digest_match", digest[i], digest_out[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for (wire, word) in message.iter().zip(bytes_to_le_words(input)) {
			w[*wire] = Word(word);
		}
		// Read the vector back in the same little-endian word order the circuit produces.
		for i in 0..8 {
			let bytes: [u8; 4] = expected[i * 4..i * 4 + 4].try_into().unwrap();
			w[digest_out[i]] = Word(u32::from_le_bytes(bytes) as u64);
		}
		circuit
			.populate_wire_witness(&mut w)
			.unwrap_or_else(|e| panic!("digest disagreed with the specification vector: {e:?}"));
	}

	#[test]
	fn draft_b1_digest_matches_spec() {
		// Fixture state: Appendix B.1 of the specification draft.
		// A 4-byte message is one block, one chunk, and its own tree root.
		check_digest(
			b"IETF",
			hex!("83a2de1ee6f4e6ab686889248f4ec0cf4cc5709446a682ffd1cbb4d6165181e2"),
		);
	}

	#[test]
	fn draft_b2_digest_matches_spec() {
		// Fixture state: Appendix B.2 of the specification draft.
		// Two full chunks, so 32 block compressions folded through one parent node.
		//
		//     chunk 0: 1024 bytes of 0xaa --\
		//                                    >-- parent (root) --> digest
		//     chunk 1: 1024 bytes of 0xbb --/
		//
		// The trace is unkeyed despite its section title.
		// - Its chaining value is the standard initial value, not a key.
		// - Its flags carry no keyed-hash bit.
		let mut input = vec![0xaau8; CHUNK_BYTES];
		input.extend_from_slice(&[0xbbu8; CHUNK_BYTES]);
		check_digest(
			&input,
			hex!("e79d2838915accd3b21bb0ba76b5edf8dc08d3d78d0db65b713f0f37ec58c346"),
		);
	}

	proptest! {
		// Every case compiles a whole hashing circuit, so the sample stays small.
		// The length range spans 0 to 10 blocks, which covers both parities.
		// Odd block counts are what reach the one-lane trailing compression.
		#![proptest_config(ProptestConfig::with_cases(12))]

		#[test]
		fn fixed_matches_blake3_crate(input in prop::collection::vec(any::<u8>(), 0..=600)) {
			// Random content at a random length, checked against the reference crate.
			check(&input);
		}
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

	/// Run `blake3_keyed_fixed` over `input` and assert it matches the reference crate keyed with
	/// `key` zero-padded to 32 bytes.
	///
	/// The key wires are witness, so `garbage_padding` populates the bytes past the key length
	/// with 0xff instead of zero: the digest must not move, since the gadget masks them.
	fn check_keyed(input: &[u8], key: &[u8], garbage_padding: bool) {
		let builder = CircuitBuilder::new();
		let message: Vec<Wire> = (0..input.len().div_ceil(4))
			.map(|_| builder.add_witness())
			.collect();
		let key_data: Vec<Wire> = (0..key.len().div_ceil(8))
			.map(|_| builder.add_witness())
			.collect();
		let key_vec = ByteVec::new_const_len(&builder, key_data, key.len());
		let digest = blake3_keyed_fixed(&builder, &message, input.len(), &key_vec);
		let digest_out: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("digest_match", digest[i], digest_out[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for (wire, word) in message.iter().zip(bytes_to_le_words(input)) {
			w[*wire] = Word(word);
		}
		// Pad out to whole wires, so the bytes past the key length are populated too.
		let mut key_bytes = key.to_vec();
		key_bytes.resize(key.len().next_multiple_of(8), if garbage_padding { 0xff } else { 0 });
		for (i, chunk) in key_bytes.chunks(8).enumerate() {
			w[key_vec.data[i]] = Word(u64::from_le_bytes(chunk.try_into().unwrap()));
		}

		let mut padded_key = [0u8; KEY_BYTES];
		padded_key[..key.len()].copy_from_slice(key);
		let expected = blake3::keyed_hash(&padded_key, input);
		for i in 0..8 {
			let bytes: [u8; 4] = expected.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap();
			w[digest_out[i]] = Word(u32::from_le_bytes(bytes) as u64);
		}
		circuit.populate_wire_witness(&mut w).unwrap_or_else(|e| {
			panic!(
				"blake3_keyed_fixed failed for len_bytes={}, key_len={}: {e:?}",
				input.len(),
				key.len()
			)
		});
	}

	#[test]
	fn keyed_full_length_key() {
		// The native case: a 32-byte key, across message lengths spanning the block and chunk
		// boundaries that switch the chunk and tree shape.
		let key: Vec<u8> = (0..KEY_BYTES).map(|i| (i * 7 + 3) as u8).collect();
		for &len in &[0usize, 1, 64, 65, 192, 1024, 1025, 3072] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check_keyed(&input, &key, false);
		}
	}

	#[test]
	fn keyed_short_keys() {
		// A key shorter than 32 bytes is zero-padded, including the empty key — which is the
		// all-zero key, and still not the unkeyed hash, since [`KEYED_HASH`] separates them. The
		// lengths cover both sides of the 8-byte wire boundary and the 4-byte word boundary the
		// padding mask straddles.
		for key_len in [0usize, 1, 4, 5, 8, 9, 16, 23, 31] {
			let key: Vec<u8> = (0..key_len).map(|i| (i * 11 + 5) as u8).collect();
			check_keyed(b"abc", &key, false);
		}
	}

	#[test]
	fn keyed_ignores_bytes_past_the_key_length() {
		// The key wires are witness, so a prover picks the bytes past the key length. Masking them
		// is what stops those bytes from steering the digest.
		for key_len in [0usize, 1, 5, 9, 23, 31] {
			let key: Vec<u8> = (0..key_len).map(|i| (i * 11 + 5) as u8).collect();
			check_keyed(b"abc", &key, true);
		}
	}

	proptest! {
		// Every case compiles a whole hashing circuit, so the sample stays small.
		#![proptest_config(ProptestConfig::with_cases(12))]

		#[test]
		fn keyed_matches_blake3_crate(
			input in prop::collection::vec(any::<u8>(), 0..=300),
			key in prop::collection::vec(any::<u8>(), 0..=KEY_BYTES),
		) {
			check_keyed(&input, &key, true);
		}
	}

	#[test]
	#[should_panic(expected = "key length (33) exceeds 32")]
	fn keyed_rejects_an_oversized_key() {
		let builder = CircuitBuilder::new();
		let data: Vec<Wire> = (0..5).map(|_| builder.add_witness()).collect();
		let key = ByteVec::new_const_len(&builder, data, KEY_BYTES + 1);
		blake3_keyed_fixed(&builder, &[], 0, &key);
	}

	#[test]
	#[should_panic(expected = "key length must be fixed at circuit construction time")]
	fn keyed_rejects_a_runtime_length_key() {
		let builder = CircuitBuilder::new();
		// `new_witness` leaves the length range at the full `0..=capacity`, so the gadget cannot
		// know which bytes to mask.
		let key = ByteVec::new_witness(&builder, 4);
		blake3_keyed_fixed(&builder, &[], 0, &key);
	}

	/// Run the two-lane gadget over both inputs and assert each digest matches the reference crate,
	/// keyed with the matching key zero-padded to 32 bytes.
	///
	/// `garbage` sets the bits a prover picks but the specification never reads: the high half of
	/// each message word, and the key bytes past each key length.
	fn check_keyed_2x(inputs: [&[u8]; 2], keys: [&[u8]; 2], garbage: bool) {
		let len_bytes = inputs[0].len();
		assert_eq!(len_bytes, inputs[1].len(), "the two messages must have equal length");

		let builder = CircuitBuilder::new();
		let messages: [Vec<Wire>; 2] = std::array::from_fn(|_| {
			(0..len_bytes.div_ceil(4))
				.map(|_| builder.add_witness())
				.collect()
		});
		let key_vecs: [ByteVec; 2] = std::array::from_fn(|l| {
			let data = (0..keys[l].len().div_ceil(8))
				.map(|_| builder.add_witness())
				.collect();
			ByteVec::new_const_len(&builder, data, keys[l].len())
		});
		let digests = blake3_keyed_fixed_2x(
			&builder,
			[&messages[0], &messages[1]],
			len_bytes,
			[&key_vecs[0], &key_vecs[1]],
		);
		// Each digest is pinned word for word against a public expected value.
		let digests_out: [[Wire; 8]; 2] =
			std::array::from_fn(|_| std::array::from_fn(|_| builder.add_inout()));
		for l in 0..2 {
			for i in 0..8 {
				builder.assert_eq("digest_match_2x", digests[l][i], digests_out[l][i]);
			}
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		// A message word occupies the low half of its wire; the high half is the other lane's.
		let dirty_high = if garbage { 0xFFFF_FFFF_0000_0000 } else { 0 };
		for l in 0..2 {
			for (wire, word) in messages[l].iter().zip(bytes_to_le_words(inputs[l])) {
				w[*wire] = Word(word | dirty_high);
			}
			// Pad out to whole wires, so the bytes past the key length are populated too.
			let mut key_bytes = keys[l].to_vec();
			key_bytes.resize(keys[l].len().next_multiple_of(8), if garbage { 0xff } else { 0 });
			for (i, chunk) in key_bytes.chunks(8).enumerate() {
				w[key_vecs[l].data[i]] = Word(u64::from_le_bytes(chunk.try_into().unwrap()));
			}

			let mut padded_key = [0u8; KEY_BYTES];
			padded_key[..keys[l].len()].copy_from_slice(keys[l]);
			let expected = blake3::keyed_hash(&padded_key, inputs[l]);
			for i in 0..8 {
				let bytes: [u8; 4] = expected.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap();
				w[digests_out[l][i]] = Word(u32::from_le_bytes(bytes) as u64);
			}
		}
		circuit.populate_wire_witness(&mut w).unwrap_or_else(|e| {
			panic!(
				"blake3_keyed_fixed_2x failed for len_bytes={len_bytes}, key_lens=({}, {}): {e:?}",
				keys[0].len(),
				keys[1].len()
			)
		});
	}

	/// Two distinct messages of `len` bytes, and two distinct 32-byte keys.
	fn distinct_pair(len: usize) -> ([Vec<u8>; 2], [Vec<u8>; 2]) {
		let messages =
			std::array::from_fn(|l| (0..len).map(|i| (i * 37 + 1 + l * 91) as u8).collect());
		let keys =
			std::array::from_fn(|l| (0..KEY_BYTES).map(|i| (i * 7 + 3 + l * 53) as u8).collect());
		(messages, keys)
	}

	#[test]
	fn keyed_2x_block_boundaries() {
		// Lengths covering 0..=16 blocks in one chunk, both parities, and the block boundaries
		// where the trailing block turns partial.
		for &len in &[
			0usize, 1, 3, 63, 64, 65, 127, 128, 129, 192, 320, 511, 512, 1023, 1024,
		] {
			let (messages, keys) = distinct_pair(len);
			check_keyed_2x([&messages[0], &messages[1]], [&keys[0], &keys[1]], false);
		}
	}

	#[test]
	fn keyed_2x_multi_chunk() {
		// Lengths spanning 2..=10 chunks, odd counts and a partial final chunk included.
		// They reach every part of the tree: the paired parents, the lone-value promotion, the
		// root.
		for &len in &[1025usize, 2048, 2049, 3072, 5121, 7168, 8192, 9217] {
			let (messages, keys) = distinct_pair(len);
			check_keyed_2x([&messages[0], &messages[1]], [&keys[0], &keys[1]], false);
		}
	}

	#[test]
	fn keyed_2x_lanes_take_different_key_lengths() {
		// The keys are independent, so their lengths need not agree.
		// Each is zero-padded to 32 bytes on its own, the empty key included.
		for (len0, len1) in [(0usize, KEY_BYTES), (1, 31), (5, 9), (23, 4), (16, 16)] {
			let key0: Vec<u8> = (0..len0).map(|i| (i * 11 + 5) as u8).collect();
			let key1: Vec<u8> = (0..len1).map(|i| (i * 13 + 2) as u8).collect();
			check_keyed_2x([b"abc", b"xyz"], [&key0, &key1], false);
		}
	}

	#[test]
	fn keyed_2x_lanes_are_independent() {
		// Invariant: a lane's digest depends on its own inputs alone.
		//
		//     message wire = [ high 32: the other lane's word | low 32: this lane's word ]
		//
		// A prover picks that high half, so masking it is what keeps the two lanes apart.
		for &len in &[1usize, 64, 65, 1025] {
			let (messages, keys) = distinct_pair(len);
			check_keyed_2x([&messages[0], &messages[1]], [&keys[0], &keys[1]], true);
		}
	}

	#[test]
	fn keyed_2x_lanes_hashing_the_same_input_agree() {
		// Identical inputs must give identical digests, since the packing treats the lanes alike.
		// A bug that favours one lane shows up here as a disagreement.
		let key: Vec<u8> = (0..KEY_BYTES).map(|i| (i * 7 + 3) as u8).collect();
		let message: Vec<u8> = (0..200).map(|i| (i * 37 + 1) as u8).collect();
		check_keyed_2x([&message, &message], [&key, &key], true);
	}

	proptest! {
		// Every case compiles a whole two-lane hashing circuit, so the sample stays small.
		#![proptest_config(ProptestConfig::with_cases(12))]

		#[test]
		fn keyed_2x_matches_blake3_crate(
			len in 0usize..=300,
			bytes0 in prop::collection::vec(any::<u8>(), 300),
			bytes1 in prop::collection::vec(any::<u8>(), 300),
			key0 in prop::collection::vec(any::<u8>(), 0..=KEY_BYTES),
			key1 in prop::collection::vec(any::<u8>(), 0..=KEY_BYTES),
		) {
			// Random content at a random shared length, against the reference crate's keyed hash.
			check_keyed_2x([&bytes0[..len], &bytes1[..len]], [&key0, &key1], true);
		}
	}

	#[test]
	#[should_panic(expected = "key length (33) exceeds 32")]
	fn keyed_2x_rejects_an_oversized_key() {
		let builder = CircuitBuilder::new();
		let good = ByteVec::new_const_len(&builder, vec![], 0);
		let data: Vec<Wire> = (0..5).map(|_| builder.add_witness()).collect();
		let oversized = ByteVec::new_const_len(&builder, data, KEY_BYTES + 1);
		blake3_keyed_fixed_2x(&builder, [&[], &[]], 0, [&good, &oversized]);
	}

	#[test]
	#[should_panic(expected = "key length must be fixed at circuit construction time")]
	fn keyed_2x_rejects_a_runtime_length_key() {
		let builder = CircuitBuilder::new();
		let good = ByteVec::new_const_len(&builder, vec![], 0);
		// `new_witness` leaves the length range at the full `0..=capacity`, so the gadget cannot
		// know which bytes to mask.
		let runtime = ByteVec::new_witness(&builder, 4);
		blake3_keyed_fixed_2x(&builder, [&[], &[]], 0, [&good, &runtime]);
	}

	/// AND-constraint counts for hashing two messages under two 32-byte keys.
	///
	/// The one-lane count, from two calls, then the two-lane count.
	fn and_counts(len_bytes: usize) -> (usize, usize) {
		let key = |builder: &CircuitBuilder| {
			let data: Vec<Wire> = (0..KEY_BYTES / 8).map(|_| builder.add_witness()).collect();
			ByteVec::new_const_len(builder, data, KEY_BYTES)
		};
		let message = |builder: &CircuitBuilder| -> Vec<Wire> {
			(0..len_bytes.div_ceil(4))
				.map(|_| builder.add_witness())
				.collect()
		};
		// Pin the digests to public wires, so no compression is dead code the builder may drop.
		let expose = |builder: &CircuitBuilder, digest: [Wire; 8]| {
			for word in digest {
				let out = builder.add_inout();
				builder.assert_eq("digest", word, out);
			}
		};

		let single = {
			let builder = CircuitBuilder::new();
			for _ in 0..2 {
				let digest =
					blake3_keyed_fixed(&builder, &message(&builder), len_bytes, &key(&builder));
				expose(&builder, digest);
			}
			CircuitStat::collect(&builder.build()).n_and_constraints
		};
		let paired = {
			let builder = CircuitBuilder::new();
			let (m0, m1) = (message(&builder), message(&builder));
			let (k0, k1) = (key(&builder), key(&builder));
			for digest in blake3_keyed_fixed_2x(&builder, [&m0, &m1], len_bytes, [&k0, &k1]) {
				expose(&builder, digest);
			}
			CircuitStat::collect(&builder.build()).n_and_constraints
		};
		(single, paired)
	}

	#[test]
	fn keyed_2x_never_costs_more_than_two_single_hashes() {
		// The shapes where the two paths differ:
		//
		//     0..=64 bytes  one block, the shape a tweakable hash uses
		//     65..=128      two blocks, where the single-hash path also fills both lanes
		//     129..=192     three blocks, where it falls back to a lone compression
		//     1025+         several chunks, so the parent tree runs too
		for &len in &[0usize, 1, 64, 65, 128, 192, 1024, 2048] {
			let (single, paired) = and_counts(len);
			assert!(
				paired <= single,
				"blake3_keyed_fixed_2x costs {paired} AND constraints at len={len}, \
				 against {single} for two single hashes"
			);
		}

		// A one-block message is the shape the gadget exists for, and the only one where it wins
		// outright: a lone compression splits its own rounds across the lanes, so the single-hash
		// path leaves nothing idle for the pair to reclaim, and the pair's saving is the 3 rounds
		// the split spends twice rather than a whole second core.
		let (single, paired) = and_counts(BLOCK_BYTES);
		assert!(
			paired < single,
			"a one-block pair costs {paired} AND constraints, no less than the {single} two \
			 single hashes cost"
		);
	}

	#[test]
	#[should_panic(expected = "message.len() (2) must equal len_bytes.div_ceil(4) (3)")]
	fn keyed_2x_rejects_a_message_of_the_wrong_length() {
		// Both messages must be exactly the shared length.
		let builder = CircuitBuilder::new();
		let key = ByteVec::new_const_len(&builder, vec![], 0);
		let long: Vec<Wire> = (0..3).map(|_| builder.add_witness()).collect();
		let short: Vec<Wire> = (0..2).map(|_| builder.add_witness()).collect();
		blake3_keyed_fixed_2x(&builder, [&long, &short], 9, [&key, &key]);
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

	/// Hashes `input` with [`Blake3Compress2x`] as a chip, and checks the digest and the system.
	///
	/// The digest wires are public and filled with the reference digest, as in [`check`], so a
	/// disagreement fails to populate. What the chip adds is checked after: the calls each
	/// compression made have to be served by an instance that recomputes the same words.
	fn check_with_compress_chip(input: &[u8]) {
		let builder = CircuitBuilder::new();
		builder.register_chip(Blake3Compress2x, &[]);

		let message: Vec<Wire> = (0..input.len().div_ceil(4))
			.map(|_| builder.add_witness())
			.collect();
		let digest = blake3_fixed(&builder, &message, input.len());
		let digest_out: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("digest_match", digest[i], digest_out[i]);
		}

		let circuit = builder.build_m4();
		circuit.validate().unwrap();
		let cs = circuit.to_constraint_system();
		cs.validate().unwrap();

		let expected = blake3::hash(input);
		let expected_words: [u32; 8] = std::array::from_fn(|i| {
			u32::from_le_bytes(expected.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap())
		});

		let witness = circuit
			.generate_witness(|w| {
				for (wire, word) in message.iter().zip(bytes_to_le_words(input)) {
					w[*wire] = Word(word);
				}
				for i in 0..8 {
					w[digest_out[i]] = Word(expected_words[i] as u64);
				}
			})
			.unwrap_or_else(|e| panic!("blake3_fixed failed for len_bytes={}: {e:?}", input.len()));

		witness.verify(&cs).unwrap();
	}

	// The gadgets between `blake3_fixed` and `blake3_compress_2x` are untouched by the chip: the
	// compressions the chunk pairs reach and the ones the parent tree reaches both land as calls
	// because the builder holds the chip, not because anything in between was told.
	//
	// Lengths run from two blocks, which is the shortest message reaching a paired compression at
	// all, up through an odd chunk count, whose tree carries a two-lane parent as well as a
	// single-lane one.
	#[test]
	fn a_registered_chip_serves_every_compression() {
		for &len in &[128usize, 320, 1025, 5121] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check_with_compress_chip(&input);
		}
	}

	// A message short enough to compress one block at a time never reaches the paired gadget, so
	// its chip goes uncalled and the system it leaves is not one that can be populated.
	#[test]
	fn a_chip_no_compression_reaches_leaves_an_uncalled_chip() {
		let builder = CircuitBuilder::new();
		builder.register_chip(Blake3Compress2x, &[]);
		blake3_fixed(&builder, &[builder.add_witness()], 4);

		let error = builder.build_m4().validate().unwrap_err();
		assert!(matches!(error, binius_frontend::CircuitM4Error::NeverCalled { .. }), "{error:?}");
	}
}
