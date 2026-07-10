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
//! - [`blake3_variable`] — full hash gadget for messages of runtime-variable length, bounded by a
//!   compile-time capacity.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::{multiplexer::multi_wire_multiplex, util::clear_high_bits};

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

/// Computes the BLAKE3 hash of a runtime-variable-length message bounded by a compile-time maximum.
///
/// The BLAKE3 analog of [`Sha256::new`](crate::sha256::Sha256): the circuit shape is fixed at
/// construction time by the capacity `max_len_bytes`, while the actual message length is a runtime
/// witness `len_bytes` in `0..=max_len_bytes`. Where [`blake3_fixed`] bakes the exact chunk/tree
/// shape and the trailing partial-block length into the circuit, this gadget must resolve them at
/// proving time.
///
/// # Construction
///
/// The digest of a length-`len` message is fully determined by three runtime quantities: the index
/// `bd` of the final message block, that block's byte length `fbl` (`1..=64`, or `0` for the empty
/// message), and the message content. Every block before `bd` is a full 64-byte block, so the whole
/// chunk/tree shape follows from `bd` alone. The gadget therefore:
///
/// 1. Masks the message tail: byte `4*i + j` of word `i` survives iff `4*i + j < len_bytes`. This
///    zero-pads the (runtime) final block per BLAKE3's rules and clears every word's high 32 bits.
/// 2. Builds one candidate digest per possible final-block index `p` in `0..max_blocks`, each an
///    exact replica of the [`blake3_fixed`] computation for a message occupying blocks `0..=p`,
///    with the sole difference that block `p`'s `block_len` is the runtime wire `fbl` rather than a
///    constant. This makes the candidate for `p == bd` reproduce the true hash.
/// 3. Selects the candidate at index `bd` with a [`multi_wire_multiplex`].
///
/// Full interior chunks are shared across candidates. The selector stays in range because
/// `len_bytes <= max_len_bytes` is enforced, which bounds `bd < max_blocks`.
///
/// # Arguments
///
/// - `builder`: Circuit builder.
/// - `message`: Input message as 32-bit little-endian words (4 bytes per wire), of length
///   `max_len_bytes.div_ceil(4)`. Words (and the bytes within the boundary word) beyond `len_bytes`
///   are ignored — the gadget masks them — so they may hold any value.
/// - `len_bytes`: Runtime message length in bytes; constrained to `0..=max_len_bytes`.
/// - `max_len_bytes`: Compile-time capacity in bytes.
///
/// # Soundness: the caller must constrain `len_bytes`
///
/// This gadget hashes the *claimed* length. It enforces
/// `digest == blake3(message[..len_bytes])` and, in-circuit, that `len_bytes <= max_len_bytes`. It
/// does **not** — and cannot — constrain the value on `len_bytes` itself; that is the caller's
/// responsibility.
///
/// Concretely, the returned digest equals `blake3(message[..len_bytes])` for a prover-chosen
/// `len_bytes`; it does **not** equal `blake3(message)` for the full committed `message`. If
/// `len_bytes` is a free witness, the constraint system is satisfiable for **any prefix** of the
/// message: a prover can pick any `l <= max_len_bytes`, set `len_bytes = l`, and the masking +
/// final-block-length logic reproduces `blake3(message[..l])`. An unconstrained `len_bytes`
/// therefore binds "the digest of *some* prefix", not "the digest of *the* message".
///
/// The attack requires the prover to be able to choose the digest. If the digest is itself a
/// verifier-fixed public input, a free `len_bytes` is harmless — the prover cannot substitute the
/// prefix digest. But a composed circuit that consumes the digest *internally* (feeding it into
/// further constraints) has no such protection, so in that setting a free `len_bytes` is a
/// soundness hole. Callers who intend a specific length **must** constrain `len_bytes` — bind it to
/// a public input, derive it from other constrained wires, or, for a length known at construction
/// time, use [`blake3_fixed`] (which is also cheaper, as it needs no runtime length checks or
/// candidate multiplexing).
///
/// Relatedly, message content beyond the claimed length is unconstrained: bytes at or above
/// `len_bytes` are masked to zero before hashing, so a satisfying witness may leave them arbitrary.
/// Only `message[..len_bytes]` is bound.
///
/// # Returns
///
/// The BLAKE3 digest as 8 wires, each holding a 32-bit little-endian word in its low 32 bits.
///
/// # Panics
///
/// Panics if `message.len()` does not equal `max_len_bytes.div_ceil(4)`.
pub fn blake3_variable(
	builder: &CircuitBuilder,
	message: &[Wire],
	len_bytes: Wire,
	max_len_bytes: usize,
) -> [Wire; 8] {
	assert_eq!(
		message.len(),
		max_len_bytes.div_ceil(4),
		"blake3_variable: message.len() ({}) must equal max_len_bytes.div_ceil(4) ({})",
		message.len(),
		max_len_bytes.div_ceil(4),
	);

	let zero = builder.add_constant(Word::ZERO);

	// Reject len_bytes > max_len_bytes: the final-block selector below must stay in range for the
	// multiplexer to be sound.
	let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
	builder.assert_false("blake3_variable.len_check", too_long);

	let blocks_per_chunk = CHUNK_BYTES / BLOCK_BYTES; // 16
	let max_blocks = max_len_bytes.div_ceil(BLOCK_BYTES).max(1);
	let max_chunks = max_len_bytes.div_ceil(CHUNK_BYTES).max(1);

	// ---- Mask the message tail to zero.
	//
	// BLAKE3 zero-pads the final (partial) block and records its true byte count in `block_len`.
	// The boundary is a runtime value, so rather than masking a single boundary word (as
	// `blake3_fixed` does for its compile-time boundary) we mask every word against `len_bytes`.
	// This zero-extends the tail for whichever candidate is ultimately selected and, as a side
	// effect, clears the high 32 bits of every word — a precondition of the compression gadget.
	// Trailing wires past the message capacity are simply zero.
	let n_words = max_blocks * blocks_per_chunk;
	let masked: Vec<Wire> = (0..n_words)
		.map(|i| {
			if i >= message.len() {
				return zero;
			}
			let mut mask = zero;
			for j in 0..4 {
				let byte_pos = builder.add_constant_64((i * 4 + j) as u64);
				let valid = builder.icmp_ult(byte_pos, len_bytes);
				let byte_mask = builder.add_constant_64(0xFFu64 << (j * 8));
				mask = builder.bor(mask, builder.select(valid, byte_mask, zero));
			}
			builder.band(message[i], mask)
		})
		.collect();

	// ---- Final-block index `bd` and final-block length `fbl`.
	//
	// For a non-empty message the final block is the one containing byte `len_bytes - 1`, and its
	// length is `len_bytes - bd * 64` (in `1..=64`). The empty message is a single block of length
	// 0 at index 0. These mirror `blake3_fixed`'s `n_blocks - 1` and trailing `block_len`.
	let one = builder.add_constant_64(1);
	let is_empty = builder.icmp_eq(len_bytes, zero);
	let (len_m1, _) = builder.isub_bin_bout(len_bytes, one, zero);
	let last_byte = builder.select(is_empty, zero, len_m1);
	let bd = builder.shr(last_byte, 6);
	let bd_bytes = builder.shl(bd, 6);
	let (fbl_nonempty, _) = builder.isub_bin_bout(len_bytes, bd_bytes, zero);
	let fbl = builder.select(is_empty, zero, fbl_nonempty);

	let const_block_bytes = builder.add_constant_64(BLOCK_BYTES as u64);
	let block_at =
		|j: usize| -> [Wire; 16] { std::array::from_fn(|i| masked[j * blocks_per_chunk + i]) };

	// ---- Chaining values of the full interior chunks, shared across candidates.
	//
	// Chunk `c` is interior (a full 16-block chunk, no ROOT) in every candidate whose final block
	// lies in a later chunk. The highest such chunk is `max_chunks - 2`.
	let full_chunk_cvs: Vec<[Wire; 8]> = (0..max_chunks.saturating_sub(1))
		.map(|c| {
			let blocks: Vec<[Wire; 16]> = (0..blocks_per_chunk)
				.map(|b| block_at(c * blocks_per_chunk + b))
				.collect();
			let block_lens = vec![const_block_bytes; blocks_per_chunk];
			blake3_chunk(
				&builder.subcircuit(format!("blake3_variable_full_chunk[{c}]")),
				&blocks,
				&block_lens,
				c as u64,
				0,
			)
		})
		.collect();

	// ---- One candidate digest per possible final-block index.
	//
	// Candidate `p` is the digest of a message occupying blocks `0..=p`: the last chunk spans its
	// first block through block `p` (whose `block_len` is the runtime `fbl`), preceded by the full
	// interior chunks. ROOT lands on the lone chunk when `p` is in the first chunk, otherwise on
	// the tree root — exactly as in `blake3_fixed`.
	let candidates: Vec<[Wire; 8]> = (0..max_blocks)
		.map(|p| {
			let last_chunk = p / blocks_per_chunk;
			let first_block = last_chunk * blocks_per_chunk;
			let n_blk = p - first_block + 1;
			let blocks: Vec<[Wire; 16]> = (0..n_blk).map(|b| block_at(first_block + b)).collect();
			let block_lens: Vec<Wire> = (0..n_blk)
				.map(|b| {
					if b + 1 == n_blk {
						fbl
					} else {
						const_block_bytes
					}
				})
				.collect();
			let last_flags_extra = if last_chunk == 0 { ROOT } else { 0 };
			let sub = builder.subcircuit(format!("blake3_variable_candidate[{p}]"));
			let last_cv =
				blake3_chunk(&sub, &blocks, &block_lens, last_chunk as u64, last_flags_extra);
			if last_chunk == 0 {
				last_cv
			} else {
				let mut leaves: Vec<[Wire; 8]> = full_chunk_cvs[..last_chunk].to_vec();
				leaves.push(last_cv);
				blake3_tree_root(&sub, leaves)
			}
		})
		.collect();

	// ---- Select the digest for the actual final-block index.
	let inputs: Vec<&[Wire]> = candidates.iter().map(|d| &d[..]).collect();
	let digest = multi_wire_multiplex(builder, &inputs, bd);
	std::array::from_fn(|i| digest[i])
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

	/// Run `blake3_variable` with capacity `max_len_bytes` over `input` and assert the circuit
	/// digest matches `blake3::hash(input)`.
	fn check_variable(max_len_bytes: usize, input: &[u8]) {
		assert!(input.len() <= max_len_bytes);

		let builder = CircuitBuilder::new();
		let message: Vec<Wire> = (0..max_len_bytes.div_ceil(4))
			.map(|_| builder.add_witness())
			.collect();
		let len_wire = builder.add_witness();
		let digest = blake3_variable(&builder, &message, len_wire, max_len_bytes);
		let digest_out: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("digest_match", digest[i], digest_out[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();

		// Fill every message wire: real words for the message, zero for the tail (the gadget masks
		// the tail, so these values are irrelevant to the digest, but every witness must be set).
		let words = bytes_to_le_words(input);
		for (i, wire) in message.iter().enumerate() {
			w[*wire] = Word(words.get(i).copied().unwrap_or(0));
		}
		w[len_wire] = Word(input.len() as u64);

		let expected = blake3::hash(input);
		let expected_words: [u32; 8] = std::array::from_fn(|i| {
			u32::from_le_bytes(expected.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap())
		});
		for i in 0..8 {
			w[digest_out[i]] = Word(expected_words[i] as u64);
		}
		circuit.populate_wire_witness(&mut w).unwrap_or_else(|e| {
			panic!("blake3_variable failed for max={max_len_bytes} len={}: {e:?}", input.len())
		});
	}

	#[test]
	fn variable_single_block_capacity() {
		// Capacity within one block: exercises the empty message and the lone-candidate multiplex.
		for len in [0usize, 1, 13, 32, 63, 64] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check_variable(64, &input);
		}
	}

	#[test]
	fn variable_single_chunk_capacity() {
		// Capacity of one full chunk: every length shares the same circuit, selecting a different
		// final block (and a different partial `block_len`) at proving time.
		for len in [
			0usize, 1, 3, 64, 65, 100, 127, 128, 200, 512, 1000, 1023, 1024,
		] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check_variable(1024, &input);
		}
	}

	#[test]
	fn variable_multi_chunk_capacity() {
		// Capacity spanning several chunks: exercises the shared interior chunks, the parent tree,
		// and the ROOT node moving from the lone chunk to the tree root as the length grows.
		let max = 3072;
		for len in [0usize, 1, 1024, 1025, 1536, 2048, 2049, 3000, 3072] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check_variable(max, &input);
		}
	}

	#[test]
	fn variable_odd_capacity() {
		// A capacity that is neither block- nor chunk-aligned: the final message word is partial
		// and the final chunk is short.
		for len in [0usize, 1, 50, 100, 1500, 2500, 2555] {
			let input: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check_variable(2555, &input);
		}
	}

	#[test]
	fn variable_rejects_overlong_length() {
		// A `len_bytes` beyond the capacity must make witness population fail (len_check).
		let builder = CircuitBuilder::new();
		let message: Vec<Wire> = (0..16).map(|_| builder.add_witness()).collect();
		let len_wire = builder.add_witness();
		let digest = blake3_variable(&builder, &message, len_wire, 64);
		let digest_out: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("digest_match", digest[i], digest_out[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for wire in &message {
			w[*wire] = Word::ZERO;
		}
		w[len_wire] = Word(65); // exceeds capacity of 64
		let expected = blake3::hash(&[0u8; 65][..64]);
		let expected_words: [u32; 8] = std::array::from_fn(|i| {
			u32::from_le_bytes(expected.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap())
		});
		for i in 0..8 {
			w[digest_out[i]] = Word(expected_words[i] as u64);
		}
		assert!(circuit.populate_wire_witness(&mut w).is_err());
	}

	// ---------------------------------------------------------------------------
	// Soundness characterization tests for the `len_bytes` caller obligation.
	//
	// `blake3_variable` hashes the *claimed* length (the value on the `len_bytes`
	// wire); it does not constrain what that length is (see the gadget doc). These
	// tests pin down the resulting semantics so the contract stays documented and
	// covered.
	//
	// The message wires here are `add_inout` (a fixed, committed statement) while
	// `len_bytes` is a free witness. That models the composed-circuit case where the
	// digest is an internal / prover-influenced value; a verifier-fixed public digest
	// would neutralize the prefix substitution (the prover could not swap it), but a
	// free `len_bytes` remains a hazard whenever the digest is not so pinned.
	// ---------------------------------------------------------------------------

	/// Build a `blake3_variable` over `max_len_bytes` with `len_bytes` a *free witness*, commit the
	/// full `message` on `add_inout` wires, claim `claimed_len` and `claimed_digest`, then run the
	/// circuit. Returns `Ok(())` iff the constraint system is satisfiable.
	fn run_variable_claim(
		max_len_bytes: usize,
		message: &[u8],
		claimed_len: usize,
		claimed_digest: &[u8; 32],
	) -> Result<(), String> {
		assert!(message.len() <= max_len_bytes);
		let builder = CircuitBuilder::new();
		let msg_wires: Vec<Wire> = (0..max_len_bytes.div_ceil(4))
			.map(|_| builder.add_inout())
			.collect();
		let len_wire = builder.add_witness();
		let digest = blake3_variable(&builder, &msg_wires, len_wire, max_len_bytes);
		let digest_out: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("digest_match", digest[i], digest_out[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		let words = bytes_to_le_words(message);
		for (i, wire) in msg_wires.iter().enumerate() {
			w[*wire] = Word(words.get(i).copied().unwrap_or(0));
		}
		w[len_wire] = Word(claimed_len as u64);
		let digest_words: [u32; 8] = std::array::from_fn(|i| {
			u32::from_le_bytes(claimed_digest[i * 4..i * 4 + 4].try_into().unwrap())
		});
		for i in 0..8 {
			w[digest_out[i]] = Word(digest_words[i] as u64);
		}
		circuit
			.populate_wire_witness(&mut w)
			.map(|_| ())
			.map_err(|e| format!("{e:?}"))
	}

	/// SOUNDNESS CHARACTERIZATION (a caller obligation, not a bug in this gadget):
	/// with `len_bytes` wired as a free witness, a prover can prove the BLAKE3 hash of an arbitrary
	/// *prefix* of the committed message. A 100-byte message is committed, but the prover claims
	/// `len = 50` and supplies `digest = blake3(message[..50])`, and the circuit is satisfiable.
	/// Callers must constrain `len_bytes` (see the `blake3_variable` doc).
	#[test]
	fn variable_unconstrained_len_admits_prefix_hash() {
		const MAX: usize = 1024;
		let msg: Vec<u8> = (0..100u32).map(|i| (i * 7 + 3) as u8).collect();

		// Honest control: claiming the true length verifies with the true digest.
		let full = blake3::hash(&msg);
		run_variable_claim(MAX, &msg, msg.len(), full.as_bytes())
			.expect("honest full-length witness must verify");

		// Attack: claim a shorter length and prove the prefix digest against the *same* committed
		// message. The tail-masking + runtime final-block length reproduce `blake3(message[..50])`.
		let prefix = &msg[..50];
		let prefix_digest = blake3::hash(prefix);
		run_variable_claim(MAX, &msg, prefix.len(), prefix_digest.as_bytes())
			.expect("a free len_bytes admits the prefix digest (caller obligation)");

		// Positive-binding control: `message[..len_bytes]` *is* bound. Under the same claimed
		// `len = 50`, the full-message digest must be rejected — the gadget hashes only the prefix.
		assert!(
			run_variable_claim(MAX, &msg, prefix.len(), full.as_bytes()).is_err(),
			"digest must equal blake3(message[..len_bytes]), not blake3(message)"
		);
	}
}
