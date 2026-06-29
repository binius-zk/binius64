// Copyright 2025 Irreducible Inc.
//! BLAKE3 chain hash for Winternitz hash chains.
//!
//! One BLAKE3 compression is the tweakable hash for a chain step:
//!
//! ```text
//! Th(prev) = compress(cv = prev, block = param || 0x00 || epoch || chain_index || position)
//! ```
//!
//! - The 32-byte previous chain value sits in the 8-word chaining value.
//! - The tweak fills the 16-word message block, zero-padded.
//! - This is the same 2-to-1 compression BLAKE3 uses for tree parents.
//! - Collision and preimage resistance rest on the BLAKE3 compression function.
//! - One compression covers parameters of up to 57 bytes.
//!
//! Two properties make this cheap on Binius64:
//! - The output is derived from the input wires, so a chain step needs no witness values.
//! - Two chains share one compression as its two 32-bit lanes, near 316 AND per hash.

use binius_frontend::{CircuitBuilder, Wire};

use crate::{
	blake3::{blake3_compress, blake3_compress_2x, ref_compress},
	concat::concat,
	fixed_byte_vec::ByteVec,
};

/// Tweak separator byte for chain hashing.
///
/// Chain steps use `0x00`, internal tree nodes `0x01`, message hashing `0x02`, the leaf `0x03`.
pub const CHAIN_TWEAK: u8 = 0x00;

/// Compact byte widths of the chain-hash tweak.
/// - 1 byte: tweak separator
/// - 4 bytes: epoch (supports lifetimes up to `2^32`)
/// - 1 byte: chain index (supports dimension up to 256)
/// - 1 byte: position (supports chain length up to 256, i.e. `coordinate_resolution_bits <= 8`)
const EPOCH_BYTES: usize = 4;
const CHAIN_INDEX_BYTES: usize = 1;
const POSITION_BYTES: usize = 1;

/// Mask selecting the low 32 bits of a 64-bit word.
const LOW32_MASK: u64 = 0xFFFF_FFFF;

/// BLAKE3 block size in bytes (one compression).
const BLOCK_BYTES: usize = 64;

/// Number of 32-bit words in a BLAKE3 block / chaining value.
const BLOCK_WORDS: usize = 16;
const CV_WORDS: usize = 8;

/// Tweak content length in bytes for a given parameter length:
/// `param || 0x00 || epoch(4) || chain_index(1) || position(1)`.
pub const fn chain_tweak_len(param_len: usize) -> usize {
	param_len + 1 + EPOCH_BYTES + CHAIN_INDEX_BYTES + POSITION_BYTES
}

/// Splits each 64-bit wire into two little-endian 32-bit word wires (low half, then high half).
///
/// Returns exactly `num_words` words.
fn split_u32_words(builder: &CircuitBuilder, data: &[Wire], num_words: usize) -> Vec<Wire> {
	let mask = builder.add_constant_64(LOW32_MASK);
	let mut words = Vec::with_capacity(num_words);
	for &w in data {
		if words.len() >= num_words {
			break;
		}
		words.push(builder.band(w, mask));
		if words.len() >= num_words {
			break;
		}
		words.push(builder.shr(w, 32));
	}
	while words.len() < num_words {
		words.push(builder.add_constant_64(0));
	}
	words
}

/// Splits a byte vector into `num_words` 32-bit words, forcing every byte at index `>= valid_bytes`
/// to zero.
///
/// The zeroing closes a malleability gap:
/// - the concatenation primitive leaves bytes past a vector's length unconstrained,
/// - a BLAKE3 compression mixes the whole 64-byte block, including those bytes,
/// - pinning them stops a prover from choosing the padding to alter the digest.
///
/// Each word is handled by its position relative to the content:
/// - fully inside: passed through,
/// - fully past: replaced by the zero constant,
/// - straddling the boundary: masked to its low valid bytes.
fn zeroed_u32_words(
	builder: &CircuitBuilder,
	data: &[Wire],
	valid_bytes: usize,
	num_words: usize,
) -> Vec<Wire> {
	let raw = split_u32_words(builder, data, num_words);
	let zero = builder.add_constant_64(0);
	(0..num_words)
		.map(|i| {
			let word_start = 4 * i;
			if word_start + 4 <= valid_bytes {
				raw[i]
			} else if word_start >= valid_bytes {
				zero
			} else {
				let keep = valid_bytes - word_start;
				let mask = builder.add_constant_64((1u64 << (keep * 8)) - 1);
				builder.band(raw[i], mask)
			}
		})
		.collect()
}

/// Builds the 16-word BLAKE3 message block holding the chain tweak.
///
/// Layout: `param || 0x00 || epoch(4) || chain_index(1) || position(1)`, zero-padded to 64 bytes.
fn tweak_block(
	builder: &CircuitBuilder,
	domain_param_wires: &[Wire],
	param_len: usize,
	epoch: Wire,
	chain_index: u8,
	position: u8,
) -> [Wire; BLOCK_WORDS] {
	let terms = vec![
		ByteVec::new_const_len(builder, domain_param_wires.to_vec(), param_len),
		ByteVec::new_const_len(builder, vec![builder.add_constant_64(CHAIN_TWEAK as u64)], 1),
		ByteVec::new_const_len(builder, vec![epoch], EPOCH_BYTES),
		ByteVec::new_const_len(
			builder,
			vec![builder.add_constant_64(chain_index as u64)],
			CHAIN_INDEX_BYTES,
		),
		ByteVec::new_const_len(
			builder,
			vec![builder.add_constant_64(position as u64)],
			POSITION_BYTES,
		),
	];
	let bv = concat(builder, &terms);
	let words = zeroed_u32_words(builder, &bv.data, chain_tweak_len(param_len), BLOCK_WORDS);
	std::array::from_fn(|i| words[i])
}

/// Splits a 32-byte chain value (4x64-bit LE) into the 8-word BLAKE3 chaining value (8x32-bit LE).
fn hash4_to_cv8(builder: &CircuitBuilder, hash: [Wire; 4]) -> [Wire; CV_WORDS] {
	let words = split_u32_words(builder, &hash, CV_WORDS);
	std::array::from_fn(|i| words[i])
}

/// Packs the 8-word BLAKE3 output (8x32-bit LE) back into a 32-byte chain value (4x64-bit LE).
fn cv8_to_hash4(builder: &CircuitBuilder, cv: &[Wire; CV_WORDS]) -> [Wire; 4] {
	std::array::from_fn(|k| {
		let hi = builder.shl(cv[2 * k + 1], 32);
		builder.bxor(cv[2 * k], hi)
	})
}

/// Computes one Winternitz chain step with a single BLAKE3 compression.
///
/// - Returns the 32-byte digest as four 64-bit little-endian wires.
/// - The digest is derived from the inputs, so no witness values are needed.
/// - The chain index and position are circuit constants, so the tweak is fixed, not prover-chosen.
///
/// # Panics
///
/// - If the parameter wire count is not `ceil(param_len / 8)`.
/// - If the tweak exceeds one 64-byte BLAKE3 block, i.e. `param_len > 57`.
pub fn circuit_chain_hash_blake3(
	builder: &CircuitBuilder,
	domain_param_wires: &[Wire],
	param_len: usize,
	epoch: Wire,
	hash: [Wire; 4],
	chain_index: u8,
	position: u8,
) -> [Wire; 4] {
	assert_eq!(domain_param_wires.len(), param_len.div_ceil(8));
	let msg_len = chain_tweak_len(param_len);
	assert!(msg_len <= BLOCK_BYTES, "chain tweak ({msg_len} bytes) exceeds one BLAKE3 block");

	let cv = hash4_to_cv8(builder, hash);
	let block = tweak_block(builder, domain_param_wires, param_len, epoch, chain_index, position);
	let counter = builder.add_constant_64(0);
	let block_len = builder.add_constant_64(msg_len as u64);
	let flags = builder.add_constant_64(0);

	let out = blake3_compress(builder, cv, block, counter, block_len, flags);
	cv8_to_hash4(builder, &out)
}

/// Computes one chain step for two independent chains at once.
///
/// - Lane 0 advances the first chain, lane 1 the second.
/// - Both lanes share the same position, epoch, and parameter.
/// - The two lanes are independent compressions packed into one BLAKE3 core.
/// - This is the cheapest chain step per hash.
///
/// # Panics
///
/// Same conditions as the single-chain step.
#[allow(clippy::too_many_arguments)]
pub fn circuit_chain_step_2x_blake3(
	builder: &CircuitBuilder,
	domain_param_wires: &[Wire],
	param_len: usize,
	epoch: Wire,
	hash0: [Wire; 4],
	hash1: [Wire; 4],
	chain_index0: u8,
	chain_index1: u8,
	position: u8,
) -> ([Wire; 4], [Wire; 4]) {
	assert_eq!(domain_param_wires.len(), param_len.div_ceil(8));
	let msg_len = chain_tweak_len(param_len);
	assert!(msg_len <= BLOCK_BYTES, "chain tweak ({msg_len} bytes) exceeds one BLAKE3 block");

	let cv0 = hash4_to_cv8(builder, hash0);
	let cv1 = hash4_to_cv8(builder, hash1);
	let block0 = tweak_block(builder, domain_param_wires, param_len, epoch, chain_index0, position);
	let block1 = tweak_block(builder, domain_param_wires, param_len, epoch, chain_index1, position);

	// Pack lane 0 into the low 32 bits and lane 1 into the high 32 bits of each word.
	// Inputs already have their high 32 bits clear, so `lo ^ (hi << 32)` equals the OR.
	let pack = |lo: Wire, hi: Wire| builder.bxor(lo, builder.shl(hi, 32));
	let cv: [Wire; CV_WORDS] = std::array::from_fn(|i| pack(cv0[i], cv1[i]));
	let block: [Wire; BLOCK_WORDS] = std::array::from_fn(|i| pack(block0[i], block1[i]));

	let zero = builder.add_constant_64(0);
	let block_len = builder.add_constant_64((msg_len as u64) | ((msg_len as u64) << 32));

	let out = blake3_compress_2x(builder, cv, block, zero, zero, block_len, zero);

	// Unpack: lane 0 from the low halves, lane 1 from the high halves.
	let mask = builder.add_constant_64(LOW32_MASK);
	let out0: [Wire; CV_WORDS] = std::array::from_fn(|i| builder.band(out[i], mask));
	let out1: [Wire; CV_WORDS] = std::array::from_fn(|i| builder.shr(out[i], 32));
	(cv8_to_hash4(builder, &out0), cv8_to_hash4(builder, &out1))
}

/// Reference (out-of-circuit) tweak block bytes, matching the circuit layout, padded to 64 bytes.
fn ref_tweak_block(param: &[u8], epoch: u32, chain_index: u8, position: u8) -> ([u32; 16], u32) {
	let mut tweak = Vec::with_capacity(BLOCK_BYTES);
	tweak.extend_from_slice(param);
	tweak.push(CHAIN_TWEAK);
	tweak.extend_from_slice(&epoch.to_le_bytes());
	tweak.push(chain_index);
	tweak.push(position);
	let msg_len = tweak.len() as u32;
	assert!(tweak.len() <= BLOCK_BYTES, "chain tweak exceeds one BLAKE3 block");
	let mut block_bytes = [0u8; BLOCK_BYTES];
	block_bytes[..tweak.len()].copy_from_slice(&tweak);
	let block: [u32; 16] = std::array::from_fn(|i| {
		u32::from_le_bytes(block_bytes[4 * i..4 * i + 4].try_into().unwrap())
	});
	(block, msg_len)
}

/// Reference (out-of-circuit) single chain step, matching the in-circuit step exactly.
///
/// Used to derive the chain endpoints when filling witnesses.
pub fn ref_chain_step_blake3(
	param: &[u8],
	epoch: u32,
	chain_index: u8,
	position: u8,
	prev: &[u8; 32],
) -> [u8; 32] {
	let cv: [u32; 8] =
		std::array::from_fn(|i| u32::from_le_bytes(prev[4 * i..4 * i + 4].try_into().unwrap()));
	let (block, msg_len) = ref_tweak_block(param, epoch, chain_index, position);
	let out = ref_compress(&cv, &block, 0, msg_len, 0);
	let mut next = [0u8; 32];
	for (i, word) in out.iter().enumerate() {
		next[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
	}
	next
}

/// Walks `num_hashes` BLAKE3 chain steps starting at chain position `start_pos`.
///
/// - Step `i` hashes at position `start_pos + i + 1`.
/// - This matches the verifier, which advances a chain only past its coordinate `x_i`.
pub fn hash_chain_blake3(
	param: &[u8],
	epoch: u32,
	chain_index: u8,
	start_hash: &[u8; 32],
	start_pos: usize,
	num_hashes: usize,
) -> [u8; 32] {
	let mut current = *start_hash;
	for i in 0..num_hashes {
		let position = (start_pos + i + 1) as u8;
		current = ref_chain_step_blake3(param, epoch, chain_index, position, &current);
	}
	current
}

/// Maximum domain (tweak) length in bytes that fits the 32-byte chaining value.
pub const TH_DOMAIN_MAX_BYTES: usize = 32;

/// General compress-based BLAKE3 tweakable hash.
///
/// - The domain (tweak) seeds the 8-word chaining value, zero-padded to 32 bytes.
/// - The data is absorbed in 64-byte blocks, each folded in with one compression.
/// - The output is the final chaining value, as four 64-bit little-endian wires.
///
/// This mirrors the chain step but with the roles generalized:
/// the chain step is the special case of a 32-byte domain and a single data block.
///
/// # Panics
///
/// - If the concatenated domain exceeds 32 bytes.
pub fn circuit_blake3_th(
	builder: &CircuitBuilder,
	domain_terms: &[ByteVec],
	data_terms: &[ByteVec],
) -> [Wire; 4] {
	// Compile-time byte lengths: each term has a constant length (start == end of its range).
	let domain_len: usize = domain_terms.iter().map(|t| *t.len_range.end()).sum();
	let data_len: usize = data_terms.iter().map(|t| *t.len_range.end()).sum();
	assert!(
		domain_len <= TH_DOMAIN_MAX_BYTES,
		"BLAKE3 tweakable-hash domain ({domain_len} bytes) exceeds the {TH_DOMAIN_MAX_BYTES}-byte chaining value"
	);

	// Seed the chaining value from the domain, zero-padded to 8 words.
	let domain = concat(builder, domain_terms);
	let cv_words = zeroed_u32_words(builder, &domain.data, domain_len, CV_WORDS);
	let mut cv: [Wire; CV_WORDS] = std::array::from_fn(|i| cv_words[i]);

	// Absorb the data in 64-byte blocks, one compression each.
	let data = concat(builder, data_terms);
	let num_blocks = data_len.div_ceil(BLOCK_BYTES).max(1);
	let data_words = zeroed_u32_words(builder, &data.data, data_len, num_blocks * BLOCK_WORDS);
	let counter = builder.add_constant_64(0);
	let flags = builder.add_constant_64(0);
	for b in 0..num_blocks {
		let block: [Wire; BLOCK_WORDS] = std::array::from_fn(|i| data_words[b * BLOCK_WORDS + i]);
		let block_len = (data_len - b * BLOCK_BYTES).min(BLOCK_BYTES);
		let block_len_w = builder.add_constant_64(block_len as u64);
		cv = blake3_compress(builder, cv, block, counter, block_len_w, flags);
	}
	cv8_to_hash4(builder, &cv)
}

/// Reference (out-of-circuit) form of `circuit_blake3_th`.
///
/// - `domain` seeds the chaining value, zero-padded to 32 bytes.
/// - `data` is absorbed in 64-byte blocks via the compression reference.
///
/// Built directly on the BLAKE3 compression function used as a tweakable hash:
/// - the domain replaces the initial chaining value, acting as the key and tweak,
/// - the data plays the role of the message blocks fed through the compression.
pub fn ref_blake3_th(domain: &[u8], data: &[u8]) -> [u8; 32] {
	// The domain seeds the 32-byte chaining value, so it cannot be longer.
	assert!(domain.len() <= TH_DOMAIN_MAX_BYTES, "domain exceeds 32 bytes");

	// Zero-pad the domain to 32 bytes, then read it as the 8-word chaining value (BLAKE3's IV
	// slot).
	let mut cv_bytes = [0u8; TH_DOMAIN_MAX_BYTES];
	cv_bytes[..domain.len()].copy_from_slice(domain);
	let mut cv: [u32; 8] =
		std::array::from_fn(|i| u32::from_le_bytes(cv_bytes[4 * i..4 * i + 4].try_into().unwrap()));

	// Absorb the data in 64-byte blocks, chaining each compression into the next.
	let total = data.len();
	// At least one compression runs, so empty data still maps to a defined digest.
	let num_blocks = total.div_ceil(BLOCK_BYTES).max(1);
	for b in 0..num_blocks {
		// This block's byte range; the last block may be shorter than 64 bytes.
		let start = b * BLOCK_BYTES;
		let end = (start + BLOCK_BYTES).min(total);

		// Pack the block into 16 little-endian words, trailing bytes left zero.
		let mut block_bytes = [0u8; BLOCK_BYTES];
		block_bytes[..end - start].copy_from_slice(&data[start..end]);
		let block: [u32; 16] = std::array::from_fn(|i| {
			u32::from_le_bytes(block_bytes[4 * i..4 * i + 4].try_into().unwrap())
		});

		// Bare compression (counter/flags = 0); block_len is the real byte count.
		cv = ref_compress(&cv, &block, 0, (end - start) as u32, 0);
	}

	// Serialize the final chaining value to 32 little-endian bytes: the digest.
	let mut out = [0u8; 32];
	for (i, word) in cv.iter().enumerate() {
		out[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
	}
	out
}

#[cfg(test)]
mod tests {
	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::util::pack_bytes_into_wires_le;
	use proptest::prelude::*;

	use super::*;

	/// Packs a 32-byte value, sets epoch, and reads back a 4x64-bit output as 32 bytes.
	fn pack_param(param_len: usize) -> Vec<u8> {
		(0..param_len.div_ceil(8) * 8)
			.map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
			.collect()
	}

	proptest! {
		#[test]
		fn single_step_matches_reference(
			param_len in 1usize..=57,
			epoch in 0u32..=u32::MAX,
			chain_index in 0u8..=255,
			position in 0u8..=255,
			prev in prop::array::uniform32(any::<u8>()),
		) {
			// One in-circuit chain step must equal the ref_compress-based reference.
			let b = CircuitBuilder::new();
			let pw = param_len.div_ceil(8);
			let param: Vec<Wire> = (0..pw).map(|_| b.add_inout()).collect();
			let epoch_w = b.add_inout();
			let hash: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
			let out = circuit_chain_hash_blake3(&b, &param, param_len, epoch_w, hash, chain_index, position);
			let expected: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
			for k in 0..4 {
				b.assert_eq("step_out", out[k], expected[k]);
			}

			let circuit = b.build();
			let mut w = circuit.new_witness_filler();
			let param_bytes = pack_param(param_len);
			pack_bytes_into_wires_le(&mut w, &param, &param_bytes);
			w[epoch_w] = Word::from_u64(epoch as u64);
			pack_bytes_into_wires_le(&mut w, &hash, &prev);
			let reference = ref_chain_step_blake3(&param_bytes[..param_len], epoch, chain_index, position, &prev);
			pack_bytes_into_wires_le(&mut w, &expected, &reference);

			circuit.populate_wire_witness(&mut w).unwrap();
			verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
		}

		#[test]
		fn generic_th_matches_reference(
			domain_len in 1usize..=32,
			data_len in 1usize..=200,
			domain in prop::collection::vec(any::<u8>(), 32),
			data in prop::collection::vec(any::<u8>(), 200),
		) {
			// The generic compress-based tweakable hash must equal its reference for any
			// domain (<= 32 bytes) and any data length.
			let b = CircuitBuilder::new();
			let dom_w: Vec<Wire> = (0..domain_len.div_ceil(8)).map(|_| b.add_inout()).collect();
			let dat_w: Vec<Wire> = (0..data_len.div_ceil(8)).map(|_| b.add_inout()).collect();
			let dom_term = ByteVec::new_const_len(&b, dom_w.clone(), domain_len);
			let dat_term = ByteVec::new_const_len(&b, dat_w.clone(), data_len);
			let out = circuit_blake3_th(&b, &[dom_term], &[dat_term]);
			let expected: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
			for k in 0..4 {
				b.assert_eq("th_out", out[k], expected[k]);
			}

			let circuit = b.build();
			let mut w = circuit.new_witness_filler();
			pack_bytes_into_wires_le(&mut w, &dom_w, &domain[..domain_len]);
			pack_bytes_into_wires_le(&mut w, &dat_w, &data[..data_len]);
			let reference = ref_blake3_th(&domain[..domain_len], &data[..data_len]);
			pack_bytes_into_wires_le(&mut w, &expected, &reference);

			circuit.populate_wire_witness(&mut w).unwrap();
			verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
		}

		#[test]
		fn two_lane_step_matches_two_singles(
			param_len in 1usize..=57,
			epoch in 0u32..=u32::MAX,
			cidx0 in 0u8..=255,
			cidx1 in 0u8..=255,
			position in 0u8..=255,
			prev0 in prop::array::uniform32(any::<u8>()),
			prev1 in prop::array::uniform32(any::<u8>()),
		) {
			// The 2x step's two lanes must each equal the single-lane reference.
			let b = CircuitBuilder::new();
			let pw = param_len.div_ceil(8);
			let param: Vec<Wire> = (0..pw).map(|_| b.add_inout()).collect();
			let epoch_w = b.add_inout();
			let h0: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
			let h1: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
			let (o0, o1) = circuit_chain_step_2x_blake3(&b, &param, param_len, epoch_w, h0, h1, cidx0, cidx1, position);
			let e0: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
			let e1: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
			for k in 0..4 {
				b.assert_eq("lane0", o0[k], e0[k]);
				b.assert_eq("lane1", o1[k], e1[k]);
			}

			let circuit = b.build();
			let mut w = circuit.new_witness_filler();
			let param_bytes = pack_param(param_len);
			pack_bytes_into_wires_le(&mut w, &param, &param_bytes);
			w[epoch_w] = Word::from_u64(epoch as u64);
			pack_bytes_into_wires_le(&mut w, &h0, &prev0);
			pack_bytes_into_wires_le(&mut w, &h1, &prev1);
			let r0 = ref_chain_step_blake3(&param_bytes[..param_len], epoch, cidx0, position, &prev0);
			let r1 = ref_chain_step_blake3(&param_bytes[..param_len], epoch, cidx1, position, &prev1);
			pack_bytes_into_wires_le(&mut w, &e0, &r0);
			pack_bytes_into_wires_le(&mut w, &e1, &r1);

			circuit.populate_wire_witness(&mut w).unwrap();
			verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn multi_step_chain_is_iterated_single_step() {
		// hash_chain_blake3 walking r steps equals r manual ref_chain_step compositions.
		let param = b"chain-blake3-param";
		let epoch = 0xDEAD_BEEFu32;
		let chain_index = 9u8;
		let start = [3u8; 32];
		let start_pos = 2usize;
		let r = 5usize;

		let mut cur = start;
		for i in 0..r {
			let pos = (start_pos + i + 1) as u8;
			cur = ref_chain_step_blake3(param, epoch, chain_index, pos, &cur);
		}
		assert_eq!(cur, hash_chain_blake3(param, epoch, chain_index, &start, start_pos, r));
	}
}
