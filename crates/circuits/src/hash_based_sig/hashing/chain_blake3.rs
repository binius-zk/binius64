// Copyright 2026 The Binius Developers
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
	blake3::{KEY_BYTES, blake3_compress, blake3_compress_2x, blake3_keyed_fixed, ref_compress},
	concat::concat,
	fixed_byte_vec::ByteVec,
	util::{clear_high_bits, split_u32_words, zeroed_u32_words},
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

/// Mask selecting the high 32 bits of a 64-bit word.
const HIGH32_MASK: u64 = 0xFFFF_FFFF_0000_0000;

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
///
/// The even word's high half is masked: [`blake3_compress`] leaves its discarded lane there, and
/// this is where the two halves of a result meet.
fn cv8_to_hash4(builder: &CircuitBuilder, cv: &[Wire; CV_WORDS]) -> [Wire; 4] {
	std::array::from_fn(|k| {
		let hi = builder.shl(cv[2 * k + 1], 32);
		builder.bxor(clear_high_bits(builder, cv[2 * k], 32), hi)
	})
}

/// Interleaves two 32-byte chain values into the 8 two-lane words a paired core consumes.
///
/// # Memory Layout
///
/// Each chain value arrives as four 64-bit wires holding eight 32-bit words:
///
/// ```text
///     first  value: [ a0 | a1 ] [ a2 | a3 ] [ a4 | a5 ] [ a6 | a7 ]
///     second value: [ b0 | b1 ] [ b2 | b3 ] [ b4 | b5 ] [ b6 | b7 ]
/// ```
///
/// The core wants one wire per word index, first value low, second value high:
///
/// ```text
///     word 0: [ b0 | a0 ]   word 1: [ b1 | a1 ]   ...   word 7: [ b7 | a7 ]
/// ```
///
/// Every output word is one masked half of each input word.
///
/// So the two-lane layout is reached directly, with no single-lane split in between.
fn interleave_cv_2x(
	builder: &CircuitBuilder,
	hash0: [Wire; 4],
	hash1: [Wire; 4],
) -> [Wire; CV_WORDS] {
	let high = builder.add_constant_64(HIGH32_MASK);
	std::array::from_fn(|i| {
		// Two output words come from one input wire, so the wire index advances at half the rate.
		let k = i / 2;
		if i % 2 == 0 {
			// An even word is each value's lower half.
			// The first value already sits there; the second lifts into the upper half.
			builder.bxor(clear_high_bits(builder, hash0[k], 32), builder.shl(hash1[k], 32))
		} else {
			// An odd word is each value's upper half.
			// The first value drops into the lower half; the second already sits in place.
			builder.bxor(builder.shr(hash0[k], 32), builder.band(hash1[k], high))
		}
	})
}

/// Splits a paired core's output back into the two 32-byte chain values it carries.
///
/// The exact inverse of the interleaving above.
///
/// - The first chain value is built from the low halves.
/// - The second is built from the high halves.
/// - Each 64-bit result wire consumes two consecutive output words.
fn deinterleave_cv_2x(builder: &CircuitBuilder, out: &[Wire; CV_WORDS]) -> ([Wire; 4], [Wire; 4]) {
	let high = builder.add_constant_64(HIGH32_MASK);
	// Low halves: the even word stays put, the odd word lifts into the upper half.
	let lane0 = std::array::from_fn(|k| {
		builder.bxor(clear_high_bits(builder, out[2 * k], 32), builder.shl(out[2 * k + 1], 32))
	});
	// High halves: the even word drops down, the odd word already sits in the upper half.
	let lane1 = std::array::from_fn(|k| {
		builder.bxor(builder.shr(out[2 * k], 32), builder.band(out[2 * k + 1], high))
	});
	(lane0, lane1)
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

	// Interleave the two chain values straight into the two-lane layout.
	// Splitting each into single-lane words first would mask the same halves twice.
	let cv = interleave_cv_2x(builder, hash0, hash1);
	let block0 = tweak_block(builder, domain_param_wires, param_len, epoch, chain_index0, position);
	let block1 = tweak_block(builder, domain_param_wires, param_len, epoch, chain_index1, position);

	// Pack lane 0 into the low 32 bits and lane 1 into the high 32 bits of each word.
	// The tweak words are built already masked, so the exclusive-or acts as a disjoint merge.
	let pack = |lo: Wire, hi: Wire| builder.bxor(lo, builder.shl(hi, 32));
	let block: [Wire; BLOCK_WORDS] = std::array::from_fn(|i| pack(block0[i], block1[i]));

	let zero = builder.add_constant_64(0);
	let block_len = builder.add_constant_64((msg_len as u64) | ((msg_len as u64) << 32));

	let out = blake3_compress_2x(builder, cv, block, zero, zero, block_len, zero);

	deinterleave_cv_2x(builder, &out)
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

/// Maximum tweak length in bytes, set by the size of a BLAKE3 key.
pub const TH_DOMAIN_MAX_BYTES: usize = KEY_BYTES;

/// BLAKE3 tweakable hash: the keyed hash of `data` under the tweak as key.
///
/// - The tweak is zero-padded to a 32-byte key, which is BLAKE3's native keying.
/// - The data is the message, hashed by the full BLAKE3 construction.
/// - The output is the 32-byte digest, as four 64-bit little-endian wires.
///
/// The full construction, rather than a bare chain of compressions, is what separates the domains:
/// - Every compression carries the keyed-hash flag.
/// - A chunk's first and last block are marked, and the root again.
/// - So no digest extends into the digest of a longer message, which an unpadded chain allows.
///
/// A tweak shorter than 32 bytes is zero-padded, so two tweaks equal after padding collide.
/// Every caller here lays its tweak out at fixed widths, which keeps them apart.
///
/// # Panics
///
/// - If the concatenated tweak exceeds the 32-byte key.
/// - If any tweak term's length is not fixed at circuit construction time.
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
		"BLAKE3 tweakable-hash tweak ({domain_len} bytes) exceeds the {TH_DOMAIN_MAX_BYTES}-byte key"
	);

	// The tweak is the key; `blake3_keyed_fixed` zero-pads it to 32 bytes and masks the rest.
	let key = concat(builder, domain_terms);

	// The data is the message, as 32-bit little-endian words with the bytes past its length zeroed.
	let data = concat(builder, data_terms);
	let message = zeroed_u32_words(builder, &data.data, data_len, data_len.div_ceil(4));

	cv8_to_hash4(builder, &blake3_keyed_fixed(builder, &message, data_len, &key))
}

/// Reference (out-of-circuit) form of [`circuit_blake3_th`].
///
/// One call to the BLAKE3 crate, so the circuit is pinned against the reference implementation
/// rather than a second hand-written copy of it.
///
/// # Panics
///
/// - If `domain` exceeds the 32-byte key.
pub fn ref_blake3_th(domain: &[u8], data: &[u8]) -> [u8; 32] {
	assert!(
		domain.len() <= TH_DOMAIN_MAX_BYTES,
		"BLAKE3 tweakable-hash tweak ({} bytes) exceeds the {TH_DOMAIN_MAX_BYTES}-byte key",
		domain.len(),
	);

	let mut key = [0u8; TH_DOMAIN_MAX_BYTES];
	key[..domain.len()].copy_from_slice(domain);
	*blake3::keyed_hash(&key, data).as_bytes()
}

#[cfg(test)]
mod tests {
	use binius_core::Word;
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
			w.pack_bytes_le(&param, &param_bytes);
			w[epoch_w] = Word::from_u64(epoch as u64);
			w.pack_bytes_le(&hash, &prev);
			let reference = ref_chain_step_blake3(&param_bytes[..param_len], epoch, chain_index, position, &prev);
			w.pack_bytes_le(&expected, &reference);

			circuit.populate_wire_witness(&mut w).unwrap();
			circuit.constraint_system().verify(&w.into_value_vec()).unwrap();
		}

		#[test]
		fn generic_th_matches_reference(
			domain_len in 0usize..=32,
			data_len in 0usize..=200,
			domain in prop::collection::vec(any::<u8>(), 32),
			data in prop::collection::vec(any::<u8>(), 200),
		) {
			// The tweakable hash must equal its reference for any tweak (<= 32 bytes) and any data
			// length, the empty ones included.
			check_th(&domain[..domain_len], &data[..data_len]);
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
			w.pack_bytes_le(&param, &param_bytes);
			w[epoch_w] = Word::from_u64(epoch as u64);
			w.pack_bytes_le(&h0, &prev0);
			w.pack_bytes_le(&h1, &prev1);
			let r0 = ref_chain_step_blake3(&param_bytes[..param_len], epoch, cidx0, position, &prev0);
			let r1 = ref_chain_step_blake3(&param_bytes[..param_len], epoch, cidx1, position, &prev1);
			w.pack_bytes_le(&e0, &r0);
			w.pack_bytes_le(&e1, &r1);

			circuit.populate_wire_witness(&mut w).unwrap();
			circuit.constraint_system().verify(&w.into_value_vec()).unwrap();
		}
	}

	/// Build the tweakable hash over `tweak` and `data`, and pin the digest to the reference.
	fn check_th(tweak: &[u8], data: &[u8]) {
		let b = CircuitBuilder::new();
		let tweak_w: Vec<Wire> = (0..tweak.len().div_ceil(8))
			.map(|_| b.add_inout())
			.collect();
		let data_w: Vec<Wire> = (0..data.len().div_ceil(8)).map(|_| b.add_inout()).collect();
		let tweak_term = ByteVec::new_const_len(&b, tweak_w.clone(), tweak.len());
		let data_term = ByteVec::new_const_len(&b, data_w.clone(), data.len());
		let out = circuit_blake3_th(&b, &[tweak_term], &[data_term]);
		let expected: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		for k in 0..4 {
			b.assert_eq("th_out", out[k], expected[k]);
		}

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&tweak_w, tweak);
		w.pack_bytes_le(&data_w, data);
		w.pack_bytes_le(&expected, &ref_blake3_th(tweak, data));

		circuit.populate_wire_witness(&mut w).unwrap_or_else(|e| {
			panic!("th disagreed for tweak_len={}, data_len={}: {e:?}", tweak.len(), data.len())
		});
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.unwrap();
	}

	#[test]
	fn th_is_the_keyed_hash_of_the_zero_padded_tweak() {
		// The tweak is the key, so a short tweak must agree with the same key padded by hand.
		let tweak = b"tweak";
		let data = b"the quick brown fox";
		let mut key = [0u8; TH_DOMAIN_MAX_BYTES];
		key[..tweak.len()].copy_from_slice(tweak);
		assert_eq!(ref_blake3_th(tweak, data), *blake3::keyed_hash(&key, data).as_bytes());
	}

	#[test]
	fn th_spans_chunks_and_blocks() {
		// Data past 1024 bytes builds a parent tree, a path a bare chain of compressions never has.
		//
		//     2304 bytes = 2 full chunks + 256 -> 3 chunk CVs -> 2 parent nodes
		//
		// 2304 is the real leaf-hash size for a 72-chain Winternitz instance.
		for &len in &[63usize, 64, 65, 1023, 1024, 1025, 2304] {
			let data: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
			check_th(b"chunk-spanning tweak", &data);
		}
	}

	#[test]
	#[should_panic(expected = "tweak (33 bytes) exceeds the 32-byte key")]
	fn th_rejects_an_oversized_tweak() {
		// The tweak becomes the key, so it cannot outgrow one.
		ref_blake3_th(&[0u8; TH_DOMAIN_MAX_BYTES + 1], b"data");
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
