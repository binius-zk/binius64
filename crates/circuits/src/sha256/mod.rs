// Copyright 2025 Irreducible Inc.
pub mod compress;

use binius_core::{consts::LOG_BYTE_BITS, word::Word};
use binius_frontend::{CircuitBuilder, Wire};
pub use compress::{
	State, populate_message_block, ref_compress, sha256_compress, sha256_compress_2x,
	sha256_compress_2x_seq,
};

use crate::{
	bytes::swap_bytes_32,
	fixed_byte_vec::ByteVec,
	multiplexer::{multi_wire_multiplex, single_wire_multiplex},
};

/// Computes SHA-256 hash of a fixed-length message.
///
/// This function creates a subcircuit that computes the SHA-256 hash of a message
/// with a compile-time known length. Unlike `sha256_varlen`, which handles
/// variable-length inputs, this function is optimized for fixed-length inputs where
/// the length is known at circuit construction time.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `message` - Input message as 32-bit words (4 bytes per wire) in big-endian format. Each wire
///   must have the high 32 bits set to zero (enforced as a precondition).
/// * `len_bytes` - The fixed length of the message in bytes (known at compile time)
///
/// # Returns
/// * `[Wire; 8]` - The SHA-256 digest as 8 wires, each containing a 32-bit word in the low 32 bits
///   (high 32 bits are zero) in big-endian order
///
/// # Panics
/// * If `message.len()` does not equal exactly `len_bytes.div_ceil(4)`
/// * If the message length in bits cannot fit in 32 bits
///
/// # Example
/// ```rust,ignore
/// use binius_frontend::crate::sha256::sha256_fixed;
/// use binius_frontend::compiler::CircuitBuilder;
///
/// let mut builder = CircuitBuilder::new();
///
/// // Create input wires for a 32-byte message (8 32-bit words)
/// let message: Vec<_> = (0..8).map(|_| builder.add_witness()).collect();
///
/// // Compute SHA-256 of the 32-byte message
/// let digest = sha256_fixed(&builder, &message, 32);
/// ```
pub fn sha256_fixed(builder: &CircuitBuilder, message: &[Wire], len_bytes: usize) -> [Wire; 8] {
	// Validate that message.len() equals exactly len_bytes.div_ceil(4)
	assert_eq!(
		message.len(),
		len_bytes.div_ceil(4),
		"message.len() ({}) must equal len_bytes.div_ceil(4) ({})",
		message.len(),
		len_bytes.div_ceil(4)
	);

	// Ensure message length in bits fits in 32 bits
	assert!(
		(len_bytes as u64)
			.checked_mul(8)
			.is_some_and(|bits| bits <= u32::MAX as u64),
		"Message length in bits must fit in 32 bits"
	);

	// Calculate padding requirements
	// SHA-256 requires: message || 0x80 || zeros || 64-bit length field
	// The 64-bit length field goes in the last 8 bytes of a block
	// We need at least 9 bytes for padding (1 for 0x80 + 8 for length)
	let n_blocks = (len_bytes + 9).div_ceil(64);
	let n_padded_words = n_blocks * 16; // 16 32-bit words per block

	// Create padded message
	let mut padded_message = Vec::with_capacity(n_padded_words);

	// Add message words
	let n_message_words = len_bytes / 4;
	let boundary_bytes = len_bytes % 4;

	// Add complete message words
	padded_message.extend_from_slice(&message[0..n_message_words]);

	// Handle partial word at boundary
	if boundary_bytes > 0 {
		// The last message word contains partial data
		let last_word = message[n_message_words];

		// Mask out the unused bytes and add delimiter
		let shift_amount = (4 - boundary_bytes) * 8;
		let mask = builder.add_constant(Word((0xFFFFFFFFu64 >> shift_amount) << shift_amount));
		let masked = builder.band(last_word, mask);

		// Add 0x80 delimiter at the right position
		let delimiter_shift = (3 - boundary_bytes) * 8;
		let delimiter = builder.add_constant(Word(0x80u64 << delimiter_shift));
		let boundary_word = builder.bxor(masked, delimiter);

		padded_message.push(boundary_word);
	} else {
		// Message ends at word boundary - delimiter goes in new word
		padded_message.push(builder.add_constant(Word(0x80000000)));
	}

	// Fill with zeros until we reach the length field position
	let zero = builder.add_constant(Word::ZERO);
	padded_message.resize(n_padded_words - 2, zero);

	// Add the length field (64 bits total)
	padded_message.push(zero); // High 32 bits of length (always 0 for us)
	let bitlen = (len_bytes as u64) * 8;
	padded_message.push(builder.add_constant(Word(bitlen)));

	// Process compression blocks two at a time.
	// Consecutive blocks chain, so each pair runs in the two lanes of one parallel core.
	// A pair costs ~half the AND count of two single-lane compressions.
	// A trailing odd block has no partner and is compressed single-lane.
	let blocks: Vec<[Wire; 16]> = padded_message
		.chunks_exact(16)
		.map(|block| block.try_into().unwrap())
		.collect();
	let n_blocks = blocks.len();

	let mask32 = builder.add_constant(Word::MASK_32);
	let mut state = State::iv(builder);
	let mut block_idx = 0;
	while block_idx + 1 < n_blocks {
		let out = sha256_compress_2x_seq(
			&builder.subcircuit(format!("sha256_fixed_compress[{block_idx}..{}]", block_idx + 2)),
			state,
			[blocks[block_idx], blocks[block_idx + 1]],
		);
		// The chaining state after the pair is the second compression's output.
		// It sits in the low 32 bits of each word; the mask restores the empty high half.
		state = State::new(std::array::from_fn(|i| builder.band(out.0[i], mask32)));
		block_idx += 2;
	}
	if block_idx < n_blocks {
		state = sha256_compress(
			&builder.subcircuit(format!("sha256_fixed_compress[{block_idx}]")),
			state,
			blocks[block_idx],
		);
	}

	// Return the final state as 8 32-bit words
	state.0
}

/// Computes the SHA-256 hash of a variable-length message.
///
/// This gadget consumes a [`ByteVec`] whose actual length is runtime-determined and returns the
/// 256-bit digest as 4 wires of 64 bits each in big-endian order, matching [`sha256_fixed`]'s
/// output layout (produced by [`State::pack_4x64b`]).
///
/// Internally the gadget *computes* each 32-bit word of the SHA-256 padded message as a derived
/// wire, classifying every word position with the flags `is_message_word`, `is_boundary_word`, and
/// `is_length_block`. The word at the message/padding boundary mixes the trailing message bytes
/// with the `0x80` delimiter; padding words are zero except word 15 of the length block, which
/// holds the bit length. The compression chain is run over every possible block and the final state
/// is selected via a multiplexer indexed by the runtime length block.
///
/// Unlike a checker gadget that asserts a caller-supplied digest, this function takes no such
/// digest and performs no digest assertion: the returned digest is a single-valued function of
/// `(data, len_bytes)`, so a free `len_bytes` can only select which message prefix is hashed, never
/// an arbitrary digest. The caller remains responsible for constraining `len_bytes` to its intended
/// value.
///
/// The input [`ByteVec`] packs bytes little-endian, whereas the compression function consumes
/// big-endian words, so the data wires are byte-swapped up front. SHA-256's 32-bit schedule words
/// are half the width of a `ByteVec` word, so each data word yields two consecutive schedule words.
///
/// # Arguments
/// * `builder` - Circuit builder
/// * `message` - Input message as a [`ByteVec`]. Its `len_bytes` wire holds the actual message
///   length.
///
/// # Returns
/// * `[Wire; 4]` - The SHA-256 digest as 4 wires of 64 bits each in big-endian order.
///
/// # Panics
/// * If the maximum message bit length cannot be represented in the 32-bit length field.
pub fn sha256_varlen(builder: &CircuitBuilder, message: &ByteVec) -> [Wire; 4] {
	// ---- 1. Input validation and setup
	//
	// Cap the maximum bit length so the 64-bit length field's low 32 bits suffice, compute the
	// number of compression blocks (accounting for the minimum 9 bytes of padding), and verify the
	// actual length is within bounds.
	let len_bytes = message.len_bytes;
	assert!(
		message.data.len() << Word::LOG_BITS <= u32::MAX as usize,
		"length of message in bits must fit within 32 bits"
	);

	let max_len_bytes = message.data.len() << (Word::LOG_BITS - LOG_BYTE_BITS);
	let n_blocks = (message.data.len() + 2).div_ceil(8);
	let n_words: usize = n_blocks << 4; // 16 words per block

	let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
	builder.assert_false("len_check", too_long);

	// `ByteVec` packs bytes little-endian; `sha256_compress` consumes big-endian 32-bit words. Each
	// 64-bit data word carries two schedule words: `swap_bytes_32` byte-reverses within each 32-bit
	// half, so the low half becomes the big-endian schedule word for the first four bytes and the
	// high half the schedule word for the next four. Split each into two low-32 wires.
	let mask32 = builder.add_constant(Word::MASK_32);
	let mut message_be: Vec<Wire> = Vec::with_capacity(message.data.len() * 2);
	for &word in &message.data {
		let swapped = swap_bytes_32(builder, word);
		message_be.push(builder.band(swapped, mask32));
		message_be.push(builder.shr(swapped, 32));
	}

	// ---- 2a. SHA-256 padding position calculation
	let zero = builder.add_constant(Word::ZERO);
	let w_bd = builder.shr(len_bytes, 2);
	let len_mod_4 = builder.band(len_bytes, builder.add_constant_zx_8(3));
	let bitlen = builder.shl(len_bytes, 3);

	// end_block_index = floor((len + 8) / 64) using a 64-bit add.
	let (sum, _carry) = builder.iadd(len_bytes, builder.add_constant_64(8));
	let end_block_index = builder.shr(sum, 6);

	// ---- Boundary word construction
	//
	// The 32-bit word at index `w_bd` mixes the trailing message bytes with the 0x80 delimiter.
	// Build the four candidate words (keeping `i` leading message bytes and placing the delimiter
	// at byte `i`) and select the one for `len_mod_4`. When `len_mod_4 == 0` the chosen candidate
	// is `0x80000000` independent of the (possibly out-of-range) boundary message word, so the
	// multiplexer's result is irrelevant in that case.
	let boundary_message_word = single_wire_multiplex(builder, &message_be, w_bd);
	let candidates: Vec<Wire> = (0..4)
		.map(|i| {
			let mask = builder.add_constant_64((0xFFFFFFFFu64 << ((4 - i) << 3)) & 0xFFFFFFFF);
			let padding_byte = builder.add_constant_64(0x80000000u64 >> (i << 3));
			let message_low = builder.band(boundary_message_word, mask);
			builder.bxor(message_low, padding_byte)
		})
		.collect();
	let boundary_word = single_wire_multiplex(builder, &candidates, len_mod_4);

	// ---- Padded message words
	//
	// Compute each 32-bit padded word as a derived wire, classifying its position:
	//
	//     1. word_index <  w_bd - pure message word
	//     2. word_index == w_bd - boundary word (message bytes + 0x80 delimiter)
	//     3. word_index >  w_bd - pure padding, except word 15 of the length block (the bit length)
	let padded_message: Vec<Wire> = (0..n_words)
		.map(|word_index| {
			let block_index = word_index >> 4;
			let column_index = word_index & 15;

			let is_message_word =
				builder.icmp_ult(builder.add_constant_64(word_index as u64), w_bd);
			let is_boundary_word =
				builder.icmp_eq(builder.add_constant_64(word_index as u64), w_bd);
			let is_length_block =
				builder.icmp_eq(builder.add_constant_64(block_index as u64), end_block_index);

			// Pure message words select the corresponding schedule word. This is only ever selected
			// when word_index < w_bd ≤ max_len_bytes >> 2 == message_be.len(), so the index is in
			// range; the zero fallback for word_index ≥ message_be.len() is never chosen.
			let msg_word = if word_index < message_be.len() {
				message_be[word_index]
			} else {
				zero
			};

			// Padding words are zero, except word 15 of the length block which holds the bit
			// length. (Word 14 — the high 32 bits of the 64-bit length — stays zero, since only
			// ≤ 32-bit bit lengths are supported.)
			let past_word = if column_index == 15 {
				builder.select(is_length_block, bitlen, zero)
			} else {
				zero
			};

			let boundary_or_past = builder.select(is_boundary_word, boundary_word, past_word);
			builder.select(is_message_word, msg_word, boundary_or_past)
		})
		.collect();

	// ---- Compression chain
	//
	// Compress two chained blocks per step through one parallel core: `sha256_compress_2x_seq` runs
	// both in the two 32-bit lanes of a 64-bit word, for ~the AND cost of a single compression. The
	// paired output packs the state after the first block in the high 32 bits and the state after
	// the second block in the low 32 bits. `states[k]` therefore ends up being the state after
	// block `k - 1`, exactly as a single-lane chain would produce, so the digest multiplexer is
	// unchanged.
	let mut states = Vec::with_capacity(n_blocks + 1);
	states.push(State::iv(builder));
	let mk_m = |block_no: usize| -> [Wire; 16] {
		padded_message[block_no << 4..(block_no + 1) << 4]
			.try_into()
			.unwrap()
	};
	let mut block_no = 0;
	while block_no + 1 < n_blocks {
		let out = sha256_compress_2x_seq(
			&builder.subcircuit(format!("compress[{block_no}..{}]", block_no + 2)),
			states[block_no].clone(),
			[mk_m(block_no), mk_m(block_no + 1)],
		);
		// The mask restores the empty high half that the single-lane digest packing relies on.
		let state_first = State::new(std::array::from_fn(|i| builder.shr(out.0[i], 32)));
		let state_second = State::new(std::array::from_fn(|i| builder.band(out.0[i], mask32)));
		states.push(state_first);
		states.push(state_second);
		block_no += 2;
	}
	// A trailing odd block has no partner, so compress it single-lane.
	if block_no < n_blocks {
		let state_out = sha256_compress(
			&builder.subcircuit(format!("compress[{block_no}]")),
			states[block_no].clone(),
			mk_m(block_no),
		);
		states.push(state_out);
	}

	// ---- Final digest selection
	//
	// The digest is the state after processing the block containing the length field, packed into
	// four 64-bit big-endian words. No caller-supplied digest is asserted — the packed selected
	// state IS the return value.
	let block_digests: Vec<[Wire; 4]> = states[1..].iter().map(|s| s.pack_4x64b(builder)).collect();
	let inputs: Vec<&[Wire]> = block_digests.iter().map(|d| &d[..]).collect();
	let final_digest_vec = multi_wire_multiplex(builder, &inputs, end_block_index);
	final_digest_vec.try_into().unwrap()
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::{CircuitBuilder, Wire};
	use hex_literal::hex;
	use sha2::Digest;

	use super::*;

	// ---- Tests for sha256_varlen function ----

	/// Builds a circuit with the given `max_len_bytes` capacity, runs `sha256_varlen` on a
	/// `ByteVec` populated with `message_bytes`, and asserts the computed digest equals
	/// `expected_digest`.
	fn test_sha256_varlen_with_input(
		message_bytes: &[u8],
		expected_digest: [u8; 32],
		max_len_bytes: usize,
	) {
		assert!(message_bytes.len() <= max_len_bytes);

		let builder = CircuitBuilder::new();
		let max_len_words = max_len_bytes.div_ceil(8);
		let input = ByteVec::new_inout(&builder, max_len_words);
		let expected_digest_wires: [Wire; 4] = array::from_fn(|_| builder.add_witness());

		let computed_digest = sha256_varlen(&builder, &input);
		for i in 0..4 {
			builder.assert_eq(format!("digest[{i}]"), computed_digest[i], expected_digest_wires[i]);
		}

		let circuit = builder.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		input.populate_data(&mut w, message_bytes);
		input.populate_len_bytes(&mut w, message_bytes.len());

		for (i, bytes) in expected_digest.chunks(8).enumerate() {
			let word = u64::from_be_bytes(bytes.try_into().unwrap());
			w[expected_digest_wires[i]] = Word(word);
		}

		circuit.populate_wire_witness(&mut w).unwrap();
		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	#[test]
	fn test_sha256_varlen_empty() {
		test_sha256_varlen_with_input(
			b"",
			hex!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
			64,
		);
	}

	#[test]
	fn test_sha256_varlen_abc() {
		test_sha256_varlen_with_input(
			b"abc",
			hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
			64,
		);
	}

	#[test]
	fn test_sha256_varlen_two_block_boundary() {
		// 56 bytes forces a second block (56 + 1 delimiter + 8 length > 64).
		test_sha256_varlen_with_input(
			&[b'a'; 56],
			hex!("b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"),
			128,
		);
	}

	#[test]
	fn test_sha256_varlen_various_sizes() {
		use rand::prelude::*;

		// Boundary-rich sizes around SHA-256's 64-byte block (word, block, and length-field
		// boundaries), plus 0.
		let sizes: Vec<usize> = vec![
			0, 1, 3, 4, 5, 31, 32, 33, 55, 56, 63, 64, 65, 119, 120, 128, 256,
		];
		// Fixed capacity larger than every test message exercises the variable-length path.
		let max_len_bytes = 320;

		let mut rng = StdRng::seed_from_u64(0);
		for size in sizes {
			let mut message = vec![0u8; size];
			rng.fill(&mut message[..]);

			let expected = sha2::Sha256::digest(&message);
			let expected_bytes: [u8; 32] = expected.into();

			test_sha256_varlen_with_input(&message, expected_bytes, max_len_bytes);
		}
	}

	#[test]
	fn test_sha256_varlen_length_exceeds_max_rejection() {
		// A `len_bytes` wire exceeding the ByteVec capacity must be rejected by the in-circuit
		// `len_check` guard (the gadget bounds `len_bytes <= capacity` from its own data length).
		let builder = CircuitBuilder::new();
		let max_len_bytes = 64usize;
		let max_len_words = max_len_bytes.div_ceil(8);
		let input = ByteVec::new_inout(&builder, max_len_words);
		let _ = sha256_varlen(&builder, &input);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		input.populate_data(&mut w, b"");
		// Claim a length one byte past the capacity; the `len_check` assertion must fail.
		w[input.len_bytes] = Word(max_len_bytes as u64 + 1);
		assert!(circuit.populate_wire_witness(&mut w).is_err());
	}

	// Helper function for sha256_fixed tests
	fn test_sha256_fixed_with_input(message: &[u8], expected_bytes: [u8; 32]) {
		let b = CircuitBuilder::new();

		// Pack message into 32-bit words
		let n_words = message.len().div_ceil(4);
		let mut message_wires = Vec::new();

		for word_idx in 0..n_words {
			let mut packed = 0u32;
			for i in 0..4 {
				let byte_idx = word_idx * 4 + i;
				if byte_idx < message.len() {
					packed |= (message[byte_idx] as u32) << (24 - i * 8);
				}
			}
			message_wires.push(b.add_constant(Word(packed as u64)));
		}

		// Create expected digest wires (8 32-bit words)
		let expected_digest_wires = array::from_fn::<_, 8, _>(|_| b.add_inout());

		// Compute the digest
		let computed_digest = sha256_fixed(&b, &message_wires, message.len());

		// Assert that computed digest equals expected digest
		for i in 0..8 {
			b.assert_eq(format!("digest[{}]", i), computed_digest[i], expected_digest_wires[i]);
		}

		let circuit = b.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		// Populate the expected digest wires
		for i in 0..8 {
			let mut word = 0u32;
			for j in 0..4 {
				word |= (expected_bytes[i * 4 + j] as u32) << (24 - j * 8);
			}
			w[expected_digest_wires[i]] = Word(word as u64);
		}

		circuit.populate_wire_witness(&mut w).unwrap();
		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	#[test]
	#[should_panic(expected = "message.len() (1) must equal len_bytes.div_ceil(4) (2)")]
	fn test_sha256_fixed_with_insufficient_wires() {
		use super::sha256_fixed;
		let builder = CircuitBuilder::new();

		// Create only 1 wire but claim message is 5 bytes (which needs 2 wires)
		let message_wires: Vec<Wire> = vec![builder.add_witness()];

		// This should panic because message.len() (1) != len_bytes.div_ceil(4) (2)
		sha256_fixed(&builder, &message_wires, 5);
	}

	#[test]
	fn test_sha256_fixed_various_sizes() {
		use rand::prelude::*;

		// Test various message sizes to ensure padding works correctly
		let sizes = vec![
			0,   // empty
			1,   // single byte
			3,   // "abc" test vector
			4,   // exactly one word
			5,   // just over word boundary
			31,  // just under half block
			32,  // exactly half block
			33,  // just over half block
			55,  // max single block
			56,  // forces two blocks
			63,  // one byte from block boundary
			64,  // exactly one block
			65,  // just over one block
			119, // max two blocks
			120, // forces three blocks
			128, // exactly two blocks
			256, // exactly four blocks
		];

		let mut rng = StdRng::seed_from_u64(0);

		for size in sizes {
			// Generate random payload
			let mut message = vec![0u8; size];
			rng.fill(&mut message[..]);

			// Compute expected hash using sha2 crate
			let expected = sha2::Sha256::digest(&message);
			let expected_bytes: [u8; 32] = expected.into();

			// Test with our circuit
			test_sha256_fixed_with_input(&message, expected_bytes);
		}
	}
}
