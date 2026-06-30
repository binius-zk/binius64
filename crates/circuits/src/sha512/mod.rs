// Copyright 2025 Irreducible Inc.
pub mod compress;

use binius_core::{
	consts::{LOG_BYTE_BITS, LOG_WORD_SIZE_BITS},
	word::Word,
};
use binius_frontend::{CircuitBuilder, Wire};
pub use compress::{State, compress, pack_message_block};

use crate::{
	bytes::swap_bytes,
	fixed_byte_vec::ByteVec,
	multiplexer::{multi_wire_multiplex, single_wire_multiplex},
};

/// Computes SHA-512 hash of a fixed-length message.
///
/// This function creates a subcircuit that computes the SHA-512 hash of a message
/// with a compile-time known length. Unlike [`sha512_varlen`], which handles a runtime
/// length, this function is optimized for fixed-length inputs where the length is known at
/// circuit construction time.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `message` - Input message as packed 64-bit words (8 bytes per wire) in big-endian format. The
///   words should already be encoded from bytes in big-endian order, matching SHA-512's
///   byte-to-word conversion.
/// * `len_bytes` - The fixed length of the message in bytes (known at compile time)
///
/// # Returns
/// * `[Wire; 8]` - The SHA-512 digest as 8 wires of 64 bits each in big-endian order
///
/// # Panics
/// * If `message.len()` does not equal exactly `len_bytes.div_ceil(8)`
/// * If the message length in bits cannot fit in 64 bits
///
/// # Example
/// ```rust,ignore
/// use binius_frontend::crate::sha512::sha512_fixed;
/// use binius_frontend::compiler::CircuitBuilder;
///
/// let mut builder = CircuitBuilder::new();
///
/// // Create input wires for a 32-byte message
/// let message: Vec<_> = (0..4).map(|_| builder.add_witness()).collect();
///
/// // Compute SHA-512 of the 32-byte message
/// let digest = sha512_fixed(&builder, &message, 32);
/// ```
pub fn sha512_fixed(builder: &CircuitBuilder, message: &[Wire], len_bytes: usize) -> [Wire; 8] {
	// Validate that message.len() equals exactly len_bytes.div_ceil(8)
	assert_eq!(
		message.len(),
		len_bytes.div_ceil(8),
		"message.len() ({}) must equal len_bytes.div_ceil(8) ({})",
		message.len(),
		len_bytes.div_ceil(8)
	);

	// Ensure message length in bits fits in 64 bits
	assert!(
		(len_bytes as u64).checked_mul(8).is_some(),
		"Message length in bits must fit in 64 bits"
	);

	// Calculate padding requirements
	// SHA-512 requires: message || 0x80 || zeros || 128-bit length field
	// The 128-bit length field goes in the last 16 bytes of a block
	// We need at least 17 bytes for padding (1 for 0x80 + 16 for length)
	let n_blocks = (len_bytes + 17).div_ceil(128);
	let n_padded_words = n_blocks * 16; // 16 words per block

	// Create padded message wires
	let mut padded_message = Vec::with_capacity(n_padded_words);
	if len_bytes.is_multiple_of(8) {
		// Message ends at a word boundary - all words are complete
		padded_message.extend_from_slice(message);
		// Next word starts with 0x80 delimiter
		padded_message.push(builder.add_constant(Word(0x8000000000000000)));
	} else {
		// Message ends mid-word - need to handle boundary word
		padded_message.extend_from_slice(&message[..message.len() - 1]);

		// Handle the last message word which is partial
		let last_idx = message.len() - 1;
		let boundary_byte_in_word = len_bytes % 8;

		// Use shift operations to extract valid bytes and add delimiter
		// Shift right to remove unwanted bytes, then shift left to restore position
		let shift_amount = (8 - boundary_byte_in_word) * 8;
		let shifted_right = builder.shr(message[last_idx], shift_amount as u32);
		let shifted_back = builder.shl(shifted_right, shift_amount as u32);

		// Add 0x80 delimiter at the right position
		let delimiter_shift = (7 - boundary_byte_in_word) * 8;
		let delimiter = builder.add_constant(Word(0x80u64 << delimiter_shift));
		let boundary_word = builder.bxor(shifted_back, delimiter);
		padded_message.push(boundary_word);
	}

	// Fill with zeros until we reach the length field position
	let zero = builder.add_constant(Word::ZERO);
	padded_message.resize(n_padded_words - 2, zero);

	// Add the length field (128 bits, but high 64 bits are always 0 for us)
	padded_message.push(zero);

	let bitlen = (len_bytes as u64) * 8;
	padded_message.push(builder.add_constant(Word(bitlen))); // Low 64 bits of length

	// Process compression blocks
	let state_out = padded_message.chunks(16).enumerate().fold(
		State::iv(builder),
		|state, (block_idx, block)| {
			let block_message: [Wire; 16] = block
				.try_into()
				.expect("padded_message.len() must be divisible by 16");
			compress(
				&builder.subcircuit(format!("sha512_fixed_compress[{}]", block_idx)),
				state,
				block_message,
			)
		},
	);

	// Return the final state as the digest
	state_out.0
}

/// Computes SHA-512 hash of a variable-length message.
///
/// This gadget consumes a [`ByteVec`] whose actual length is runtime-determined and returns the
/// 512-bit digest as 8 wires in big-endian order, matching [`sha512_fixed`]'s output layout.
///
/// Internally the gadget *computes* each word of the SHA-512 padded message as a derived wire,
/// classifying every word position with the flags `is_message_word`, `is_boundary_word`, and
/// `is_length_block`. The word at the message/padding boundary mixes the trailing message bytes
/// with the `0x80` delimiter; padding words are zero except word 15 of the length block, which
/// holds the 64-bit bit length. The compression chain is then run over every possible block and
/// the final state is selected via a multiplexer indexed by the runtime length block.
///
/// The input [`ByteVec`] packs bytes little-endian, whereas the compression function consumes
/// big-endian words, so the data wires are byte-swapped up front.
///
/// # Arguments
/// * `builder` - Circuit builder
/// * `message` - Input message as a [`ByteVec`]. Its `len_bytes` wire holds the actual message
///   length.
///
/// # Returns
/// * `[Wire; 8]` - The SHA-512 digest as 8 wires of 64 bits each in big-endian order.
///
/// # Panics
/// * If the maximum message bit length cannot be represented in a 64-bit wire.
pub fn sha512_varlen(builder: &CircuitBuilder, message: &ByteVec) -> [Wire; 8] {
	// ---- 1. Input validation and setup
	//
	// Cap the maximum bit length so the 128-bit length field's low 64 bits suffice, compute the
	// number of compression blocks (accounting for the minimum 17 bytes of padding), and verify
	// the actual length is within bounds.
	let len_bytes = message.len_bytes;
	assert!(
		message.data.len() << LOG_WORD_SIZE_BITS <= u64::MAX as usize,
		"length of message in bits must fit in 64-bit wire"
	);

	let max_len_bytes = message.data.len() << (LOG_WORD_SIZE_BITS - LOG_BYTE_BITS);
	let n_blocks = (message.data.len() + 3).div_ceil(16);
	let n_words: usize = n_blocks << 4; // 16 words per block

	let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
	builder.assert_false("len_check", too_long);

	// `ByteVec` packs bytes little-endian; `compress` consumes big-endian words. Convert once.
	let message_be: Vec<Wire> = message
		.data
		.iter()
		.map(|&word| swap_bytes(builder, word))
		.collect();

	// ---- 2a. SHA-512 padding position calculation
	let zero = builder.add_constant(Word::ZERO);
	let w_bd = builder.shr(len_bytes, 3);
	let len_mod_8 = builder.band(len_bytes, builder.add_constant_zx_8(7));
	let bitlen = builder.shl(len_bytes, 3);

	// end_block_index = floor((len + 16) / 128) using 64-bit add
	let (sum, _carry) = builder.iadd(len_bytes, builder.add_constant_64(16));
	let end_block_index = builder.shr(sum, 7);

	// ---- Boundary word construction
	//
	// The word at index `w_bd` mixes the trailing message bytes with the 0x80 delimiter. Build the
	// eight candidate words (keeping `i` leading message bytes and placing the delimiter at byte
	// `i`) and select the one for `len_mod_8`. When `len_mod_8 == 0` the
	// chosen candidate is `0x80…00` independent of the (possibly out-of-range) boundary message
	// word, so the multiplexer's result is irrelevant in that case.
	let boundary_message_word = single_wire_multiplex(builder, &message_be, w_bd);
	let candidates: Vec<Wire> = (0..8)
		.map(|i| {
			let mask = builder.add_constant_64(0xFFFFFFFFFFFFFF00 << ((7 - i) << 3));
			let padding_byte = builder.add_constant_64(0x8000000000000000 >> (i << 3));
			let message_low = builder.band(boundary_message_word, mask);
			builder.bxor(message_low, padding_byte)
		})
		.collect();
	let boundary_word = single_wire_multiplex(builder, &candidates, len_mod_8);

	// ---- Padded message words
	//
	// Compute each padded word as a derived wire, classifying its position:
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

			// Pure message words select the corresponding input word. This is only ever selected
			// when word_index < w_bd ≤ max_len_bytes >> 3 == message_be.len(), so the index is in
			// range; the zero fallback for word_index ≥ message_be.len() is never chosen.
			let msg_word = if word_index < message_be.len() {
				message_be[word_index]
			} else {
				zero
			};

			// Padding words are zero, except word 15 of the length block which holds the bit
			// length. (Word 14 — the high 64 bits of the 128-bit length — stays zero, since only
			// ≤ 64-bit bit lengths are supported.)
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
	// Daisy-chain the compression function over every block, starting from the SHA-512 IV.
	let mut states = Vec::with_capacity(n_blocks + 1);
	states.push(State::iv(builder));
	for block_no in 0..n_blocks {
		let m: [Wire; 16] = padded_message[block_no << 4..(block_no + 1) << 4]
			.try_into()
			.unwrap();
		let state_out = compress(
			&builder.subcircuit(format!("compress[{block_no}]")),
			states[block_no].clone(),
			m,
		);
		states.push(state_out);
	}

	// ---- Final digest selection
	//
	// The digest is the state after processing the block containing the length field.
	let inputs: Vec<&[Wire]> = states[1..].iter().map(|s| &s.0[..]).collect();
	let final_digest_vec = multi_wire_multiplex(builder, &inputs, end_block_index);
	final_digest_vec.try_into().unwrap()
}

#[cfg(test)]
mod tests {
	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::{CircuitBuilder, Wire};
	use hex_literal::hex;
	use sha2::Digest;

	use super::{sha512_fixed, sha512_varlen};
	use crate::fixed_byte_vec::ByteVec;

	// ---- Tests for sha512_fixed function ----

	/// Helper function to test sha512_fixed with a specific message
	fn test_sha512_fixed_with_input(message_bytes: &[u8], expected_digest: [u8; 64]) {
		let builder = CircuitBuilder::new();

		// Create message wires
		let n_words = message_bytes.len().div_ceil(8);
		let message_wires: Vec<Wire> = (0..n_words).map(|_| builder.add_witness()).collect();

		// Create digest output wires
		let expected_digest_wires: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());

		// Call sha512_fixed
		let computed_digest = sha512_fixed(&builder, &message_wires, message_bytes.len());

		// Assert computed digest equals expected
		for i in 0..8 {
			builder.assert_eq(format!("digest[{i}]"), computed_digest[i], expected_digest_wires[i]);
		}

		let circuit = builder.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		// Populate message wires
		for (i, wire) in message_wires.iter().enumerate() {
			let byte_start = i * 8;
			let byte_end = ((i + 1) * 8).min(message_bytes.len());

			let mut word = 0u64;
			for j in byte_start..byte_end {
				word |= (message_bytes[j] as u64) << (56 - (j - byte_start) * 8);
			}
			w[*wire] = Word(word);
		}

		// Populate expected digest wires
		for (i, bytes) in expected_digest.chunks(8).enumerate() {
			let word = u64::from_be_bytes(bytes.try_into().unwrap());
			w[expected_digest_wires[i]] = Word(word);
		}

		circuit.populate_wire_witness(&mut w).unwrap();
		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	#[test]
	#[should_panic(expected = "message.len() (1) must equal len_bytes.div_ceil(8) (2)")]
	fn test_sha512_fixed_with_insufficient_wires() {
		let builder = CircuitBuilder::new();

		// Create only 1 wire but claim message is 10 bytes (which needs 2 wires)
		let message_wires: Vec<Wire> = vec![builder.add_witness()];

		// This should panic because message.len() (1) != len_bytes.div_ceil(8) (2)
		sha512_fixed(&builder, &message_wires, 10);
	}

	#[test]
	fn test_sha512_fixed_exact_wire_count() {
		let builder = CircuitBuilder::new();

		// Test that the function requires exact wire count

		// Empty message: 0 bytes requires 0 wires
		let empty: Vec<Wire> = vec![];
		let _ = sha512_fixed(&builder, &empty, 0);

		// 8 bytes requires exactly 1 wire
		let one_wire: Vec<Wire> = vec![builder.add_witness()];
		let _ = sha512_fixed(&builder, &one_wire, 8);

		// 10 bytes requires exactly 2 wires (10.div_ceil(8) = 2)
		let two_wires: Vec<Wire> = vec![builder.add_witness(), builder.add_witness()];
		let _ = sha512_fixed(&builder, &two_wires, 10);

		// 16 bytes requires exactly 2 wires
		let two_wires_full: Vec<Wire> = vec![builder.add_witness(), builder.add_witness()];
		let _ = sha512_fixed(&builder, &two_wires_full, 16);

		// 17 bytes requires exactly 3 wires (17.div_ceil(8) = 3)
		let three_wires: Vec<Wire> = vec![
			builder.add_witness(),
			builder.add_witness(),
			builder.add_witness(),
		];
		let _ = sha512_fixed(&builder, &three_wires, 17);
	}

	#[test]
	fn test_sha512_fixed_various_sizes() {
		use rand::prelude::*;

		// Test various message sizes to ensure padding works correctly
		let sizes = vec![
			0,   // empty
			1,   // single byte
			7,   // just under word boundary
			8,   // exactly one word
			9,   // just over word boundary
			63,  // just under half block
			64,  // exactly half block
			65,  // just over half block
			111, // max single block
			112, // forces two blocks
			127, // one byte from block boundary
			128, // exactly one block
			129, // just over one block
			239, // max two blocks
			240, // forces three blocks
			256, // exactly two blocks
		];

		let mut rng = StdRng::seed_from_u64(0);

		for size in sizes {
			// Generate random payload
			let mut message = vec![0u8; size];
			rng.fill(&mut message[..]);

			// Compute expected hash using sha2 crate
			let expected = sha2::Sha512::digest(&message);
			let expected_bytes: [u8; 64] = expected.into();

			// Test with our circuit
			test_sha512_fixed_with_input(&message, expected_bytes);
		}
	}

	// ---- Tests for sha512_varlen function ----

	/// Helper that builds a circuit with the given `max_len_bytes` capacity, runs
	/// `sha512_varlen` on a `ByteVec` populated with `message_bytes`, and asserts the
	/// computed digest equals `expected_digest`.
	fn test_sha512_varlen_with_input(
		message_bytes: &[u8],
		expected_digest: [u8; 64],
		max_len_bytes: usize,
	) {
		assert!(message_bytes.len() <= max_len_bytes);

		let builder = CircuitBuilder::new();
		let max_len_words = max_len_bytes.div_ceil(8);
		let input = ByteVec::new_inout(&builder, max_len_words);
		let expected_digest_wires: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());

		let computed_digest = sha512_varlen(&builder, &input);
		for i in 0..8 {
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
	fn test_sha512_varlen_empty() {
		test_sha512_varlen_with_input(
			b"",
			hex!(
				"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
			),
			128,
		);
	}

	#[test]
	fn test_sha512_varlen_abc() {
		test_sha512_varlen_with_input(
			b"abc",
			hex!(
				"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
			),
			128,
		);
	}

	#[test]
	fn test_sha512_varlen_various_sizes() {
		use rand::prelude::*;

		// Same boundary-rich set used by test_sha512_fixed_various_sizes, plus 0.
		let sizes: Vec<usize> = vec![
			0, 1, 7, 8, 9, 63, 64, 65, 111, 112, 127, 128, 129, 239, 240, 256,
		];
		// Fixed capacity larger than every test message exercises the variable-length path.
		let max_len_bytes = 320;

		let mut rng = StdRng::seed_from_u64(0);
		for size in sizes {
			let mut message = vec![0u8; size];
			rng.fill(&mut message[..]);

			let expected = sha2::Sha512::digest(&message);
			let expected_bytes: [u8; 64] = expected.into();

			test_sha512_varlen_with_input(&message, expected_bytes, max_len_bytes);
		}
	}
}
