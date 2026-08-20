// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::{
	SHA3_256_DIGEST_WORDS, SHA3_256_RATE_BYTES, SHA3_384_DIGEST_WORDS, SHA3_384_RATE_BYTES,
	SHA3_512_DIGEST_WORDS, SHA3_512_RATE_BYTES, SHA3_DELIMITER_BYTE,
};
use crate::{
	fixed_byte_vec::ByteVec,
	keccak::{N_WORDS_PER_STATE, permutation::keccak_f1600},
	multiplexer::{multi_wire_multiplex, single_wire_multiplex},
};

/// Computes a FIPS 202 SHA-3 digest of a variable-length message.
///
/// This is the shared core for every SHA-3 variant in this module.
///
/// Only the rate and the digest length differ between them.
///
/// The message length is a runtime value bounded by a fixed maximum, so the padded words are
/// derived with runtime multiplexers instead of computed directly.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message with a runtime length wire and a fixed maximum capacity.
/// * `rate_bytes` - The sponge rate for this hash variant, in bytes.
/// * `digest_words` - The digest length for this hash variant, in 64-bit words.
fn sha3_varlen(
	builder: &CircuitBuilder,
	message: &ByteVec,
	rate_bytes: usize,
	digest_words: usize,
) -> Vec<Wire> {
	let len_bytes = message.len_bytes;
	let data = &message.data;

	let n_words_per_block = rate_bytes / 8;
	let max_len_bytes = data.len() << 3;
	// A message that exactly fills its blocks still needs one more block for the padding.
	let n_blocks = (max_len_bytes + 1).div_ceil(rate_bytes);
	let n_words = n_blocks * n_words_per_block;

	// Reject any claimed length past the wire capacity before it drives any derivation below.
	let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
	builder.assert_false("len_check", too_long);

	let zero = builder.add_constant(Word::ZERO);
	let msb_one = builder.add_constant(Word::MSB_ONE);

	// Split the length into a word index and a byte offset within that word.
	//
	// The word at that index is where the padding's domain suffix belongs.
	let w_bd = builder.shr(len_bytes, 3);
	let len_mod_8 = builder.band(len_bytes, builder.add_constant_64(7));

	// Find which block holds the padding, by scanning every block's byte range.
	//
	// The rate is not a power of two, so this cannot be found with a shift.
	let mut end_block_index = zero;
	for block_no in 0..n_blocks {
		let block_start = builder.add_constant_64((block_no * rate_bytes) as u64);
		let block_end = builder.add_constant_64(((block_no + 1) * rate_bytes) as u64);
		let gte_start = builder.icmp_ule(block_start, len_bytes);
		let lt_end = builder.icmp_ult(len_bytes, block_end);
		let is_final_block = builder.band(gte_start, lt_end);
		end_block_index = builder.select(
			is_final_block,
			builder.add_constant_64(block_no as u64),
			end_block_index,
		);
	}

	// Build every possible boundary word, one per byte offset the length could land on.
	//
	// Each candidate keeps the low message bytes up to that offset and places the domain
	// suffix right after them.
	//
	// At offset zero the message contributes nothing, so the candidate is the suffix alone.
	let boundary_message_word = single_wire_multiplex(builder, data, w_bd);
	let candidates: Vec<Wire> = (0..8)
		.map(|i| {
			let mask = builder.add_constant_64(0x00FFFFFFFFFFFFFF >> ((7 - i) << 3));
			let delimiter = builder.add_constant_64(SHA3_DELIMITER_BYTE << (i << 3));
			let message_low = builder.band(boundary_message_word, mask);
			builder.bxor(message_low, delimiter)
		})
		.collect();
	// Select the one candidate that matches the actual byte offset.
	let boundary_word = single_wire_multiplex(builder, &candidates, len_mod_8);

	// Derive every padded word, one at a time, classified by its position relative to the
	// boundary word and to the block holding the padding.
	let padded_message: Vec<Wire> = (0..n_words)
		.map(|word_index| {
			let block_index = word_index / n_words_per_block;
			let column_index = word_index % n_words_per_block;
			let word_idx_wire = builder.add_constant_64(word_index as u64);

			let is_message_word = builder.icmp_ult(word_idx_wire, w_bd);
			let is_boundary_word = builder.icmp_eq(word_idx_wire, w_bd);
			let is_end_block =
				builder.icmp_eq(builder.add_constant_64(block_index as u64), end_block_index);

			// A word strictly before the boundary is plain message data.
			//
			// This index is only ever selected when it is within the message capacity, so the
			// zero fallback below it is never actually read.
			let msg_word = if word_index < data.len() {
				data[word_index]
			} else {
				zero
			};

			// The last word of the last block carries the padding rule's closing bit.
			//
			// It combines with the boundary word when the boundary falls there too, and
			// otherwise stands alone as a plain padding word.
			let delimiter = if column_index == n_words_per_block - 1 {
				builder.select(is_end_block, msb_one, zero)
			} else {
				zero
			};
			let boundary_val = if column_index == n_words_per_block - 1 {
				builder.bxor(boundary_word, delimiter)
			} else {
				boundary_word
			};

			let boundary_or_padding = builder.select(is_boundary_word, boundary_val, delimiter);
			builder.select(is_message_word, msg_word, boundary_or_padding)
		})
		.collect();

	// Absorb one padded block at a time.
	//
	// XOR it into the front of the state, then permute the whole state.
	//
	// Every intermediate state is kept, since the block holding the padding is only known at
	// runtime.
	let mut states: Vec<[Wire; N_WORDS_PER_STATE]> = Vec::with_capacity(n_blocks + 1);
	states.push([zero; N_WORDS_PER_STATE]);
	for block_no in 0..n_blocks {
		let mut state = states[block_no];
		for i in 0..n_words_per_block {
			state[i] = builder.bxor(state[i], padded_message[block_no * n_words_per_block + i]);
		}
		keccak_f1600(builder, &mut state);
		states.push(state);
	}

	// Select the digest out of the one state that followed the block holding the padding.
	let inputs: Vec<&[Wire]> = states[1..].iter().map(|s| &s[..]).collect();
	let digest_vec = multi_wire_multiplex(builder, &inputs, end_block_index);
	digest_vec[..digest_words].to_vec()
}

/// Computes the SHA3-256 digest of a variable-length message.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message with a runtime length wire and a fixed maximum capacity.
///
/// # Returns
/// The digest as 4 little-endian 64-bit words.
pub fn sha3_256_varlen(
	builder: &CircuitBuilder,
	message: &ByteVec,
) -> [Wire; SHA3_256_DIGEST_WORDS] {
	sha3_varlen(builder, message, SHA3_256_RATE_BYTES, SHA3_256_DIGEST_WORDS)
		.try_into()
		.unwrap()
}

/// Computes the SHA3-384 digest of a variable-length message.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message with a runtime length wire and a fixed maximum capacity.
///
/// # Returns
/// The digest as 6 little-endian 64-bit words.
pub fn sha3_384_varlen(
	builder: &CircuitBuilder,
	message: &ByteVec,
) -> [Wire; SHA3_384_DIGEST_WORDS] {
	sha3_varlen(builder, message, SHA3_384_RATE_BYTES, SHA3_384_DIGEST_WORDS)
		.try_into()
		.unwrap()
}

/// Computes the SHA3-512 digest of a variable-length message.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message with a runtime length wire and a fixed maximum capacity.
///
/// # Returns
/// The digest as 8 little-endian 64-bit words.
pub fn sha3_512_varlen(
	builder: &CircuitBuilder,
	message: &ByteVec,
) -> [Wire; SHA3_512_DIGEST_WORDS] {
	sha3_varlen(builder, message, SHA3_512_RATE_BYTES, SHA3_512_DIGEST_WORDS)
		.try_into()
		.unwrap()
}

#[cfg(test)]
mod tests {
	use binius_core::Word;
	use binius_frontend::{CircuitBuilder, Wire};
	use rand::prelude::*;
	use rstest::rstest;
	use sha3::Digest;

	use super::*;

	// Builds a circuit with the given capacity, runs one hash function on a message that fits
	// within it, and checks the computed digest against a reference implementation.
	fn test_varlen<const DIGEST_WORDS: usize>(
		message: &[u8],
		max_message_len_bytes: usize,
		hash_fn: impl FnOnce(&CircuitBuilder, &ByteVec) -> [Wire; DIGEST_WORDS],
		reference: impl FnOnce(&[u8]) -> Vec<u8>,
	) {
		assert!(message.len() <= max_message_len_bytes);

		let expected_digest = reference(message);
		assert_eq!(expected_digest.len(), DIGEST_WORDS * 8);

		let b = CircuitBuilder::new();
		let max_len_words = max_message_len_bytes.div_ceil(8);
		let input = ByteVec::new_inout(&b, max_len_words);
		let expected_digest_wires: [Wire; DIGEST_WORDS] = std::array::from_fn(|_| b.add_witness());

		// Constrain the circuit's digest to equal the expected one, wire by wire.
		let computed_digest = hash_fn(&b, &input);
		for i in 0..DIGEST_WORDS {
			b.assert_eq(format!("digest[{i}]"), computed_digest[i], expected_digest_wires[i]);
		}

		let circuit = b.build();
		let cs = circuit.constraint_system();
		let mut witness = circuit.new_witness_filler();

		// Populate the message below its capacity and record its true length.
		input.populate_data(&mut witness, message);
		input.populate_len_bytes(&mut witness, message.len());
		// Populate the expected digest wires.
		for (i, bytes) in expected_digest.chunks(8).enumerate() {
			witness[expected_digest_wires[i]] = Word(u64::from_le_bytes(bytes.try_into().unwrap()));
		}

		// Evaluating the circuit fills in every internal wire from the message and digest inputs.
		circuit
			.populate_wire_witness(&mut witness)
			.expect("Circuit should accept valid witness");
		// Checking the constraint system confirms the computed digest equals the reference
		// digest, not just that the witness filled in without panicking.
		cs.verify(&witness.into_value_vec())
			.expect("All constraints should be satisfied");
	}

	// Deterministic random message of the given length, seeded from both lengths so every
	// combination of message length and capacity gets an independent, reproducible message.
	fn random_message(message_len_bytes: usize, max_message_len_bytes: usize) -> Vec<u8> {
		let seed = ((message_len_bytes as u64) << 32) | (max_message_len_bytes as u64);
		let mut rng = StdRng::seed_from_u64(seed);
		let mut message = vec![0u8; message_len_bytes];
		rng.fill_bytes(&mut message);
		message
	}

	#[rstest]
	#[case(0, 100)] // Empty message
	#[case(1, 100)] // Single byte, well below capacity
	#[case(1, 144)] // Single byte, capacity spans two blocks
	#[case(135, 136)] // One byte before the block boundary
	#[case(136, 136)] // Exactly one block
	#[case(137, 272)] // Crosses the block boundary
	#[case(271, 272)] // One byte before two blocks
	#[case(272, 272)] // Exactly two blocks
	fn test_sha3_256_varlen(
		#[case] message_len_bytes: usize,
		#[case] max_message_len_bytes: usize,
	) {
		let message = random_message(message_len_bytes, max_message_len_bytes);
		test_varlen(&message, max_message_len_bytes, sha3_256_varlen, |m| {
			sha3::Sha3_256::digest(m).to_vec()
		});
	}

	#[rstest]
	#[case(0, 100)] // Empty message
	#[case(1, 100)] // Single byte, well below capacity
	#[case(103, 104)] // One byte before the block boundary
	#[case(104, 104)] // Exactly one block
	#[case(105, 208)] // Crosses the block boundary
	fn test_sha3_384_varlen(
		#[case] message_len_bytes: usize,
		#[case] max_message_len_bytes: usize,
	) {
		let message = random_message(message_len_bytes, max_message_len_bytes);
		test_varlen(&message, max_message_len_bytes, sha3_384_varlen, |m| {
			sha3::Sha3_384::digest(m).to_vec()
		});
	}

	#[rstest]
	#[case(0, 100)] // Empty message
	#[case(1, 100)] // Single byte, well below capacity
	#[case(71, 72)] // One byte before the block boundary
	#[case(72, 72)] // Exactly one block
	#[case(73, 144)] // Crosses the block boundary
	fn test_sha3_512_varlen(
		#[case] message_len_bytes: usize,
		#[case] max_message_len_bytes: usize,
	) {
		let message = random_message(message_len_bytes, max_message_len_bytes);
		test_varlen(&message, max_message_len_bytes, sha3_512_varlen, |m| {
			sha3::Sha3_512::digest(m).to_vec()
		});
	}
}
