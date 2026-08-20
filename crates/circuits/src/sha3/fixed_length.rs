// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::{
	SHA3_256_DIGEST_WORDS, SHA3_256_RATE_BYTES, SHA3_384_DIGEST_WORDS, SHA3_384_RATE_BYTES,
	SHA3_512_DIGEST_WORDS, SHA3_512_RATE_BYTES, SHA3_DELIMITER_BYTE,
};
use crate::keccak::{N_WORDS_PER_STATE, permutation::keccak_f1600};

/// Computes a FIPS 202 SHA-3 digest of a fixed-length message.
///
/// This is the shared core for every SHA-3 variant in this module.
///
/// Only the rate and the digest length differ between them.
///
/// The message length is a compile-time constant, so the padded words are computed directly
/// instead of derived with runtime multiplexers.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message as packed 64-bit words, 8 bytes per wire, little-endian.
/// * `len_bytes` - The exact length of the message in bytes.
/// * `rate_bytes` - The sponge rate for this hash variant, in bytes.
/// * `digest_words` - The digest length for this hash variant, in 64-bit words.
///
/// # Panics
/// * If the message wire count does not equal the message length divided into 64-bit words, rounded
///   up.
fn sha3_fixed(
	builder: &CircuitBuilder,
	message: &[Wire],
	len_bytes: usize,
	rate_bytes: usize,
	digest_words: usize,
) -> Vec<Wire> {
	// The caller must supply exactly one wire per 8-byte word of the message.
	assert_eq!(
		message.len(),
		len_bytes.div_ceil(8),
		"message.len() ({}) must equal len_bytes.div_ceil(8) ({})",
		message.len(),
		len_bytes.div_ceil(8)
	);

	let n_words_per_block = rate_bytes / 8;
	// A message that exactly fills whole blocks still needs one more block for the padding.
	let n_blocks = (len_bytes + 1).div_ceil(rate_bytes);
	let n_padded_words = n_blocks * n_words_per_block;

	let mut padded_message = Vec::with_capacity(n_padded_words);

	// FIPS 202 folds a two-bit domain suffix into the first padding byte.
	//
	// A byte-aligned message either ends exactly on a word boundary, in which case the suffix
	// becomes a whole new word, or ends partway through a word, in which case the suffix is
	// combined with the trailing message bytes in that same word.
	if len_bytes.is_multiple_of(8) {
		// Every message word is already complete, so the suffix becomes its own word.
		padded_message.extend_from_slice(message);
		padded_message.push(builder.add_constant(Word(SHA3_DELIMITER_BYTE)));
	} else {
		// Every word before the last one is complete message data.
		padded_message.extend_from_slice(&message[..message.len() - 1]);

		let last_idx = message.len() - 1;
		let byte_in_word = len_bytes % 8;

		// Keep only the valid low bytes of the trailing word.
		let mask = (1u64 << (byte_in_word * 8)) - 1;
		let masked_word = builder.band(message[last_idx], builder.add_constant(Word(mask)));

		// Place the domain suffix right after the last valid message byte.
		let padding_bit = SHA3_DELIMITER_BYTE << (byte_in_word * 8);
		let boundary_word = builder.bxor(masked_word, builder.add_constant(Word(padding_bit)));
		padded_message.push(boundary_word);
	}

	// Every word after the boundary word is zero padding, until the final byte set below.
	let zero = builder.add_constant(Word::ZERO);
	padded_message.resize(n_padded_words, zero);

	// The padding rule always sets the top bit of the last byte of the last block.
	//
	// XOR combines it correctly even when the domain suffix already landed in that same word.
	let last_byte_mask = 0x80u64 << 56;
	let last_idx = n_padded_words - 1;
	padded_message[last_idx] =
		builder.bxor(padded_message[last_idx], builder.add_constant(Word(last_byte_mask)));

	// Absorb one padded block at a time.
	//
	// XOR it into the front of the state, then permute the whole state.
	let mut state = [zero; N_WORDS_PER_STATE];
	for block in padded_message.chunks(n_words_per_block) {
		for (i, &word) in block.iter().enumerate() {
			state[i] = builder.bxor(state[i], word);
		}
		keccak_f1600(builder, &mut state);
	}

	// The digest length never exceeds the rate for any FIPS 202 hash function.
	//
	// So the digest can be read straight out of the state after the last permutation.
	state[..digest_words].to_vec()
}

/// Computes the SHA3-256 digest of a fixed-length message.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message as packed 64-bit words, 8 bytes per wire, little-endian.
/// * `len_bytes` - The exact length of the message in bytes.
///
/// # Returns
/// The digest as 4 little-endian 64-bit words.
///
/// # Panics
/// * If the message wire count does not equal the message length divided into 64-bit words, rounded
///   up.
pub fn sha3_256(
	builder: &CircuitBuilder,
	message: &[Wire],
	len_bytes: usize,
) -> [Wire; SHA3_256_DIGEST_WORDS] {
	// Delegate to the shared core with this variant's rate and digest length.
	sha3_fixed(builder, message, len_bytes, SHA3_256_RATE_BYTES, SHA3_256_DIGEST_WORDS)
		.try_into()
		.unwrap()
}

/// Computes the SHA3-384 digest of a fixed-length message.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message as packed 64-bit words, 8 bytes per wire, little-endian.
/// * `len_bytes` - The exact length of the message in bytes.
///
/// # Returns
/// The digest as 6 little-endian 64-bit words.
///
/// # Panics
/// * If the message wire count does not equal the message length divided into 64-bit words, rounded
///   up.
pub fn sha3_384(
	builder: &CircuitBuilder,
	message: &[Wire],
	len_bytes: usize,
) -> [Wire; SHA3_384_DIGEST_WORDS] {
	// Delegate to the shared core with this variant's rate and digest length.
	sha3_fixed(builder, message, len_bytes, SHA3_384_RATE_BYTES, SHA3_384_DIGEST_WORDS)
		.try_into()
		.unwrap()
}

/// Computes the SHA3-512 digest of a fixed-length message.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints.
/// * `message` - Input message as packed 64-bit words, 8 bytes per wire, little-endian.
/// * `len_bytes` - The exact length of the message in bytes.
///
/// # Returns
/// The digest as 8 little-endian 64-bit words.
///
/// # Panics
/// * If the message wire count does not equal the message length divided into 64-bit words, rounded
///   up.
pub fn sha3_512(
	builder: &CircuitBuilder,
	message: &[Wire],
	len_bytes: usize,
) -> [Wire; SHA3_512_DIGEST_WORDS] {
	// Delegate to the shared core with this variant's rate and digest length.
	sha3_fixed(builder, message, len_bytes, SHA3_512_RATE_BYTES, SHA3_512_DIGEST_WORDS)
		.try_into()
		.unwrap()
}

#[cfg(test)]
mod tests {
	use binius_frontend::CircuitBuilder;
	use rand::prelude::*;
	use rstest::rstest;
	use sha3::Digest;

	use super::*;

	// Builds a circuit around one hash function, runs it on a random message of the given
	// length, and checks the computed digest against a reference implementation.
	fn test_fixed<const DIGEST_WORDS: usize>(
		message_len_bytes: usize,
		hash_fn: impl FnOnce(&CircuitBuilder, &[Wire], usize) -> [Wire; DIGEST_WORDS],
		reference: impl FnOnce(&[u8]) -> Vec<u8>,
	) {
		// Deterministic random message, so every run of a given case is reproducible.
		let seed = message_len_bytes as u64;
		let mut rng = StdRng::seed_from_u64(seed);
		let mut message = vec![0u8; message_len_bytes];
		rng.fill_bytes(&mut message);

		let expected_digest = reference(&message);
		assert_eq!(expected_digest.len(), DIGEST_WORDS * 8);

		let builder = CircuitBuilder::new();
		let n_words = message_len_bytes.div_ceil(8);
		let message_wires: Vec<_> = (0..n_words).map(|_| builder.add_witness()).collect();
		let expected_digest_wires: [Wire; DIGEST_WORDS] =
			std::array::from_fn(|_| builder.add_witness());

		// Constrain the circuit's digest to equal the expected one, wire by wire.
		let computed_digest = hash_fn(&builder, &message_wires, message_len_bytes);
		for i in 0..DIGEST_WORDS {
			builder.assert_eq(format!("digest[{i}]"), computed_digest[i], expected_digest_wires[i]);
		}

		let circuit = builder.build();
		let cs = circuit.constraint_system();
		let mut witness = circuit.new_witness_filler();

		// Populate the message wires, 8 bytes per wire, little-endian.
		for (i, chunk) in message.chunks(8).enumerate() {
			let mut word_bytes = [0u8; 8];
			word_bytes[..chunk.len()].copy_from_slice(chunk);
			witness[message_wires[i]] = Word(u64::from_le_bytes(word_bytes));
		}
		// Populate the expected digest wires the same way.
		for (i, chunk) in expected_digest.chunks(8).enumerate() {
			witness[expected_digest_wires[i]] = Word(u64::from_le_bytes(chunk.try_into().unwrap()));
		}

		// Evaluating the circuit fills in every internal wire from the message and digest inputs.
		circuit.populate_wire_witness(&mut witness).unwrap();
		// Checking the constraint system confirms the computed digest equals the reference
		// digest, not just that the witness filled in without panicking.
		cs.verify(&witness.into_value_vec())
			.expect("Circuit constraints should be satisfied");
	}

	#[rstest]
	#[case(0)] // Empty message
	#[case(1)] // Single byte
	#[case(8)] // Exactly one word
	#[case(135)] // One byte before the block boundary
	#[case(136)] // Exactly one block
	#[case(137)] // One byte over the block boundary
	#[case(272)] // Exactly two blocks
	#[case(500)] // Arbitrary larger message
	fn test_sha3_256(#[case] message_len_bytes: usize) {
		test_fixed(message_len_bytes, sha3_256, |m| sha3::Sha3_256::digest(m).to_vec());
	}

	#[rstest]
	#[case(0)] // Empty message
	#[case(1)] // Single byte
	#[case(103)] // One byte before the block boundary
	#[case(104)] // Exactly one block
	#[case(105)] // One byte over the block boundary
	#[case(500)] // Arbitrary larger message
	fn test_sha3_384(#[case] message_len_bytes: usize) {
		test_fixed(message_len_bytes, sha3_384, |m| sha3::Sha3_384::digest(m).to_vec());
	}

	#[rstest]
	#[case(0)] // Empty message
	#[case(1)] // Single byte
	#[case(71)] // One byte before the block boundary
	#[case(72)] // Exactly one block
	#[case(73)] // One byte over the block boundary
	#[case(500)] // Arbitrary larger message
	fn test_sha3_512(#[case] message_len_bytes: usize) {
		test_fixed(message_len_bytes, sha3_512, |m| sha3::Sha3_512::digest(m).to_vec());
	}

	#[test]
	#[should_panic(expected = "message.len() (1) must equal len_bytes.div_ceil(8) (2)")]
	fn test_sha3_256_wrong_wire_count() {
		// One wire claims to hold 10 bytes, but 10 bytes needs 2 wires.
		let builder = CircuitBuilder::new();
		let message_wires = vec![builder.add_witness()];
		sha3_256(&builder, &message_wires, 10);
	}
}
