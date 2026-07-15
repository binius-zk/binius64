// Copyright 2025 Irreducible Inc.

pub mod fixed_length;
pub mod permutation;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};
use permutation::Permutation;

use crate::{
	fixed_byte_vec::ByteVec,
	multiplexer::{multi_wire_multiplex, single_wire_multiplex},
};

pub const N_WORDS_PER_DIGEST: usize = 4;
pub const N_WORDS_PER_STATE: usize = 25;
pub const RATE_BYTES: usize = 136;
pub const N_WORDS_PER_BLOCK: usize = RATE_BYTES / 8;

/// Computes the Keccak-256 hash of a variable-length message.
///
/// This gadget consumes a [`ByteVec`] whose actual length is runtime-determined and returns the
/// 256-bit digest as 4 wires (little-endian 64-bit words), matching
/// [`fixed_length::keccak256`]'s output layout.
///
/// Keccak is a sponge: the message is `pad10*1`-padded (the `0x01` domain byte immediately after
/// the message, then zeros, with `0x80` set in the final byte of the last rate block), split into
/// 136-byte (17-word) blocks that are XORed into the 1600-bit state and permuted by
/// [`Permutation::keccak_f1600`]. Each padded word is *computed* as a derived wire, classified by
/// its position relative to the runtime boundary word `w_bd = len_bytes >> 3`:
///
///   1. `word_index <  w_bd` - pure message word,
///   2. `word_index == w_bd` - boundary word (trailing message bytes mixed with the `0x01`
///      delimiter, plus `0x80` when it is also the last word of the final block),
///   3. `word_index >  w_bd` - padding (zero, except the final block's last word, which carries the
///      `0x80` delimiter).
///
/// The digest is the first four state words after the block that contains the padding, selected
/// via a multiplexer indexed by the runtime length. Because the padded words are derived (not
/// witnessed), no padding-correctness constraints are needed.
///
/// Both [`ByteVec`] and Keccak pack bytes little-endian, so — unlike
/// [`crate::sha512::sha512_varlen`] — no byte swap is needed.
///
/// # Arguments
/// * `builder` - Circuit builder
/// * `message` - Input message as a [`ByteVec`]; its `len_bytes` wire holds the actual length.
///
/// # Returns
/// * `[Wire; 4]` - The Keccak-256 digest as 4 little-endian 64-bit words.
pub fn keccak256_varlen(builder: &CircuitBuilder, message: &ByteVec) -> [Wire; N_WORDS_PER_DIGEST] {
	let len_bytes = message.len_bytes;
	let data = &message.data;

	let max_len_bytes = data.len() << 3;
	// A message that exactly fills its blocks still needs one more block for the padding, hence +1.
	let n_blocks = (max_len_bytes + 1).div_ceil(RATE_BYTES);
	let n_words = n_blocks * N_WORDS_PER_BLOCK;

	// Constrain the claimed length to lie within capacity.
	let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
	builder.assert_false("len_check", too_long);

	let zero = builder.add_constant(Word::ZERO);
	let msb_one = builder.add_constant(Word::MSB_ONE);

	// `w_bd` is the word holding the first padding byte (`0x01`); `len_mod_8` is its byte offset
	// within that word.
	let w_bd = builder.shr(len_bytes, 3);
	let len_mod_8 = builder.band(len_bytes, builder.add_constant_64(7));

	// `end_block_index` is the block that contains the padding (the block holding byte
	// `len_bytes`). 136 is not a power of two, so this is found by a linear scan rather than a
	// shift.
	let mut end_block_index = zero;
	for block_no in 0..n_blocks {
		let block_start = builder.add_constant_64((block_no * RATE_BYTES) as u64);
		let block_end = builder.add_constant_64(((block_no + 1) * RATE_BYTES) as u64);
		let gte_start = builder.icmp_ule(block_start, len_bytes);
		let lt_end = builder.icmp_ult(len_bytes, block_end);
		let is_final_block = builder.band(gte_start, lt_end);
		end_block_index = builder.select(
			is_final_block,
			builder.add_constant_64(block_no as u64),
			end_block_index,
		);
	}

	// Boundary word: keep the `len_mod_8` low message bytes and place the `0x01` delimiter at byte
	// `len_mod_8`. When `len_mod_8 == 0` the chosen candidate is `0x01` independent of the
	// (possibly out-of-range) boundary message word.
	let boundary_message_word = single_wire_multiplex(builder, data, w_bd);
	let candidates: Vec<Wire> = (0..8)
		.map(|i| {
			let mask = builder.add_constant_64(0x00FFFFFFFFFFFFFF >> ((7 - i) << 3));
			let delimiter = builder.add_constant_64(1u64 << (i << 3));
			let message_low = builder.band(boundary_message_word, mask);
			builder.bxor(message_low, delimiter)
		})
		.collect();
	let boundary_word = single_wire_multiplex(builder, &candidates, len_mod_8);

	// Compute every padded word as a derived wire, classified by position.
	let padded_message: Vec<Wire> = (0..n_words)
		.map(|word_index| {
			let block_index = word_index / N_WORDS_PER_BLOCK;
			let column_index = word_index % N_WORDS_PER_BLOCK;
			let word_idx_wire = builder.add_constant_64(word_index as u64);

			let is_message_word = builder.icmp_ult(word_idx_wire, w_bd);
			let is_boundary_word = builder.icmp_eq(word_idx_wire, w_bd);
			let is_end_block =
				builder.icmp_eq(builder.add_constant_64(block_index as u64), end_block_index);

			// Message words select the corresponding input word. Only ever chosen when
			// `word_index < w_bd <= max_len_bytes >> 3 == data.len()`, so the index is in range and
			// the zero fallback is never selected.
			let msg_word = if word_index < data.len() {
				data[word_index]
			} else {
				zero
			};

			// The final word of the final block carries the `0x80` delimiter — folded into the
			// boundary word when the boundary is that word, and otherwise standing alone as a
			// padding word.
			let delimiter = if column_index == N_WORDS_PER_BLOCK - 1 {
				builder.select(is_end_block, msb_one, zero)
			} else {
				zero
			};
			let boundary_val = if column_index == N_WORDS_PER_BLOCK - 1 {
				builder.bxor(boundary_word, delimiter)
			} else {
				boundary_word
			};

			let boundary_or_padding = builder.select(is_boundary_word, boundary_val, delimiter);
			builder.select(is_message_word, msg_word, boundary_or_padding)
		})
		.collect();

	// Sponge: XOR each block into the state and permute.
	let mut states: Vec<[Wire; N_WORDS_PER_STATE]> = Vec::with_capacity(n_blocks + 1);
	states.push([zero; N_WORDS_PER_STATE]);
	for block_no in 0..n_blocks {
		let mut state = states[block_no];
		for i in 0..N_WORDS_PER_BLOCK {
			state[i] = builder.bxor(state[i], padded_message[block_no * N_WORDS_PER_BLOCK + i]);
		}
		Permutation::keccak_f1600(builder, &mut state);
		states.push(state);
	}

	// Digest = the first four state words after the block that contains the padding.
	let inputs: Vec<&[Wire]> = states[1..].iter().map(|s| &s[..]).collect();
	let digest_vec = multi_wire_multiplex(builder, &inputs, end_block_index);
	digest_vec[..N_WORDS_PER_DIGEST].try_into().unwrap()
}

#[cfg(test)]
mod tests {
	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::{CircuitBuilder, Wire};
	use rand::prelude::*;
	use rstest::rstest;
	use sha3::Digest;

	use super::*;
	use crate::fixed_byte_vec::ByteVec;

	/// Builds a circuit with the given `max_message_len_bytes` capacity, runs `keccak256_varlen`
	/// on a `ByteVec` populated with `message`, and asserts the computed digest equals the sha3
	/// reference.
	fn test_keccak_varlen_with_input(message: &[u8], max_message_len_bytes: usize) {
		assert!(
			message.len() <= max_message_len_bytes,
			"Message length {} exceeds max capacity {} bytes",
			message.len(),
			max_message_len_bytes
		);

		// Compute expected digest using sha3 crate
		let mut hasher = sha3::Keccak256::new();
		hasher.update(message);
		let expected_digest: [u8; 32] = hasher.finalize().into();

		let b = CircuitBuilder::new();
		let max_len_words = max_message_len_bytes.div_ceil(8);
		let input = ByteVec::new_inout(&b, max_len_words);
		let expected_digest_wires: [Wire; N_WORDS_PER_DIGEST] =
			std::array::from_fn(|_| b.add_witness());

		let computed_digest = keccak256_varlen(&b, &input);
		for i in 0..N_WORDS_PER_DIGEST {
			b.assert_eq(format!("digest[{i}]"), computed_digest[i], expected_digest_wires[i]);
		}

		let circuit = b.build();
		let cs = circuit.constraint_system();
		let mut witness = circuit.new_witness_filler();

		input.populate_data(&mut witness, message);
		input.populate_len_bytes(&mut witness, message.len());
		for (i, bytes) in expected_digest.chunks(8).enumerate() {
			witness[expected_digest_wires[i]] = Word(u64::from_le_bytes(bytes.try_into().unwrap()));
		}

		circuit
			.populate_wire_witness(&mut witness)
			.expect("Circuit should accept valid witness");
		verify_constraints(cs, &witness.into_value_vec())
			.expect("All constraints should be satisfied");
	}

	#[rstest]
	#[case(0, 100)] // Empty message
	#[case(1, 100)] // Single byte - minimal non-empty
	#[case(1, 144)] // Single byte - minimal non-empty
	#[case(135, 136)] // 135 bytes - one byte before block boundary
	#[case(136, 136)] // 136 bytes - exactly one block
	#[case(137, 272)] // 137 bytes - crosses block boundary
	#[case(271, 272)] // 271 bytes - one byte before two blocks
	#[case(272, 272)] // 272 bytes - exactly two blocks
	fn test_keccak_varlen(#[case] message_len_bytes: usize, #[case] max_message_len_bytes: usize) {
		// Create test message with deterministic random bytes seeded by the length inputs
		let seed = ((message_len_bytes as u64) << 32) | (max_message_len_bytes as u64);
		let mut rng = StdRng::seed_from_u64(seed);
		let mut message = vec![0u8; message_len_bytes];
		rng.fill_bytes(&mut message);

		test_keccak_varlen_with_input(&message, max_message_len_bytes);
	}
}
