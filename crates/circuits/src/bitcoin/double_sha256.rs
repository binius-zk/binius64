// Copyright 2025 Irreducible Inc.
//! The Bitcoin double-SHA256 hash function.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use sha2::Digest;

use crate::{bytes::swap_bytes_32, sha256::sha256_fixed};

/// Retained only so callers can obtain the reference digest via [`Self::populate_inner`].
///
/// (Hopefully this struct is not needed in the future, and only the [`Self::construct_circuit`]
/// method is needed without any return value.)
pub struct DoubleSha256;

impl DoubleSha256 {
	/// Constructs a circuit that asserts that `digest = SHA256(SHA256(message))`.
	/// The message length in bytes is fixed at compile time to be `message.len() * 8`.
	///
	/// `message` and `digest` are Bitcoin little-endian packed (8 bytes per wire).
	///
	/// # Preconditions
	///
	/// - `message.len() * 8 == message_len`
	pub fn construct_circuit(
		builder: &CircuitBuilder,
		message: Vec<Wire>,
		digest: [Wire; 4],
	) -> Self {
		let mask32 = builder.add_constant(Word::MASK_32);

		// First SHA-256. `message` is little-endian 8-byte wires; `sha256_fixed` consumes
		// big-endian 32-bit schedule words, so byte-swap within each 32-bit half and split each
		// 64-bit wire into its two schedule words (mirrors `sha256::sha256_varlen`'s input
		// prologue).
		let mut message_be: Vec<Wire> = Vec::with_capacity(message.len() * 2);
		for &w in &message {
			let swapped = swap_bytes_32(builder, w);
			message_be.push(builder.band(swapped, mask32));
			message_be.push(builder.shr(swapped, 32));
		}
		let digest_0_be = sha256_fixed(builder, &message_be, message.len() * 8); // [Wire; 8] BE

		// Second SHA-256 over the 32-byte first digest. Its output words are already the big-endian
		// 32-bit schedule words the second hash expects, so feed them straight in (no swap).
		let digest_1_be = sha256_fixed(builder, &digest_0_be, 32); // [Wire; 8] BE

		// Repack the big-endian 32-bit output words into little-endian 64-bit wires (the old
		// `digest_to_le_wires` contract that `merkle_path`/`header_chain` depend on) and assert.
		let computed: [Wire; 4] = std::array::from_fn(|i| {
			let lo = swap_bytes_32(builder, digest_1_be[2 * i]);
			let hi = swap_bytes_32(builder, digest_1_be[2 * i + 1]);
			builder.bxor(lo, builder.shl(hi, 32))
		});
		builder.assert_eq_v("double sha256 digest", computed, digest);

		Self
	}

	/// Returns `SHA256(SHA256(message))`.
	///
	/// The circuit derives every internal wire from the input `message`/`digest` wires, so unlike
	/// the old struct this populates nothing; it is retained only so callers that chain on the
	/// returned digest bytes continue to compile.
	pub fn populate_inner(&self, _filler: &mut WitnessFiller, message: &[u8]) -> [u8; 32] {
		let digest_0: [u8; 32] = sha2::Sha256::digest(message).into();
		sha2::Sha256::digest(digest_0).into()
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_core::verify::verify_constraints;
	use binius_frontend::util::pack_bytes_into_wires_le;
	use hex_literal::hex;

	use super::*;

	#[test]
	fn test_valid() {
		// construct circuit
		let builder = CircuitBuilder::new();
		let block_header: [Wire; 10] = array::from_fn(|_| builder.add_witness());
		let block_hash: [Wire; 4] = array::from_fn(|_| builder.add_witness());
		let double_sha_256 =
			DoubleSha256::construct_circuit(&builder, block_header.to_vec(), block_hash);
		let circuit = builder.build();

		// populate_witness
		let mut filler = circuit.new_witness_filler();
		let block_header_value = hex!(
			"000000264a14e21adad047d981c06a26446e345eda3d8beb807401000000000000000000fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83642ea468b32c0217d14ba4d1"
		);
		let block_hash_value =
			hex!("228561b085b7524957e515605725901238299ff2793300000000000000000000");
		pack_bytes_into_wires_le(&mut filler, &block_header, &block_header_value);
		pack_bytes_into_wires_le(&mut filler, &block_hash, &block_hash_value);
		double_sha_256.populate_inner(&mut filler, &block_header_value);
		circuit.populate_wire_witness(&mut filler).unwrap();

		// check
		let constraint_system = circuit.constraint_system();
		verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
	}

	#[test]
	fn test_invalid() {
		// construct circuit
		let builder = CircuitBuilder::new();
		let block_header: [Wire; 10] = array::from_fn(|_| builder.add_witness());
		let block_hash: [Wire; 4] = array::from_fn(|_| builder.add_witness());
		let double_sha_256 =
			DoubleSha256::construct_circuit(&builder, block_header.to_vec(), block_hash);
		let circuit = builder.build();

		// populate_witness
		let mut filler = circuit.new_witness_filler();
		let block_header_value = hex!(
			"000000264a14e21adad047d981c06a26446e345eda3d8beb807401000000000000000000fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83642ea468b32c0217d14ba4d1"
		);
		let block_hash_value =
			hex!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
		pack_bytes_into_wires_le(&mut filler, &block_header, &block_header_value);
		pack_bytes_into_wires_le(&mut filler, &block_hash, &block_hash_value);
		double_sha_256.populate_inner(&mut filler, &block_header_value);
		// should fail because the hash is wrong
		circuit.populate_wire_witness(&mut filler).unwrap_err();
	}
}
