// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! The Bitcoin double-SHA256 hash function.

use binius_frontend::{CircuitBuilder, Wire};

use crate::{bytes::swap_bytes_32, sha256::sha256_fixed, util::clear_high_bits};

/// Returns `SHA256(SHA256(message))`.
///
/// The message length in bytes is fixed at compile time to be `message.len() * 8`.
///
/// `message` and the returned digest are Bitcoin little-endian packed (8 bytes per wire).
pub fn double_sha256(builder: &CircuitBuilder, message: &[Wire]) -> [Wire; 4] {
	// First SHA-256. `message` is little-endian 8-byte wires; `sha256_fixed` consumes
	// big-endian 32-bit schedule words, so byte-swap within each 32-bit half and split each
	// 64-bit wire into its two schedule words (mirrors `sha256::sha256_varlen`'s input
	// prologue).
	let mut message_be: Vec<Wire> = Vec::with_capacity(message.len() * 2);
	for &w in message {
		let swapped = swap_bytes_32(builder, w);
		message_be.push(clear_high_bits(builder, swapped, 32));
		message_be.push(builder.shr(swapped, 32));
	}
	let digest_0_be = sha256_fixed(builder, &message_be, message.len() * 8); // [Wire; 8] BE

	// Second SHA-256 over the 32-byte first digest. Its output words are already the big-endian
	// 32-bit schedule words the second hash expects, so feed them straight in (no swap).
	let digest_1_be = sha256_fixed(builder, &digest_0_be, 32); // [Wire; 8] BE

	// Repack the big-endian 32-bit output words into little-endian 64-bit wires, the form
	// `merkle_path`/`header_chain` chain on.
	std::array::from_fn(|i| {
		let lo = swap_bytes_32(builder, digest_1_be[2 * i]);
		let hi = swap_bytes_32(builder, digest_1_be[2 * i + 1]);
		builder.bxor(lo, builder.shl(hi, 32))
	})
}

#[cfg(test)]
mod tests {
	use std::array;

	use hex_literal::hex;

	use super::*;

	/// Builds a circuit asserting `double_sha256(header) == hash`, then runs it on the given
	/// values.
	fn check_double_sha256(header_value: &[u8], hash_value: &[u8]) -> anyhow::Result<()> {
		let builder = CircuitBuilder::new();
		let block_header: [Wire; 10] = array::from_fn(|_| builder.add_witness());
		let block_hash: [Wire; 4] = array::from_fn(|_| builder.add_witness());
		builder.assert_eq_v("block hash", double_sha256(&builder, &block_header), block_hash);
		let circuit = builder.build();

		let mut filler = circuit.new_witness_filler();
		filler.pack_bytes_le(&block_header, header_value);
		filler.pack_bytes_le(&block_hash, hash_value);
		circuit.populate_wire_witness(&mut filler)?;

		let constraint_system = circuit.constraint_system();
		constraint_system.verify(&filler.into_value_vec())?;
		Ok(())
	}

	const BLOCK_HEADER: [u8; 80] = hex!(
		"000000264a14e21adad047d981c06a26446e345eda3d8beb807401000000000000000000fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83642ea468b32c0217d14ba4d1"
	);

	#[test]
	fn test_valid() {
		let block_hash = hex!("228561b085b7524957e515605725901238299ff2793300000000000000000000");
		check_double_sha256(&BLOCK_HEADER, &block_hash).unwrap();
	}

	#[test]
	fn test_invalid() {
		let block_hash = hex!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
		check_double_sha256(&BLOCK_HEADER, &block_hash).unwrap_err();
	}
}
