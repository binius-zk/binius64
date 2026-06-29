// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::chain_blake3::{circuit_blake3_th, ref_blake3_th};
use crate::fixed_byte_vec::ByteVec;

/// Tweak separator for internal Merkle-tree node hashing.
pub const TREE_TWEAK: u8 = 0x01;

/// Computes an internal Merkle-tree node hash, returning its 32-byte digest as four 64-bit wires.
///
/// Evaluates the BLAKE3 tweakable hash with:
/// - domain (chaining value) `param || 0x01 || level || index`,
/// - data (absorbed block) `left || right`.
///
/// The level and index place each node in its own hash domain, so a node at one position can never
/// be reused at another.
///
/// # Arguments
///
/// * `builder` - Circuit builder.
/// * `domain_param_wires` - Per-signer parameter, eight bytes per wire.
/// * `domain_param_len` - Parameter length in bytes; at most 23 so the domain fits the 32-byte cv.
/// * `left` - Left child hash, 32 bytes as four 64-bit little-endian wires.
/// * `right` - Right child hash, 32 bytes as four 64-bit little-endian wires.
/// * `level` - Tree level, low four bytes used.
/// * `index` - Node index at this level, low four bytes used.
///
/// # Returns
///
/// The 32-byte parent digest as four 64-bit little-endian wires.
#[allow(clippy::too_many_arguments)]
pub fn circuit_tree_hash(
	builder: &CircuitBuilder,
	domain_param_wires: Vec<Wire>,
	domain_param_len: usize,
	left: [Wire; 4],
	right: [Wire; 4],
	level: Wire,
	index: Wire,
) -> [Wire; 4] {
	let domain = vec![
		ByteVec::new_const_len(builder, domain_param_wires, domain_param_len),
		ByteVec::new_const_len(builder, vec![builder.add_constant_64(TREE_TWEAK as u64)], 1),
		ByteVec::new_const_len(builder, vec![level], 4),
		ByteVec::new_const_len(builder, vec![index], 4),
	];
	let data = vec![
		ByteVec::new_const_len(builder, left.to_vec(), 32),
		ByteVec::new_const_len(builder, right.to_vec(), 32),
	];
	circuit_blake3_th(builder, &domain, &data)
}

/// Reference (out-of-circuit) internal tree-node hash, matching `circuit_tree_hash` exactly.
///
/// # Arguments
///
/// * `param` - Per-signer parameter bytes.
/// * `left` - Left child hash.
/// * `right` - Right child hash.
/// * `level` - Tree level, encoded as four little-endian bytes.
/// * `index` - Node index at this level, encoded as four little-endian bytes.
pub fn hash_tree_node(
	param: &[u8],
	left: &[u8; 32],
	right: &[u8; 32],
	level: u32,
	index: u32,
) -> [u8; 32] {
	let mut domain = Vec::with_capacity(param.len() + 1 + 4 + 4);
	domain.extend_from_slice(param);
	domain.push(TREE_TWEAK);
	domain.extend_from_slice(&level.to_le_bytes());
	domain.extend_from_slice(&index.to_le_bytes());

	let mut data = Vec::with_capacity(64);
	data.extend_from_slice(left);
	data.extend_from_slice(right);

	ref_blake3_th(&domain, &data)
}

#[cfg(test)]
mod tests {
	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::util::pack_bytes_into_wires_le;
	use proptest::prelude::*;

	use super::*;

	// Build the gadget, populate the inputs, and return the evaluated 32-byte parent digest.
	fn run_circuit(
		param_bytes: &[u8],
		left: &[u8; 32],
		right: &[u8; 32],
		level: u32,
		index: u32,
	) -> [u8; 32] {
		let b = CircuitBuilder::new();
		let param: Vec<Wire> = (0..param_bytes.len().div_ceil(8))
			.map(|_| b.add_inout())
			.collect();
		let left_w: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		let right_w: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		let level_w = b.add_inout();
		let index_w = b.add_inout();

		let digest = circuit_tree_hash(
			&b,
			param.clone(),
			param_bytes.len(),
			left_w,
			right_w,
			level_w,
			index_w,
		);
		let out: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		for k in 0..4 {
			b.assert_eq("tree_digest", digest[k], out[k]);
		}

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		pack_bytes_into_wires_le(&mut w, &param, param_bytes);
		pack_bytes_into_wires_le(&mut w, &left_w, left);
		pack_bytes_into_wires_le(&mut w, &right_w, right);
		w[level_w] = Word::from_u64(level as u64);
		w[index_w] = Word::from_u64(index as u64);
		let reference = hash_tree_node(param_bytes, left, right, level, index);
		pack_bytes_into_wires_le(&mut w, &out, &reference);

		circuit.populate_wire_witness(&mut w).unwrap();
		verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
		reference
	}

	#[test]
	fn matches_reference_18_byte_param() {
		// Real-parameter shape: 18-byte parameter, two 32-byte children.
		let digest = run_circuit(b"test_param_18bytes", &[1u8; 32], &[2u8; 32], 5, 123);
		assert_ne!(digest, [0u8; 32]);
	}

	proptest! {
		#[test]
		fn matches_reference_property_based(
			param_len in 1usize..=23,
			level in 0u32..=32,
			index in 0u32..=u32::MAX,
			left in prop::array::uniform32(any::<u8>()),
			right in prop::array::uniform32(any::<u8>()),
			seed in any::<u64>(),
		) {
			use rand::{Rng, SeedableRng, rngs::StdRng};

			// Random parameter of the sampled length.
			let mut rng = StdRng::seed_from_u64(seed);
			let mut param = vec![0u8; param_len];
			rng.fill_bytes(&mut param);

			run_circuit(&param, &left, &right, level, index);
		}
	}
}
