// Copyright 2025 Irreducible Inc.
use std::array;

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::hashing::circuit_tree_hash;

/// Reconstructs a Merkle tree root from a leaf and its authentication path.
///
/// The circuit climbs one tree level per authentication-path sibling, ordering each pair by the
/// corresponding bit of `leaf_index`. To verify membership, the caller compares the returned root
/// against the committed one.
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param` - Cryptographic domain parameter (32 bytes as 4x64-bit LE wires)
/// * `domain_param_len` - Actual byte length of the parameter (must be less than or equal to
///   domain_param.len() * 8)
/// * `leaf_hash` - The leaf hash to verify (32 bytes as 4x64-bit LE wires)
/// * `leaf_index` - Index of the leaf in the tree (as a wire)
/// * `auth_path` - Authentication path: sibling hashes from leaf to root
///
/// # Returns
///
/// The reconstructed root hash (32 bytes as 4x64-bit LE wires). The BLAKE3 node digests along the
/// path are derived from the inputs by the evaluator.
pub fn circuit_merkle_path(
	builder: &CircuitBuilder,
	domain_param: &[Wire],
	domain_param_len: usize,
	leaf_hash: &[Wire; 4],
	leaf_index: Wire,
	auth_path: &[[Wire; 4]],
) -> [Wire; 4] {
	assert!(
		domain_param_len <= domain_param.len() * 8,
		"domain_param_len {} exceeds maximum capacity {} of domain_param wires",
		domain_param_len,
		domain_param.len() * 8
	);

	// Climb one tree level per authentication-path sibling.
	auth_path
		.iter()
		.enumerate()
		.fold(*leaf_hash, |current_hash, (level, sibling_hash)| {
			let is_right = builder.shl(leaf_index, (Word::BITS - 1 - level) as u32);

			// Order the pair as (left, right): swap in the sibling on the opposite side.
			let left_hash = array::from_fn::<_, 4, _>(|i| {
				builder.select(is_right, sibling_hash[i], current_hash[i])
			});
			let right_hash = array::from_fn::<_, 4, _>(|i| {
				builder.select(is_right, current_hash[i], sibling_hash[i])
			});

			// The parent index drops the low bit of the current node's index.
			let parent_index = builder.shr(leaf_index, level as u32 + 1);

			// Hash the ordered pair into the parent; the digest is gate-derived.
			let level_wire = builder.add_constant_64(level as u64);
			circuit_tree_hash(
				builder,
				domain_param.to_vec(),
				domain_param_len,
				left_hash,
				right_hash,
				level_wire,
				parent_index,
			)
		})
}

#[cfg(test)]
mod tests {
	use binius_core::Word;

	use super::*;
	use crate::hash_based_sig::hashing::hash_tree_node;

	// Build a 4-leaf tree, run the path circuit for the given leaf, and return the verification
	// result. The internal node digests are derived by the evaluator, so only inputs are populated.
	//
	//          root
	//         /    \
	//        n2     n3
	//       / \    / \
	//      l0 l1  l2 l3
	fn run_path(
		leaf_index: u64,
		leaf: &[u8; 32],
		sibling0: &[u8; 32],
		sibling1: &[u8; 32],
		root: &[u8; 32],
	) -> Result<(), String> {
		let builder = CircuitBuilder::new();
		let param: Vec<Wire> = (0..PARAM.len().div_ceil(8))
			.map(|_| builder.add_inout())
			.collect();
		let leaf_w: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
		let index_w = builder.add_inout();
		let root_w: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
		let path: Vec<[Wire; 4]> = (0..2)
			.map(|_| std::array::from_fn(|_| builder.add_inout()))
			.collect();

		let computed_root =
			circuit_merkle_path(&builder, &param, PARAM.len(), &leaf_w, index_w, &path);
		builder.assert_eq_v("merkle_root_check", computed_root, root_w);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param, PARAM);
		w.pack_bytes_le(&leaf_w, leaf);
		w[index_w] = Word::from_u64(leaf_index);
		w.pack_bytes_le(&path[0], sibling0);
		w.pack_bytes_le(&path[1], sibling1);
		w.pack_bytes_le(&root_w, root);

		circuit
			.populate_wire_witness(&mut w)
			.map_err(|e| format!("populate: {e:?}"))?;
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.map_err(|e| format!("verify: {e:?}"))
	}

	const PARAM: &[u8; 18] = b"merkle_tree_param!";
	const L0: &[u8; 32] = b"leaf_0_hash_value_32_bytes!!!!!!";
	const L1: &[u8; 32] = b"leaf_1_hash_value_32_bytes!!!!!!";
	const L2: &[u8; 32] = b"leaf_2_hash_value_32_bytes!!!!!!";
	const L3: &[u8; 32] = b"leaf_3_hash_value_32_bytes!!!!!!";

	// The three internal nodes of the fixture tree, as the BLAKE3 reference computes them.
	fn fixture_nodes() -> ([u8; 32], [u8; 32], [u8; 32]) {
		let n2 = hash_tree_node(PARAM, L0, L1, 0, 0);
		let n3 = hash_tree_node(PARAM, L2, L3, 0, 1);
		let root = hash_tree_node(PARAM, &n2, &n3, 1, 0);
		(n2, n3, root)
	}

	#[test]
	fn valid_path_accepts() {
		// Leaf 1: siblings are l0 (level 0) and n3 (level 1).
		let (_n2, n3, root) = fixture_nodes();
		run_path(1, L1, L0, &n3, &root).unwrap();
	}

	#[test]
	fn wrong_sibling_rejects() {
		// Leaf 1's valid path is (L0, n3); corrupting the level-1 sibling breaks the root.
		let (_n2, n3, root) = fixture_nodes();
		let mut bad_sibling = n3;
		bad_sibling[0] ^= 0xFF;
		let result = run_path(1, L1, L0, &bad_sibling, &root);
		assert!(result.is_err(), "corrupted sibling must fail to match the root");
	}

	#[test]
	fn wrong_root_rejects() {
		// The correct path against a corrupted root must fail.
		let (_n2, n3, mut root) = fixture_nodes();
		root[0] ^= 0xFF;
		let result = run_path(1, L1, L0, &n3, &root);
		assert!(result.is_err(), "corrupted root must be rejected");
	}
}
