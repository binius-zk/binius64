// Copyright 2025 Irreducible Inc.
//! Witness population utilities for hash-based signature verification.
//!
//! This module provides helper functions for populating witness data
//! in hash-based signature circuits, including XMSS and Winternitz OTS.

use rand::prelude::*;

use super::{
	hashing::{hash_chain_blake3, hash_public_key, hash_tree_node},
	winternitz_ots::{NONCE_LENGTH_BYTES, WinternitzSpec, grind_nonce},
};

/// Builds a complete Merkle tree from leaf nodes.
///
/// This function assumes the number of leaves is a power of 2.
///
/// # Returns
/// A tuple containing:
/// - Vector of tree levels (index 0 = leaves, last index = root)
/// - The root hash
///
/// # Panics
/// Panics if leaves.len() is not a power of 2
pub fn build_merkle_tree(param: &[u8], leaves: &[[u8; 32]]) -> (Vec<Vec<[u8; 32]>>, [u8; 32]) {
	assert!(leaves.len().is_power_of_two(), "Number of leaves must be a power of 2");

	let tree_depth = leaves.len().trailing_zeros() as usize;
	let mut tree_levels = vec![leaves.to_vec()];

	for level in 0..tree_depth {
		let current_level = &tree_levels[level];
		let mut next_level = Vec::new();

		for i in (0..current_level.len()).step_by(2) {
			let parent = hash_tree_node(
				param,
				&current_level[i],
				&current_level[i + 1],
				level as u32,
				(i / 2) as u32,
			);
			next_level.push(parent);
		}

		tree_levels.push(next_level);
	}

	let root = tree_levels[tree_depth][0];
	(tree_levels, root)
}

/// Extracts the authentication path for a given leaf index in a Merkle tree.
///
/// This function assumes the tree has power-of-2 leaves.
///
/// # Arguments
/// * `tree_levels` - All levels of the tree (from build_merkle_tree)
/// * `leaf_index` - Index of the leaf to build path for
///
/// # Returns
/// Vector of sibling hashes from leaf to root
pub fn extract_auth_path(tree_levels: &[Vec<[u8; 32]>], leaf_index: usize) -> Vec<[u8; 32]> {
	let mut auth_path = Vec::new();
	let mut idx = leaf_index;
	let tree_height = tree_levels.len() - 1;

	for level in 0..tree_height {
		let sibling_idx = idx ^ 1;
		auth_path.push(tree_levels[level][sibling_idx]);
		idx /= 2;
	}

	auth_path
}

/// Helper structure containing signature data for a validator.
///
/// This is useful for generating test data or populating witness values
/// in multi-signature scenarios.
pub struct ValidatorSignatureData {
	/// Root hash of the validator's Merkle tree
	pub root: [u8; 32],
	/// Nonce feeding the message hash.
	pub nonce: [u8; NONCE_LENGTH_BYTES],
	/// Signature hashes for each Winternitz chain
	pub signature_hashes: Vec<[u8; 32]>,
	/// Public key hashes for each Winternitz chain
	pub public_key_hashes: Vec<[u8; 32]>,
	/// Authentication path in the Merkle tree
	pub auth_path: Vec<[u8; 32]>,
	/// Codeword coordinates
	pub coords: Vec<u8>,
}

impl ValidatorSignatureData {
	/// Generate a valid signature for a validator at a given epoch.
	///
	/// This function generates all the cryptographic data needed for a validator's
	/// signature including the Winternitz OTS signature, public key, and Merkle tree
	/// authentication path.
	///
	/// # Panics
	/// Panics if:
	/// - The epoch is greater than the number of leaves in the tree.
	/// - A `grind_nonce` fails to find a valid nonce
	/// - A coordinate returned by `grind_nonce` is invalid.
	pub fn generate(
		rng: &mut StdRng,
		param_bytes: &[u8],
		message_bytes: &[u8; 32],
		epoch: u32,
		spec: &WinternitzSpec,
		tree_height: usize,
	) -> Self {
		assert!(
			tree_height <= 31,
			"Tree height {} exceeds maximum supported height of 31",
			tree_height,
		);

		// Validate epoch is within valid range for the tree
		let num_leaves = 1usize << tree_height;
		assert!(
			(epoch as usize) < num_leaves,
			"Epoch {} exceeds maximum leaf index {} for tree height {}",
			epoch,
			num_leaves - 1,
			tree_height
		);

		let grind_result = grind_nonce(spec, rng, param_bytes, epoch as u64, message_bytes)
			.expect("Failed to find valid nonce");

		let mut nonce = [0u8; NONCE_LENGTH_BYTES];
		nonce.copy_from_slice(&grind_result.nonce);
		let coords = grind_result.coords;

		// Generate Winternitz signature and public key
		let mut signature_hashes = Vec::new();
		let mut public_key_hashes = Vec::new();

		for (chain_idx, &coord) in coords.iter().enumerate() {
			assert!(
				(coord as usize) < spec.chain_len(),
				"Coordinate {} exceeds chain length {}",
				coord,
				spec.chain_len()
			);

			let mut sig_hash = [0u8; 32];
			rng.fill_bytes(&mut sig_hash);
			signature_hashes.push(sig_hash);

			let pk_hash = hash_chain_blake3(
				param_bytes,
				epoch,
				chain_idx as u8,
				&sig_hash,
				coord as usize,
				spec.chain_len() - 1 - coord as usize,
			);
			public_key_hashes.push(pk_hash);
		}

		// Build a Merkle tree with 2^tree_height leaves
		let mut leaves = vec![[0u8; 32]; num_leaves];
		leaves[epoch as usize] = hash_public_key(param_bytes, epoch as u64, &public_key_hashes);
		for (i, leaf) in leaves.iter_mut().enumerate() {
			if i != epoch as usize {
				rng.fill_bytes(leaf);
			}
		}

		let (tree_levels, root) = build_merkle_tree(param_bytes, &leaves);
		let auth_path = extract_auth_path(&tree_levels, epoch as usize);

		ValidatorSignatureData {
			root,
			nonce,
			signature_hashes,
			public_key_hashes,
			auth_path,
			coords,
		}
	}
}
