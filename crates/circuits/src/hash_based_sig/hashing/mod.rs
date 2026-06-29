// Copyright 2025 Irreducible Inc.
//! BLAKE3 tweakable-hash circuits for hash-based signatures.
mod chain_blake3;
mod message;
mod public_key;
mod tree;

pub use chain_blake3::{
	chain_tweak_len, circuit_blake3_th, circuit_chain_hash_blake3, circuit_chain_step_2x_blake3,
	hash_chain_blake3, ref_blake3_th, ref_chain_step_blake3,
};
pub use message::{MESSAGE_TWEAK, circuit_message_hash, hash_message};
pub use public_key::{PUBLIC_KEY_TWEAK, circuit_public_key_hash, hash_public_key};
pub use tree::{TREE_TWEAK, circuit_tree_hash, hash_tree_node};
