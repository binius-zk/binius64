// Copyright 2026 The Binius Developers
//! The truncated-BIP32 circuit statement and its reference derivation oracle.
//!
//! The statement proves knowledge of a private witness whose derived compressed secp256k1 public
//! key hashes (SHA-256) to a public digest. The witness derives that key by either:
//!
//! * starting from the seed master key (`use_seed`), or
//! * taking a single hardened child step from a supplied parent extended private key,
//!
//! and then following a non-hardened-only path. Seed, parent key, hardened index, path, and depth
//! are all private; only the SHA-256 digest is public (`inout`).

use std::{array, iter};

use anyhow::{Result, bail};
use binius_circuits::{
	bignum::select as select_biguint,
	bitcoin::p2pkh_signature::compress_pubkey,
	ecdsa::scalar_mul::scalar_mul,
	multiplexer::multi_wire_multiplex,
	secp256k1::{Secp256k1, Secp256k1Affine},
	sha256::sha256_fixed,
};
use binius_core::word::Word;
use binius_examples::circuits::bip32::{
	extended_key_parts, hardened_child, master_from_seed, non_hardened_child,
};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use bitcoin::{
	NetworkKind,
	bip32::{ChainCode, ChildNumber, Fingerprint, Xpriv},
	secp256k1::{PublicKey, Secp256k1 as BtcSecp256k1, SecretKey},
};
use sha2::{Digest, Sha256};

/// Bit 31 of a derivation index marks a hardened child; valid indices are below it.
const HARDENED_BIT: u32 = 0x8000_0000;

/// Bit 63 of the packed index word carries the `use_seed` flag.
const USE_SEED_BIT: u64 = 1 << 63;

/// Number of 4-byte big-endian words in a SHA-256 digest.
const N_HASH_WORDS: usize = 8;

/// Circuit-native inputs for one truncated-BIP32 derivation.
pub struct DerivationInputs {
	/// 64-byte BIP32 seed (used only when `use_seed`).
	pub seed: [u8; 64],
	/// 64-byte extended private key `private_key || chain_code` (used when `!use_seed`).
	pub parent_xprivkey: [u8; 64],
	/// Hardened child index (31-bit) applied to `parent_xprivkey` (used when `!use_seed`).
	pub hardened_index: u32,
	/// Start from the seed master key instead of a hardened step from `parent_xprivkey`.
	pub use_seed: bool,
	/// Non-hardened derivation path; each index must be below `2^31`.
	pub path: Vec<u32>,
}

/// The truncated-BIP32 circuit: asserts that the SHA-256 of the derived compressed public key
/// equals the public digest.
pub struct TruncatedBip32 {
	seed: [Wire; 8],
	parent_xprivkey: [Wire; 8],
	hardened_index: Wire,
	path: Vec<Wire>,
	path_length: Wire,
	expected_hash: [Wire; N_HASH_WORDS],
	max_depth: usize,
}

impl TruncatedBip32 {
	/// Build the circuit for a non-hardened chain of up to `max_depth` steps.
	pub fn build(max_depth: usize, builder: &mut CircuitBuilder) -> Self {
		let seed: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let parent_xprivkey: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let hardened_index = builder.add_witness();
		let path: Vec<Wire> = (0..max_depth).map(|_| builder.add_witness()).collect();
		let path_length = builder.add_witness();

		let pubkey = truncated_derive_compressed(
			builder,
			&seed,
			&parent_xprivkey,
			hardened_index,
			&path,
			path_length,
		);

		// SHA-256 of the 33-byte compressed public key is the only public input.
		let digest = sha256_fixed(builder, &pubkey, 33);
		let expected_hash: [Wire; N_HASH_WORDS] = array::from_fn(|_| builder.add_inout());
		for (idx, (&computed, &expected)) in digest.iter().zip(&expected_hash).enumerate() {
			builder.assert_eq(format!("pubkey_hash[{idx}]"), computed, expected);
		}

		Self {
			seed,
			parent_xprivkey,
			hardened_index,
			path,
			path_length,
			expected_hash,
			max_depth,
		}
	}

	/// Populate the witness for `inputs`, computing the expected digest with a `bitcoin`-crate
	/// oracle.
	pub fn populate_witness(&self, inputs: &DerivationInputs, w: &mut WitnessFiller) -> Result<()> {
		if inputs.path.len() > self.max_depth {
			bail!("path depth {} exceeds max_depth {}", inputs.path.len(), self.max_depth);
		}
		if inputs.hardened_index >= HARDENED_BIT {
			bail!("hardened index {} out of range (must be < 2^31)", inputs.hardened_index);
		}
		if let Some(&idx) = inputs.path.iter().find(|&&i| i >= HARDENED_BIT) {
			bail!("path index {idx} must be non-hardened (< 2^31)");
		}

		pack_be_words(&inputs.seed, &self.seed, w);
		pack_be_words(&inputs.parent_xprivkey, &self.parent_xprivkey, w);

		let mut index_word = inputs.hardened_index as u64;
		if inputs.use_seed {
			index_word |= USE_SEED_BIT;
		}
		w[self.hardened_index] = Word::from_u64(index_word);

		// Pad unused tail levels with index 0 (a normal child, never selected).
		for i in 0..self.max_depth {
			let idx = inputs.path.get(i).copied().unwrap_or(0);
			w[self.path[i]] = Word::from_u64(idx as u64);
		}
		w[self.path_length] = Word::from_u64(inputs.path.len() as u64);

		let pubkey = oracle_compressed_pubkey(inputs)?;
		let hash: [u8; 32] = Sha256::digest(pubkey).into();
		for (&wire, chunk) in iter::zip(&self.expected_hash, hash.chunks_exact(4)) {
			let word = u32::from_be_bytes(chunk.try_into().expect("4-byte chunk"));
			w[wire] = Word::from_u64(word as u64);
		}

		tracing::info!(
			"truncated BIP32 compressed pubkey {} -> SHA-256 {} (use_seed {}, depth {})",
			hex::encode(pubkey),
			hex::encode(hash),
			inputs.use_seed,
			inputs.path.len()
		);
		Ok(())
	}
}

/// Derive the compressed public key for the truncated tree: start from the seed master key or one
/// hardened step from `parent_xprivkey` (selected on bit 63 of `hardened_index`), then run a
/// non-hardened-only chain over `path`, selecting the level at `path_length` with a multiplexer.
///
/// Composed from the per-step gadgets in [`binius_examples::circuits::bip32`].
fn truncated_derive_compressed(
	b: &mut CircuitBuilder,
	seed: &[Wire; 8],
	parent_xprivkey: &[Wire; 8],
	hardened_index: Wire,
	path: &[Wire],
	path_length: Wire,
) -> Vec<Wire> {
	let curve = Secp256k1::new(b);

	// Starting extended private key: the seed master key, or one hardened step from the parent.
	let (k_seed, c_seed) = master_from_seed(b, seed);
	let (k_par, c_par) = extended_key_parts(parent_xprivkey);
	let (k_hard, c_hard) = hardened_child(b, &curve, &k_par, &c_par, hardened_index);

	// Bit 63 of `hardened_index` is the (MSB) `use_seed` flag selecting the master branch.
	let mut k = select_biguint(b, hardened_index, &k_seed, &k_hard);
	let mut c: [Wire; 4] = array::from_fn(|i| b.select(hardened_index, c_seed[i], c_hard[i]));

	// Non-hardened chain. Run the full max_depth and select the public key at `path_length`. The
	// compressed pubkey at each level is the parent for the next CKD and a multiplexer candidate.
	let mut serp_levels: Vec<Vec<Wire>> = Vec::with_capacity(path.len() + 1);
	for &index in path {
		let point = scalar_mul(b, &curve, &k, Secp256k1Affine::generator(b));
		serp_levels.push(compress_pubkey(b, &point.x, &point.y));
		(k, c) = non_hardened_child(b, &curve, &k, &c, &point, index);
	}
	let point = scalar_mul(b, &curve, &k, Secp256k1Affine::generator(b));
	serp_levels.push(compress_pubkey(b, &point.x, &point.y));

	let refs: Vec<&[Wire]> = serp_levels.iter().map(Vec::as_slice).collect();
	multi_wire_multiplex(b, &refs, path_length)
}

/// Pack 64 big-endian bytes into eight 64-bit words.
fn pack_be_words(bytes: &[u8; 64], wires: &[Wire; 8], w: &mut WitnessFiller) {
	for (&wire, chunk) in iter::zip(wires, bytes.chunks_exact(8)) {
		w[wire] = Word::from_u64(u64::from_be_bytes(chunk.try_into().expect("8-byte chunk")));
	}
}

/// Reference oracle mirroring the circuit statement: returns the 33-byte compressed public key.
fn oracle_compressed_pubkey(inputs: &DerivationInputs) -> Result<[u8; 33]> {
	let secp = BtcSecp256k1::new();

	// Starting extended private key: seed master, or one hardened step from the parent.
	let start = if inputs.use_seed {
		Xpriv::new_master(NetworkKind::Main, &inputs.seed)
			.map_err(|e| anyhow::anyhow!("invalid master key: {e}"))?
	} else {
		let parent = xpriv_from_raw(&inputs.parent_xprivkey)?;
		let child = ChildNumber::from_hardened_idx(inputs.hardened_index)
			.map_err(|e| anyhow::anyhow!("invalid hardened index: {e}"))?;
		parent
			.derive_priv(&secp, &[child])
			.map_err(|e| anyhow::anyhow!("hardened derivation failed: {e}"))?
	};

	let children: Vec<ChildNumber> = inputs
		.path
		.iter()
		.map(|&idx| {
			ChildNumber::from_normal_idx(idx)
				.map_err(|e| anyhow::anyhow!("invalid path index {idx}: {e}"))
		})
		.collect::<Result<_>>()?;
	let derived = start
		.derive_priv(&secp, &children)
		.map_err(|e| anyhow::anyhow!("path derivation failed: {e}"))?;
	let pubkey = PublicKey::from_secret_key(&secp, &derived.private_key);
	Ok(pubkey.serialize())
}

/// Reconstruct an [`Xpriv`] from a raw 64-byte `private_key || chain_code` extended private key.
/// Only the private key and chain code affect derivation, so the metadata fields are filled with
/// the master-key defaults.
fn xpriv_from_raw(raw: &[u8; 64]) -> Result<Xpriv> {
	let private_key = SecretKey::from_slice(&raw[..32])
		.map_err(|e| anyhow::anyhow!("invalid parent private key: {e}"))?;
	let chain_code = ChainCode::from(<[u8; 32]>::try_from(&raw[32..]).expect("32-byte tail"));
	Ok(Xpriv {
		network: NetworkKind::Main,
		depth: 0,
		parent_fingerprint: Fingerprint::from([0u8; 4]),
		child_number: ChildNumber::from_normal_idx(0).expect("0 is a valid normal index"),
		private_key,
		chain_code,
	})
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;
	use binius_frontend::CircuitBuilder;

	use super::*;

	/// Build the circuit, populate it for `inputs`, and verify all constraints hold (the witness
	/// population already checks the derived key against the `bitcoin`-crate oracle).
	fn check(inputs: &DerivationInputs, max_depth: usize) {
		let mut builder = CircuitBuilder::new();
		let circuit_def = TruncatedBip32::build(max_depth, &mut builder);
		let circuit = builder.build();

		let mut w = circuit.new_witness_filler();
		circuit_def
			.populate_witness(inputs, &mut w)
			.expect("populate witness");
		circuit
			.populate_wire_witness(&mut w)
			.expect("witness population");
		verify_constraints(circuit.constraint_system(), &w.into_value_vec())
			.expect("constraints satisfied");
	}

	fn test_seed() -> [u8; 64] {
		array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(1))
	}

	/// A deterministic 64-byte parent extended private key (a valid scalar followed by a chain
	/// code).
	fn test_parent_xprivkey() -> [u8; 64] {
		array::from_fn(|i| (i as u8).wrapping_mul(5).wrapping_add(3))
	}

	#[test]
	fn use_seed_with_non_hardened_path() {
		check(
			&DerivationInputs {
				seed: test_seed(),
				parent_xprivkey: test_parent_xprivkey(),
				hardened_index: 0,
				use_seed: true,
				path: vec![1, 2],
			},
			2,
		);
	}

	#[test]
	fn hardened_parent_with_non_hardened_path() {
		check(
			&DerivationInputs {
				seed: test_seed(),
				parent_xprivkey: test_parent_xprivkey(),
				hardened_index: 7,
				use_seed: false,
				path: vec![5, 1_000_000_000],
			},
			2,
		);
	}

	#[test]
	fn empty_path_outputs_starting_key() {
		// path_length 0 must output the starting xprivkey's public key, for both branches.
		check(
			&DerivationInputs {
				seed: test_seed(),
				parent_xprivkey: test_parent_xprivkey(),
				hardened_index: 0,
				use_seed: true,
				path: vec![],
			},
			2,
		);
		check(
			&DerivationInputs {
				seed: test_seed(),
				parent_xprivkey: test_parent_xprivkey(),
				hardened_index: 3,
				use_seed: false,
				path: vec![],
			},
			2,
		);
	}

	#[test]
	fn short_path_with_padding() {
		// depth 1 within a max-depth-3 circuit exercises the multiplexer and the ignored tail.
		check(
			&DerivationInputs {
				seed: test_seed(),
				parent_xprivkey: test_parent_xprivkey(),
				hardened_index: 0,
				use_seed: false,
				path: vec![9],
			},
			3,
		);
	}

	#[test]
	fn full_depth_hardened() {
		check(
			&DerivationInputs {
				seed: test_seed(),
				parent_xprivkey: test_parent_xprivkey(),
				hardened_index: HARDENED_BIT - 1,
				use_seed: false,
				path: vec![0, 1, 2],
			},
			3,
		);
	}

	/// The truncated derivation of a full BIP44 path (`split_derivation` → one in-circuit hardened
	/// step → non-hardened suffix) must reproduce the same compressed public key as deriving the
	/// whole path directly. This is what binds a generated proof to the wallet's real address.
	#[test]
	fn matches_full_bip44_derivation() {
		use crate::address::{
			AddressType, bip44_path, derive_compressed_pubkey, pubkey_sha256, split_derivation,
		};

		let seed = test_seed();
		let path = bip44_path(AddressType::P2pkh, 3, 1, 7);
		let inputs = split_derivation(&seed, &path, 2).expect("split BIP44 path");

		// `check` confirms the circuit agrees with the oracle; the oracle here is shown to
		// reproduce the full-path derivation. Together: the circuit reproduces the wallet's real
		// public key.
		check(&inputs, 2);
		let truncated: [u8; 32] = Sha256::digest(oracle_compressed_pubkey(&inputs).unwrap()).into();
		let full = derive_compressed_pubkey(&seed, &path).expect("full derivation");
		assert_eq!(truncated, pubkey_sha256(&full));
	}

	/// `split_derivation` must handle hardened levels *following* non-hardened ones: it splits at
	/// the last hardened level, folding the (mixed) prefix into the offline parent.
	#[test]
	fn matches_mixed_hardened_path() {
		use crate::address::{derive_compressed_pubkey, pubkey_sha256, split_derivation};

		let seed = test_seed();
		let path = vec![0, 1 | HARDENED_BIT, 2]; // m/0/1'/2 — a hardened level after a non-hardened one
		let inputs = split_derivation(&seed, &path, 2).expect("split mixed path");
		assert!(!inputs.use_seed);

		check(&inputs, 2);
		let truncated: [u8; 32] = Sha256::digest(oracle_compressed_pubkey(&inputs).unwrap()).into();
		let full = derive_compressed_pubkey(&seed, &path).expect("full derivation");
		assert_eq!(truncated, pubkey_sha256(&full));
	}

	#[test]
	fn rejects_hardened_path_index() {
		let mut builder = CircuitBuilder::new();
		let circuit_def = TruncatedBip32::build(2, &mut builder);
		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		let err = circuit_def
			.populate_witness(
				&DerivationInputs {
					seed: test_seed(),
					parent_xprivkey: test_parent_xprivkey(),
					hardened_index: 0,
					use_seed: true,
					path: vec![HARDENED_BIT],
				},
				&mut w,
			)
			.unwrap_err();
		assert!(err.to_string().contains("non-hardened"));
	}
}
