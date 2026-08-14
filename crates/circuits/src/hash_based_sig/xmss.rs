// Copyright 2026 The Binius Developers
// Copyright (c) 2026 leanEthereum
//! XMSS: a Merkle tree of `2^LOG_LIFETIME` WOTS public-key hashes.
//!
//! A signature is a WOTS signature at one epoch plus the authentication path linking that epoch's
//! leaf to the committed root. Verification recovers the WOTS public key from the chain tips,
//! hashes it into the leaf, and climbs the path.
//!
//! XMSS is stateful: signing twice at one epoch breaks the one-time signature, and so the key.

use std::iter;

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};
use rand::CryptoRng;

use super::{
	DIGEST_LEN, DIGEST_WIRES, Digest, LOG_LIFETIME, MESSAGE_WIRES, Message, PUBLIC_PARAM_LEN,
	PUBLIC_PARAM_WIRES, PublicParam, RANDOMNESS_WIRES, Randomness, V,
	hashing::{TWEAK_TYPE_MERKLE, circuit_tweak_hash, tweak_hash},
	wots::{
		circuit_recover_public_key, circuit_wots_encode, circuit_wots_public_key_hash,
		find_randomness_for_wots_encoding, iterate_hash, recover_public_key, wots_encode,
		wots_public_key_hash,
	},
};

/// An XMSS public key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XmssPublicKey {
	pub merkle_root: Digest,
	pub public_param: PublicParam,
}

/// An XMSS signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmssSignature {
	/// The ground randomness the encoding is drawn from.
	pub randomness: Randomness,
	/// Each chain walked as far as its digit.
	pub chain_tips: [Digest; V],
	/// Sibling nodes from the leaf up to the root.
	pub merkle_path: [Digest; LOG_LIFETIME],
}

/// The wires an [`XmssSignature`] occupies in circuit.
#[derive(Debug, Clone)]
pub struct XmssSignatureWires {
	pub randomness: [Wire; RANDOMNESS_WIRES],
	pub chain_tips: [[Wire; DIGEST_WIRES]; V],
	pub merkle_path: [[Wire; DIGEST_WIRES]; LOG_LIFETIME],
}

impl XmssSignatureWires {
	/// Allocates the signature as private witness wires.
	pub fn new_witness(builder: &CircuitBuilder) -> Self {
		Self {
			randomness: std::array::from_fn(|_| builder.add_witness()),
			chain_tips: std::array::from_fn(|_| std::array::from_fn(|_| builder.add_witness())),
			merkle_path: std::array::from_fn(|_| std::array::from_fn(|_| builder.add_witness())),
		}
	}

	/// Populates the wires from a signature.
	pub fn populate(&self, w: &mut binius_frontend::WitnessFiller, signature: &XmssSignature) {
		w.pack_bytes_le(&self.randomness, &signature.randomness);
		for (wires, tip) in iter::zip(&self.chain_tips, &signature.chain_tips) {
			w.pack_bytes_le(wires, tip);
		}
		for (wires, node) in iter::zip(&self.merkle_path, &signature.merkle_path) {
			w.pack_bytes_le(wires, node);
		}
	}
}

/// Why a signature failed to verify.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum XmssVerifyError {
	/// The randomness does not encode to a valid codeword.
	InvalidWots,
	/// The authentication path does not reach the committed root.
	InvalidMerklePath,
}

/// A Merkle parent, hashed under the tweak that fixes its level and index.
///
/// The level and index place each node in its own hash domain, so a node computed at one position
/// cannot be replayed at another.
pub fn merkle_node(
	public_param: &PublicParam,
	level: usize,
	index: u32,
	left: &Digest,
	right: &Digest,
) -> Digest {
	let mut data = [0u8; 2 * DIGEST_LEN];
	data[..DIGEST_LEN].copy_from_slice(left);
	data[DIGEST_LEN..].copy_from_slice(right);
	tweak_hash(public_param, TWEAK_TYPE_MERKLE, level as u32, index, &data)
}

/// Climbs an authentication path from a leaf to the root it implies.
fn climb(
	public_param: &PublicParam,
	leaf: &Digest,
	epoch: u32,
	merkle_path: &[Digest; LOG_LIFETIME],
) -> Digest {
	merkle_path
		.iter()
		.enumerate()
		.fold(*leaf, |current, (level, sibling)| {
			let is_left = ((epoch >> level) & 1) == 0;
			let (left, right) = if is_left {
				(current, *sibling)
			} else {
				(*sibling, current)
			};
			// The parent sits one level up, at half the index.
			let parent_index = ((epoch as u64) >> (level + 1)) as u32;
			merkle_node(public_param, level + 1, parent_index, &left, &right)
		})
}

/// Verifies an XMSS signature.
pub fn xmss_verify(
	public_key: &XmssPublicKey,
	message: &Message,
	signature: &XmssSignature,
	epoch: u32,
) -> Result<(), XmssVerifyError> {
	let encoding = wots_encode(message, epoch, &public_key.public_param, &signature.randomness)
		.ok_or(XmssVerifyError::InvalidWots)?;
	let chain_ends =
		recover_public_key(&signature.chain_tips, &encoding, epoch, &public_key.public_param);
	let leaf = wots_public_key_hash(&public_key.public_param, epoch, &chain_ends);
	if climb(&public_key.public_param, &leaf, epoch, &signature.merkle_path)
		== public_key.merkle_root
	{
		Ok(())
	} else {
		Err(XmssVerifyError::InvalidMerklePath)
	}
}

/// In-circuit form of [`xmss_verify`].
///
/// Three checks are stacked:
/// 1. the randomness encodes to a valid codeword, and the chains walk from their tips to the
///    Winternitz public key,
/// 2. those chain ends hash into the Merkle leaf,
/// 3. the authentication path links that leaf to the committed root.
///
/// Every digest is derived from the inputs, so this emits constraints and returns nothing.
///
/// # Arguments
///
/// - `builder`: circuit builder.
/// - `public_param`: the signer's public parameter.
/// - `merkle_root`: the signer's committed root.
/// - `message`: the 32-byte message.
/// - `epoch`: the leaf index the signature is at.
/// - `signature`: the signature's witness wires.
pub fn circuit_xmss_verify(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	merkle_root: &[Wire; DIGEST_WIRES],
	message: &[Wire; MESSAGE_WIRES],
	epoch: Wire,
	signature: &XmssSignatureWires,
) {
	// An epoch is a `u32` in the reference, and only its low four bytes reach a tweak. Bounding
	// it here keeps epochs that agree modulo 2^32 from sharing every tweak in the instance.
	builder.assert_zero("xmss_epoch_in_range", builder.shr(epoch, LOG_LIFETIME as u32));

	let digits = circuit_wots_encode(builder, public_param, epoch, message, &signature.randomness);
	let chain_ends =
		circuit_recover_public_key(builder, public_param, epoch, &signature.chain_tips, &digits);
	let leaf = circuit_wots_public_key_hash(builder, public_param, epoch, &chain_ends);

	let root = signature
		.merkle_path
		.iter()
		.enumerate()
		.fold(leaf, |current, (level, sibling)| {
			// `select` reads its condition's most significant bit, so the epoch's level-th bit is
			// shifted up to it: set means this node is the right child.
			let is_right = builder.shl(epoch, (Word::BITS - 1 - level) as u32);
			let left: [Wire; DIGEST_WIRES] =
				std::array::from_fn(|k| builder.select(is_right, sibling[k], current[k]));
			let right: [Wire; DIGEST_WIRES] =
				std::array::from_fn(|k| builder.select(is_right, current[k], sibling[k]));

			let parent_index = builder.shr(epoch, level as u32 + 1);
			let payload = [left[0], left[1], right[0], right[1]];
			circuit_tweak_hash(
				builder,
				public_param,
				TWEAK_TYPE_MERKLE,
				builder.add_constant_64(level as u64 + 1),
				parent_index,
				&payload,
			)
		});

	builder.assert_eq_v("xmss_merkle_root", root, *merkle_root);
}

/// Generates a key and a signature on `message` at `epoch`, for witness generation.
///
/// The tree exists only along the authentication path. Its sibling nodes are drawn at random and
/// the root is whatever climbing the path from the leaf produces, which is what the reference's
/// secret key does for every node outside the range of epochs it can sign — so a `2^32`-leaf tree
/// costs 32 hashes here rather than `2^32`.
///
/// This is enough to exercise a verifier and no more: there is no secret key to sign with again,
/// and no second epoch under the same root.
pub fn generate_signature(
	rng: &mut impl CryptoRng,
	message: &Message,
	epoch: u32,
) -> (XmssPublicKey, XmssSignature) {
	let mut public_param = [0u8; PUBLIC_PARAM_LEN];
	rng.fill_bytes(&mut public_param);

	let (randomness, encoding) =
		find_randomness_for_wots_encoding(message, epoch, &public_param, rng);

	// A chain's secret preimage walked as far as its digit is what the signature reveals.
	let chain_tips: [Digest; V] = std::array::from_fn(|i| {
		let mut pre_image = [0u8; DIGEST_LEN];
		rng.fill_bytes(&mut pre_image);
		iterate_hash(&pre_image, encoding[i] as usize, &public_param, epoch, i, 0)
	});

	let chain_ends = recover_public_key(&chain_tips, &encoding, epoch, &public_param);
	let leaf = wots_public_key_hash(&public_param, epoch, &chain_ends);

	let merkle_path: [Digest; LOG_LIFETIME] = std::array::from_fn(|_| {
		let mut node = [0u8; DIGEST_LEN];
		rng.fill_bytes(&mut node);
		node
	});
	let merkle_root = climb(&public_param, &leaf, epoch, &merkle_path);

	(
		XmssPublicKey {
			merkle_root,
			public_param,
		},
		XmssSignature {
			randomness,
			chain_tips,
			merkle_path,
		},
	)
}

#[cfg(test)]
mod tests {
	use rand::{Rng, SeedableRng, rngs::StdRng};
	use rstest::rstest;

	use super::*;
	use crate::hash_based_sig::MESSAGE_LEN;

	/// Builds the verification circuit, populates it, and returns the result of checking it.
	fn run(
		public_key: &XmssPublicKey,
		message: &Message,
		signature: &XmssSignature,
		epoch: u32,
	) -> Result<(), String> {
		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let root_w: [Wire; DIGEST_WIRES] = std::array::from_fn(|_| b.add_inout());
		let message_w: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| b.add_inout());
		let epoch_w = b.add_inout();
		let sig_w = XmssSignatureWires::new_witness(&b);

		circuit_xmss_verify(&b, &param_w, &root_w, &message_w, epoch_w, &sig_w);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, &public_key.public_param);
		w.pack_bytes_le(&root_w, &public_key.merkle_root);
		w.pack_bytes_le(&message_w, message);
		w[epoch_w] = Word::from_u64(epoch as u64);
		sig_w.populate(&mut w, signature);

		circuit
			.populate_wire_witness(&mut w)
			.map_err(|e| format!("populate: {e:?}"))?;
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.map_err(|e| format!("verify: {e:?}"))
	}

	fn generate(seed: u64, epoch: u32) -> (XmssPublicKey, Message, XmssSignature) {
		let mut rng = StdRng::seed_from_u64(seed);
		let mut message = [0u8; MESSAGE_LEN];
		rng.fill_bytes(&mut message);
		let (public_key, signature) = generate_signature(&mut rng, &message, epoch);
		(public_key, message, signature)
	}

	#[rstest]
	#[case::first_epoch(0)]
	#[case::odd_epoch(1)]
	#[case::interior_epoch(0x1234_5678)]
	#[case::last_epoch(u32::MAX)]
	fn a_generated_signature_verifies(#[case] epoch: u32) {
		let (public_key, message, signature) = generate(1, epoch);
		// The native verifier and the circuit must agree that it is valid.
		xmss_verify(&public_key, &message, &signature, epoch).unwrap();
		run(&public_key, &message, &signature, epoch).unwrap();
	}

	#[test]
	fn a_tampered_path_node_is_rejected() {
		let (public_key, message, mut signature) = generate(2, 77);
		signature.merkle_path[0][0] ^= 0xFF;
		assert_eq!(
			xmss_verify(&public_key, &message, &signature, 77),
			Err(XmssVerifyError::InvalidMerklePath)
		);
		assert!(run(&public_key, &message, &signature, 77).is_err());
	}

	#[test]
	fn a_tampered_root_is_rejected() {
		let (mut public_key, message, signature) = generate(3, 77);
		public_key.merkle_root[0] ^= 0xFF;
		assert!(run(&public_key, &message, &signature, 77).is_err());
	}

	#[test]
	fn another_message_is_rejected() {
		let (public_key, mut message, signature) = generate(4, 77);
		message[0] ^= 0xFF;
		assert!(run(&public_key, &message, &signature, 77).is_err());
	}

	#[test]
	fn another_epoch_is_rejected() {
		// Every tweak on the path carries the level and index, and the leaf and chains carry the
		// epoch, so a signature does not move to a neighbouring leaf.
		let (public_key, message, signature) = generate(5, 77);
		assert!(run(&public_key, &message, &signature, 78).is_err());
	}

	#[test]
	fn an_epoch_past_the_lifetime_is_rejected() {
		// Only the low four bytes of an epoch reach a tweak, so epochs 2^32 apart would otherwise
		// share every hash in the instance.
		let (public_key, message, signature) = generate(6, 5);
		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let root_w: [Wire; DIGEST_WIRES] = std::array::from_fn(|_| b.add_inout());
		let message_w: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| b.add_inout());
		let epoch_w = b.add_inout();
		let sig_w = XmssSignatureWires::new_witness(&b);
		circuit_xmss_verify(&b, &param_w, &root_w, &message_w, epoch_w, &sig_w);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, &public_key.public_param);
		w.pack_bytes_le(&root_w, &public_key.merkle_root);
		w.pack_bytes_le(&message_w, &message);
		w[epoch_w] = Word::from_u64((1u64 << LOG_LIFETIME) + 5);
		sig_w.populate(&mut w, &signature);

		assert!(
			circuit.populate_wire_witness(&mut w).is_err(),
			"an epoch outside the lifetime must not verify"
		);
	}

	#[test]
	fn the_path_climbs_the_side_its_index_says() {
		// Leaf 1 is a right child at level 0 and its subtree a left child at level 1, so the
		// ordering at each level is what the epoch's bits dictate.
		let pp = [1u8; PUBLIC_PARAM_LEN];
		let leaf = [2u8; DIGEST_LEN];
		let sibling0 = [3u8; DIGEST_LEN];
		let sibling1 = [4u8; DIGEST_LEN];
		let parent = merkle_node(&pp, 1, 0, &sibling0, &leaf);
		let grandparent = merkle_node(&pp, 2, 0, &parent, &sibling1);

		let mut path = [[0u8; DIGEST_LEN]; LOG_LIFETIME];
		path[0] = sibling0;
		path[1] = sibling1;
		let mut current = grandparent;
		for (level, sibling) in path.iter().enumerate().skip(2) {
			current = merkle_node(&pp, level + 1, 0, &current, sibling);
		}
		assert_eq!(climb(&pp, &leaf, 1, &path), current);
	}
}
