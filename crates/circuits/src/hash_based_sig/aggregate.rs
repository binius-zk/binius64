// Copyright 2026 The Binius Developers
//! Aggregation of XMSS signatures on a common message.
//!
//! Each signer has an independent tree, so a public key is a `(root, public parameter)` pair and
//! both are per-signer. All signers sign the same message at the same epoch, and one proof stands
//! for every signature.

use std::iter;

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use super::{
	DIGEST_WIRES, MESSAGE_WIRES, Message, PUBLIC_PARAM_WIRES,
	xmss::{XmssPublicKey, XmssSignature, XmssSignatureWires, circuit_xmss_verify},
};

/// One signer's public key and signature wires.
#[derive(Debug, Clone)]
pub struct SignerWires {
	pub public_param: [Wire; PUBLIC_PARAM_WIRES],
	pub merkle_root: [Wire; DIGEST_WIRES],
	pub signature: XmssSignatureWires,
}

/// The wires an aggregate verification occupies.
///
/// The message, the epoch and every signer's public key are public inputs; the signatures are
/// private witnesses.
#[derive(Debug, Clone)]
pub struct MultiSigWires {
	pub message: [Wire; MESSAGE_WIRES],
	pub epoch: Wire,
	pub signers: Vec<SignerWires>,
}

impl MultiSigWires {
	/// Allocates the wires for `num_signers` signers.
	pub fn new(builder: &CircuitBuilder, num_signers: usize) -> Self {
		Self {
			message: std::array::from_fn(|_| builder.add_inout()),
			epoch: builder.add_inout(),
			signers: (0..num_signers)
				.map(|_| SignerWires {
					public_param: std::array::from_fn(|_| builder.add_inout()),
					merkle_root: std::array::from_fn(|_| builder.add_inout()),
					signature: XmssSignatureWires::new_witness(builder),
				})
				.collect(),
		}
	}

	/// Populates every wire from a message, an epoch and one key-and-signature pair per signer.
	///
	/// # Panics
	///
	/// If the number of pairs is not the number of signers the wires were allocated for.
	pub fn populate(
		&self,
		w: &mut WitnessFiller,
		message: &Message,
		epoch: u32,
		signatures: &[(XmssPublicKey, XmssSignature)],
	) {
		assert_eq!(
			signatures.len(),
			self.signers.len(),
			"expected {} signatures, got {}",
			self.signers.len(),
			signatures.len()
		);

		w.pack_bytes_le(&self.message, message);
		w[self.epoch] = Word::from_u64(epoch as u64);
		for (wires, (public_key, signature)) in iter::zip(&self.signers, signatures) {
			w.pack_bytes_le(&wires.public_param, &public_key.public_param);
			w.pack_bytes_le(&wires.merkle_root, &public_key.merkle_root);
			wires.signature.populate(w, signature);
		}
	}
}

/// Verifies every signer's XMSS signature on the common message at the common epoch.
///
/// Sharing one epoch wire across the signers is what makes the epoch common: there is no
/// per-signer epoch to disagree with.
pub fn circuit_xmss_multisig(builder: &CircuitBuilder, wires: &MultiSigWires) {
	for (i, signer) in wires.signers.iter().enumerate() {
		let builder = builder.subcircuit(format!("signer[{i}]"));
		circuit_xmss_verify(
			&builder,
			&signer.public_param,
			&signer.merkle_root,
			&wires.message,
			wires.epoch,
			&signer.signature,
		);
	}
}

#[cfg(test)]
mod tests {
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::hash_based_sig::{MESSAGE_LEN, xmss::generate_signature};

	/// Generates `num_signers` independent signatures on one message at one epoch.
	fn generate(
		seed: u64,
		num_signers: usize,
		epoch: u32,
	) -> (Message, Vec<(XmssPublicKey, XmssSignature)>) {
		let mut rng = StdRng::seed_from_u64(seed);
		let mut message = [0u8; MESSAGE_LEN];
		rng.fill_bytes(&mut message);
		let signatures = (0..num_signers)
			.map(|_| generate_signature(&mut rng, &message, epoch))
			.collect();
		(message, signatures)
	}

	fn run(
		message: &Message,
		epoch: u32,
		signatures: &[(XmssPublicKey, XmssSignature)],
	) -> Result<(), String> {
		let b = CircuitBuilder::new();
		let wires = MultiSigWires::new(&b, signatures.len());
		circuit_xmss_multisig(&b, &wires);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		wires.populate(&mut w, message, epoch, signatures);

		circuit
			.populate_wire_witness(&mut w)
			.map_err(|e| format!("populate: {e:?}"))?;
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.map_err(|e| format!("verify: {e:?}"))
	}

	#[test]
	fn independent_signers_verify_together() {
		let (message, signatures) = generate(1, 3, 42);
		run(&message, 42, &signatures).unwrap();
	}

	#[test]
	fn one_bad_signature_fails_the_aggregate() {
		let (message, mut signatures) = generate(2, 3, 42);
		signatures[1].1.chain_tips[0][0] ^= 0xFF;
		assert!(run(&message, 42, &signatures).is_err());
	}

	#[test]
	fn a_signature_on_another_message_fails_the_aggregate() {
		// Signers share the message wires, so a signature over anything else has nowhere to hide.
		let (message, mut signatures) = generate(3, 2, 42);
		let mut rng = StdRng::seed_from_u64(4);
		let mut other = [0u8; MESSAGE_LEN];
		rng.fill_bytes(&mut other);
		signatures[0] = generate_signature(&mut rng, &other, 42);
		assert!(run(&message, 42, &signatures).is_err());
	}
}
