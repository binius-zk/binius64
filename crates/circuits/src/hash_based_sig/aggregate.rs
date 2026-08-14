// Copyright 2026 The Binius Developers
//! Aggregation of XMSS signatures on a common message.
//!
//! Each signer has an independent tree, so a public key is a `(root, public parameter)` pair and
//! both are per-signer. All signers sign the same message at the same epoch, and one proof stands
//! for every signature.
//!
//! The same aggregate is built two ways over one set of wires: [`circuit_xmss_multisig`] emits
//! every signer's verification inline, and [`circuit_xmss_multisig_chip`] states it once as an M4
//! chip and calls that chip per signer.

use std::iter;

use binius_core::Word;
use binius_frontend::{CircuitBuilder, CircuitM4, Wire, WitnessFiller};

use super::{
	DIGEST_WIRES, MESSAGE_WIRES, Message, PUBLIC_PARAM_WIRES,
	xmss::{XmssPublicKey, XmssSignature, XmssSignatureWires, circuit_xmss_verify},
};
use crate::blake3::Blake3Compress2x;

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

/// The words one call to the XMSS chip passes, in the chip's inout order.
///
/// [`xmss_verify_chip`] allocates its interface in this same order, so a call's operands line up
/// with the chip's inout segment position by position.
fn chip_call_words(
	signer: &SignerWires,
	message: &[Wire; MESSAGE_WIRES],
	epoch: Wire,
) -> Vec<Wire> {
	let XmssSignatureWires {
		randomness,
		chain_tips,
		merkle_path,
	} = &signer.signature;
	signer
		.public_param
		.iter()
		.chain(&signer.merkle_root)
		.chain(message)
		.chain(iter::once(&epoch))
		.chain(randomness)
		.chain(chain_tips.iter().flatten())
		.chain(merkle_path.iter().flatten())
		.copied()
		.collect()
}

/// One XMSS verification, as a system of its own for a caller to embed as a chip.
///
/// Every word the check relates is an inout word, so a call supplies the signer's public key, the
/// message, the epoch and the whole signature, and the chip derives the rest. There is nothing for
/// the caller to compute on the chip's behalf: the chip has no outputs, only the constraints that
/// make its operands a valid signature.
///
/// The system registers [`Blake3Compress2x`] for itself, so the paired chain compressions inside it
/// are calls to a chip of its own. Embedding this system therefore brings in two chips: this one,
/// and the compression it calls.
fn xmss_verify_chip() -> CircuitM4 {
	let builder = CircuitBuilder::new();
	builder.register_chip(Blake3Compress2x, &[]);

	// The inout segment is ordered by wire creation, so allocating in `chip_call_words` order is
	// what makes the interface that order. The assertion below holds the two together.
	let public_param = std::array::from_fn(|_| builder.add_inout());
	let merkle_root = std::array::from_fn(|_| builder.add_inout());
	let message: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| builder.add_inout());
	let epoch = builder.add_inout();
	let signature = XmssSignatureWires {
		randomness: std::array::from_fn(|_| builder.add_inout()),
		chain_tips: std::array::from_fn(|_| std::array::from_fn(|_| builder.add_inout())),
		merkle_path: std::array::from_fn(|_| std::array::from_fn(|_| builder.add_inout())),
	};
	let signer = SignerWires {
		public_param,
		merkle_root,
		signature,
	};

	circuit_xmss_verify(
		&builder,
		&signer.public_param,
		&signer.merkle_root,
		&message,
		epoch,
		&signer.signature,
	);

	let system = builder.build_m4();
	assert_eq!(
		system.main.circuit.inout(),
		chip_call_words(&signer, &message, epoch),
		"the chip's inout segment must be the order its call sites pass"
	);
	system
}

/// [`circuit_xmss_multisig`] with each signer's verification dispatched to a chip.
///
/// The main circuit is left holding no hash gates at all: it declares the statement, witnesses the
/// signatures, and passes both to one chip call per signer. Every signer's verification is the same
/// relation over different words, which is what a chip is for — the constraints are stated once and
/// the trace pays for them once per instance.
///
/// The wires are [`MultiSigWires`] unchanged, so [`MultiSigWires::populate`] fills this circuit as
/// it does the inline one. The built circuit carries chips, though, so it builds with
/// [`CircuitBuilder::build_m4`] rather than [`CircuitBuilder::build`].
pub fn circuit_xmss_multisig_chip(builder: &CircuitBuilder, wires: &MultiSigWires) {
	let chip = builder.add_chip(xmss_verify_chip());
	for signer in &wires.signers {
		builder.call_chip(chip, &chip_call_words(signer, &wires.message, wires.epoch));
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

	/// [`run`] over the chip-dispatching circuit, checking the chips as well as the main circuit.
	///
	/// The main circuit populating is only half of it: a call whose operands its chip's instance
	/// does not reproduce shows up in [`WitnessM4::verify`](binius_core::m4::WitnessM4::verify)
	/// alone, which is what the final check here covers.
	fn run_chip(
		message: &Message,
		epoch: u32,
		signatures: &[(XmssPublicKey, XmssSignature)],
	) -> Result<(), String> {
		let b = CircuitBuilder::new();
		let wires = MultiSigWires::new(&b, signatures.len());
		circuit_xmss_multisig_chip(&b, &wires);

		let circuit = b.build_m4();
		circuit.validate().map_err(|e| format!("validate: {e:?}"))?;
		let cs = circuit.to_constraint_system();
		cs.validate().map_err(|e| format!("validate cs: {e:?}"))?;

		let witness = circuit
			.generate_witness(|w| wires.populate(w, message, epoch, signatures))
			.map_err(|e| format!("populate: {e:?}"))?;
		witness.verify(&cs).map_err(|e| format!("verify: {e:?}"))
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

	#[test]
	fn independent_signers_verify_together_through_the_chip() {
		let (message, signatures) = generate(1, 2, 42);
		run_chip(&message, 42, &signatures).unwrap();
	}

	#[test]
	fn one_bad_signature_fails_the_chip_aggregate() {
		let (message, mut signatures) = generate(2, 2, 42);
		signatures[1].1.chain_tips[0][0] ^= 0xFF;
		assert!(run_chip(&message, 42, &signatures).is_err());
	}
}
