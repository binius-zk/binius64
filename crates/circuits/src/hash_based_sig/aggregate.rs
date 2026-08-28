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
//!
//! Both publish every signer's key, so their statement grows with the number of signers.
//!
//! A third form publishes one commitment to the whole set instead.
//! That is what a statement has to look like before an aggregate can be a step in a tree.

use std::iter;

use binius_core::Word;
use binius_frontend::{CircuitBuilder, CircuitM4, Wire, WitnessFiller};

use super::{
	DIGEST_LEN, DIGEST_WIRES, MESSAGE_WIRES, Message, PUBLIC_PARAM_LEN, PUBLIC_PARAM_WIRES,
	xmss::{XmssPublicKey, XmssSignature, XmssSignatureWires, circuit_xmss_verify},
};
use crate::{
	bignum::{BigUint, biguint_lt},
	blake3::{Blake3Compress2x, blake3_fixed},
	util::split_u32_words,
};

/// One signer's public key and signature wires.
#[derive(Debug, Clone)]
pub struct SignerWires {
	pub public_param: [Wire; PUBLIC_PARAM_WIRES],
	pub merkle_root: [Wire; DIGEST_WIRES],
	pub signature: XmssSignatureWires,
}

impl SignerWires {
	/// Allocates a signer whose key is public and whose signature is private.
	pub fn new_inout(builder: &CircuitBuilder) -> Self {
		Self {
			public_param: std::array::from_fn(|_| builder.add_inout()),
			merkle_root: std::array::from_fn(|_| builder.add_inout()),
			signature: XmssSignatureWires::new_witness(builder),
		}
	}

	/// Allocates a signer whose key is private as well.
	///
	/// A key held privately reaches the statement only through a commitment to the whole set.
	pub fn new_witness(builder: &CircuitBuilder) -> Self {
		Self {
			public_param: std::array::from_fn(|_| builder.add_witness()),
			merkle_root: std::array::from_fn(|_| builder.add_witness()),
			signature: XmssSignatureWires::new_witness(builder),
		}
	}

	/// Fills the key and the signature.
	fn populate(
		&self,
		w: &mut WitnessFiller<'_>,
		public_key: &XmssPublicKey,
		signature: &XmssSignature,
	) {
		w.pack_bytes_le(&self.public_param, &public_key.public_param);
		w.pack_bytes_le(&self.merkle_root, &public_key.merkle_root);
		self.signature.populate(w, signature);
	}

	/// The key read as one integer, least significant limb first.
	///
	/// The packing puts the key's first byte in the lowest limb's lowest byte.
	/// So the integer is the key's bytes read little-endian end to end.
	fn key_as_integer(&self) -> BigUint {
		BigUint {
			limbs: self
				.public_param
				.iter()
				.chain(&self.merkle_root)
				.copied()
				.collect(),
		}
	}
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
				.map(|_| SignerWires::new_inout(builder))
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
		w: &mut WitnessFiller<'_>,
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
			wires.populate(w, public_key, signature);
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

/// Bytes one signer contributes to a signer-set commitment.
pub const SIGNER_BYTES: usize = PUBLIC_PARAM_LEN + DIGEST_LEN;

/// Wires holding a signer-set commitment.
///
/// Eight words, each holding a little-endian 32-bit half.
/// That is the form the hash gadget returns.
pub const SIGNER_SET_WIRES: usize = 8;

/// The order signers are declared in.
///
/// Only totality matters here, not which order it is.
///
/// This one is the order the wire packing already induces.
/// So nothing in circuit has to rearrange a key to compare it.
///
/// A key's bytes read as one little-endian integer.
/// An integer comparison is therefore a comparison of the reversed byte string.
pub fn signer_order_key(public_key: &XmssPublicKey) -> [u8; SIGNER_BYTES] {
	let mut bytes = [0u8; SIGNER_BYTES];
	bytes[..PUBLIC_PARAM_LEN].copy_from_slice(&public_key.public_param);
	bytes[PUBLIC_PARAM_LEN..].copy_from_slice(&public_key.merkle_root);
	bytes.reverse();
	bytes
}

/// Puts signatures into the declared order.
pub fn sort_by_signer(signatures: &mut [(XmssPublicKey, XmssSignature)]) {
	signatures.sort_unstable_by_key(|(public_key, _)| signer_order_key(public_key));
}

/// The commitment to a declared signer set.
///
/// The keys are hashed in the order given, so this binds an order as well as a set.
/// Requiring that order to be strictly increasing is what makes the two the same thing.
///
/// The digest is the full 256 bits rather than the scheme's truncated 128.
///
/// A signature needs its hashes to make a *chosen* target hard to collide.
/// 128 bits carry that.
///
/// A set commitment needs *any* two sets to be hard to collide.
/// 128 bits leave that at 64.
pub fn signer_set_digest(public_keys: &[XmssPublicKey]) -> [u8; 32] {
	// The layout is the wire layout: each signer's parameter, then its root, in declared order.
	let mut bytes = Vec::with_capacity(public_keys.len() * SIGNER_BYTES);
	for public_key in public_keys {
		bytes.extend_from_slice(&public_key.public_param);
		bytes.extend_from_slice(&public_key.merkle_root);
	}
	*blake3::hash(&bytes).as_bytes()
}

/// The wires a committed aggregate occupies.
///
/// The message, the epoch and one commitment to the signer set are public.
/// Every key and every signature is private.
///
/// So the statement is the same size whatever the number of signers.
#[derive(Debug, Clone)]
pub struct CommittedMultiSigWires {
	pub message: [Wire; MESSAGE_WIRES],
	pub epoch: Wire,
	pub signer_set: [Wire; SIGNER_SET_WIRES],
	pub signers: Vec<SignerWires>,
}

impl CommittedMultiSigWires {
	/// Allocates the wires for `num_signers` signers.
	///
	/// # Panics
	///
	/// If there are no signers.
	/// A commitment to nothing states nothing.
	/// So an empty aggregate is a mistake, not a degenerate case worth supporting.
	pub fn new(builder: &CircuitBuilder, num_signers: usize) -> Self {
		assert!(num_signers > 0, "an aggregate needs at least one signer");
		Self {
			message: std::array::from_fn(|_| builder.add_inout()),
			epoch: builder.add_inout(),
			signer_set: std::array::from_fn(|_| builder.add_inout()),
			signers: (0..num_signers)
				.map(|_| SignerWires::new_witness(builder))
				.collect(),
		}
	}

	/// Populates every wire from a message, an epoch and one key-and-signature pair per signer.
	///
	/// The commitment is computed over the pairs in the order given.
	/// Nothing is sorted here.
	///
	/// So a caller that has not sorted gets a statement its own circuit rejects.
	/// It never gets a statement about a set it did not mean.
	///
	/// # Panics
	///
	/// If the number of pairs is not the number of signers the wires were allocated for.
	pub fn populate(
		&self,
		w: &mut WitnessFiller<'_>,
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
			wires.populate(w, public_key, signature);
		}

		// The commitment goes in as the hash gadget reports it.
		// One little-endian 32-bit word per wire, in the wire's low half.
		let public_keys: Vec<_> = signatures
			.iter()
			.map(|&(public_key, _)| public_key)
			.collect();
		let digest = signer_set_digest(&public_keys);
		for (&wire, chunk) in iter::zip(&self.signer_set, digest.chunks_exact(4)) {
			let word = u32::from_le_bytes(chunk.try_into().expect("a chunk is four bytes"));
			w[wire] = Word::from_u64(word as u64);
		}
	}
}

/// The aggregate verification against a committed signer set rather than a published one.
///
/// Three things are checked, and all three are needed:
///
/// - every signature verifies, as in the published form,
/// - the keys strictly increase, so no two signers are the same signer,
/// - they hash to the published commitment.
///
/// # Why all three
///
/// Verification alone counts signatures, not signers: one key repeated across every slot passes it.
///
/// The commitment alone fixes a sequence, not a set.
/// One set of keys in two orders gives two commitments, and so does one key repeated.
///
/// A verifier holding the set would not know which of them to expect.
///
/// Strict increase settles both at once.
/// A repeat is not an increase, so repeats are out.
/// One order survives per set, so the commitment commits to the set.
pub fn circuit_xmss_multisig_committed(builder: &CircuitBuilder, wires: &CommittedMultiSigWires) {
	// Every signer's key contributes its bytes to the commitment, in declared order.
	let mut payload = Vec::with_capacity(wires.signers.len() * (PUBLIC_PARAM_WIRES + DIGEST_WIRES));

	for (i, signer) in wires.signers.iter().enumerate() {
		let signer_builder = builder.subcircuit(format!("signer[{i}]"));
		circuit_xmss_verify(
			&signer_builder,
			&signer.public_param,
			&signer.merkle_root,
			&wires.message,
			wires.epoch,
			&signer.signature,
		);

		// Each neighbouring pair is compared once.
		// The chain of those comparisons orders the whole list.
		if i > 0 {
			let previous = &wires.signers[i - 1];
			let increases =
				biguint_lt(&signer_builder, &previous.key_as_integer(), &signer.key_as_integer());
			signer_builder.assert_true("signer keys strictly increase", increases);
		}

		payload.extend_from_slice(&signer.public_param);
		payload.extend_from_slice(&signer.merkle_root);
	}

	// The hash gadget reads 32-bit words and a signer's key is 64-bit wires.
	// So the payload is split in half word by word.
	let len_bytes = wires.signers.len() * SIGNER_BYTES;
	let message = split_u32_words(builder, &payload, len_bytes / 4);
	let digest = blake3_fixed(builder, &message, len_bytes);

	builder.assert_eq_v("signer set commitment", digest, wires.signer_set);
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

	/// Runs the committed aggregate, letting a caller disturb the witness before it is checked.
	///
	/// The commitment is a public input.
	/// So a test that wants a wrong one overwrites its wires after populating.
	fn run_committed(
		message: &Message,
		epoch: u32,
		signatures: &[(XmssPublicKey, XmssSignature)],
		disturb: impl FnOnce(&mut WitnessFiller<'_>, &CommittedMultiSigWires),
	) -> Result<(), String> {
		let b = CircuitBuilder::new();
		let wires = CommittedMultiSigWires::new(&b, signatures.len());
		circuit_xmss_multisig_committed(&b, &wires);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		wires.populate(&mut w, message, epoch, signatures);
		disturb(&mut w, &wires);

		circuit
			.populate_wire_witness(&mut w)
			.map_err(|e| format!("populate: {e:?}"))?;
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.map_err(|e| format!("verify: {e:?}"))
	}

	/// Signatures in declared order, which is what an honest caller supplies.
	fn generate_sorted(
		seed: u64,
		num_signers: usize,
		epoch: u32,
	) -> (Message, Vec<(XmssPublicKey, XmssSignature)>) {
		let (message, mut signatures) = generate(seed, num_signers, epoch);
		sort_by_signer(&mut signatures);
		(message, signatures)
	}

	#[test]
	fn a_committed_aggregate_verifies() {
		let (message, signatures) = generate_sorted(1, 3, 42);
		run_committed(&message, 42, &signatures, |_, _| ()).unwrap();
	}

	#[test]
	fn the_statement_is_the_same_size_whatever_the_number_of_signers() {
		// Invariant: the public segment holds the message, the epoch and the commitment.
		// Nothing in it is per-signer.
		//
		//     published:  message (4) + epoch (1) + commitment (8) = 13 words
		//     witnessed:  every key, every signature
		let published = |num_signers| {
			let b = CircuitBuilder::new();
			let wires = CommittedMultiSigWires::new(&b, num_signers);
			circuit_xmss_multisig_committed(&b, &wires);
			b.build().constraint_system().n_inout
		};
		assert_eq!(published(1), published(8));
	}

	#[test]
	fn the_published_form_grows_with_the_signers_and_this_one_does_not() {
		// The defect this aggregate exists to fix, stated as a comparison.
		let published_inout = |num_signers| {
			let b = CircuitBuilder::new();
			let wires = MultiSigWires::new(&b, num_signers);
			circuit_xmss_multisig(&b, &wires);
			b.build().constraint_system().n_inout
		};
		assert!(published_inout(8) > published_inout(1));
	}

	#[test]
	fn a_repeated_signer_fails_the_committed_aggregate() {
		// Invariant: a set of three signers means three *distinct* signers.
		//
		// Mutation: give slot 1 slot 0's key and signature.
		//
		//     keys:  [k0, k0, k2]
		//     -> the pair (0, 1) is not an increase
		//
		// Both slots verify, since each carries its own valid signature.
		// So verification alone does not catch this.
		let (message, mut signatures) = generate_sorted(5, 3, 42);
		signatures[1] = signatures[0].clone();
		let err = run_committed(&message, 42, &signatures, |_, _| ())
			.expect_err("a repeated signer must be rejected");
		assert!(err.contains("strictly increase"), "unexpected failure: {err}");
	}

	#[test]
	fn signers_out_of_order_fail_the_committed_aggregate() {
		// Invariant: one order per set, so a verifier holding the set knows what to recompute.
		//
		// Mutation: swap two neighbours of an otherwise sorted list.
		//
		// The commitment still matches the witness, because it is computed over the order given.
		// What fails is the ordering check.
		let (message, mut signatures) = generate_sorted(6, 3, 42);
		signatures.swap(0, 1);
		let err = run_committed(&message, 42, &signatures, |_, _| ())
			.expect_err("an unsorted signer list must be rejected");
		assert!(err.contains("strictly increase"), "unexpected failure: {err}");
	}

	#[test]
	fn a_wrong_commitment_fails_the_committed_aggregate() {
		// Invariant: the published commitment is the one the witnessed keys hash to.
		//
		// Mutation: flip one bit of the published commitment, leaving every signature valid.
		let (message, signatures) = generate_sorted(7, 2, 42);
		let err = run_committed(&message, 42, &signatures, |w, wires| {
			w[wires.signer_set[0]] = Word(w[wires.signer_set[0]].0 ^ 1);
		})
		.expect_err("a commitment the keys do not hash to must be rejected");
		assert!(err.contains("commitment"), "unexpected failure: {err}");
	}

	#[test]
	fn a_bad_signature_fails_the_committed_aggregate() {
		// A private key is still a key that has to sign.
		let (message, mut signatures) = generate_sorted(8, 3, 42);
		signatures[1].1.chain_tips[0][0] ^= 0xFF;
		assert!(run_committed(&message, 42, &signatures, |_, _| ()).is_err());
	}

	#[test]
	fn the_commitment_is_over_the_keys_and_nothing_else() {
		// Invariant: the native commitment is what the circuit recomputes.
		// So a caller can predict the statement without building a circuit.
		//
		// Two sets differing in one byte of one key must commit differently.
		// The same set must commit the same way twice.
		let (_, signatures) = generate_sorted(9, 3, 42);
		let keys: Vec<_> = signatures.iter().map(|&(k, _)| k).collect();

		let mut other = keys.clone();
		other[2].merkle_root[0] ^= 1;

		assert_eq!(signer_set_digest(&keys), signer_set_digest(&keys));
		assert_ne!(signer_set_digest(&keys), signer_set_digest(&other));
	}

	#[test]
	fn the_declared_order_is_the_order_the_circuit_compares_in() {
		// The native ordering and the in-circuit comparison have to agree.
		// Otherwise an honestly sorted caller builds a statement its own circuit rejects.
		//
		// Sorting natively and running the circuit is the check.
		// The ordering assertions pass only if the two orders are the same order.
		let (message, signatures) = generate_sorted(10, 4, 42);
		let keys: Vec<_> = signatures.iter().map(|&(k, _)| k).collect();

		// The native order is strictly increasing, so no two keys collide in it either.
		for pair in keys.windows(2) {
			assert!(signer_order_key(&pair[0]) < signer_order_key(&pair[1]));
		}

		run_committed(&message, 42, &signatures, |_, _| ()).unwrap();
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
