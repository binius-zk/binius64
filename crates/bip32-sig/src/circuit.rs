// Copyright 2026 The Binius Developers
//! Glue over the reused `Bip32Example` circuit: setup/serialization, proving, and verification.

use std::{
	fs,
	path::{Path, PathBuf},
	time::{Duration, Instant},
};

use anyhow::{Result, bail};
use binius_core::{
	constraint_system::{ConstraintSystem, ValueVecLayout},
	word::Word,
};
use binius_examples::{
	ExampleCircuit,
	circuits::bip32::{Bip32Example, Instance, Params},
	setup_zk,
};
use binius_frontend::{Circuit, CircuitBuilder};
use binius_hash::StdHashSuite;
use binius_utils::serialization::{DeserializeBytes, SerializeBytes};
use binius_verifier::{
	config::StdChallenger,
	transcript::{ProverTranscript, VerifierTranscript},
	zk_config::ZKVerifier,
};

/// BIP32 tree depth supported by the circuit (matches the `bip32` example default). A BIP44 path
/// `m/purpose'/coin'/account'/change/index` is exactly this deep.
pub const MAX_DEPTH: usize = 5;

/// Log of the inverse rate for the proof system. Must match between prove and verify.
pub const LOG_INV_RATE: usize = 3;

/// Number of public `inout` words the circuit exposes (the SHA-256 digest, 8 × 32-bit words).
const N_HASH_WORDS: usize = 8;

/// Filename of the cached constraint system. It lives in the working directory, alongside the
/// proof files the tool produces.
pub const CS_CACHE_FILE: &str = "bip32-sig-constraint-system.bin";

/// Where the serialized constraint system is cached between runs (the working directory).
pub fn cs_cache_path() -> PathBuf {
	PathBuf::from(CS_CACHE_FILE)
}

/// Build the circuit and the example wrapper used to populate witnesses.
pub fn build_circuit() -> Result<(Circuit, Bip32Example)> {
	let mut builder = CircuitBuilder::new();
	let example = Bip32Example::build(
		Params {
			max_depth: MAX_DEPTH,
		},
		&mut builder,
	)?;
	let circuit = builder.build();
	Ok((circuit, example))
}

/// Serialize a constraint system to `path`, creating parent directories as needed.
fn save_cs(cs: &ConstraintSystem, path: &Path) -> Result<()> {
	if let Some(parent) = path.parent()
		&& !parent.as_os_str().is_empty()
	{
		fs::create_dir_all(parent)
			.map_err(|e| anyhow::anyhow!("failed to create '{}': {e}", parent.display()))?;
	}
	let mut buf = Vec::new();
	cs.serialize(&mut buf)?;
	fs::write(path, &buf)
		.map_err(|e| anyhow::anyhow!("failed to write '{}': {e}", path.display()))?;
	Ok(())
}

/// Deserialize a constraint system from `path`.
pub fn load_cs(path: &Path) -> Result<ConstraintSystem> {
	let buf =
		fs::read(path).map_err(|e| anyhow::anyhow!("failed to read '{}': {e}", path.display()))?;
	ConstraintSystem::deserialize(buf.as_slice())
		.map_err(|e| anyhow::anyhow!("failed to deserialize constraint system: {e}"))
}

/// Build the circuit and write its constraint system to the cache path.
pub fn create_and_cache_cs(path: &Path) -> Result<ConstraintSystem> {
	let (circuit, _) = build_circuit()?;
	let cs = circuit.constraint_system().clone();
	save_cs(&cs, path)?;
	Ok(cs)
}

/// Load the cached constraint system, building and caching it first if absent.
pub fn load_or_create_cs(path: &Path) -> Result<ConstraintSystem> {
	if path.exists() {
		load_cs(path)
	} else {
		create_and_cache_cs(path)
	}
}

/// Result of proving: the proof bytes, the SHA-256 public-key hash (the public input, read back
/// from the witness so it is guaranteed to match what the proof commits to), and a timing breakdown
/// of the three phases.
pub struct Proof {
	pub bytes: Vec<u8>,
	pub pubkey_sha256: [u8; 32],
	/// Building the circuit and setting up the prover/verifier (witness-independent).
	pub setup_time: Duration,
	/// Generating the witness and producing the proof.
	pub proving_time: Duration,
	/// Self-verifying the freshly produced proof.
	pub verify_time: Duration,
}

/// Generate a signature-of-knowledge proof for `(seed, path)` over `message`, and self-verify it.
///
/// The three phases are timed independently: setup (circuit build + prover/verifier setup) is
/// witness-independent and is the part a future cache could elide; proving covers witness
/// generation and `prove_sig`; verification is the self-check and is kept out of the proving time.
pub fn prove(seed: &[u8; 64], path: &[u32], message: &[u8]) -> Result<Proof> {
	// Phase 1 — setup: build the circuit and the prover/verifier. None of this depends on the
	// witness, so it is the work a serialized setup would let us skip.
	let setup_start = Instant::now();
	let (circuit, example) = build_circuit()?;
	let cs = circuit.constraint_system().clone();
	let (verifier, prover) = setup_zk::<StdHashSuite>(cs, LOG_INV_RATE)?;
	let setup_time = setup_start.elapsed();

	// Phase 2 — witness generation + proving.
	let proving_start = Instant::now();
	let mut filler = circuit.new_witness_filler();
	let instance = Instance {
		seed: Some(hex::encode(seed)),
		path: path.to_vec(),
	};
	example.populate_witness(instance, &mut filler)?;
	circuit.populate_wire_witness(&mut filler)?;
	let witness = filler.into_value_vec();

	let layout = circuit.constraint_system().value_vec_layout.clone();
	let pubkey_sha256 = public_inout_to_sha256(&layout, witness.public())?;

	let challenger = StdChallenger::default();
	let mut prover_transcript = ProverTranscript::new(challenger.clone());
	let mut rng = rand::rng();
	prover.prove_sig(witness.clone(), message, &mut rng, &mut prover_transcript)?;
	let bytes = prover_transcript.finalize();
	let proving_time = proving_start.elapsed();

	// Phase 3 — verification: self-check the freshly produced proof (not counted as proving time).
	let verify_start = Instant::now();
	let mut verifier_transcript = VerifierTranscript::new(challenger, bytes.clone());
	verifier.verify_sig(witness.public(), message, &mut verifier_transcript)?;
	verifier_transcript.finalize()?;
	let verify_time = verify_start.elapsed();

	Ok(Proof {
		bytes,
		pubkey_sha256,
		setup_time,
		proving_time,
		verify_time,
	})
}

/// Timing breakdown of verification: setting up the verifier vs. checking the proof.
pub struct Verification {
	/// Setting up the verifier from the constraint system.
	pub setup_time: Duration,
	/// Checking the signature of knowledge.
	pub verify_time: Duration,
}

/// Verify a signature-of-knowledge proof against a constraint system, public-key hash, and message.
pub fn verify(
	cs: &ConstraintSystem,
	pubkey_sha256: &[u8; 32],
	proof: &[u8],
	message: &[u8],
) -> Result<Verification> {
	let public = public_from_sha256(cs, pubkey_sha256)?;

	let setup_start = Instant::now();
	let verifier = ZKVerifier::<StdHashSuite>::setup(cs.clone(), LOG_INV_RATE)?;
	let setup_time = setup_start.elapsed();

	let verify_start = Instant::now();
	let challenger = StdChallenger::default();
	let mut verifier_transcript = VerifierTranscript::new(challenger, proof.to_vec());
	verifier.verify_sig(&public, message, &mut verifier_transcript)?;
	verifier_transcript.finalize()?;
	let verify_time = verify_start.elapsed();

	Ok(Verification {
		setup_time,
		verify_time,
	})
}

/// Read the 8 public `inout` words (the SHA-256 digest) out of a public values vector.
fn public_inout_to_sha256(layout: &ValueVecLayout, public: &[Word]) -> Result<[u8; 32]> {
	if layout.n_inout != N_HASH_WORDS {
		bail!("unexpected circuit: {} inout words (expected {N_HASH_WORDS})", layout.n_inout);
	}
	let mut digest = [0u8; 32];
	for i in 0..N_HASH_WORDS {
		let word = public[layout.offset_inout + i].0 as u32;
		digest[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
	}
	Ok(digest)
}

/// Reconstruct the full public values vector (constants + the SHA-256 inout words) from a
/// constraint system and a stored digest, matching the layout the prover used.
fn public_from_sha256(cs: &ConstraintSystem, sha256: &[u8; 32]) -> Result<Vec<Word>> {
	let layout = &cs.value_vec_layout;
	if layout.n_inout != N_HASH_WORDS {
		bail!("unexpected circuit: {} inout words (expected {N_HASH_WORDS})", layout.n_inout);
	}
	let mut public = vec![Word::ZERO; layout.offset_witness];
	public[..layout.n_const].copy_from_slice(&cs.constants);
	for i in 0..N_HASH_WORDS {
		let word = u32::from_be_bytes(sha256[4 * i..4 * i + 4].try_into().expect("4-byte chunk"));
		public[layout.offset_inout + i] = Word::from_u64(word as u64);
	}
	Ok(public)
}
