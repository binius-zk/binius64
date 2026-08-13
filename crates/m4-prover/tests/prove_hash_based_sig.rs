// Copyright 2026 The Binius Developers
//! M4 proof of one hash-based signature verification, compressing through a BLAKE3 chip.
//!
//! The companion to `prove_hash_primitives`, one level up: that file proves a batch of bare
//! primitives, this one proves a whole XMSS verification and lets the primitive inside it land in
//! a chip. The circuit verifies a single signature, so the main chip runs once; the paired BLAKE3
//! compressions its Winternitz chains reach become instances of the [`Blake3Compress2x`] chip, and
//! the two are proved together as one composite system.
//!
//! The proof runs inside a timing span.
//! With tracing enabled, the prover's internal spans nest beneath that span.
//! The per-phase breakdown of proving is then visible, split between main and the chip.
//!
//! The run exists to be read rather than to gate anything, so it is `#[ignore]`d and needs
//! `--ignored` to run. Run it `--release` as well: an unoptimized prover over this circuit is
//! slower by orders of magnitude.
//!
//! Run it with the timing tree:
//!
//! ```text
//! RUST_LOG=debug cargo test --release -p binius-m4-prover --test prove_hash_based_sig \
//!     -- --ignored --nocapture
//! ```

use binius_circuits::{
	blake3::Blake3Compress2x,
	hash_based_sig::{
		winternitz_ots::{NONCE_LENGTH_BYTES, NONCE_WIRES_COUNT, WinternitzSpec},
		witness_utils::ValidatorSignatureData,
		xmss::{XmssSignature, circuit_xmss},
	},
};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, CircuitM4, CircuitStat, Wire, WitnessFiller};
use binius_hash::StdHashSuite;
use binius_m4_prover::ProverM4;
use binius_m4_verifier::VerifierM4;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use rand::prelude::*;
use tracing::{debug, info_span, level_filters::LevelFilter};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching `prove_hash_primitives`.
const LOG_INV_RATE: usize = 1;

/// A 32-byte digest occupies four 64-bit little-endian wires.
const HASH_WIRES: usize = 4;

/// Reads the environment variable `var`, or `default` if unset.
///
/// Lets the run be resized from the command line, e.g.
/// `WINTERNITZ_SPEC=2 cargo test --release -p binius-m4-prover --test prove_hash_based_sig`.
///
/// # Panics
///
/// Panics if `var` is set but does not parse as a `usize`.
fn usize_from_env(var: &str, default: usize) -> usize {
	match std::env::var(var) {
		Ok(val) => val
			.parse()
			.unwrap_or_else(|_| panic!("{var} must be a non-negative integer, got {val:?}")),
		Err(_) => default,
	}
}

/// Installs a timing-tree tracing subscriber, once per test binary.
///
/// The subscriber prints each root span's duration and its children's share of it.
/// Verbosity defaults to `WARN`; `RUST_LOG` overrides that to reach the prover's `DEBUG` spans.
/// A call after a subscriber is already installed does nothing.
fn init_tracing() {
	let env_filter = EnvFilter::builder()
		.with_default_directive(LevelFilter::WARN.into())
		.from_env_lossy();

	// A second call does nothing once a global subscriber is installed.
	let _ = tracing_subscriber::registry()
		.with(env_filter)
		.with(ForestLayer::default())
		.try_init();
}

/// The public wires of one XMSS verification.
struct XmssWires {
	/// The signer's domain parameter, eight bytes per wire.
	domain_param: Vec<Wire>,
	/// The signed message, 32 bytes.
	message: [Wire; HASH_WIRES],
	/// The committed Merkle root the authentication path must reach.
	root_hash: [Wire; HASH_WIRES],
	/// The signature itself, whose wires carry the nonce, epoch, chain ends and path.
	signature: XmssSignature,
}

/// Builds a circuit verifying one XMSS signature, with the paired BLAKE3 compression as a chip.
///
/// Registering the chip is the whole opt-in: `circuit_xmss` reaches `blake3_compress_2x` through
/// the Winternitz chain steps without knowing whether it lands in gates or a chip call. The
/// single-lane compressions the message hash, the public-key hash and the Merkle path use stay
/// inline, so the chip serves the chain steps alone.
fn build_xmss_circuit(spec: &WinternitzSpec, tree_height: usize) -> (CircuitM4, XmssWires) {
	let builder = CircuitBuilder::new();
	builder.register_chip(Blake3Compress2x, &[]);

	// Everything the verifier is given is public: the statement is the signature and what it
	// signs. Nothing here is a witness, so the committed trace holds only derived values.
	let domain_param: Vec<Wire> = (0..spec.domain_param_len.div_ceil(8))
		.map(|_| builder.add_inout())
		.collect();
	let message: [Wire; HASH_WIRES] = std::array::from_fn(|_| builder.add_inout());
	let root_hash: [Wire; HASH_WIRES] = std::array::from_fn(|_| builder.add_inout());
	let signature = XmssSignature {
		nonce: (0..NONCE_WIRES_COUNT)
			.map(|_| builder.add_inout())
			.collect(),
		epoch: builder.add_inout(),
		signature_hashes: (0..spec.dimension())
			.map(|_| std::array::from_fn(|_| builder.add_inout()))
			.collect(),
		public_key_hashes: (0..spec.dimension())
			.map(|_| std::array::from_fn(|_| builder.add_inout()))
			.collect(),
		auth_path: (0..tree_height)
			.map(|_| std::array::from_fn(|_| builder.add_inout()))
			.collect(),
	};

	circuit_xmss(&builder, spec, &domain_param, &message, &signature, &root_hash);

	let circuit = builder.build_m4();
	(
		circuit,
		XmssWires {
			domain_param,
			message,
			root_hash,
			signature,
		},
	)
}

/// Generates a valid signature and writes it into the circuit's public wires.
///
/// Every digest the circuit checks is BLAKE3 over these inputs, so the evaluator derives the rest.
fn fill_xmss(
	wires: &XmssWires,
	spec: &WinternitzSpec,
	tree_height: usize,
	w: &mut WitnessFiller<'_>,
) {
	let mut rng = StdRng::seed_from_u64(0);

	let mut param_bytes = vec![0u8; spec.domain_param_len];
	rng.fill_bytes(&mut param_bytes);
	let mut message_bytes = [0u8; 32];
	rng.fill_bytes(&mut message_bytes);
	let epoch = rng.next_u32() % (1u32 << tree_height);

	// Grinding the nonce and walking the chains out of circuit is what makes the assignment a
	// signature the circuit accepts rather than arbitrary words.
	let data = ValidatorSignatureData::generate(
		&mut rng,
		&param_bytes,
		&message_bytes,
		epoch,
		spec,
		tree_height,
	);

	// The parameter is padded up to its wire capacity; the rest fill their wires exactly.
	let mut padded_param = vec![0u8; wires.domain_param.len() * 8];
	padded_param[..param_bytes.len()].copy_from_slice(&param_bytes);
	w.pack_bytes_le(&wires.domain_param, &padded_param);
	w.pack_bytes_le(&wires.message, &message_bytes);
	w.pack_bytes_le(&wires.root_hash, &data.root);

	let mut nonce_padded = vec![0u8; NONCE_LENGTH_BYTES];
	nonce_padded[..data.nonce.len()].copy_from_slice(&data.nonce);
	w.pack_bytes_le(&wires.signature.nonce, &nonce_padded);
	w[wires.signature.epoch] = Word::from_u64(epoch as u64);

	for (wire_set, hash) in wires
		.signature
		.signature_hashes
		.iter()
		.zip(&data.signature_hashes)
	{
		w.pack_bytes_le(wire_set, hash);
	}
	for (wire_set, hash) in wires
		.signature
		.public_key_hashes
		.iter()
		.zip(&data.public_key_hashes)
	{
		w.pack_bytes_le(wire_set, hash);
	}
	for (wire_set, node) in wires.signature.auth_path.iter().zip(&data.auth_path) {
		w.pack_bytes_le(wire_set, node);
	}
}

// Proves one XMSS verification through M4 and verifies it.
//
// The Winternitz spec and the tree height are overridable via `WINTERNITZ_SPEC` and
// `XMSS_TREE_HEIGHT`. The spec is the cost driver: it fixes the number of chains and their length,
// and so the chip's instance count.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_hash_based_sig() {
	init_tracing();

	let spec = match usize_from_env("WINTERNITZ_SPEC", 1) {
		1 => WinternitzSpec::spec_1(),
		2 => WinternitzSpec::spec_2(),
		other => panic!("WINTERNITZ_SPEC must be 1 or 2, got {other}"),
	};
	let tree_height = usize_from_env("XMSS_TREE_HEIGHT", 8);

	let (circuit, wires) = build_xmss_circuit(&spec, tree_height);
	circuit
		.validate()
		.expect("the system can be populated in one pass");

	// Report what each sub-system costs. The chip's instance count is what the chain steps
	// produced, and the spare capacity lines show how much of each padded section is wasted.
	debug!("xmss main circuit stats:\n{}", CircuitStat::collect(&circuit.main.circuit));
	for (id, (chip, instances)) in circuit.chips.iter().enumerate() {
		debug!(
			"xmss chip[{id}] over {instances} instances:\n{}",
			CircuitStat::collect(&chip.circuit)
		);
	}

	// Generate the witness in its own span: for this circuit that includes evaluating every
	// BLAKE3 compression, which is the bulk of it.
	let witness = info_span!("witness_generation", primitive = "xmss")
		.in_scope(|| circuit.generate_witness(|w| fill_xmss(&wires, &spec, tree_height, w)))
		.expect("the generated signature satisfies the circuit");

	let cs = circuit.to_constraint_system();
	cs.validate().unwrap();

	// The witness satisfying the system is what says the timing below is of a real proof, not of
	// a batch of chip instances that never had to agree with main.
	witness
		.verify(&cs)
		.expect("the signature verifies in circuit");

	let verifier = VerifierM4::<StdHashSuite>::setup(&cs, LOG_INV_RATE).unwrap();
	let prover = ProverM4::<OptimalPackedB128, StdHashSuite>::setup(&verifier);

	// Warm up first, discarding the proof. This pays the one-time costs — thread-pool spin-up,
	// lazily built tables, page faults on the prover's scratch buffers — so the timed run below
	// measures steady-state proving.
	let mut warmup_transcript = ProverTranscript::new(StdChallenger::default());
	info_span!("prove_warmup", primitive = "xmss")
		.in_scope(|| prover.prove(&witness, &mut warmup_transcript))
		.unwrap();

	// Prove in a span. The main chip's and each chip's sub-proof spans nest beneath it.
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	info_span!("prove", primitive = "xmss")
		.in_scope(|| prover.prove(&witness, &mut prover_transcript))
		.unwrap();

	// The proof must verify, and must leave no trailing transcript data.
	let mut verifier_transcript = prover_transcript.into_verifier();
	verifier
		.verify(witness.main.inout(), &mut verifier_transcript)
		.expect("the proof verifies");
	verifier_transcript
		.finalize()
		.expect("no trailing proof data");
}
