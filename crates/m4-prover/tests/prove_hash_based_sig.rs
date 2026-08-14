// Copyright 2026 The Binius Developers
//! M4 proof of one XMSS verification, compressing through a BLAKE3 chip.
//!
//! The companion to `prove_hash_primitives`, one level up: that file proves a batch of bare
//! primitives, this one proves a whole signature verification — the encoding, the 42 Winternitz
//! chains, the leaf and the 32-level authentication path.
//!
//! Registering the chip is the whole opt-in: `circuit_xmss_verify` reaches `blake3_compress_2x`
//! through the paired chain steps without knowing whether it lands in gates or a chip call. The
//! lone compressions the encoding, the leaf and the path use stay inline, so the chip serves the
//! chain steps alone.
//!
//! The proof runs inside a timing span. With tracing enabled, the prover's internal spans nest
//! beneath that span, so the per-phase breakdown of proving is visible.
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
		DIGEST_WIRES, MESSAGE_LEN, MESSAGE_WIRES, Message, PUBLIC_PARAM_WIRES,
		xmss::{XmssSignatureWires, circuit_xmss_verify, generate_signature},
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
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing::{debug, info_span, level_filters::LevelFilter};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching `prove_hash_primitives`.
const LOG_INV_RATE: usize = 1;

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
	public_param: [Wire; PUBLIC_PARAM_WIRES],
	message: [Wire; MESSAGE_WIRES],
	merkle_root: [Wire; DIGEST_WIRES],
	epoch: Wire,
	signature: XmssSignatureWires,
}

/// Builds a circuit verifying one XMSS signature, with the paired BLAKE3 compression as a chip.
fn build_xmss_circuit() -> (CircuitM4, XmssWires) {
	let builder = CircuitBuilder::new();
	builder.register_chip(Blake3Compress2x, &[]);

	// Everything the verifier is given is public: the statement is the signature and what it
	// signs. Nothing here is a witness, so the committed trace holds only derived values.
	let public_param: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| builder.add_inout());
	let message: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| builder.add_inout());
	let merkle_root: [Wire; DIGEST_WIRES] = std::array::from_fn(|_| builder.add_inout());
	let epoch = builder.add_inout();
	let signature = XmssSignatureWires {
		randomness: std::array::from_fn(|_| builder.add_inout()),
		chain_tips: std::array::from_fn(|_| std::array::from_fn(|_| builder.add_inout())),
		merkle_path: std::array::from_fn(|_| std::array::from_fn(|_| builder.add_inout())),
	};

	circuit_xmss_verify(&builder, &public_param, &merkle_root, &message, epoch, &signature);

	let circuit = builder.build_m4();
	(
		circuit,
		XmssWires {
			public_param,
			message,
			merkle_root,
			epoch,
			signature,
		},
	)
}

/// Generates a valid signature and writes it into the circuit's public wires.
///
/// Every digest the circuit checks is derived from these inputs, so the evaluator fills the rest.
fn fill_xmss(wires: &XmssWires, w: &mut WitnessFiller<'_>) {
	let mut rng = StdRng::seed_from_u64(0);

	let mut message: Message = [0u8; MESSAGE_LEN];
	rng.fill_bytes(&mut message);
	let mut epoch_bytes = [0u8; 4];
	rng.fill_bytes(&mut epoch_bytes);
	let epoch = u32::from_le_bytes(epoch_bytes);

	// Grinding the randomness and walking the chains out of circuit is what makes the assignment
	// a signature the circuit accepts rather than arbitrary words.
	let (public_key, signature) = generate_signature(&mut rng, &message, epoch);

	w.pack_bytes_le(&wires.public_param, &public_key.public_param);
	w.pack_bytes_le(&wires.message, &message);
	w.pack_bytes_le(&wires.merkle_root, &public_key.merkle_root);
	w[wires.epoch] = Word::from_u64(epoch as u64);
	wires.signature.populate(w, &signature);
}

// Proves one XMSS verification through M4 and verifies it.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_hash_based_sig() {
	init_tracing();

	let (circuit, wires) = build_xmss_circuit();
	circuit
		.validate()
		.expect("the system can be populated in one pass");

	// Report what each sub-system costs. The chip's instance count is what the chain steps
	// produced, and the spare-capacity lines show how much of each padded section is wasted.
	debug!("xmss main circuit stats:\n{}", CircuitStat::collect(&circuit.main.circuit));
	for (id, (chip, instances)) in circuit.chips.iter().enumerate() {
		debug!(
			"xmss chip[{id}] over {instances} instances:\n{}",
			CircuitStat::collect(&chip.circuit)
		);
	}

	// Generate the witness in its own span: for this circuit that is every BLAKE3 compression,
	// which is the bulk of it.
	let witness = info_span!("witness_generation", primitive = "xmss")
		.in_scope(|| circuit.generate_witness(|w| fill_xmss(&wires, w)))
		.expect("the generated signature satisfies the circuit");

	let cs = circuit.to_constraint_system();
	cs.validate().unwrap();

	// The witness satisfying the system is what says the timing below is of a real proof.
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

	// Prove in a span, so the prover's phases nest beneath it.
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
