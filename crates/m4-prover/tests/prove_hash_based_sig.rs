// Copyright 2026 The Binius Developers
//! M4 proof of a batch of XMSS verifications, each dispatched to a chip.
//!
//! The companion to `prove_hash_primitives`, one level up: that file proves a batch of bare
//! primitives, this one proves whole signature verifications — the encoding, the 42 Winternitz
//! chains, the leaf and the 32-level authentication path, once per signer.
//!
//! The circuit is two chips deep. The main circuit holds no hash gates at all: it declares the
//! statement — the message, the epoch and every signer's public key — witnesses the signatures, and
//! dispatches one call per signer to the XMSS chip. That chip in turn reaches `blake3_compress_2x`
//! through the paired chain steps as a chip of its own. So each verification's constraints are
//! stated once and the trace pays for them once per instance, at both levels.
//!
//! The proof runs inside a timing span. With tracing enabled, the prover's internal spans nest
//! beneath that span, so the per-phase breakdown of proving is visible.
//!
//! The run exists to be read rather than to gate anything, so it is `#[ignore]`d and needs
//! `--ignored` to run. Run it `--release` as well: an unoptimized prover over this circuit is
//! slower by orders of magnitude.
//!
//! Run it with the timing tree. Signing, witness generation and proving all parallelize behind the
//! `rayon` feature, so pass it or read single-threaded times:
//!
//! ```text
//! RUST_LOG=debug cargo test --release -p binius-m4-prover --test prove_hash_based_sig \
//!     --features rayon -- --ignored --nocapture
//! ```

use binius_circuits::hash_based_sig::{
	MESSAGE_LEN, Message,
	aggregate::{MultiSigWires, circuit_xmss_multisig_chip},
	xmss::{XmssPublicKey, XmssSignature, generate_signature},
};
use binius_frontend::{CircuitBuilder, CircuitM4, CircuitStat};
use binius_hash::StdHashSuite;
use binius_m4_prover::ProverM4;
use binius_m4_verifier::VerifierM4;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_utils::rayon::prelude::*;
use binius_verifier::config::StdChallenger;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use tracing::{debug, info_span, level_filters::LevelFilter};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching `prove_hash_primitives`.
const LOG_INV_RATE: usize = 1;

/// The number of signatures the batch verifies.
///
/// A power of two fills the XMSS chip's instances exactly. The BLAKE3 chip beneath it takes a
/// number of calls per signature that is not a power of two, so its own count pads whatever this
/// is set to — filling both at once is not on offer. The count is what makes the run a batch
/// rather than a single verification; it trades proving time for a timing tree whose per-phase
/// costs are legible.
const NUM_SIGNERS: usize = 512;

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

/// Builds a circuit verifying `NUM_SIGNERS` XMSS signatures on one message, each through the chip.
fn build_multisig_circuit() -> (CircuitM4, MultiSigWires) {
	let builder = CircuitBuilder::new();
	let wires = MultiSigWires::new(&builder, NUM_SIGNERS);
	circuit_xmss_multisig_chip(&builder, &wires);
	(builder.build_m4(), wires)
}

/// Generates the message and epoch the batch is over, and one valid signature per signer.
///
/// Grinding the randomness and walking the chains out of circuit is what makes each assignment a
/// signature the chip accepts rather than arbitrary words. The grind is what the work is: the
/// target sum sits far enough above the mean digit sum that a signature costs tens of thousands of
/// encoding hashes. The signers are independent, so each draws its own seed and they run in
/// parallel; only the message and the epoch are shared.
fn generate_statement() -> (Message, u32, Vec<(XmssPublicKey, XmssSignature)>) {
	let mut rng = StdRng::seed_from_u64(0);

	let mut message: Message = [0u8; MESSAGE_LEN];
	rng.fill_bytes(&mut message);
	let mut epoch_bytes = [0u8; 4];
	rng.fill_bytes(&mut epoch_bytes);
	let epoch = u32::from_le_bytes(epoch_bytes);

	// Drawing the seeds in order keeps the batch a function of the one seed above, whatever order
	// the signers then run in.
	let seeds = (0..NUM_SIGNERS)
		.map(|_| rng.random::<u64>())
		.collect::<Vec<_>>();
	let signatures = seeds
		.into_par_iter()
		.map(|seed| generate_signature(&mut StdRng::seed_from_u64(seed), &message, epoch))
		.collect();

	(message, epoch, signatures)
}

// Proves a batch of XMSS verifications through M4 and verifies it.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_hash_based_sig() {
	init_tracing();

	let (circuit, wires) = build_multisig_circuit();
	circuit
		.validate()
		.expect("the system can be populated in one pass");

	// Report what each sub-system costs. Each chip's instance count is what its callers produced,
	// and the spare-capacity lines show how much of each padded section is wasted.
	debug!("xmss main circuit stats:\n{}", CircuitStat::collect(&circuit.main.circuit));
	for (id, (chip, instances)) in circuit.chips.iter().enumerate() {
		debug!(
			"xmss chip[{id}] over {instances} instances:\n{}",
			CircuitStat::collect(&chip.circuit)
		);
	}

	// Signing is the test's own work rather than the circuit's, and at this batch size it costs
	// more than proving does, so it is timed on its own and outside the span below.
	let (message, epoch, signatures) =
		info_span!("sign", primitive = "xmss").in_scope(generate_statement);

	// Generate the witness in its own span: for this circuit that is every BLAKE3 compression,
	// which is the bulk of it.
	let witness = info_span!("witness_generation", primitive = "xmss")
		.in_scope(|| circuit.generate_witness(|w| wires.populate(w, &message, epoch, &signatures)))
		.expect("the generated signatures satisfy the circuit");

	let cs = circuit.to_constraint_system();
	cs.validate().unwrap();

	// The witness satisfying the system is what says the timing below is of a real proof.
	witness
		.verify(&cs)
		.expect("the signatures verify in circuit");

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
	let _scope = info_span!("verify", primitive = "xmss").entered();
	verifier
		.verify(witness.main.inout(), &mut verifier_transcript)
		.expect("the proof verifies");
	verifier_transcript
		.finalize()
		.expect("no trailing proof data");
}
