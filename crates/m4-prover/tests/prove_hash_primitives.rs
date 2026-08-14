// Copyright 2026 The Binius Developers
//! M4 proofs of a batch of BLAKE3 compressions, SHA-256 compressions, Keccak permutations, or
//! 64-bit multiplications.
//!
//! Each test proves one primitive, batched across many M4 instances, and verifies it. The circuits
//! come from [`binius_m4_prover::test_circuits`], which the `prove_hash_primitives` benchmark
//! shares.
//! Each test's batch size is controlled by its own env var (see the `#[test]` doc comments below),
//! expressed in the primitive's natural unit — compressions, permutations, or instances — rather
//! than the raw M4 instance count, since some circuits pack more than one primitive per instance.
//! The proof runs inside a timing span.
//! With tracing enabled, the prover's internal spans nest beneath that span.
//! The per-phase breakdown of proving is then visible.
//!
//! The batched runs exist to be read rather than to gate anything, and each proves a workload
//! sized for a legible timing tree, so they are `#[ignore]`d and need `--ignored` to run. Run them
//! `--release` as well: an unoptimized prover over these batch sizes is slower by orders of
//! magnitude. `prove_integer_multiplication_single_instance` is the exception — it proves a batch
//! of one to cover a degenerate case, costs a fraction of a second, and runs by default.
//!
//! Run one with the timing tree:
//!
//! ```text
//! RUST_LOG=debug cargo test --release -p binius-m4-prover --test prove_hash_primitives \
//!     prove_blake3_compression -- --ignored --nocapture
//! ```

use binius_frontend::CircuitStat;
use binius_hash::StdHashSuite;
use binius_m4_prover::{
	Prover,
	test_circuits::{
		Blake3Circuit, ImulCircuit, KeccakCircuit, Secp256k1AddIncompleteCircuit, Sha256Circuit,
		TestCircuit,
	},
};
use binius_m4_verifier::Verifier;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use tracing::{debug, info_span, level_filters::LevelFilter};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const LOG_INV_RATE: usize = 1;

/// Reads a base-2 logarithm of a size from the environment variable `var`, or `default` if unset.
///
/// Lets each test's instance count be tuned from the command line, e.g.
/// `LOG_BLAKE3_COMPRESSIONS=16 cargo test -p binius-m4-prover --test prove_hash_primitives`.
///
/// # Panics
///
/// Panics if `var` is set but does not parse as a `usize`.
fn log_size_from_env(var: &str, default: usize) -> usize {
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
/// Verbosity defaults to `DEBUG`, the level at which the prover's spans are recorded.
/// `RUST_LOG` overrides that default.
/// A call after a subscriber is already installed does nothing.
fn init_tracing() {
	// Default to DEBUG, the level the prover's spans are recorded at.
	// RUST_LOG overrides this default.
	let env_filter = EnvFilter::builder()
		.with_default_directive(LevelFilter::WARN.into())
		.from_env_lossy();

	// A second call does nothing once a global subscriber is installed.
	let _ = tracing_subscriber::registry()
		.with(env_filter)
		.with(ForestLayer::default())
		.try_init();
}

/// Proves a batch of `2^log_instances` instances of `C`'s circuit through M4 and verifies it.
///
/// Witness generation runs in one span.
/// Proving runs twice: a warmup run whose proof is discarded, then the run that is kept.
/// Each proving run gets its own span, and the prover's internal spans nest beneath it.
///
/// # Panics
///
/// Panics if the witness inputs do not satisfy the circuit.
/// Panics if the proof fails to verify.
fn prove_once<C: TestCircuit>(name: &str, log_instances: usize) {
	init_tracing();

	let test_circuit = C::new();
	let circuit = test_circuit.circuit();

	// Report the circuit's constraint counts and value-vector occupancy.
	// The spare-capacity lines show how much of each padded section is wasted.
	debug!("{name} circuit stats:\n{}", CircuitStat::collect(circuit));

	// Generate the batch witness in its own span.
	let table = info_span!("witness_generation", primitive = name)
		.in_scope(|| circuit.populate_batch_parallel(log_instances, |i, w| test_circuit.fill(i, w)))
		.unwrap();

	// Clone and validate the shared single-instance constraint system.
	let cs = circuit.constraint_system().clone();
	cs.validate().unwrap();

	// Set up the verifier.
	// Build the prover from it, sharing its FRI parameters.
	let verifier = Verifier::setup(&cs, log_instances, LOG_INV_RATE);
	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(&verifier);

	// Warm up first, discarding the proof.
	// This pays the one-time costs — thread-pool spin-up, lazily built tables, page faults on the
	// prover's scratch buffers — so the timed run below measures steady-state proving.
	let mut warmup_transcript = ProverTranscript::new(StdChallenger::default());
	info_span!("prove_warmup", primitive = name)
		.in_scope(|| prover.prove_chip(&table, &mut warmup_transcript));

	// Prove in a span.
	// The prover's commit, reduction, and opening spans nest beneath it.
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	info_span!("prove", primitive = name)
		.in_scope(|| prover.prove_chip(&table, &mut prover_transcript));

	// The proof must verify.
	// It must also leave no trailing transcript data.
	let _scope = info_span!("verify", primitive = name).entered();
	let mut verifier_transcript = prover_transcript.into_verifier();
	verifier
		.verify_chip(&mut verifier_transcript)
		.expect("the proof verifies");
	verifier_transcript
		.finalize()
		.expect("no trailing proof data");
}

// Proves BLAKE3 compressions through M4 and verifies them.
//
// Each instance packs two lanes, so the instance count is half the compression count. Overridable
// via `LOG_BLAKE3_COMPRESSIONS`.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_blake3_compression() {
	let log_compressions = log_size_from_env("LOG_BLAKE3_COMPRESSIONS", 14);
	assert!(log_compressions >= 1, "LOG_BLAKE3_COMPRESSIONS must be at least 1");
	prove_once::<Blake3Circuit<1>>("blake3", log_compressions - 1);
}

// Proves three independent two-lane BLAKE3 compressions per instance through M4 and verifies them,
// like `prove_keccak_permutation_3x` does for Keccak.
//
// Three two-lane compressions per instance pack the value vector more densely than one, so a
// smaller share of the committed trace is padding. The instance count is overridable via
// `LOG_BLAKE3_3X_INSTANCES`; each instance holds 6 compressions.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_blake3_compression_3x() {
	let log_instances = log_size_from_env("LOG_BLAKE3_3X_INSTANCES", 13);
	prove_once::<Blake3Circuit<3>>("blake3_3x", log_instances);
}

// Proves SHA-256 compressions through M4 and verifies them.
//
// Each instance packs two lanes, so the instance count is half the compression count. Overridable
// via `LOG_SHA256_COMPRESSIONS`.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_sha256_compression() {
	let log_compressions = log_size_from_env("LOG_SHA256_COMPRESSIONS", 14);
	assert!(log_compressions >= 1, "LOG_SHA256_COMPRESSIONS must be at least 1");
	prove_once::<Sha256Circuit<1>>("sha256", log_compressions - 1);
}

// Proves three independent two-lane SHA-256 compressions per instance through M4 and verifies them.
//
// Three two-lane compressions per instance pack the value vector more densely than one, so a
// smaller share of the committed trace is padding. The instance count is overridable via
// `LOG_SHA256_3X_INSTANCES`; each instance holds 6 compressions.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_sha256_compression_3x() {
	let log_instances = log_size_from_env("LOG_SHA256_3X_INSTANCES", 13);
	prove_once::<Sha256Circuit<3>>("sha256_3x", log_instances);
}

// Proves Keccak-f1600 permutations through M4 and verifies them.
//
// Overridable via `LOG_KECCAK_PERMUTATIONS`.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_keccak_permutation() {
	let log_permutations = log_size_from_env("LOG_KECCAK_PERMUTATIONS", 13);
	prove_once::<KeccakCircuit<1>>("keccak", log_permutations);
}

// Proves three independent Keccak-f1600 permutations per instance through M4 and verifies them.
//
// Three permutations per instance pack the value vector more densely than one, so a smaller share
// of the committed trace is padding. The instance count is overridable via
// `LOG_KECCAK_3X_INSTANCES`; each instance holds 3 permutations.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_keccak_permutation_3x() {
	let log_instances = log_size_from_env("LOG_KECCAK_3X_INSTANCES", 13);
	prove_once::<KeccakCircuit<3>>("keccak_3x", log_instances);
}

// Proves 64×64→128-bit multiplications through M4 and verifies them.
//
// This is the only primitive here with IMUL constraints, so it covers the IntMul pushforward oracle
// spec that `IOPVerifier::oracle_specs` derives — a wrong spec list would make the shared
// prover/verifier compiler disagree and fail the trace opening. Overridable via
// `LOG_IMUL_INSTANCES`.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_integer_multiplication() {
	let log_instances = log_size_from_env("LOG_IMUL_INSTANCES", 13);
	prove_once::<ImulCircuit>("imul", log_instances);
}

// Proves incomplete secp256k1 point additions through M4 and verifies them.
//
// The heaviest primitive here: one addition is three 256-bit modular multiplications plus a
// modular-division hint, so its circuit dwarfs a hash compression. Overridable via
// `LOG_SECP256K1_ADDITIONS`.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_secp256k1_add_incomplete() {
	let log_additions = log_size_from_env("LOG_SECP256K1_ADDITIONS", 10);
	prove_once::<Secp256k1AddIncompleteCircuit>("secp256k1_add", log_additions);
}

// A batch of one instance, with IMUL constraints.
//
// The re-randomization still runs here, over no rounds, since a batch of one is still a batch.
// Nothing else covers that degenerate sumcheck, on either side.
#[test]
fn prove_integer_multiplication_single_instance() {
	prove_once::<ImulCircuit>("imul-1", 0);
}
