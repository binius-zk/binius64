// Copyright 2026 The Binius Developers
//! M4 proofs of a batch of BLAKE3 compressions, Keccak permutations, or 64-bit multiplications.
//!
//! Each test proves one primitive, batched across many M4 instances, and verifies it.
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

use std::array;

use binius_circuits::{blake3::blake3_compress_2x, keccak::permutation::keccak_f1600};
use binius_core::word::Word;
use binius_frontend::{BatchWitnessFiller, Circuit, CircuitBuilder, CircuitStat, Wire};
use binius_m4_prover::Prover;
use binius_m4_verifier::Verifier;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use rand::prelude::*;
use tracing::{debug, info_span, level_filters::LevelFilter};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const LOG_INV_RATE: usize = 1;

/// The number of 64-bit lanes in a Keccak-f1600 state.
const KECCAK_STATE_LANES: usize = 25;

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

/// Proves one instance of `circuit` through M4 and verifies it.
///
/// Witness generation runs in one span.
/// Proving runs twice: a warmup run whose proof is discarded, then the run that is kept.
/// Each proving run gets its own span, and the prover's internal spans nest beneath it.
///
/// # Panics
///
/// Panics if the witness inputs do not satisfy the circuit.
/// Panics if the proof fails to verify.
fn prove_once<F>(name: &str, circuit: &Circuit, log_instances: usize, fill: F)
where
	F: Fn(usize, &mut BatchWitnessFiller<'_, '_>),
{
	init_tracing();

	// Report the circuit's constraint counts and value-vector occupancy.
	// The spare-capacity lines show how much of each padded section is wasted.
	debug!("{name} circuit stats:\n{}", CircuitStat::collect(circuit));

	// Generate the single-instance witness in its own span.
	let table = info_span!("witness_generation", primitive = name)
		.in_scope(|| circuit.populate_batch(log_instances, fill).unwrap());

	// Clone and validate the shared single-instance constraint system.
	let cs = circuit.constraint_system().clone();
	cs.validate().unwrap();

	// Set up the verifier.
	// Build the prover from it, sharing its FRI parameters.
	let verifier = Verifier::setup(&cs, log_instances, LOG_INV_RATE);
	let prover = Prover::<OptimalPackedB128>::setup(&verifier);

	// Warm up first, discarding the proof.
	// This pays the one-time costs — thread-pool spin-up, lazily built tables, page faults on the
	// prover's scratch buffers — so the timed run below measures steady-state proving.
	let mut warmup_transcript = ProverTranscript::new(StdChallenger::default());
	info_span!("prove_warmup", primitive = name)
		.in_scope(|| prover.prove(&table, &mut warmup_transcript));

	// Prove in a span.
	// The prover's commit, reduction, and opening spans nest beneath it.
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	info_span!("prove", primitive = name).in_scope(|| prover.prove(&table, &mut prover_transcript));

	// The proof must verify.
	// It must also leave no trailing transcript data.
	let mut verifier_transcript = prover_transcript.into_verifier();
	verifier
		.verify(&mut verifier_transcript)
		.expect("the proof verifies");
	verifier_transcript
		.finalize()
		.expect("no trailing proof data");
}

/// The public wires of one two-lane BLAKE3 compression.
///
/// Each wire packs two independent compressions.
/// Lane 0 sits in the low 32 bits, lane 1 in the high 32.
struct Blake3Wires {
	/// The 8-word input chaining value, two lanes per word.
	cv: [Wire; 8],
	/// The 16-word message block, two lanes per word.
	block: [Wire; 16],
	/// The low 32 bits of the 64-bit block counter, two lanes.
	counter_lo: Wire,
	/// The high 32 bits of the 64-bit block counter, two lanes.
	counter_hi: Wire,
	/// The block length in bytes, two lanes.
	block_len: Wire,
	/// The domain-separation flags, two lanes.
	flags: Wire,
}

/// Builds a circuit for `N` independent two-lane BLAKE3 compressions.
///
/// Each two-lane compression is itself two independent compressions, so the circuit proves
/// `2 * N` compressions total.
/// Packing several into one circuit fills the value vector more densely, so less of it is padding.
fn build_blake3_circuit<const N: usize>() -> (Circuit, [Blake3Wires; N]) {
	let builder = CircuitBuilder::new();

	// Every compression's inputs are public.
	let wires: [Blake3Wires; N] = array::from_fn(|_| Blake3Wires {
		cv: array::from_fn(|_| builder.add_inout()),
		block: array::from_fn(|_| builder.add_inout()),
		counter_lo: builder.add_inout(),
		counter_hi: builder.add_inout(),
		block_len: builder.add_inout(),
		flags: builder.add_inout(),
	});

	for &Blake3Wires {
		cv,
		block,
		counter_lo,
		counter_hi,
		block_len,
		flags,
	} in &wires
	{
		// Promote each output chaining-value word to a public output.
		// That promotion keeps the compression alive under dead-code elimination.
		let out = blake3_compress_2x(&builder, cv, block, counter_lo, counter_hi, block_len, flags);
		for wire in out {
			builder.mark_inout(wire);
		}
	}

	(builder.build(), wires)
}

/// Packs two independent 32-bit lane values into one 64-bit word.
const fn pack_lanes(lane0: u32, lane1: u32) -> Word {
	Word((lane0 as u64) | ((lane1 as u64) << 32))
}

/// Fills every BLAKE3 instance's inputs with two independent 32-bit lanes per word.
///
/// The compression derives its output from these inputs.
/// So any assignment is valid.
fn fill_blake3<const N: usize>(
	wires: &[Blake3Wires; N],
	_instance: usize,
	w: &mut BatchWitnessFiller<'_, '_>,
) {
	let mut rng = StdRng::seed_from_u64(0);

	for wire_set in wires {
		// A 32-bit value per chaining-value word, per lane.
		for wire in wire_set.cv {
			w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
		}
		// A 32-bit value per message word, per lane.
		for wire in wire_set.block {
			w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
		}
		// The 64-bit counter, split into low and high halves, per lane.
		w[wire_set.counter_lo] = pack_lanes(rng.next_u32(), rng.next_u32());
		w[wire_set.counter_hi] = pack_lanes(rng.next_u32(), rng.next_u32());
		// A byte length in 0..=64, per lane.
		w[wire_set.block_len] = pack_lanes(rng.next_u32() % 65, rng.next_u32() % 65);
		// Arbitrary domain-separation flags, per lane.
		w[wire_set.flags] = pack_lanes(rng.next_u32(), rng.next_u32());
	}
}

/// Builds a circuit for `N` independent Keccak-f1600 permutations.
///
/// The permutations share no wires.
/// Packing several into one circuit fills the value vector more densely, so less of it is padding.
fn build_keccak_circuit<const N: usize>() -> (Circuit, [[Wire; KECCAK_STATE_LANES]; N]) {
	let builder = CircuitBuilder::new();

	// Each permutation gets its own 25-lane public input state.
	let inputs: [[Wire; KECCAK_STATE_LANES]; N] =
		array::from_fn(|_| array::from_fn(|_| builder.add_inout()));

	for input in inputs {
		// Permute a copy of the input in place.
		// After the call, `state` holds the output lanes.
		let mut state = input;
		keccak_f1600(&builder, &mut state);

		// Promote the permuted lanes to public outputs.
		// That promotion keeps the permutation alive under dead-code elimination.
		for wire in state {
			builder.mark_inout(wire);
		}
	}

	(builder.build(), inputs)
}

/// Fills every Keccak input state lane with a random 64-bit word.
///
/// Keccak proving is data-independent.
/// So any state is valid.
fn fill_keccak<const N: usize>(
	inputs: &[[Wire; KECCAK_STATE_LANES]; N],
	_instance: usize,
	w: &mut BatchWitnessFiller<'_, '_>,
) {
	let mut rng = StdRng::seed_from_u64(0);

	// One random 64-bit word per state lane, across all permutations.
	for &wire in inputs.iter().flatten() {
		w[wire] = Word(rng.next_u64());
	}
}

/// Builds a circuit for one 64×64→128-bit integer multiplication.
///
/// Unlike the hash primitives (which are purely bitwise / carry-adder based), this circuit has IMUL
/// constraints, so its M4 proof commits the extra IntMul logup* pushforward oracle — exercising the
/// IMUL branch of `IOPVerifier::oracle_specs`.
fn build_imul_circuit() -> (Circuit, [Wire; 2]) {
	let builder = CircuitBuilder::new();

	// Two 64-bit public factors.
	let a = builder.add_inout();
	let b = builder.add_inout();

	// Promoting the product halves keeps the multiplication alive under dead-code elimination.
	let (hi, lo) = builder.imul(a, b);
	builder.mark_inout(hi);
	builder.mark_inout(lo);

	(builder.build(), [a, b])
}

/// Fills the multiplication instance's two factor wires with random 64-bit words.
///
/// The product is derived from these inputs, so any assignment is valid.
fn fill_imul(inputs: &[Wire; 2], _instance: usize, w: &mut BatchWitnessFiller<'_, '_>) {
	let mut rng = StdRng::seed_from_u64(0);

	for &wire in inputs {
		w[wire] = Word(rng.next_u64());
	}
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
	let log_instances = log_compressions - 1;
	let (circuit, inputs) = build_blake3_circuit::<1>();
	prove_once("blake3", &circuit, log_instances, |instance, w| fill_blake3(&inputs, instance, w));
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
	let (circuit, inputs) = build_blake3_circuit::<3>();
	prove_once("blake3_3x", &circuit, log_instances, |instance, w| {
		fill_blake3(&inputs, instance, w)
	});
}

// Proves Keccak-f1600 permutations through M4 and verifies them.
//
// Overridable via `LOG_KECCAK_PERMUTATIONS`.
#[test]
#[ignore = "proving run for its timing tree; use --ignored --release"]
fn prove_keccak_permutation() {
	let log_permutations = log_size_from_env("LOG_KECCAK_PERMUTATIONS", 13);
	let (circuit, inputs) = build_keccak_circuit::<1>();
	prove_once("keccak", &circuit, log_permutations, |instance, w| {
		fill_keccak(&inputs, instance, w)
	});
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
	let (circuit, inputs) = build_keccak_circuit::<3>();
	prove_once("keccak_3x", &circuit, log_instances, |instance, w| {
		fill_keccak(&inputs, instance, w)
	});
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
	let (circuit, inputs) = build_imul_circuit();
	prove_once("imul", &circuit, log_instances, |instance, w| fill_imul(&inputs, instance, w));
}

// A batch of one instance, with IMUL constraints.
//
// The re-randomization still runs here, over no rounds, since a batch of one is still a batch.
// Nothing else covers that degenerate sumcheck, on either side.
#[test]
fn prove_integer_multiplication_single_instance() {
	let (circuit, inputs) = build_imul_circuit();
	prove_once("imul-1", &circuit, 0, |instance, w| fill_imul(&inputs, instance, w));
}
