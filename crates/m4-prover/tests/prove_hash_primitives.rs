// Copyright 2026 The Binius Developers
//! Single-instance M4 proofs of a BLAKE3 compression and a Keccak permutation.
//!
//! Each test proves one primitive a single time and verifies it.
//! The proof runs inside a timing span.
//! With tracing enabled, the prover's internal spans nest beneath that span.
//! The per-phase breakdown of proving is then visible.
//!
//! Run one with the timing tree:
//!
//! ```text
//! RUST_LOG=debug cargo test -p binius-m4-prover --test prove_hash_primitives \
//!     prove_blake3_compression -- --nocapture
//! ```

use std::array;

use binius_circuits::{blake3::blake3_compress_2x, keccak::permutation::keccak_f1600};
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::{BatchWitnessFiller, Prover, ValueTable};
use binius_m4_verifier::Verifier;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use rand::prelude::*;
use tracing::{info_span, level_filters::LevelFilter};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Base-2 logarithm of the instance count.
const LOG_INSTANCES: usize = 13;

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const LOG_INV_RATE: usize = 1;

/// The number of 64-bit lanes in a Keccak-f1600 state.
const KECCAK_STATE_LANES: usize = 25;

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
/// Proving runs in another.
/// The prover's internal spans nest beneath the proving span.
///
/// # Panics
///
/// Panics if the witness inputs do not satisfy the circuit.
/// Panics if the proof fails to verify.
fn prove_once<F>(name: &str, circuit: &Circuit, fill: F)
where
	F: Fn(usize, &mut BatchWitnessFiller<'_, '_>),
{
	init_tracing();

	// Generate the single-instance witness in its own span.
	let table = info_span!("witness_generation", primitive = name)
		.in_scope(|| ValueTable::populate(circuit, LOG_INSTANCES, fill).unwrap());

	// Prepare the shared constraint system.
	let mut cs = circuit.constraint_system().clone();
	cs.validate_and_prepare().unwrap();

	// Set up the verifier.
	// Build the prover from it, sharing its FRI parameters.
	let verifier = Verifier::setup(&cs, LOG_INSTANCES, LOG_INV_RATE);
	let prover = Prover::<OptimalPackedB128>::setup(&verifier);

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

/// The witness input wires of one two-lane BLAKE3 compression.
///
/// Each wire packs two independent compressions.
/// Lane 0 sits in the low 32 bits, lane 1 in the high 32.
/// The output is force-committed, so the circuit has no inout wires.
struct Blake3Inputs {
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

/// Builds a circuit for one two-lane BLAKE3 compression and force-commits its output.
fn build_blake3_circuit() -> (Circuit, Blake3Inputs) {
	let builder = CircuitBuilder::new();

	// Every compression input is a witness wire.
	let cv = array::from_fn(|_| builder.add_witness());
	let block = array::from_fn(|_| builder.add_witness());
	let counter_lo = builder.add_witness();
	let counter_hi = builder.add_witness();
	let block_len = builder.add_witness();
	let flags = builder.add_witness();

	// Force-commit each output word.
	// This keeps the compression alive under dead-code elimination.
	let out = blake3_compress_2x(&builder, cv, block, counter_lo, counter_hi, block_len, flags);
	for wire in out {
		builder.force_commit(wire);
	}

	(
		builder.build(),
		Blake3Inputs {
			cv,
			block,
			counter_lo,
			counter_hi,
			block_len,
			flags,
		},
	)
}

/// Packs two independent 32-bit lane values into one 64-bit word.
const fn pack_lanes(lane0: u32, lane1: u32) -> Word {
	Word((lane0 as u64) | ((lane1 as u64) << 32))
}

/// Fills the BLAKE3 instance's inputs with two independent 32-bit lanes per word.
///
/// The compression derives its output from these inputs.
/// So any assignment is valid.
fn fill_blake3(inputs: &Blake3Inputs, _instance: usize, w: &mut BatchWitnessFiller<'_, '_>) {
	let mut rng = StdRng::seed_from_u64(0);

	// A 32-bit value per chaining-value word, per lane.
	for wire in inputs.cv {
		w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
	}
	// A 32-bit value per message word, per lane.
	for wire in inputs.block {
		w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
	}
	// The 64-bit counter, split into low and high halves, per lane.
	w[inputs.counter_lo] = pack_lanes(rng.next_u32(), rng.next_u32());
	w[inputs.counter_hi] = pack_lanes(rng.next_u32(), rng.next_u32());
	// A byte length in 0..=64, per lane.
	w[inputs.block_len] = pack_lanes(rng.next_u32() % 65, rng.next_u32() % 65);
	// Arbitrary domain-separation flags, per lane.
	w[inputs.flags] = pack_lanes(rng.next_u32(), rng.next_u32());
}

/// Builds a circuit for one Keccak-f1600 permutation and force-commits its output state.
fn build_keccak_circuit() -> (Circuit, [Wire; KECCAK_STATE_LANES]) {
	let builder = CircuitBuilder::new();

	// The 25-lane input state is witness input.
	let input: [Wire; KECCAK_STATE_LANES] = array::from_fn(|_| builder.add_witness());

	// Permute a copy of the input in place.
	// After the call, `state` holds the output lanes.
	let mut state = input;
	keccak_f1600(&builder, &mut state);

	// Force-commit the output lanes.
	// This keeps the permutation alive under dead-code elimination.
	for wire in state {
		builder.force_commit(wire);
	}

	(builder.build(), input)
}

/// Fills the Keccak instance's 25 input state lanes with random 64-bit words.
///
/// Keccak proving is data-independent.
/// So any state is valid.
fn fill_keccak(
	input: &[Wire; KECCAK_STATE_LANES],
	_instance: usize,
	w: &mut BatchWitnessFiller<'_, '_>,
) {
	let mut rng = StdRng::seed_from_u64(0);

	// One random 64-bit word per state lane.
	for &wire in input {
		w[wire] = Word(rng.next_u64());
	}
}

/// Builds a circuit for one 64×64→128-bit integer multiplication and force-commits its output.
///
/// Unlike the hash primitives (which are purely bitwise / carry-adder based), this circuit has IMUL
/// constraints, so its M4 proof commits the extra IntMul logup* pushforward oracle — exercising the
/// IMUL branch of `IOPVerifier::oracle_specs`.
fn build_imul_circuit() -> (Circuit, [Wire; 2]) {
	let builder = CircuitBuilder::new();

	// Two 64-bit witness factors.
	let a = builder.add_witness();
	let b = builder.add_witness();

	// Force-commit the 128-bit product halves so the multiplication survives dead-code elimination.
	let (hi, lo) = builder.imul(a, b);
	builder.force_commit(hi);
	builder.force_commit(lo);

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

// Proves one BLAKE3 compression through M4 and verifies it.
#[test]
fn prove_blake3_compression() {
	let (circuit, inputs) = build_blake3_circuit();
	prove_once("blake3", &circuit, |instance, w| fill_blake3(&inputs, instance, w));
}

// Proves one Keccak-f1600 permutation through M4 and verifies it.
#[test]
fn prove_keccak_permutation() {
	let (circuit, input) = build_keccak_circuit();
	prove_once("keccak", &circuit, |instance, w| fill_keccak(&input, instance, w));
}

// Proves one 64×64→128-bit multiplication through M4 and verifies it.
//
// This is the only primitive here with IMUL constraints, so it covers the IntMul pushforward oracle
// spec that `IOPVerifier::oracle_specs` derives — a wrong spec list would make the shared
// prover/verifier compiler disagree and fail the trace opening.
#[test]
fn prove_integer_multiplication() {
	let (circuit, inputs) = build_imul_circuit();
	prove_once("imul", &circuit, |instance, w| fill_imul(&inputs, instance, w));
}
