// Copyright 2026 The Binius Developers
//! End-to-end M4 proving throughput for batches of independent hash primitives.
//!
//! One benchmark runs per primitive, each proving a batch of instances of a single-instance circuit
//! from [`binius_m4_prover::test_circuits`], which the `prove_hash_primitives` timing tests share.
//! The `_3x` variants pack three primitives into one instance instead of one, filling the value
//! vector more densely, so a smaller share of the committed trace is padding. Throughput is
//! reported in primitives, not instances: the BLAKE3 and SHA-256 cores each compute two
//! independent compressions from the two 32-bit lanes of a 64-bit word.
//!
//! Witness generation is inside the timed unit. Published Flock figures include it in the proof
//! path, so the measured unit matches.
//!
//! Environment overrides:
//! - `LOG_INSTANCES`: base-2 log of the instance count (default 13).
//! - `LOG_INV_RATE`: base-2 log of the inverse Reed-Solomon rate (default 1 = rate 1/2).

use std::env;

use binius_frontend::BatchWitnessFiller;
use binius_hash::StdHashSuite;
use binius_m4_prover::{
	Prover,
	test_circuits::{Blake3Circuit, KeccakCircuit, Sha256Circuit, TestCircuit},
};
use binius_m4_verifier::Verifier;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use criterion::{
	BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
	measurement::WallTime,
};

/// Base-2 logarithm of the instance count: 2^13 = 8192 instances.
const DEFAULT_LOG_INSTANCES: usize = 13;

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const DEFAULT_LOG_INV_RATE: usize = 1;

/// Times end-to-end proving of a batch of independent instances of one primitive's circuit.
///
/// Setup is built once, outside the timed loop. Before timing, one batch is proved and verified as
/// a correctness gate. The timed closure then regenerates the batch witness and proves it.
///
/// The prover uses the platform-optimal packed field, matching the production prover.
///
/// # Panics
///
/// Panics if the correctness gate fails to verify.
fn bench_primitive<C: TestCircuit>(
	group: &mut BenchmarkGroup<'_, WallTime>,
	name: &str,
	log_instances: usize,
	log_inv_rate: usize,
) {
	// One circuit, replicated across the batch below.
	let test_circuit = C::new();
	let circuit = test_circuit.circuit();
	let fill = |i: usize, w: &mut BatchWitnessFiller<'_, '_>| test_circuit.fill(i, w);

	// Clone and validate the shared single-instance constraint system once.
	let cs = circuit.constraint_system().clone();
	cs.validate()
		.expect("circuit produces a valid constraint system");

	// The verifier fixes the shape and FRI parameters; the prover inherits them.
	let verifier = Verifier::setup(&cs, log_instances, log_inv_rate);
	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(&verifier);

	// Correctness gate: prove and verify one batch before timing.
	// A bench that measures a proof of nothing is worse than no bench.
	{
		let table = circuit
			.populate_batch_parallel(log_instances, fill)
			.expect("witness inputs satisfy the circuit");
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		// Replay the same transcript; a faithful proof verifies and leaves no trailing data.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
			.expect("the batch proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// The batch proves this many primitives per instance, over all instances.
	group.throughput(Throughput::Elements(C::PRIMITIVES_PER_INSTANCE << log_instances));

	group.bench_function(BenchmarkId::new(name, log_instances), |b| {
		b.iter(|| {
			// Regenerate the batch witness, then prove it: the per-batch work of a real prover.
			let table = circuit
				.populate_batch_parallel(log_instances, fill)
				.unwrap();
			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			prover.prove_chip(&table, &mut prover_transcript);
			prover_transcript
		});
	});
}

fn bench_prove_hash_primitives(c: &mut Criterion) {
	// Batch size and code rate are environment-tunable for sweeping.
	let log_instances = env_usize("LOG_INSTANCES").unwrap_or(DEFAULT_LOG_INSTANCES);
	let log_inv_rate = env_usize("LOG_INV_RATE").unwrap_or(DEFAULT_LOG_INV_RATE);

	let mut group = c.benchmark_group("prove_hash_primitives");
	// One batch per iteration is heavy, so keep the sample count modest.
	group.sample_size(10);

	bench_primitive::<Blake3Circuit<1>>(&mut group, "blake3", log_instances, log_inv_rate);
	bench_primitive::<Blake3Circuit<3>>(&mut group, "blake3_3x", log_instances, log_inv_rate);
	bench_primitive::<Sha256Circuit<1>>(&mut group, "sha256", log_instances, log_inv_rate);
	bench_primitive::<Sha256Circuit<3>>(&mut group, "sha256_3x", log_instances, log_inv_rate);
	bench_primitive::<KeccakCircuit<1>>(&mut group, "keccak", log_instances, log_inv_rate);
	bench_primitive::<KeccakCircuit<3>>(&mut group, "keccak_3x", log_instances, log_inv_rate);

	group.finish();
}

/// Reads a `usize` environment variable, returning `None` when unset or not a number.
fn env_usize(key: &str) -> Option<usize> {
	env::var(key).ok().and_then(|s| s.parse().ok())
}

criterion_group!(prove_hash_primitives, bench_prove_hash_primitives);
criterion_main!(prove_hash_primitives);
