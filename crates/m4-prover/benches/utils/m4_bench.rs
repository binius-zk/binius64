// Copyright 2026 The Binius Developers
//! Shared end-to-end proving driver for M4 throughput benchmarks.
//!
//! A benchmark supplies one single-instance circuit and a per-instance input filler.
//! The driver replicates that circuit across many independent instances.
//! It then times the batched prover from witness generation through the proof.
//!
//! Witness generation is inside the timed unit.
//! Published Flock figures include it in the proof path, so the measured unit matches.

use binius_frontend::Circuit;
use binius_m4_prover::{BatchWitnessFiller, Prover, ValueTable};
use binius_m4_verifier::Verifier;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use criterion::{BenchmarkId, Criterion, Throughput};

/// Times end-to-end proving of many independent instances of one circuit.
///
/// Setup is built once, outside the timed loop.
/// Before timing, one batch is proved and verified as a correctness gate.
/// The timed closure then regenerates the batch witness and proves it.
///
/// The prover uses the platform-optimal packed field, matching the production prover.
///
/// # Arguments
///
/// - `c`: the Criterion harness.
/// - `group_name`: the benchmark group name.
/// - `circuit`: the single-instance circuit; it must have no inout wires.
/// - `log_instances`: base-2 logarithm of the instance count.
/// - `log_inv_rate`: base-2 logarithm of the inverse Reed-Solomon rate.
/// - `elements_per_instance`: primitives computed by one instance, used only for the throughput
///   count (one instance may pack several primitives, e.g. two lanes of a compression).
/// - `fill`: assigns instance `i`'s witness inputs; it must set every witness input.
///
/// # Panics
///
/// Panics if the circuit has inout wires or IMUL constraints.
/// Panics if the correctness gate fails to verify.
pub fn bench_m4_proving<F>(
	c: &mut Criterion,
	group_name: &str,
	circuit: &Circuit,
	log_instances: usize,
	log_inv_rate: usize,
	elements_per_instance: u64,
	fill: F,
) where
	F: Fn(usize, &mut BatchWitnessFiller<'_, '_>),
{
	// Prepare the shared single-instance constraint system once.
	let mut cs = circuit.constraint_system().clone();
	cs.validate_and_prepare()
		.expect("circuit produces a valid constraint system");

	// The verifier fixes the shape and FRI parameters; the prover inherits them.
	let verifier = Verifier::setup(&cs, log_instances, log_inv_rate);
	let prover = Prover::<OptimalPackedB128>::setup(&verifier);

	// Correctness gate: prove and verify one batch before timing.
	// A bench that measures a proof of nothing is worse than no bench.
	{
		let table = ValueTable::populate_parallel(circuit, log_instances, &fill)
			.expect("witness inputs satisfy the circuit");
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		// Replay the same transcript; a faithful proof verifies and leaves no trailing data.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("the batch proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// The batch proves `elements_per_instance` primitives per instance, over all instances.
	let n_elements = elements_per_instance << log_instances;

	let mut group = c.benchmark_group(group_name);
	// One batch per iteration is heavy, so keep the sample count modest.
	group.sample_size(10);
	group.throughput(Throughput::Elements(n_elements));

	group.bench_function(BenchmarkId::from_parameter(log_instances), |b| {
		b.iter(|| {
			// Regenerate the batch witness, then prove it: the per-batch work of a real prover.
			let table = ValueTable::populate_parallel(circuit, log_instances, &fill).unwrap();
			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			prover.prove(&table, &mut prover_transcript);
			prover_transcript
		});
	});

	group.finish();
}
