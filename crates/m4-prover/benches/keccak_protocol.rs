// Copyright 2026 The Binius Developers
//! End-to-end M4 proving throughput for independent Keccak-f[1600] permutations.
//!
//! One Keccak-f[1600] permutation runs per instance over a 25-word witness-input state. The whole
//! batch is proved together, so the primitive count is one per instance.
//!
//! Environment overrides:
//! - `LOG_INSTANCES`: base-2 log of the instance count (default 13).
//! - `LOG_INV_RATE`: base-2 log of the inverse Reed-Solomon rate (default 1 = rate 1/2).

#[path = "utils/m4_bench.rs"]
mod m4_bench;

use std::{array, env};

use binius_circuits::keccak::permutation::keccak_f1600;
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::BatchWitnessFiller;
use criterion::{Criterion, criterion_group, criterion_main};
use m4_bench::bench_m4_proving;

/// Base-2 logarithm of the instance count: 2^13 = 8192 permutations.
const DEFAULT_LOG_INSTANCES: usize = 13;

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const DEFAULT_LOG_INV_RATE: usize = 1;

/// Keccak-f[1600] state width in 64-bit lanes.
const STATE_LANES: usize = 25;

/// Permutations computed per instance.
const PERMUTATIONS_PER_INSTANCE: u64 = 1;

/// The witness input wires of one Keccak-f[1600] permutation instance.
///
/// The output is force-committed, so the circuit has no inout wires.
#[derive(Clone, Copy)]
struct KeccakInputs {
	/// The 25-word input state.
	state: [Wire; STATE_LANES],
}

/// Builds a circuit for one Keccak-f[1600] permutation and force-commits its output.
///
/// Force-committing the output keeps the permutation alive under dead-code elimination.
fn build_keccak_circuit() -> (Circuit, KeccakInputs) {
	let builder = CircuitBuilder::new();
	let input = array::from_fn(|_| builder.add_witness());
	let mut state = input;

	keccak_f1600(&builder, &mut state);

	for wire in state {
		builder.force_commit(wire);
	}

	(builder.build(), KeccakInputs { state: input })
}

/// A deterministic, instance- and lane-dependent input word.
///
/// Keccak's circuit shape and timing are data-independent, so these only need to be
/// non-degenerate and reproducible.
const fn input_word(instance: usize, lane: usize) -> Word {
	let mixed = (instance as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
		^ (lane as u64).wrapping_mul(0x0100_0000_01b3);
	Word(mixed)
}

/// Assigns one instance's Keccak input state.
fn fill_instance(inputs: &KeccakInputs, i: usize, w: &mut BatchWitnessFiller<'_, '_>) {
	for lane in 0..STATE_LANES {
		w[inputs.state[lane]] = input_word(i, lane);
	}
}

fn bench_prove_keccak_permutations(c: &mut Criterion) {
	// Batch size and code rate are environment-tunable for sweeping.
	let log_instances = env_usize("LOG_INSTANCES").unwrap_or(DEFAULT_LOG_INSTANCES);
	let log_inv_rate = env_usize("LOG_INV_RATE").unwrap_or(DEFAULT_LOG_INV_RATE);

	// One circuit, replicated across the batch by the shared driver.
	let (circuit, inputs) = build_keccak_circuit();

	bench_m4_proving(
		c,
		"keccak_permutations",
		&circuit,
		log_instances,
		log_inv_rate,
		PERMUTATIONS_PER_INSTANCE,
		|i, w| fill_instance(&inputs, i, w),
	);
}

/// Reads a `usize` environment variable, returning `None` when unset or not a number.
fn env_usize(key: &str) -> Option<usize> {
	env::var(key).ok().and_then(|s| s.parse().ok())
}

criterion_group!(keccak_protocol, bench_prove_keccak_permutations);
criterion_main!(keccak_protocol);
