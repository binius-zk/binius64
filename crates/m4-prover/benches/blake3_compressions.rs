// Copyright 2026 The Binius Developers
//! End-to-end M4 proving throughput for independent BLAKE3 compressions.
//!
//! One two-lane [`blake3_compress_2x`] runs per instance, computing two independent compressions
//! from the two 32-bit lanes of each 64-bit word. The whole batch is proved together, so the
//! primitive count is two per instance. Reusing the two-lane core matches the efficient
//! representation the signature circuits use.
//!
//! Environment overrides:
//! - `LOG_INSTANCES`: base-2 log of the instance count (default 13); the compression count is twice
//!   that.
//! - `LOG_INV_RATE`: base-2 log of the inverse Reed-Solomon rate (default 1 = rate 1/2).

#[path = "utils/m4_bench.rs"]
mod m4_bench;

use std::{array, env};

use binius_circuits::blake3::blake3_compress_2x;
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::BatchWitnessFiller;
use criterion::{Criterion, criterion_group, criterion_main};
use m4_bench::bench_m4_proving;
use rand::prelude::*;

/// Base-2 logarithm of the instance count: 2^13 = 8192 instances = 16384 compressions.
const DEFAULT_LOG_INSTANCES: usize = 13;

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const DEFAULT_LOG_INV_RATE: usize = 1;

/// Compressions computed per instance: the two-lane core runs two at once.
const COMPRESSIONS_PER_INSTANCE: u64 = 2;

/// The witness input wires of one two-lane BLAKE3 compression instance.
///
/// Each wire packs two independent compressions: lane 0 in the low 32 bits, lane 1 in the high 32
/// bits. The output is force-committed, so the circuit has no inout wires.
#[derive(Clone, Copy)]
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
///
/// Force-committing the output keeps the compression alive under dead-code elimination.
/// The circuit has no inout wires, as the batch witness table requires.
fn build_blake3_circuit() -> (Circuit, Blake3Inputs) {
	let builder = CircuitBuilder::new();

	// Every compression input is a witness wire, filled per instance.
	let cv = array::from_fn(|_| builder.add_witness());
	let block = array::from_fn(|_| builder.add_witness());
	let counter_lo = builder.add_witness();
	let counter_hi = builder.add_witness();
	let block_len = builder.add_witness();
	let flags = builder.add_witness();

	// Force-commit each output word so the compression survives dead-code elimination.
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

/// Assigns one instance's two-lane BLAKE3 inputs from a per-instance seeded RNG.
///
/// Each 64-bit word carries two independent 32-bit lanes, matching the two-lane core.
/// The compression derives its output from these inputs, so any assignment is valid.
/// Seeding per instance keeps the data non-degenerate and reproducible.
fn fill_instance(inputs: &Blake3Inputs, i: usize, w: &mut BatchWitnessFiller<'_, '_>) {
	// Seed from the instance index so the batch is deterministic and instance-varying.
	let mut rng = StdRng::seed_from_u64(i as u64);

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

fn bench_prove_blake3_compressions(c: &mut Criterion) {
	// Batch size and code rate are environment-tunable for sweeping.
	let log_instances = env_usize("LOG_INSTANCES").unwrap_or(DEFAULT_LOG_INSTANCES);
	let log_inv_rate = env_usize("LOG_INV_RATE").unwrap_or(DEFAULT_LOG_INV_RATE);

	// One circuit, replicated across the batch by the shared driver.
	let (circuit, inputs) = build_blake3_circuit();

	bench_m4_proving(
		c,
		"blake3_compressions",
		&circuit,
		log_instances,
		log_inv_rate,
		COMPRESSIONS_PER_INSTANCE,
		|i, w| fill_instance(&inputs, i, w),
	);
}

/// Reads a `usize` environment variable, returning `None` when unset or not a number.
fn env_usize(key: &str) -> Option<usize> {
	env::var(key).ok().and_then(|s| s.parse().ok())
}

criterion_group!(blake3_compressions, bench_prove_blake3_compressions);
criterion_main!(blake3_compressions);
