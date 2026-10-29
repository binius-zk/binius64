// Copyright 2026 The Binius Developers

use std::{array, num::NonZero, thread::available_parallelism};

use binius_compute::BufferPool;
use binius_core::word::Word;
use binius_frontend::{BatchWitnessFiller, Circuit, CircuitBuilder, Wire};
use binius_utils::rayon::ThreadPoolBuilder;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

mod fixtures;

use fixtures::{STATE_LANES, env_usize, permutation};

/// Base-2 logarithm of the instance count the batched paths fill.
const DEFAULT_LOG_INSTANCES: usize = 13;

/// A second, smaller batch size.
const SMALL_LOG_INSTANCES: usize = 10;

/// Thread counts the parallel path is measured at, below the machine's core count.
const THREAD_STEPS: [usize; 3] = [1, 4, 16];

/// Tile sizes the sweep compares.
const TILE_SIZES: [usize; 6] = [64, 128, 256, 512, 1024, 2048];

/// Base-2 logarithm of the instance count the correctness gate fills.
const GATE_LOG_INSTANCES: usize = 8;

/// Tile size the correctness gate uses, small enough to split its batch across several tiles.
const GATE_TILE_SIZE: usize = 64;

/// The thread counts to measure: the fixed steps that fit, then every core.
fn thread_counts() -> Vec<usize> {
	let cores = available_parallelism().map_or(1, NonZero::get);
	let mut counts: Vec<usize> = THREAD_STEPS.into_iter().filter(|&n| n < cores).collect();
	counts.push(cores);
	counts
}

/// Builds a circuit applying one permutation to a private input state, with a public output state.
fn build_fixture() -> (Circuit, [Wire; STATE_LANES]) {
	let b = CircuitBuilder::new();
	let input: [Wire; STATE_LANES] = array::from_fn(|_| b.add_witness());

	let mut state = input;
	permutation(&b, &mut state);

	// Promoting the permuted state keeps the permutation alive under dead-code elimination.
	for wire in state {
		b.mark_inout(wire);
	}

	(b.build(), input)
}

/// A deterministic, instance- and lane-dependent input word.
const fn input_word(instance: usize, lane: usize) -> Word {
	Word((instance as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (lane as u64).wrapping_mul(0x1b3))
}

fn bench_witness(c: &mut Criterion) {
	let log_instances = env_usize("LOG_INSTANCES").unwrap_or(DEFAULT_LOG_INSTANCES);

	let (circuit, input) = build_fixture();
	let n_eval_insn = circuit.n_eval_insn();
	println!("permutation: {} gates, {n_eval_insn} instructions", circuit.n_gates());

	let fill = |instance: usize, w: &mut BatchWitnessFiller<'_, '_>| {
		for lane in 0..STATE_LANES {
			w[input[lane]] = input_word(instance, lane);
		}
	};

	let pool = BufferPool::new();
	let alloc = &pool;

	// Correctness gate.
	{
		let serial = circuit
			.populate_batch(&alloc, GATE_LOG_INSTANCES, fill)
			.expect("the fixture asserts nothing, so any inputs populate");
		let parallel = circuit
			.populate_batch_parallel_with_stripe_width(
				&alloc,
				GATE_LOG_INSTANCES,
				GATE_TILE_SIZE,
				fill,
			)
			.expect("the fixture asserts nothing, so any inputs populate");
		assert_eq!(serial.as_words(), parallel.as_words(), "tiling changed the batch witness");

		let constants = &circuit.constraint_system().constants;
		for instance in 0..serial.n_instances() {
			let values = serial.instance_value_vec(instance, constants);
			circuit
				.constraint_system()
				.verify(&values)
				.unwrap_or_else(|e| panic!("instance {instance} failed verification: {e}"));
		}
	}

	let mut group = c.benchmark_group("witness");
	group.sample_size(10);

	group.throughput(Throughput::Elements(n_eval_insn as u64));
	group.bench_function("single_instance", |b| {
		// The value vector is allocated and filled outside the timed loop.
		let mut filler = circuit.new_witness_filler();
		for lane in 0..STATE_LANES {
			filler[input[lane]] = input_word(0, lane);
		}
		b.iter(|| circuit.populate_wire_witness(&mut filler).unwrap());
	});

	// Throughput counts word operations: one instruction per instance.
	group.throughput(Throughput::Elements((n_eval_insn << log_instances) as u64));
	group.bench_function(BenchmarkId::new("batch_serial", log_instances), |b| {
		b.iter(|| circuit.populate_batch(&alloc, log_instances, fill).unwrap());
	});

	// The parallel path's gain depends on the thread count, so each one is its own measurement.
	for n_threads in thread_counts() {
		let thread_pool = ThreadPoolBuilder::new()
			.num_threads(n_threads)
			.build()
			.expect("the thread count is positive");
		let name = format!("batch_parallel/threads={n_threads}");

		for log in [SMALL_LOG_INSTANCES, log_instances] {
			group.throughput(Throughput::Elements((n_eval_insn << log) as u64));
			group.bench_function(BenchmarkId::new(&name, log), |b| {
				thread_pool.install(|| {
					b.iter(|| circuit.populate_batch_parallel(&alloc, log, fill).unwrap());
				});
			});
		}
	}

	group.throughput(Throughput::Elements((n_eval_insn << log_instances) as u64));
	for tile_size in TILE_SIZES {
		group.bench_function(BenchmarkId::new("batch_parallel_tile", tile_size), |b| {
			b.iter(|| {
				circuit
					.populate_batch_parallel_with_stripe_width(
						&alloc,
						log_instances,
						tile_size,
						fill,
					)
					.unwrap()
			});
		});
	}

	group.finish();
}

criterion_group!(witness, bench_witness);
criterion_main!(witness);
