// Copyright 2026 The Binius Developers
//! Witness-generation benchmark for [`ValueTable`], serial and parallel.
//!
//! The circuit applies one Keccak-f1600 permutation to a 25-word state. The state words are
//! witness inputs and the permuted output words are force-committed, so the circuit has no inout
//! wires. Witness generation is timed for 8192 instances.

use std::array;

use binius_circuits::keccak::permutation::Permutation;
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::ValueTable;
use criterion::{Criterion, criterion_group, criterion_main};

/// The base-2 logarithm of the instance count: 2^13 = 8192 instances.
const LOG_INSTANCES: usize = 13;

/// The number of 64-bit lanes in a Keccak-f1600 state.
const STATE_LANES: usize = 25;

/// Candidate instance-stripe widths for parallel [`ValueTable`] witness generation.
const STRIPE_WIDTHS: [usize; 3] = [256, 512, 1024];

/// Builds a circuit that applies one Keccak-f1600 permutation to a witness-input state and
/// force-commits the permuted output words. Returns the circuit and the 25 input state wires.
fn build_keccak_circuit() -> (Circuit, [Wire; STATE_LANES]) {
	let builder = CircuitBuilder::new();
	let input: [Wire; STATE_LANES] = array::from_fn(|_| builder.add_witness());

	// Permute a copy of the input wires in place; `state` then holds the output wires.
	let mut state = input;
	Permutation::keccak_f1600(&builder, &mut state);

	// Pin the outputs so dead-code elimination keeps the whole permutation.
	for wire in state {
		builder.force_commit(wire);
	}

	(builder.build(), input)
}

/// A deterministic, instance- and lane-dependent input word. Keccak's timing is data-independent,
/// so the exact values only need to be non-degenerate.
const fn input_word(instance: usize, lane: usize) -> Word {
	let mixed = (instance as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
		^ (lane as u64).wrapping_mul(0x0100_0000_01b3);
	Word(mixed)
}

fn bench_keccak_witness_gen(c: &mut Criterion) {
	let (circuit, input) = build_keccak_circuit();

	let mut group = c.benchmark_group("keccak_f1600_witness_gen_8k");
	// Each iteration generates the witness for 8192 instances, so keep the sample count modest.
	group.sample_size(10);

	group.bench_function("value_table", |b| {
		b.iter(|| {
			ValueTable::populate(&circuit, LOG_INSTANCES, |instance, w| {
				for lane in 0..STATE_LANES {
					w[input[lane]] = input_word(instance, lane);
				}
			})
			.unwrap()
		});
	});

	for stripe_width in STRIPE_WIDTHS {
		group.bench_function(format!("value_table_parallel_{stripe_width}"), |b| {
			b.iter(|| {
				ValueTable::populate_parallel_with_stripe_width(
					&circuit,
					LOG_INSTANCES,
					stripe_width,
					|instance, w| {
						for lane in 0..STATE_LANES {
							w[input[lane]] = input_word(instance, lane);
						}
					},
				)
				.unwrap()
			});
		});
	}

	group.finish();
}

criterion_group!(benches, bench_keccak_witness_gen);
criterion_main!(benches);
