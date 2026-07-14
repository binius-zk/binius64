// Copyright 2026 The Binius Developers
//! Benchmark for assembling a batched per-operation witness — the operand-column layout an
//! operation reduction consumes.
//!
//! Currently covers the BitAnd operation via [`BatchAndCheckWitness::build`]; the IntMul
//! equivalent will be added alongside its protocol. Uses the Keccak-f1600 permutation circuit
//! (see the `keccak_witness_gen` bench) as a realistic, AND-heavy constraint system: the circuit
//! applies one permutation to a 25-word state, the state words are witness inputs and the permuted
//! outputs are force-committed. Populating the batch table and preparing the constants/constraints
//! are done once as setup; only the witness assembly is timed, over 8192 instances.

use std::array;

use binius_circuits::keccak::permutation::Permutation;
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::{BatchAndCheckWitness, ValueTable};
use criterion::{Criterion, criterion_group, criterion_main};

/// The base-2 logarithm of the instance count: 2^13 = 8192 instances.
const LOG_INSTANCES: usize = 13;

/// The number of 64-bit lanes in a Keccak-f1600 state.
const STATE_LANES: usize = 25;

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

fn bench_assemble_operation_witness(c: &mut Criterion) {
	let (circuit, input) = build_keccak_circuit();

	// Setup (not timed): populate the wire-major batch table for every instance.
	let table = ValueTable::populate(&circuit, LOG_INSTANCES, |instance, w| {
		for lane in 0..STATE_LANES {
			w[input[lane]] = input_word(instance, lane);
		}
	})
	.unwrap();

	// The circuit's constants, shared by every instance.
	let constants = circuit.constraint_system().constants.clone();

	// The per-instance AND constraints, prepared so their count is a power of two (a precondition
	// of `BatchAndCheckWitness::build`).
	let and_constraints = {
		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		cs.and_constraints
	};

	let mut group = c.benchmark_group("assemble_operation_witness");
	group.bench_function("bitand_keccak_f1600", |b| {
		b.iter(|| BatchAndCheckWitness::build(&table, &constants, &and_constraints));
	});

	group.finish();
}

criterion_group!(benches, bench_assemble_operation_witness);
criterion_main!(benches);
