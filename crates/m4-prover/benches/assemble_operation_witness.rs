// Copyright 2026 The Binius Developers
//! Benchmark for assembling a batched per-operation witness — the operand-column layout an
//! operation reduction consumes.
//!
//! Covers BitAnd via [`BatchAndCheckWitness::build`], plus the IntMul and BinMul operand-column
//! builders used by the M4 shift reduction. The BitAnd case uses Keccak-f1600 as a realistic,
//! AND-heavy constraint system. The IntMul and BinMul cases use independent operation batches,
//! large enough to exercise the constraint-major assembly path without turning setup into the
//! benchmark. Populating each batch table and preparing the constants/constraints are done once as
//! setup; only witness assembly is timed, over 8192 instances.

use std::array;

use binius_circuits::keccak::permutation::keccak_f1600;
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::{
	BatchAndCheckWitness, ValueTable, build_binmul_witness, build_intmul_witness,
};
use criterion::{Criterion, criterion_group, criterion_main};

/// The base-2 logarithm of the instance count: 2^13 = 8192 instances.
const LOG_INSTANCES: usize = 13;

/// The number of 64-bit lanes in a Keccak-f1600 state.
const STATE_LANES: usize = 25;

/// Independent arithmetic gates per IntMul/BinMul fixture.
const ARITHMETIC_OPS: usize = 256;

/// Builds a circuit that applies one Keccak-f1600 permutation to a witness-input state and
/// force-commits the permuted output words. Returns the circuit and the 25 input state wires.
fn build_keccak_circuit() -> (Circuit, [Wire; STATE_LANES]) {
	let builder = CircuitBuilder::new();
	let input: [Wire; STATE_LANES] = array::from_fn(|_| builder.add_witness());

	// Permute a copy of the input wires in place; `state` then holds the output wires.
	let mut state = input;
	keccak_f1600(&builder, &mut state);

	// Pin the outputs so dead-code elimination keeps the whole permutation.
	for wire in state {
		builder.force_commit(wire);
	}

	(builder.build(), input)
}

/// Builds a circuit with independent unsigned 64x64->128 multiplications.
fn build_intmul_circuit() -> (Circuit, [(Wire, Wire); ARITHMETIC_OPS]) {
	let builder = CircuitBuilder::new();
	let input = array::from_fn(|_| (builder.add_witness(), builder.add_witness()));

	for &(x, y) in &input {
		let (hi, lo) = builder.imul(x, y);
		builder.force_commit(hi);
		builder.force_commit(lo);
	}

	(builder.build(), input)
}

/// Builds a circuit with independent GHASH-field multiplications.
fn build_binmul_circuit() -> (Circuit, [(Wire, Wire, Wire, Wire); ARITHMETIC_OPS]) {
	let builder = CircuitBuilder::new();
	let input = array::from_fn(|_| {
		(
			builder.add_witness(),
			builder.add_witness(),
			builder.add_witness(),
			builder.add_witness(),
		)
	});

	for &(a_lo, a_hi, b_lo, b_hi) in &input {
		let (c_lo, c_hi) = builder.bmul(a_lo, a_hi, b_lo, b_hi);
		builder.force_commit(c_lo);
		builder.force_commit(c_hi);
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

/// A deterministic input word for operation fixtures.
const fn operation_input_word(instance: usize, operation: usize, lane: usize) -> Word {
	let mixed = (instance as u64)
		.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		.rotate_left((lane * 11) as u32)
		^ (operation as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
		^ (lane as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
	Word(mixed)
}

fn bench_assemble_operation_witness(c: &mut Criterion) {
	let mut group = c.benchmark_group("assemble_operation_witness");

	{
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

		// The per-instance AND constraints, prepared so their count is a power of two (a
		// precondition of `BatchAndCheckWitness::build`).
		let and_constraints = {
			let mut cs = circuit.constraint_system().clone();
			cs.validate_and_prepare().unwrap();
			cs.and_constraints
		};

		group.bench_function("bitand_keccak_f1600", |b| {
			b.iter(|| BatchAndCheckWitness::build(&table, &constants, &and_constraints));
		});
	}

	{
		let (circuit, input) = build_intmul_circuit();
		let table = ValueTable::populate(&circuit, LOG_INSTANCES, |instance, w| {
			for (operation, &(x, y)) in input.iter().enumerate() {
				w[x] = operation_input_word(instance, operation, 0);
				w[y] = operation_input_word(instance, operation, 1);
			}
		})
		.unwrap();
		let constants = circuit.constraint_system().constants.clone();
		let imul_constraints = {
			let mut cs = circuit.constraint_system().clone();
			cs.validate_and_prepare().unwrap();
			cs.imul_constraints
		};

		group.bench_function("intmul_256_ops", |b| {
			b.iter(|| build_intmul_witness(&table, &constants, &imul_constraints));
		});
	}

	{
		let (circuit, input) = build_binmul_circuit();
		let table = ValueTable::populate(&circuit, LOG_INSTANCES, |instance, w| {
			for (operation, &(a_lo, a_hi, b_lo, b_hi)) in input.iter().enumerate() {
				w[a_lo] = operation_input_word(instance, operation, 0);
				w[a_hi] = operation_input_word(instance, operation, 1);
				w[b_lo] = operation_input_word(instance, operation, 2);
				w[b_hi] = operation_input_word(instance, operation, 3);
			}
		})
		.unwrap();
		let constants = circuit.constraint_system().constants.clone();
		let bmul_constraints = {
			let mut cs = circuit.constraint_system().clone();
			cs.validate_and_prepare().unwrap();
			cs.bmul_constraints
		};

		group.bench_function("binmul_256_ops", |b| {
			b.iter(|| build_binmul_witness(&table, &constants, &bmul_constraints));
		});
	}

	group.finish();
}

criterion_group!(benches, bench_assemble_operation_witness);
criterion_main!(benches);
