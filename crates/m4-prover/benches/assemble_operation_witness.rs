// Copyright 2026 The Binius Developers
//! Benchmark for assembling one batched operation-witness column from sparse operands.
//!
//! This targets [`OperandColumns::build`] directly. Setup constructs a minimal witness-only
//! [`ValueTable`] and generates random [`Operand`]s with varied sparse row structure: number of
//! shifted value indices, value indices, shift amounts, and shift variants. Only the column
//! assembly is timed.

use binius_compute::BufferPool;
use binius_core::{
	ValueIndex, ValueTable,
	constraint_system::{Operand, Shift, ShiftVariant, ShiftedValueIndex, ZeroConstraint},
	word::Word,
};
use binius_frontend::{CircuitBuilder, Wire};
use binius_m4_prover::OperandColumns;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

/// The base-2 logarithm of the instance count: 2^13 = 8192 instances.
const LOG_INSTANCES: usize = 13;

/// Witness rows available for generated operands to read from.
const N_WITNESS_VALUES: usize = 1024;

/// Constants available for generated operands to read from.
const N_CONSTANTS: usize = 16;

/// Number of operands in each benchmark case.
const N_OPERANDS: usize = 1024;

const SHIFT_VARIANTS: [ShiftVariant; 8] = [
	ShiftVariant::Sll,
	ShiftVariant::Slr,
	ShiftVariant::Sar,
	ShiftVariant::Rotr,
	ShiftVariant::Sll32,
	ShiftVariant::Srl32,
	ShiftVariant::Sra32,
	ShiftVariant::Rotr32,
];

struct OperandCase {
	name: &'static str,
	min_terms: usize,
	max_terms: usize,
	seed: u64,
}

const CASES: [OperandCase; 3] = [
	OperandCase {
		name: "sparse_0_to_2_terms",
		min_terms: 0,
		max_terms: 2,
		seed: 0,
	},
	OperandCase {
		name: "mixed_0_to_4_terms",
		min_terms: 0,
		max_terms: 4,
		seed: 1,
	},
	OperandCase {
		name: "dense_1_to_8_terms",
		min_terms: 1,
		max_terms: 8,
		seed: 2,
	},
];

fn build_table_fixture() -> (ValueTable, Vec<Word>) {
	let builder = CircuitBuilder::new();

	for i in 0..N_CONSTANTS {
		builder.add_constant(fixture_constant_word(i));
	}

	let witnesses: Vec<Wire> = (0..N_WITNESS_VALUES)
		.map(|_| {
			let wire = builder.add_witness();
			builder.force_commit(wire);
			wire
		})
		.collect();

	let circuit = builder.build();
	let table = circuit
		.populate_batch(LOG_INSTANCES, |instance, filler| {
			for (index, &wire) in witnesses.iter().enumerate() {
				filler[wire] = fixture_witness_word(instance, index);
			}
		})
		.unwrap();

	let constants = circuit.constraint_system().constants.clone();
	assert!(constants.len() >= N_CONSTANTS);
	// Every generated private index must name a committed witness row.
	assert!(table.layout().n_witness >= N_WITNESS_VALUES);

	(table, constants)
}

const fn fixture_constant_word(index: usize) -> Word {
	Word((index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93) ^ 0xa076_1d64_78bd_642f)
}

const fn fixture_witness_word(instance: usize, index: usize) -> Word {
	let mixed = (instance as u64)
		.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		.rotate_left((index % Word::BITS) as u32)
		^ (index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	Word(mixed)
}

fn generate_constraints(case: &OperandCase, constants_len: usize) -> (Vec<ZeroConstraint>, usize) {
	let mut rng = StdRng::seed_from_u64(case.seed);
	let constraints = (0..N_OPERANDS)
		.map(|_| ZeroConstraint([random_operand(&mut rng, case, constants_len)]))
		.collect::<Vec<_>>();
	let total_shifted_indices = constraints
		.iter()
		.map(|constraint| constraint.val().len())
		.sum();

	assert!(
		total_shifted_indices > 0,
		"benchmark fixture must have at least one shifted value index"
	);

	(constraints, total_shifted_indices)
}

fn random_operand(rng: &mut impl Rng, case: &OperandCase, constants_len: usize) -> Operand {
	let n_terms = rng.random_range(case.min_terms..=case.max_terms);
	(0..n_terms)
		.map(|_| random_shifted_value_index(rng, constants_len))
		.collect()
}

fn random_shifted_value_index(rng: &mut impl Rng, constants_len: usize) -> ShiftedValueIndex {
	let value_index = random_value_index(rng, constants_len);
	let variant = SHIFT_VARIANTS[rng.random_range(0..SHIFT_VARIANTS.len())];
	let amount = rng.random_range(0..variant.max_amount());

	// A constraint system carries canonical shifts, which spell the identity one way only.
	let shift = if amount == 0 {
		Shift::IDENTITY
	} else {
		Shift::new(variant, amount)
	};

	ShiftedValueIndex::single(value_index, shift)
}

fn random_value_index(rng: &mut impl Rng, constants_len: usize) -> ValueIndex {
	if rng.random_range(0..8) == 0 {
		ValueIndex::constant(rng.random_range(0..constants_len) as u32)
	} else {
		ValueIndex::private(rng.random_range(0..N_WITNESS_VALUES) as u32)
	}
}

fn bench_assemble_operation_witness(c: &mut Criterion) {
	let (table, constants) = build_table_fixture();
	let pool = BufferPool::new();
	let alloc = &pool;

	let mut group = c.benchmark_group("assemble_operation_witness");

	for case in CASES {
		let (constraints, total_shifted_indices) = generate_constraints(&case, constants.len());

		group.throughput(Throughput::Elements(total_shifted_indices as u64));
		group.bench_with_input(
			BenchmarkId::from_parameter(case.name),
			&constraints,
			|b, constraints| {
				b.iter(|| -> OperandColumns<&BufferPool, 1> {
					OperandColumns::<_, 1>::build(&table, &constants, constraints, &alloc)
				});
			},
		);
	}

	group.finish();
}

criterion_group!(benches, bench_assemble_operation_witness);
criterion_main!(benches);
