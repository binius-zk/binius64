// Copyright 2026 The Binius Developers
//! Benchmark for assembling a batched per-operation witness — the operand-column layout an
//! operation reduction consumes.
//!
//! This targets [`OperandColumns::build`] at the BitAnd arity, the shape the M4 prover builds from
//! a constraint system's AND constraints. Two fixtures share the benchmark group:
//!
//! * `bitand_keccak_f1600` assembles the real AND constraints of a Keccak-f1600 permutation circuit
//!   (see the `keccak_witness_gen` bench). Operand density, shift mix and wire locality are
//!   whatever the frontend emits, so this is the production-shaped measurement.
//! * The `*_terms` cases assemble synthetic operands of a controlled term density, which a fixed
//!   circuit cannot vary. They read the same table over and over from uniformly drawn rows, with
//!   none of the locality a circuit's program order gives, so their per-word rate is pessimistic.
//!   Read them against each other rather than against the Keccak case.
//!
//! Populating the batch tables and preparing the constants and constraints are setup; only the
//! column assembly is timed, over 8192 instances.

use std::array;

use binius_circuits::keccak::permutation::keccak_f1600;
use binius_compute::BufferPool;
use binius_core::{
	ValueIndex, ValueTable,
	constraint_system::{
		AndConstraint, Composition, Operand, Shift, ShiftVariant, ShiftedValueIndex,
	},
	word::Word,
};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::OperandColumns;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

/// The base-2 logarithm of the instance count: 2^13 = 8192 instances.
const LOG_INSTANCES: usize = 13;

/// The number of 64-bit lanes in a Keccak-f1600 state.
const STATE_LANES: usize = 25;

/// Witness rows available for generated operands to read from.
const N_WITNESS_VALUES: usize = 1024;

/// Constants available for generated operands to read from.
const N_CONSTANTS: usize = 16;

/// Number of constraints in each generated benchmark case.
const N_CONSTRAINTS: usize = 1024;

/// One term in this many names a constant, the rest naming witness rows.
///
/// A constant is splatted from a single word and never touches the table, so the ratio moves the
/// result directly. It is kept low because a circuit's operands mostly name wires.
const CONSTANT_TERM_ODDS: u32 = 8;

/// How the three term classes are drawn, as their share of this many draws.
///
/// Each class takes its own path through the assembly: an unshifted term copies its row, a singly
/// shifted one shifts as it streams, and a doubly shifted one resolves both slots per word. The
/// shares are arbitrary, and only have to keep every path represented.
const TERM_CLASS_DRAWS: u32 = 4;

/// One generated benchmark case: a term density and the seed that generates it.
struct OperandCase {
	name: &'static str,
	min_terms: usize,
	max_terms: usize,
	seed: u64,
}

/// The generated cases, from barely-populated operands to the widest a frontend emits.
///
/// Every operand carries at least one term: an empty one is written by a single fill, which
/// measures memory bandwidth rather than operand assembly.
const CASES: [OperandCase; 3] = [
	OperandCase {
		name: "sparse_1_to_2_terms",
		min_terms: 1,
		max_terms: 2,
		seed: 0,
	},
	OperandCase {
		name: "mixed_1_to_4_terms",
		min_terms: 1,
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

/// Builds a circuit that applies one Keccak-f1600 permutation to a public input state and promotes
/// the permuted lanes to public outputs. Returns the circuit and the 25 input state wires.
fn build_keccak_circuit() -> (Circuit, [Wire; STATE_LANES]) {
	let builder = CircuitBuilder::new();
	let input: [Wire; STATE_LANES] = array::from_fn(|_| builder.add_inout());

	// Permute a copy of the input wires in place; `state` then holds the output wires.
	let mut state = input;
	keccak_f1600(&builder, &mut state);

	// Promoting the permuted state keeps the whole permutation alive under dead-code elimination.
	for wire in state {
		builder.mark_inout(wire);
	}

	(builder.build(), input)
}

/// The Keccak fixture: a populated batch table, the circuit's constants and its AND constraints.
fn build_keccak_fixture() -> (ValueTable, Vec<Word>, Vec<AndConstraint>) {
	let (circuit, input) = build_keccak_circuit();

	let table = circuit
		.populate_batch(LOG_INSTANCES, |instance, filler| {
			for lane in 0..STATE_LANES {
				filler[input[lane]] = keccak_input_word(instance, lane);
			}
		})
		.unwrap();

	let cs = circuit.constraint_system().clone();
	cs.validate().unwrap();

	(table, cs.constants, cs.and_constraints)
}

/// A deterministic, instance- and lane-dependent input word. Keccak's timing is data-independent,
/// so the exact values only need to be non-degenerate.
const fn keccak_input_word(instance: usize, lane: usize) -> Word {
	let mixed = (instance as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
		^ (lane as u64).wrapping_mul(0x0100_0000_01b3);
	Word(mixed)
}

/// The synthetic fixture: a witness-only batch table and the constants the generated terms read.
///
/// The circuit carries no constraints of its own. It exists to allocate the rows the generated
/// value indices name, and to populate them with words the shifts have something to move.
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

/// A distinct constant word per index, so the frontend cannot deduplicate the fixture's constants.
const fn fixture_constant_word(index: usize) -> Word {
	Word((index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93) ^ 0xa076_1d64_78bd_642f)
}

/// A deterministic, instance- and row-dependent witness word.
///
/// Assembly is data-independent, so the words only need to be non-degenerate under a shift.
const fn fixture_witness_word(instance: usize, index: usize) -> Word {
	let mixed = (instance as u64)
		.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		.rotate_left((index % Word::BITS) as u32)
		^ (index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	Word(mixed)
}

/// Generates one case's constraints, at the case's term density.
///
/// Only the first two operands are assembled into columns, so the third is left empty.
fn generate_constraints(case: &OperandCase, constants_len: usize) -> Vec<AndConstraint> {
	let mut rng = StdRng::seed_from_u64(case.seed);
	(0..N_CONSTRAINTS)
		.map(|_| {
			AndConstraint([
				random_operand(&mut rng, case, constants_len),
				random_operand(&mut rng, case, constants_len),
				Operand::default(),
			])
		})
		.collect()
}

/// One operand: a XOR of between `min_terms` and `max_terms` shifted values.
fn random_operand(rng: &mut impl Rng, case: &OperandCase, constants_len: usize) -> Operand {
	let n_terms = rng.random_range(case.min_terms..=case.max_terms);
	(0..n_terms)
		.map(|_| random_shifted_value_index(rng, constants_len))
		.collect()
}

/// One term, drawn across all three classes the assembly distinguishes.
fn random_shifted_value_index(rng: &mut impl Rng, constants_len: usize) -> ShiftedValueIndex {
	let value_index = random_value_index(rng, constants_len);
	match rng.random_range(0..TERM_CLASS_DRAWS) {
		0 => ShiftedValueIndex::plain(value_index),
		1 => ShiftedValueIndex::new(value_index, random_shift_pair(rng)),
		_ => ShiftedValueIndex::single(value_index, random_shift(rng)),
	}
}

/// One value index, naming a constant or a committed witness row.
fn random_value_index(rng: &mut impl Rng, constants_len: usize) -> ValueIndex {
	if rng.random_range(0..CONSTANT_TERM_ODDS) == 0 {
		ValueIndex::constant(rng.random_range(0..constants_len) as u32)
	} else {
		ValueIndex::private(rng.random_range(0..N_WITNESS_VALUES) as u32)
	}
}

/// One shift that moves the word, never the identity.
fn random_shift(rng: &mut impl Rng) -> Shift {
	let variant = ShiftVariant::ALL[rng.random_range(0..ShiftVariant::ALL.len())];
	Shift::new(variant, rng.random_range(1..variant.max_amount()))
}

/// A shift sequence that genuinely needs both slots.
///
/// A pair collapsing to one shift, or clearing the word, is not a term a constraint system carries,
/// so it is redrawn. Two shifts of one variant always compose, which is the bulk of the redraws.
fn random_shift_pair(rng: &mut impl Rng) -> [Shift; 2] {
	loop {
		let inner = random_shift(rng);
		let outer = random_shift(rng);
		if Shift::compose(inner, outer) == Composition::Pair {
			return [inner, outer];
		}
	}
}

/// The number of words a case streams, which is the throughput the assembly is rated on.
///
/// Every term of every assembled column passes over one stripe of instances, so a term costs a
/// stripe rather than a word.
///
/// # Panics
///
/// Panics if the constraints carry no terms at all, which would rate the case on nothing.
fn streamed_words(constraints: &[AndConstraint]) -> u64 {
	let n_terms: usize = constraints
		.iter()
		.map(|constraint| constraint.a().len() + constraint.b().len())
		.sum();
	assert!(n_terms > 0, "benchmark fixture must have at least one shifted value index");
	(n_terms as u64) << LOG_INSTANCES
}

fn bench_assemble_operation_witness(c: &mut Criterion) {
	// The columns are drawn from a pool that lives across the timed iterations, matching how the
	// prover recycles its working buffers between proofs.
	//
	// So this benchmark measures assembly onto recycled blocks, not onto fresh ones: every
	// iteration after the first reuses the blocks its predecessor freed. Comparing it against a
	// revision that allocated a fresh `Vec` per call therefore measures the allocator, not the
	// assembly algorithm — the per-word work is unchanged. Read a delta here as "cost of the
	// allocation strategy", never as an algorithmic speedup.
	let pool = BufferPool::new();
	let alloc = &pool;

	let mut group = c.benchmark_group("assemble_operation_witness");

	let (keccak_table, keccak_constants, keccak_constraints) = build_keccak_fixture();
	group.throughput(Throughput::Elements(streamed_words(&keccak_constraints)));
	group.bench_function("bitand_keccak_f1600", |b| {
		b.iter(|| -> OperandColumns<&BufferPool, 2> {
			OperandColumns::<_, 2>::build(
				&keccak_table,
				&keccak_constants,
				&keccak_constraints,
				&alloc,
			)
		});
	});

	let (table, constants) = build_table_fixture();
	for case in CASES {
		let constraints = generate_constraints(&case, constants.len());

		group.throughput(Throughput::Elements(streamed_words(&constraints)));
		group.bench_with_input(
			BenchmarkId::from_parameter(case.name),
			&constraints,
			|b, constraints| {
				b.iter(|| -> OperandColumns<&BufferPool, 2> {
					OperandColumns::<_, 2>::build(&table, &constants, constraints, &alloc)
				});
			},
		);
	}

	group.finish();
}

criterion_group!(benches, bench_assemble_operation_witness);
criterion_main!(benches);
