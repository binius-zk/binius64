// Copyright 2026 The Binius Developers

use std::array;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Options, Wire};
use criterion::{
	BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
	measurement::WallTime,
};

mod fixtures;

use fixtures::{BLOCK_WORDS, CHAINING_WORDS, STATE_LANES, compression, env_usize, permutation};

/// Permutations the fixture chains.
const DEFAULT_PERMUTATIONS: usize = 30;

/// Compressions the fixture chains.
const DEFAULT_COMPRESSIONS: usize = 50;

/// Repeats the correctness gate compiles.
const GATE_REPEATS: usize = 2;

/// State lanes a sponge absorbs into; the rest are capacity.
const RATE_LANES: usize = 17;

/// Block words a length-padded message leaves zeroed.
const PADDING_WORDS: usize = 6;

/// A fixture's gate graph and the witness wires that drive it.
type Graph = (CircuitBuilder, Vec<Wire>);

/// Builds a sponge: absorb into the rate lanes, permute, repeat.
fn permutation_graph(opts: Options, n_permutations: usize) -> Graph {
	let b = CircuitBuilder::with_opts(opts);
	let mut inputs = Vec::new();

	// The capacity lanes stay zero into the first permutation.
	let mut state = [b.add_constant_64(0); STATE_LANES];

	for _ in 0..n_permutations {
		for lane in state.iter_mut().take(RATE_LANES) {
			let word = b.add_witness();
			inputs.push(word);
			*lane = b.bxor(*lane, word);
		}
		permutation(&b, &mut state);
	}

	// Promoting the final state keeps the chain alive under dead-code elimination.
	for wire in state {
		b.mark_inout(wire);
	}

	(b, inputs)
}

/// Builds a chain of compressions, one length-padded block per compression.
fn compression_graph(opts: Options, n_compressions: usize) -> Graph {
	let b = CircuitBuilder::with_opts(opts);
	let mut inputs = Vec::new();

	let mut chaining: [Wire; CHAINING_WORDS] = array::from_fn(|_| b.add_witness());
	inputs.extend_from_slice(&chaining);

	// The zero tail of a padded block is what zero propagation forwards past.
	let zero = b.add_constant_64(0);
	for _ in 0..n_compressions {
		let block: [Wire; BLOCK_WORDS] = array::from_fn(|i| {
			if i < BLOCK_WORDS - PADDING_WORDS {
				b.add_witness()
			} else {
				zero
			}
		});
		inputs.extend(block.iter().take(BLOCK_WORDS - PADDING_WORDS));
		compression(&b, &mut chaining, &block);
	}

	for wire in chaining {
		b.mark_inout(wire);
	}

	(b, inputs)
}

/// Returns the default pass set with one override applied.
fn with(override_fn: impl FnOnce(&mut Options)) -> Options {
	let mut opts = Options::default();
	override_fn(&mut opts);
	opts
}

/// The pass sets to compare, each named by what it changes from the default.
fn configurations() -> [(&'static str, Options); 9] {
	[
		("default", Options::default()),
		("no_gate_fusion", with(|o| o.enable_gate_fusion = false)),
		("no_zero_propagation", with(|o| o.enable_zero_propagation = false)),
		("no_cse", with(|o| o.enable_common_subexpression_elimination = false)),
		("no_dce", with(|o| o.enable_dead_code_elimination = false)),
		("no_scratch_pooling", with(|o| o.enable_scratch_pooling = false)),
		("no_algebraic_folding", with(|o| o.enable_algebraic_folding = false)),
		("constant_propagation", with(|o| o.enable_constant_propagation = true)),
		("all_off", with(all_off)),
	]
}

/// Turns every optional pass off.
const fn all_off(opts: &mut Options) {
	opts.enable_gate_fusion = false;
	opts.enable_constant_propagation = false;
	opts.enable_common_subexpression_elimination = false;
	opts.enable_dead_code_elimination = false;
	opts.enable_algebraic_folding = false;
	opts.enable_scratch_pooling = false;
	opts.enable_zero_propagation = false;
}

/// A deterministic, position-dependent input word.
const fn input_word(position: usize) -> Word {
	Word((position as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
}

/// Compiles a small instance of the fixture and checks the circuit it produces is satisfiable.
///
/// # Panics
///
/// Panics if the compiled circuit fails to validate, populate or verify.
fn assert_compiles_correctly(graph: fn(Options, usize) -> Graph, opts: Options) {
	let (builder, inputs) = graph(opts, GATE_REPEATS);
	let circuit = builder.build();
	circuit
		.constraint_system()
		.validate()
		.expect("the compiled circuit is a valid constraint system");

	let mut filler = circuit.new_witness_filler();
	for (position, &wire) in inputs.iter().enumerate() {
		filler[wire] = input_word(position);
	}
	circuit
		.populate_wire_witness(&mut filler)
		.expect("the fixture asserts nothing, so any inputs populate");
	circuit
		.constraint_system()
		.verify(&filler.into_value_vec())
		.expect("the populated witness satisfies the circuit");
}

/// Times compilation of one fixture across every pass set.
fn bench_fixture(
	group: &mut BenchmarkGroup<'_, WallTime>,
	name: &str,
	graph: fn(Options, usize) -> Graph,
	n_repeats: usize,
) {
	// The unoptimized build keeps every gate: that count is the compiler's input size.
	let (builder, _) = graph(with(all_off), n_repeats);
	let n_gates = builder.build().n_gates();
	println!("{name}: {n_repeats} repeats, {n_gates} gates");
	group.throughput(Throughput::Elements(n_gates as u64));

	for (config_name, opts) in configurations() {
		assert_compiles_correctly(graph, opts);

		group.bench_function(BenchmarkId::new(name, config_name), |b| {
			// The graph is built in the untimed setup, so this times compilation alone.
			b.iter_batched(
				|| graph(opts, n_repeats),
				|(builder, _)| builder.build(),
				BatchSize::PerIteration,
			);
		});
	}
}

fn bench_build(c: &mut Criterion) {
	let n_permutations = env_usize("N_PERMUTATIONS").unwrap_or(DEFAULT_PERMUTATIONS);
	let n_compressions = env_usize("N_COMPRESSIONS").unwrap_or(DEFAULT_COMPRESSIONS);

	let mut group = c.benchmark_group("circuit_build");
	group.sample_size(10);

	bench_fixture(&mut group, "permutation", permutation_graph, n_permutations);
	bench_fixture(&mut group, "compression", compression_graph, n_compressions);

	group.finish();
}

criterion_group!(build, bench_build);
criterion_main!(build);
