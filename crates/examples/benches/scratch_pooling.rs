// Copyright 2026 The Binius Developers
//! Before and after sharing scratch slots, measured on the BLAKE3 circuit.
//!
//! Gate fusion inlines a linear operation into its consumers and leaves its result uncommitted.
//! Without sharing, each such result holds a value-vector slot for the whole run.
//! With sharing, the segment shrinks to the largest number of them alive at once.
//!
//! Two things are reported:
//! - the per-instance value-vector length, and what a batch of them costs in memory,
//! - the wall time of filling a witness, one instance at a time and in a batch.
//!
//! # Why the timings should not move
//!
//! The saving is memory, not time.
//!
//! - Sharing changes how many distinct addresses the writes land on, not how many writes there are.
//! - Either way there is one store per gate per instance.
//! - Both evaluation paths walk one row at a time, whatever the buffer's total height.
//! - So the working set per instruction is one row wide.
//!
//! The timing groups exist to confirm that a shorter value vector costs no throughput.

use binius_circuits::blake3::CHUNK_BYTES;
use binius_core::Word;
use binius_examples::{
	ExampleCircuit,
	circuits::{
		blake3::Blake3Example,
		utils::{HasherInstance, HasherParams},
	},
};
use binius_frontend::{Circuit, CircuitBuilder};
use binius_utils::strided_array::StridedArray2DViewMut;
use criterion::{Criterion, criterion_group, criterion_main};

/// Message length the circuit is built for.
///
/// BLAKE3's fixed gadget is single-chunk, so one full chunk is the largest circuit it builds.
const MESSAGE_BYTES: usize = CHUNK_BYTES;

/// Instance count used for the batched figures.
///
/// Large enough that both buffers are hundreds of megabytes.
/// That is the regime where the per-instance length decides how many instances fit.
const BATCH_INSTANCES: usize = 1 << 11;

/// Builds the BLAKE3 circuit under one of the two scratch layout policies.
fn build(share_scratch: bool) -> (Circuit, Blake3Example) {
	let mut builder = CircuitBuilder::new();
	// The policy has to be selected before any gate is emitted, since it is read at compile time.
	if share_scratch {
		builder.enable_scratch_pooling();
	}
	// Fixed-length mode, which is the only one the single-chunk gadget supports.
	let params = HasherParams {
		message_len: Some(MESSAGE_BYTES),
		max_message_len: None,
	};
	let example = Blake3Example::build(params, &mut builder).expect("blake3 circuit builds");
	let circuit = builder.build();
	(circuit, example)
}

/// The message to hash, held fixed so both policies measure identical work.
const fn instance() -> HasherInstance {
	HasherInstance {
		random_message: false,
		random_message_len: None,
		message: None,
	}
}

/// Fills a single instance's witness, starting from an empty value vector.
fn fill_scalar(circuit: &Circuit, example: &Blake3Example) {
	// A fresh vector each time, so the measurement includes the allocation the caller would pay.
	let mut filler = circuit.new_witness_filler();
	example
		.populate_witness(instance(), &mut filler)
		.expect("witness populates");
	circuit
		.populate_wire_witness(&mut filler)
		.expect("circuit is satisfiable");
}

/// The transposed buffer the batched path fills: one row per value index, one column per instance.
fn batch_buffer(circuit: &Circuit, example: &Blake3Example, n_instances: usize) -> Vec<Word> {
	let layout = &circuit.constraint_system().value_vec_layout;
	let full_len = layout.combined_len() + layout.n_scratch;

	// One filled instance serves as the template every column is seeded from.
	let mut filler = circuit.new_witness_filler();
	example
		.populate_witness(instance(), &mut filler)
		.expect("witness populates");
	let template = filler.value_vec().combined_witness().to_vec();

	let mut data = vec![Word::ZERO; full_len * n_instances];
	let mut view = StridedArray2DViewMut::without_stride(&mut data, full_len, n_instances)
		.expect("buffer matches the requested shape");
	// Copy the template down every column, so all instances start from the same inputs.
	for instance in 0..n_instances {
		for (row, &word) in template.iter().enumerate() {
			view[(row, instance)] = word;
		}
	}
	data
}

/// Prints the layout figures for both policies side by side.
fn report_layout(unpooled: &Circuit, pooled: &Circuit) {
	// Committed length, segment length, and their sum, which is one instance's storage.
	let describe = |circuit: &Circuit| {
		let layout = &circuit.constraint_system().value_vec_layout;
		let committed = layout.combined_len();
		let scratch = layout.n_scratch;
		(committed, scratch, committed + scratch)
	};
	let (committed, scratch_off, full_off) = describe(unpooled);
	let (_, scratch_on, full_on) = describe(pooled);

	// Bytes a batched buffer needs at BATCH_INSTANCES, one word being 8 bytes.
	let mib = |words: usize| (words * BATCH_INSTANCES * 8) as f64 / (1024.0 * 1024.0);

	println!("BLAKE3, {MESSAGE_BYTES}-byte message, {} gates", unpooled.n_gates());
	println!("                          unpooled     pooled");
	println!("  committed words         {committed:>8}   {committed:>8}");
	println!("  scratch words           {scratch_off:>8}   {scratch_on:>8}");
	println!("  value vector words      {full_off:>8}   {full_on:>8}");
	println!(
		"  batch buffer @{BATCH_INSTANCES:<8} {:>7.1}MiB {:>7.1}MiB",
		mib(full_off),
		mib(full_on)
	);
	println!(
		"  scratch reduction      {:.1}x, value vector {:.2}x smaller",
		scratch_off as f64 / scratch_on.max(1) as f64,
		full_off as f64 / full_on as f64
	);
	println!();
}

fn bench_scratch_pooling(c: &mut Criterion) {
	// The same circuit compiled twice, differing only in how its scratch segment is laid out.
	let (unpooled, unpooled_example) = build(false);
	let (pooled, pooled_example) = build(true);

	// Guard the comparison: two builds that disagree on the committed witness are not comparable.
	let committed = |circuit: &Circuit, example: &Blake3Example| {
		let mut filler = circuit.new_witness_filler();
		example.populate_witness(instance(), &mut filler).unwrap();
		circuit.populate_wire_witness(&mut filler).unwrap();
		filler.value_vec().combined_witness().to_vec()
	};
	assert_eq!(
		committed(&unpooled, &unpooled_example),
		committed(&pooled, &pooled_example),
		"pooling changed the committed witness"
	);

	report_layout(&unpooled, &pooled);

	// Group 1: one instance at a time, the path a single proof takes.
	let mut scalar = c.benchmark_group("witness_fill_scalar");
	scalar.bench_function("unpooled", |b| b.iter(|| fill_scalar(&unpooled, &unpooled_example)));
	scalar.bench_function("pooled", |b| b.iter(|| fill_scalar(&pooled, &pooled_example)));
	scalar.finish();

	// Group 2: a whole batch at once, the path the memory saving is aimed at.
	let mut batched = c.benchmark_group("witness_fill_batched");
	// Each sample fills millions of words, so fewer samples keep the run to a few seconds.
	batched.sample_size(20);
	for (name, circuit, example) in [
		("unpooled", &unpooled, &unpooled_example),
		("pooled", &pooled, &pooled_example),
	] {
		let layout = &circuit.constraint_system().value_vec_layout;
		let full_len = layout.combined_len() + layout.n_scratch;
		// Built once outside the timed section: seeding the columns is setup, not the measurement.
		let mut data = batch_buffer(circuit, example, BATCH_INSTANCES);
		batched.bench_function(name, |b| {
			b.iter(|| {
				let mut view =
					StridedArray2DViewMut::without_stride(&mut data, full_len, BATCH_INSTANCES)
						.unwrap();
				circuit.populate_wire_witness_batched(&mut view).unwrap();
			})
		});
	}
	batched.finish();
}

criterion_group!(benches, bench_scratch_pooling);
criterion_main!(benches);
