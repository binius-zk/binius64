// Copyright 2025 Irreducible Inc.

use std::mem::size_of;

use binius_compute::BufferPool;
use binius_field::{BinaryField, ExtensionField, arch::OptimalPackedB128};
use binius_math::{
	multilinear::eq::eq_ind_partial_eval,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_prover::ring_switch::{
	LOG_SPLIT_BLOCK, fold_1b_rows_for_b128, fold_1b_rows_for_b128_split, fold_elems_inplace,
	rs_eq_ind_from_factors,
};
use binius_utils::checked_arithmetics::checked_log_2;
use binius_verifier::config::{B1, B128};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

fn bench_fold_1b_rows_for_b128(c: &mut Criterion) {
	let mut group = c.benchmark_group("fold_1b_rows_for_b128");

	let log_bits = checked_log_2(B128::N_BITS);
	for log_len in [12, 16] {
		const LOG_BITS_PER_BYTE: usize = 3;
		group.throughput(Throughput::Bytes((1 << (log_len + log_bits - LOG_BITS_PER_BYTE)) as u64));
		group.bench_function(format!("log_len={log_len}"), |b| {
			let mut rng = rand::rng();

			let mat = random_field_buffer::<B128>(&mut rng, log_len);
			let vec = random_field_buffer::<B128>(&mut rng, log_len);

			b.iter(|| fold_1b_rows_for_b128(&mat, &vec));
		});
	}

	group.finish();
}

fn bench_fold_elems_inplace(c: &mut Criterion) {
	let mut group = c.benchmark_group("ring_switch/fold_elems_inplace");

	type P = OptimalPackedB128;

	for log_len in [12, 16] {
		// Calculate throughput based on the size of elems buffer in bytes
		let elem_bytes = (1 << log_len) * size_of::<P>();
		group.throughput(Throughput::Bytes(elem_bytes as u64));
		group.bench_function(format!("log_len={log_len}"), |b| {
			let mut rng = rand::rng();

			let elems = random_field_buffer::<P>(&mut rng, log_len);
			let vec =
				random_field_buffer::<B128>(&mut rng, <B128 as ExtensionField<B1>>::LOG_DEGREE);

			b.iter_batched(
				|| elems.clone(),
				|elems| fold_elems_inplace(elems, &vec),
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

/// The row fold alone, against a materialized tensor versus against its two factors.
///
/// Both routes are handed their inputs already expanded, so this isolates the two kernels.
/// `dense` rebuilds a nibble table every 4 rows and streams the `2^n` tensor alongside the matrix.
/// `split` builds byte tables once and streams the matrix alone.
fn bench_fold_1b_rows_split(c: &mut Criterion) {
	let mut group = c.benchmark_group("ring_switch/fold_1b_rows_split");
	group.sample_size(10);

	type P = OptimalPackedB128;

	for log_len in [16, 20, 22] {
		let elem_bytes = (1usize << log_len) * size_of::<P>();
		group.throughput(Throughput::Bytes(elem_bytes as u64));

		let mut rng = rand::rng();
		let mat = random_field_buffer::<P>(&mut rng, log_len);
		let suffix: Vec<B128> = random_scalars(&mut rng, log_len);

		let tensor = eq_ind_partial_eval::<P>(&suffix);
		let (suffix_lo, suffix_hi) = suffix.split_at(LOG_SPLIT_BLOCK.min(log_len));
		let eq_lo = eq_ind_partial_eval::<B128>(suffix_lo);
		let eq_hi = eq_ind_partial_eval::<B128>(suffix_hi);

		group.bench_function(format!("dense/log_len={log_len}"), |b| {
			b.iter(|| fold_1b_rows_for_b128(&mat, &tensor));
		});
		group.bench_function(format!("split/log_len={log_len}"), |b| {
			b.iter(|| fold_1b_rows_for_b128_split(&mat, &eq_lo, &eq_hi));
		});
	}

	group.finish();
}

/// The two suffix-tensor pipelines, whole:
///
/// - `dense`: expand the full tensor, fold the matrix against it, transform it in place.
/// - `split`: expand the two factors, fold against them, build the indicator in one pass.
///
/// Both produce the identical row fold and equality indicator, so the times compare directly.
fn bench_suffix_tensor_pipeline(c: &mut Criterion) {
	let mut group = c.benchmark_group("ring_switch/suffix_tensor_pipeline");
	group.sample_size(10);

	type P = OptimalPackedB128;

	for log_len in [16, 20, 22] {
		let elem_bytes = (1usize << log_len) * size_of::<P>();
		group.throughput(Throughput::Bytes(elem_bytes as u64));

		let mut rng = rand::rng();
		let mat = random_field_buffer::<P>(&mut rng, log_len);
		let suffix: Vec<B128> = random_scalars(&mut rng, log_len);
		let row_batching: Vec<B128> =
			random_scalars(&mut rng, <B128 as ExtensionField<B1>>::LOG_DEGREE);
		let row_batch_query = eq_ind_partial_eval::<B128>(&row_batching);

		group.bench_function(format!("dense/log_len={log_len}"), |b| {
			b.iter(|| {
				let tensor = eq_ind_partial_eval::<P>(&suffix);
				let s_hat_v = fold_1b_rows_for_b128(&mat, &tensor);
				let rs_eq_ind = fold_elems_inplace(tensor, &row_batch_query);
				(s_hat_v, rs_eq_ind)
			});
		});

		group.bench_function(format!("split/log_len={log_len}"), |b| {
			// The prover caps the low factor here so its fold tables stay cache-resident.
			let (suffix_lo, suffix_hi) = suffix.split_at(LOG_SPLIT_BLOCK.min(log_len));

			// The prover draws the indicator from a pool, so the bench does too.
			// One pool spans every iteration, which is the steady state being measured:
			// the first iteration allocates a block and the rest reuse it.
			let pool = BufferPool::new();
			let alloc = &pool;

			b.iter(|| {
				let eq_lo = eq_ind_partial_eval::<B128>(suffix_lo);
				let eq_hi = eq_ind_partial_eval::<B128>(suffix_hi);
				let s_hat_v = fold_1b_rows_for_b128_split(&mat, &eq_lo, &eq_hi);
				let rs_eq_ind =
					rs_eq_ind_from_factors::<_, P>(&alloc, &eq_lo, &eq_hi, &row_batch_query);
				(s_hat_v, rs_eq_ind)
			});
		});
	}

	group.finish();
}

criterion_group!(
	ring_switch,
	bench_fold_1b_rows_for_b128,
	bench_fold_elems_inplace,
	bench_fold_1b_rows_split,
	bench_suffix_tensor_pipeline
);
criterion_main!(ring_switch);
