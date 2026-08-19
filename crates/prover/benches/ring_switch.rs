// Copyright 2025 Irreducible Inc.

use std::mem::size_of;

use binius_compute::BufferPool;
use binius_field::{ExtensionField, Random, arch::OptimalPackedB128};
use binius_math::{
	inner_product::{inner_product_packed, inner_product_subfield},
	multilinear::eq::eq_ind_partial_eval,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_prover::ring_switch::{
	LOG_SPLIT_BLOCK, fold_1b_rows_for_b128_split, rs_eq_ind_from_factors,
};
use binius_verifier::config::{B1, B128};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// The row fold alone, against the two factors of the suffix tensor.
///
/// Its inputs arrive already expanded, so this isolates the kernel from the expansion.
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

		let (suffix_lo, suffix_hi) = suffix.split_at(LOG_SPLIT_BLOCK.min(log_len));
		let eq_lo = eq_ind_partial_eval::<B128>(suffix_lo);
		let eq_hi = eq_ind_partial_eval::<B128>(suffix_hi);

		group.bench_function(format!("log_len={log_len}"), |b| {
			b.iter(|| fold_1b_rows_for_b128_split(&mat, &eq_lo, &eq_hi));
		});
	}

	group.finish();
}

/// The whole suffix-tensor pipeline: expand the two factors, fold the matrix against them, and
/// build the equality indicator in one pass. The `2^n` tensor is never materialized.
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

		// The prover caps the low factor here so its fold tables stay cache-resident.
		let (suffix_lo, suffix_hi) = suffix.split_at(LOG_SPLIT_BLOCK.min(log_len));

		// The prover draws the indicator from a pool, so the bench does too.
		// One pool spans every iteration, which is the steady state being measured:
		// the first iteration allocates a block and the rest reuse it.
		let pool = BufferPool::new();
		let alloc = &pool;

		group.bench_function(format!("log_len={log_len}"), |b| {
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

/// The sumcheck-claim inner product at its one real shape: `N = 2^log_packing = 128` terms.
///
/// In-binary A/B: the naive per-term reduce (`inner_product_subfield`, what `ring_switch::prove`
/// called before) against the deferred-reduce path (`inner_product_packed`, what it calls now).
fn bench_sumcheck_claim_inner_product(c: &mut Criterion) {
	let mut group = c.benchmark_group("ring_switch/sumcheck_claim_inner_product");

	let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
	let n = 1usize << log_packing;
	group.throughput(Throughput::Elements(n as u64));

	let mut rng = rand::rng();
	let s_hat_u: Vec<B128> = (0..n).map(|_| B128::random(&mut rng)).collect();
	let eq_r_double_prime: Vec<B128> = (0..n).map(|_| B128::random(&mut rng)).collect();

	group.bench_function("naive_per_term_reduce", |b| {
		b.iter(|| {
			inner_product_subfield::<B128, B128>(
				s_hat_u.iter().copied(),
				eq_r_double_prime.iter().copied(),
			)
		});
	});

	group.bench_function("deferred_reduce", |b| {
		b.iter(|| {
			inner_product_packed::<B128, B128>(
				log_packing,
				s_hat_u.iter().copied(),
				eq_r_double_prime.iter().copied(),
			)
		});
	});

	group.finish();
}

criterion_group!(
	ring_switch,
	bench_fold_1b_rows_split,
	bench_suffix_tensor_pipeline,
	bench_sumcheck_claim_inner_product
);
criterion_main!(ring_switch);
