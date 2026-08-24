// Copyright 2025 Irreducible Inc.

use binius_field::arch::{OptimalB128, OptimalPackedB128};
use binius_math::{
	multilinear::{
		Multilinear, MultilinearMut,
		hypercube::{Hypercube, OneCube},
	},
	test_utils::{random_field_buffer, random_scalars},
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

fn bench_multilinear_evaluate(c: &mut Criterion) {
	type F = OptimalB128;
	type P = OptimalPackedB128;

	let mut group = c.benchmark_group("multilinear_evaluate");

	// Benchmark with 20 variables
	let log_n = 20;
	let n_vars = log_n;

	// Calculate data size for throughput measurement
	let data_len = 1 << (log_n - P::LOG_WIDTH);
	let data_size = data_len * size_of::<P>();
	group.throughput(Throughput::Bytes(data_size as u64));

	// Generate random multilinear polynomial and evaluation point
	let mut rng = StdRng::seed_from_u64(0);
	let buffer = random_field_buffer::<P>(&mut rng, log_n);
	let point = random_scalars::<F>(&mut rng, n_vars);

	// Benchmark evaluate function (sqrt memory)
	group.bench_function(BenchmarkId::new("evaluate", format!("n_vars={n_vars}")), |b| {
		b.iter(|| buffer.evaluate(&point));
	});

	// Benchmark evaluate_inplace function
	group.bench_function(BenchmarkId::new("evaluate_inplace", format!("n_vars={n_vars}")), |b| {
		b.iter_batched(
			|| buffer.clone(),
			|buffer| buffer.evaluate_inplace(&point),
			BatchSize::SmallInput,
		);
	});

	// Benchmark evaluation with already-expanded tensor
	group.bench_function(
		BenchmarkId::new("evaluate with tensor", format!("n_vars={n_vars}")),
		|b| {
			let eq_tensor = OneCube::eq_ind_partial_eval::<P>(&point);
			b.iter(|| buffer.par_inner_product(&eq_tensor));
		},
	);

	group.finish();
}

criterion_group!(benches, bench_multilinear_evaluate);
criterion_main!(benches);
