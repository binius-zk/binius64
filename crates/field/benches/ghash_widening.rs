// Copyright 2026 The Binius Developers

//! Benchmark widening (deferred-reduction) `GF(2^128)` multiplication against full multiplication.
//!
//! Measures the inner-product pattern `sum_i a_i * b_i` two ways: a full multiply per term (which
//! reduces every product), and a widening multiply that accumulates unreduced products and reduces
//! once at the end.

use std::hint::black_box;

use binius_field::{GhashSq256b, PackedField, WideMul, arch::OptimalPackedB128};
use criterion::{
	BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};

fn bench_at_n<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, label: &str, n: usize) {
	let mut rng = rand::rng();
	let a_vals: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();
	let b_vals: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();

	group.throughput(Throughput::Elements((n * P::WIDTH) as u64));

	group.bench_function(format!("full_mul/{label}/n={n}"), |b| {
		b.iter(|| {
			let mut acc = P::default();
			for i in 0..n {
				acc += black_box(a_vals[i]) * black_box(b_vals[i]);
			}
			black_box(acc)
		})
	});

	group.bench_function(format!("wide_mul/{label}/n={n}"), |b| {
		b.iter(|| {
			let mut acc = <P as WideMul>::Output::default();
			for i in 0..n {
				acc += P::wide_mul(black_box(a_vals[i]), black_box(b_vals[i]));
			}
			black_box(P::reduce(acc))
		})
	});
}

fn bench_ghash_widening(c: &mut Criterion) {
	let mut group = c.benchmark_group("ghash_widening");

	let label = format!("OptimalPacked_{}x128b", OptimalPackedB128::WIDTH);
	for &n in &[16, 256, 4096] {
		bench_at_n::<OptimalPackedB128>(&mut group, &label, n);
	}

	// GF(2^256) is a degree-two extension of GHASH, so one multiply is three GHASH products.
	// Deferring their reduction across an inner product is where the widening multiply pays off.
	for &n in &[16, 256, 4096] {
		bench_at_n::<GhashSq256b>(&mut group, "GhashSq256b", n);
	}

	group.finish();
}

criterion_group!(benches, bench_ghash_widening);
criterion_main!(benches);
