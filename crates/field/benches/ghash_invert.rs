// Copyright 2026 The Binius Developers

//! Benchmark GHASH field inversion: the Itoh-Tsujii algorithm (`invert_b128`) against the
//! nibble-based `InvertOrZero` implementation.
//!
//! Both compute the same `invert-or-zero` of a `BinaryField128bGhash` scalar; this measures their
//! throughput over a batch of random elements.

use std::hint::black_box;

use binius_field::{
	BinaryField128bGhash as GhashB128, Field, Random, arch::invert_b128,
	arithmetic_traits::InvertOrZero,
};
use criterion::{
	BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};

fn bench_at_n(group: &mut BenchmarkGroup<'_, WallTime>, n: usize) {
	let mut rng = rand::rng();
	let vals: Vec<GhashB128> = (0..n).map(|_| GhashB128::random(&mut rng)).collect();

	group.throughput(Throughput::Elements(n as u64));

	group.bench_function(format!("itoh_tsujii/n={n}"), |b| {
		b.iter(|| {
			let mut acc = GhashB128::ZERO;
			for &x in &vals {
				acc += invert_b128::<GhashB128>(black_box(x));
			}
			black_box(acc)
		})
	});

	group.bench_function(format!("invert_or_zero/n={n}"), |b| {
		b.iter(|| {
			let mut acc = GhashB128::ZERO;
			for &x in &vals {
				acc += black_box(x).invert_or_zero();
			}
			black_box(acc)
		})
	});
}

fn bench_ghash_invert(c: &mut Criterion) {
	let mut group = c.benchmark_group("ghash_invert");

	for &n in &[16, 256, 4096] {
		bench_at_n(&mut group, n);
	}

	group.finish();
}

criterion_group!(benches, bench_ghash_invert);
criterion_main!(benches);
