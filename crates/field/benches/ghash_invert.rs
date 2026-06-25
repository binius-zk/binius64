// Copyright 2026 The Binius Developers

//! Benchmark GHASH field inversion (Itoh-Tsujii, which now backs `InvertOrZero`) across packing
//! widths: the scalar `BinaryField128bGhash` and the 1x/2x/4x packed fields.
//!
//! Throughput is reported in scalar elements (batch size times packing width), so the numbers are
//! directly comparable across widths.

use std::hint::black_box;

use binius_field::{
	BinaryField128bGhash as GhashB128, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
	PackedBinaryGhash4x128b, PackedField,
};
use criterion::{
	BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};

fn bench_width<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, label: &str, n: usize) {
	let mut rng = rand::rng();
	let vals: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();

	group.throughput(Throughput::Elements((n * P::WIDTH) as u64));

	group.bench_function(format!("{label}/n={n}"), |b| {
		b.iter(|| {
			let mut acc = P::zero();
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
		bench_width::<GhashB128>(&mut group, "scalar", n);
		bench_width::<PackedBinaryGhash1x128b>(&mut group, "1x128b", n);
		bench_width::<PackedBinaryGhash2x128b>(&mut group, "2x128b", n);
		bench_width::<PackedBinaryGhash4x128b>(&mut group, "4x128b", n);
	}

	group.finish();
}

criterion_group!(benches, bench_ghash_invert);
criterion_main!(benches);
