// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible Inc.

use std::array;

use binius_field::{BinaryField128bGhash as Ghash, Field, Random};
use binius_math::batch_invert::{BatchInversion, batch_invert};
use criterion::{
	BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use rand::rngs::ThreadRng;

fn bench_batch_invert(c: &mut Criterion) {
	let mut group = c.benchmark_group("Batch Invert Throughput");
	let mut rng = rand::rng();

	fn bench_for_size<const N: usize, const N2: usize>(
		group: &mut BenchmarkGroup<'_, WallTime>,
		rng: &mut ThreadRng,
	) {
		group.throughput(Throughput::Elements(N as u64));
		let mut elements: [Ghash; N] = array::from_fn(|_| <Ghash as Random>::random(&mut *rng));
		let scratchpad = &mut [Ghash::ZERO; N2];
		group.bench_function(format!("{N}"), |b| {
			b.iter(|| {
				batch_invert::<N>(&mut elements, scratchpad);
			})
		});
	}

	bench_for_size::<2, 4>(&mut group, &mut rng);
	bench_for_size::<4, 8>(&mut group, &mut rng);
	bench_for_size::<8, 16>(&mut group, &mut rng);
	bench_for_size::<16, 32>(&mut group, &mut rng);
	bench_for_size::<32, 64>(&mut group, &mut rng);
	bench_for_size::<64, 128>(&mut group, &mut rng);
	bench_for_size::<128, 256>(&mut group, &mut rng);
	bench_for_size::<256, 512>(&mut group, &mut rng);

	group.finish();
}

fn bench_batch_inversion_struct(c: &mut Criterion) {
	let mut group = c.benchmark_group("BatchInversion Struct Throughput");
	let mut rng = rand::rng();

	for n in [1, 4, 6, 64, 96, 256, 384] {
		group.throughput(Throughput::Elements(n as u64));
		let mut elements = Vec::with_capacity(n);
		for _ in 0..n {
			elements.push(<Ghash as Random>::random(&mut rng));
		}
		let mut inverter = BatchInversion::<Ghash>::new(n);
		group.bench_function(format!("{n}"), |b| {
			b.iter(|| {
				inverter.invert_or_zero(&mut elements);
			})
		});
	}

	group.finish();
}

criterion_group!(batch_invert_bench, bench_batch_invert, bench_batch_inversion_struct,);
criterion_main!(batch_invert_bench);
