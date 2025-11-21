// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField128bGhash as Ghash, Random};
use binius_math::batch_invert::BatchInversion;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

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

criterion_group!(batch_invert_bench, bench_batch_inversion_struct);
criterion_main!(batch_invert_bench);
