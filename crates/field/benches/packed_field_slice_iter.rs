// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	PackedBinaryField128x1b, PackedBinaryField256x1b, PackedBinaryField512x1b, PackedField,
};
use criterion::{
	BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};

fn benchmark_iter_impl<P: PackedField>(
	group: &mut BenchmarkGroup<'_, WallTime>,
	id: &str,
	len: usize,
) {
	let mut rng = rand::rng();
	let values = repeat_with(|| P::random(&mut rng))
		.take(len)
		.collect::<Vec<P>>();

	group.throughput(Throughput::Elements((len * P::WIDTH) as _));
	group.bench_function(id, |b| b.iter(|| PackedField::iter_slice(&values).sum::<P::Scalar>()));
}

const BATCH_SIZE: usize = 1 << 15;

macro_rules! benchmark_from_fn {
	($field:ty, $g:ident) => {
		benchmark_iter_impl::<$field>(
			&mut $g,
			&format!("{}/iter_begin", stringify!($field)),
			BATCH_SIZE / 2,
		);
	};
}

fn packed_128(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_128");

	benchmark_from_fn!(PackedBinaryField128x1b, group);
}

fn packed_256(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_256");

	benchmark_from_fn!(PackedBinaryField256x1b, group);
}

fn packed_512(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_512");

	benchmark_from_fn!(PackedBinaryField512x1b, group);
}

criterion_group!(iterate, packed_128, packed_256, packed_512);
criterion_main!(iterate);
