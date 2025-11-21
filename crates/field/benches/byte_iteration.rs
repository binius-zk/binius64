// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	PackedField,
	arch::{OptimalPackedB1, OptimalPackedB128},
	byte_iteration::create_partial_sums_lookup_tables,
	packed::PackedSlice,
};
use criterion::{
	BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};

pub fn bench_create_partial_sums<P>(
	group: &mut BenchmarkGroup<WallTime>,
	name: &str,
	counts: impl Iterator<Item = usize>,
) where
	P: PackedField + Clone,
{
	let mut rng = rand::rng();

	for count in counts {
		let count = count.next_power_of_two().max(8);
		let values = repeat_with(|| P::random(&mut rng))
			.take(count)
			.collect::<Vec<P>>();

		let values_collection = PackedSlice::new_with_len(&values, count);

		group.bench_function(format!("{name}/{count}"), |bench| {
			bench.iter(|| create_partial_sums_lookup_tables(values_collection.clone()));
		});
	}
}

fn partial_sums_benchmark(c: &mut Criterion) {
	let mut group = c.benchmark_group("create_partial_sums_lookup_tables");
	let counts = [8, 64, 128];

	bench_create_partial_sums::<OptimalPackedB1>(
		&mut group,
		"BinaryField1b",
		counts.iter().copied(),
	);
	bench_create_partial_sums::<OptimalPackedB128>(
		&mut group,
		"BinaryField128b",
		counts.iter().copied(),
	);
}

criterion_group!(binary_field_utils, partial_sums_benchmark);
criterion_main!(binary_field_utils);
