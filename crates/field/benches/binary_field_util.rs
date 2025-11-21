// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{BinaryField1b, BinaryField128bGhash, ExtensionField, PackedField};
use criterion::{
	BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};

pub fn bench_inner_product_par<FX, PX, PY>(
	group: &mut BenchmarkGroup<WallTime>,
	name: &str,
	counts: impl Iterator<Item = usize>,
) where
	PX: PackedField<Scalar = FX>,
	PY: PackedField,
	FX: ExtensionField<PY::Scalar>,
{
	let mut rng = rand::rng();
	for count in counts {
		let xs = repeat_with(|| PX::random(&mut rng))
			.take(count)
			.collect::<Vec<PX>>();
		let ys = repeat_with(|| PY::random(&mut rng))
			.take(count)
			.collect::<Vec<PY>>();
		group.bench_function(format!("{name}/{count}"), |bench| {
			bench.iter(|| binius_field::util::inner_product_par::<FX, PX, PY>(&xs, &ys));
		});
	}
}

fn inner_product_par(c: &mut Criterion) {
	let mut group = c.benchmark_group("inner_product_par");
	let counts = [128usize, 512, 1024, 8192, 1 << 20];
	bench_inner_product_par::<_, BinaryField128bGhash, BinaryField1b>(
		&mut group,
		"128bx1b",
		counts.iter().copied(),
	);
}

criterion_group!(binary_field_utils, inner_product_par);
criterion_main!(binary_field_utils);
