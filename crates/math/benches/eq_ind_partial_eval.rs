// Copyright 2025 Irreducible Inc.

use binius_field::arch::{OptimalB128, OptimalPackedB128};
use binius_math::{
	multilinear::hypercube::{Hypercube, InfCube, OneCube, eq_ind_partial_eval},
	test_utils::random_scalars,
};
use criterion::{
	BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
	measurement::WallTime,
};
use rand::{SeedableRng, rngs::StdRng};

type F = OptimalB128;
type P = OptimalPackedB128;

fn bench_cube<Cube: Hypercube>(group: &mut BenchmarkGroup<'_, WallTime>, cube: &str, point: &[F]) {
	let id = BenchmarkId::new(cube, format!("n_vars={}", point.len()));
	group.bench_function(id, |b| {
		b.iter(|| eq_ind_partial_eval::<Cube, P>(point));
	});
}

fn bench_eq_ind_partial_eval(c: &mut Criterion) {
	let mut group = c.benchmark_group("eq_ind_partial_eval");

	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in [16, 20, 24] {
		// Throughput is measured in the number of output elements, which is the size of the
		// returned tensor over the n-dimensional hypercube.
		let n_output_elems = 1u64 << n_vars;
		group.throughput(Throughput::Elements(n_output_elems));

		let point = random_scalars::<F>(&mut rng, n_vars);
		bench_cube::<OneCube>(&mut group, "one_cube", &point);
		bench_cube::<InfCube>(&mut group, "inf_cube", &point);
	}

	group.finish();
}

criterion_group!(benches, bench_eq_ind_partial_eval);
criterion_main!(benches);
