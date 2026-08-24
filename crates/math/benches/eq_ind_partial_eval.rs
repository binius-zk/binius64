// Copyright 2025 Irreducible Inc.

use binius_field::arch::{OptimalB128, OptimalPackedB128};
use binius_math::{multilinear::hypercube::Hypercube, test_utils::random_scalars};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

type F = OptimalB128;
type P = OptimalPackedB128;

fn bench_eq_ind_partial_eval(c: &mut Criterion) {
	let mut group = c.benchmark_group("eq_ind_partial_eval");

	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in [16, 20, 24] {
		// Throughput is measured in the number of output elements, which is the size of the
		// returned tensor over the n-dimensional hypercube.
		let n_output_elems = 1u64 << n_vars;
		group.throughput(Throughput::Elements(n_output_elems));

		let point = random_scalars::<F>(&mut rng, n_vars);

		// The cube is a value, so both bases are one loop rather than two instantiations.
		for (cube, name) in [(Hypercube::One, "one_cube"), (Hypercube::Inf, "inf_cube")] {
			let id = BenchmarkId::new(name, format!("n_vars={n_vars}"));
			group.bench_function(id, |b| {
				b.iter(|| cube.expand(&point).build::<P>());
			});
		}
	}

	group.finish();
}

criterion_group!(benches, bench_eq_ind_partial_eval);
criterion_main!(benches);
