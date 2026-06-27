// Copyright 2025 Irreducible Inc.

use binius_field::arch::{OptimalB128, OptimalPackedB128};
use binius_math::{multilinear::eq::eq_ind_partial_eval, test_utils::random_scalars};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

fn bench_eq_ind_partial_eval(c: &mut Criterion) {
	type F = OptimalB128;
	type P = OptimalPackedB128;

	let mut group = c.benchmark_group("eq_ind_partial_eval");

	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in [16, 20, 24] {
		// Throughput is measured in the number of output elements, which is the size of the
		// returned tensor over the n-dimensional hypercube.
		let n_output_elems = 1u64 << n_vars;
		group.throughput(Throughput::Elements(n_output_elems));

		let point = random_scalars::<F>(&mut rng, n_vars);
		group.bench_function(BenchmarkId::from_parameter(format!("n_vars={n_vars}")), |b| {
			b.iter(|| eq_ind_partial_eval::<P>(&point));
		});
	}

	group.finish();
}

criterion_group!(benches, bench_eq_ind_partial_eval);
criterion_main!(benches);
