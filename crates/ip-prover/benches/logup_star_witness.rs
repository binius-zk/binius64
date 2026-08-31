// Copyright 2026 The Binius Developers

//! Benchmarks the logUp* table-side denominator build.
//!
//! The build is one pass over a table's whole cube, so it scales with the table and not the lookup.

use binius_compute::BufferPool;
use binius_field::{FieldOps, arch::OptimalPackedB128};
use binius_ip_prover::logup_star::witness::table_denominator;
use binius_math::test_utils::random_scalars;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;

// Builds the denominator column of one table's fractional-addition circuit.
fn bench_table_denominator(criterion: &mut Criterion) {
	let mut group = criterion.benchmark_group("logup_star_witness/table_denominator");
	let mut rng = StdRng::seed_from_u64(0);

	// 2^8 is a byte table, 2^22 a large one; the span covers both ends of the realistic range.
	for table_n_vars in [8, 12, 16, 20, 22] {
		group.throughput(Throughput::Elements(1 << table_n_vars));
		group.bench_function(format!("n_vars={table_n_vars}"), |bench| {
			// One logUp challenge per table, drawn from the transcript in production.
			let c = random_scalars::<F>(&mut rng, 1)[0];
			let pool = BufferPool::new();
			let alloc = &pool;

			// Each run drops its buffer back into the pool, so no run pays for fresh pages.
			bench.iter(|| table_denominator::<_, F, P>(&alloc, c, table_n_vars));
		});
	}

	group.finish();
}

criterion_group!(logup_star_witness, bench_table_denominator);
criterion_main!(logup_star_witness);
