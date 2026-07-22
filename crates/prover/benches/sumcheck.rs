// Copyright 2025 Irreducible Inc.

use binius_compute::BufferPool;
use binius_field::{FieldOps, arch::OptimalPackedB128, packed::PackedField};
use binius_ip_prover::sumcheck::{prove_single_mlecheck, quadratic_mlecheck_prover};
use binius_math::{FieldBuffer, inner_product::inner_product_par, test_utils::random_scalars};
use binius_transcript::ProverTranscript;
use binius_utils::rayon::prelude::*;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, prelude::StdRng};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;

fn bench_mlecheck_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("mlecheck");

	let mut rng = StdRng::seed_from_u64(0);

	// Test different sizes of multilinear polynomials
	for n_vars in [12, 16, 20] {
		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("A*B/n_vars={n_vars}"), |b| {
			let a_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let b_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;
			// Build the multilinears once; each iteration clones them from the pool.
			let multilinear_a = FieldBuffer::<P>::from_values_in(&alloc, &a_scalars);
			let multilinear_b = FieldBuffer::<P>::from_values_in(&alloc, &b_scalars);

			let eval_point = random_scalars(&mut rng, n_vars);
			let eval_claim = inner_product_par(&multilinear_a, &multilinear_b);

			let mut transcript = ProverTranscript::new(StdChallenger::default());

			// Benchmark only the proving phase
			b.iter_batched(
				|| [multilinear_a.clone(), multilinear_b.clone()],
				|multilinears| {
					let prover = quadratic_mlecheck_prover(
						&alloc,
						multilinears,
						|[a, b]| a * b,
						|[a, b]| a * b,
						eval_point.clone(),
						eval_claim,
					);

					prove_single_mlecheck(prover, &mut transcript)
				},
				BatchSize::SmallInput,
			);
		});

		// Benchmark mul gate composition: a * b - c
		group.bench_function(format!("A*B-C/n_vars={n_vars}"), |b| {
			let a_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let b_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let c_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;
			// Build the multilinears once; each iteration clones them from the pool.
			let multilinear_a = FieldBuffer::<P>::from_values_in(&alloc, &a_scalars);
			let multilinear_b = FieldBuffer::<P>::from_values_in(&alloc, &b_scalars);
			let multilinear_c = FieldBuffer::<P>::from_values_in(&alloc, &c_scalars);

			let eval_point = random_scalars(&mut rng, n_vars);
			let eval_claim =
				(multilinear_a.as_ref(), multilinear_b.as_ref(), multilinear_c.as_ref())
					.into_par_iter()
					.map(|(&a_i, &b_i, &c_i)| a_i * b_i - c_i)
					.sum::<P>()
					.into_iter()
					.take(1 << n_vars)
					.sum();

			let mut transcript = ProverTranscript::new(StdChallenger::default());

			// Benchmark only the proving phase
			b.iter_batched(
				|| {
					[
						multilinear_a.clone(),
						multilinear_b.clone(),
						multilinear_c.clone(),
					]
				},
				|multilinears| {
					let prover = quadratic_mlecheck_prover(
						&alloc,
						multilinears,
						|[a, b, c]| a * b - c,
						|[a, b, _c]| a * b,
						eval_point.clone(),
						eval_claim,
					);

					prove_single_mlecheck(prover, &mut transcript)
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(sumcheck, bench_mlecheck_prove);
criterion_main!(sumcheck);
