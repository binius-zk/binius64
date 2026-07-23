// Copyright 2025-2026 The Binius Developers

use binius_compute::{BufferPool, PoolVec};
use binius_field::{FieldOps, arch::OptimalPackedB128};
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::fracaddcheck::FracAddCheckProver;
use binius_math::{
	FieldBuffer,
	multilinear::evaluate::evaluate,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;

fn bench_fracaddcheck_new(c: &mut Criterion) {
	let mut group = c.benchmark_group("fracaddcheck/new");

	for n_vars in [12, 16, 20] {
		// Full reduction: k = n_vars, so sums layer has log_len = 0.
		let k = n_vars;

		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let mut rng = rand::rng();
			let num_buffer = random_field_buffer::<P>(&mut rng, n_vars);
			let den_buffer = random_field_buffer::<P>(&mut rng, n_vars);

			let pool = BufferPool::new();
			let alloc = &pool;

			b.iter_batched(
				|| {
					(
						FieldBuffer::<_, PoolVec<_>>::clone_from_slice(&alloc, num_buffer.to_ref()),
						FieldBuffer::<_, PoolVec<_>>::clone_from_slice(&alloc, den_buffer.to_ref()),
					)
				},
				|(witness_num, witness_den)| {
					FracAddCheckProver::<_, P>::new(k, &alloc, (witness_num, witness_den))
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

fn bench_fracaddcheck_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("fracaddcheck/prove");

	for n_vars in [12, 16, 20] {
		// Full reduction: k = n_vars, so sums layer has log_len = 0.
		let k = n_vars;

		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let mut rng = rand::rng();
			let num_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let den_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;

			// Build the prover once, then clone it per iteration (untimed setup).
			let (prover, sums) = FracAddCheckProver::new(
				k,
				&alloc,
				(
					FieldBuffer::<P, _>::from_values_in(&alloc, &num_scalars),
					FieldBuffer::<P, _>::from_values_in(&alloc, &den_scalars),
				),
			);
			let sum_num_eval = evaluate(&sums.0, &[]);
			let sum_den_eval = evaluate(&sums.1, &[]);
			let claim = (
				MultilinearEvalClaim {
					eval: sum_num_eval,
					point: vec![],
				},
				MultilinearEvalClaim {
					eval: sum_den_eval,
					point: vec![],
				},
			);

			let mut transcript = ProverTranscript::new(StdChallenger::default());

			b.iter_batched(
				|| (prover.clone(), claim.clone()),
				|(prover, claim)| prover.prove(claim, &mut transcript),
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(fracaddcheck, bench_fracaddcheck_new, bench_fracaddcheck_prove);
criterion_main!(fracaddcheck);
