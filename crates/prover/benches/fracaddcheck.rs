// Copyright 2025-2026 The Binius Developers

use binius_compute::BufferPool;
use binius_field::arch::OptimalPackedB128;
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::{fracaddcheck::FracAddCheckProver, sumcheck::mle_store::pooled_copy};
use binius_math::{multilinear::evaluate::evaluate, test_utils::random_field_buffer};
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

type P = OptimalPackedB128;

fn bench_fracaddcheck_new(c: &mut Criterion) {
	let mut group = c.benchmark_group("fracaddcheck/new");

	for n_vars in [12, 16, 20] {
		// Full reduction: k = n_vars, so sums layer has log_len = 0.
		let k = n_vars;

		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let mut rng = rand::rng();
			let witness_num = random_field_buffer::<P>(&mut rng, n_vars);
			let witness_den = random_field_buffer::<P>(&mut rng, n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;

			b.iter_batched(
				|| (witness_num.clone(), witness_den.clone()),
				|(witness_num, witness_den)| {
					FracAddCheckProver::<_, P>::new(
						k,
						&alloc,
						(pooled_copy(&alloc, &witness_num), pooled_copy(&alloc, &witness_den)),
					)
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
			let witness_num = random_field_buffer::<P>(&mut rng, n_vars);
			let witness_den = random_field_buffer::<P>(&mut rng, n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;

			// Pre-compute the claim (final sums layer evaluation at empty point).
			let (_prover, sums) = FracAddCheckProver::new(
				k,
				&alloc,
				(pooled_copy(&alloc, &witness_num), pooled_copy(&alloc, &witness_den)),
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
				|| {
					let (prover, _sums) = FracAddCheckProver::new(
						k,
						&alloc,
						(pooled_copy(&alloc, &witness_num), pooled_copy(&alloc, &witness_den)),
					);
					(prover, claim.clone())
				},
				|(prover, claim)| prover.prove(claim, &mut transcript),
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(fracaddcheck, bench_fracaddcheck_new, bench_fracaddcheck_prove);
criterion_main!(fracaddcheck);
