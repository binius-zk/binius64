// Copyright 2025-2026 The Binius Developers

use binius_compute::{BufferPool, PoolVec};
use binius_field::{FieldOps, arch::OptimalPackedB128};
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::prodcheck::ProdcheckProver;
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

fn bench_prodcheck_new(c: &mut Criterion) {
	let mut group = c.benchmark_group("prodcheck/new");

	for n_vars in [12, 16, 20] {
		// Full product: k = n_vars, so products layer has log_len = 0
		let k = n_vars;

		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let mut rng = rand::rng();
			let witness_buffer = random_field_buffer::<P>(&mut rng, n_vars);

			let pool = BufferPool::new();
			let alloc = &pool;

			b.iter_batched(
				|| FieldBuffer::<_, PoolVec<_>>::clone_from_slice(&alloc, witness_buffer.to_ref()),
				|witness| ProdcheckProver::<_, P>::new(k, &alloc, witness),
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

fn bench_prodcheck_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("prodcheck/prove");

	for n_vars in [12, 16, 20] {
		// Full product: k = n_vars, so products layer has log_len = 0
		let k = n_vars;

		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let mut rng = rand::rng();
			let witness_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;

			// Build the prover once, then clone it per iteration (untimed setup).
			let (prover, products) = ProdcheckProver::new(
				k,
				&alloc,
				FieldBuffer::<P, _>::from_values_in(&alloc, &witness_scalars),
			);
			let products_eval = evaluate(&products, &[]);
			let claim = MultilinearEvalClaim {
				eval: products_eval,
				point: vec![],
			};

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

criterion_group!(prodcheck, bench_prodcheck_new, bench_prodcheck_prove);
criterion_main!(prodcheck);
