// Copyright 2025-2026 The Binius Developers

use binius_compute::BufferPool;
use binius_field::arch::OptimalPackedB128;
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::{prodcheck::ProdcheckProver, sumcheck::mle_store::pooled_copy};
use binius_math::{multilinear::evaluate::evaluate, test_utils::random_field_buffer};
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

type P = OptimalPackedB128;

fn bench_prodcheck_new(c: &mut Criterion) {
	let mut group = c.benchmark_group("prodcheck/new");

	for n_vars in [12, 16, 20] {
		// Full product: k = n_vars, so products layer has log_len = 0
		let k = n_vars;

		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let mut rng = rand::rng();
			let witness = random_field_buffer::<P>(&mut rng, n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;

			b.iter_batched(
				|| witness.clone(),
				|witness| ProdcheckProver::<_, P>::new(k, &alloc, pooled_copy(&alloc, &witness)),
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
			let witness = random_field_buffer::<P>(&mut rng, n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;

			// Pre-compute the claim (products layer evaluation at empty point)
			let (_prover, products) =
				ProdcheckProver::new(k, &alloc, pooled_copy(&alloc, &witness));
			let products_eval = evaluate(&products, &[]);
			let claim = MultilinearEvalClaim {
				eval: products_eval,
				point: vec![],
			};

			let mut transcript = ProverTranscript::new(StdChallenger::default());

			b.iter_batched(
				|| {
					let (prover, _products) =
						ProdcheckProver::new(k, &alloc, pooled_copy(&alloc, &witness));
					(prover, claim.clone())
				},
				|(prover, claim)| prover.prove(claim, &mut transcript),
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(prodcheck, bench_prodcheck_new, bench_prodcheck_prove);
criterion_main!(prodcheck);
