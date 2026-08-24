// Copyright 2025-2026 The Binius Developers

use binius_compute::BufferPool;
use binius_field::{FieldOps, arch::OptimalPackedB128};
use binius_ip::fracaddcheck::FracAddEvalClaim;
use binius_ip_prover::fracaddcheck::{FracAddCircuit, fraction::Fraction};
use binius_math::{
	FieldBuffer,
	multilinear::Multilinear,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;

fn bench_fracaddcheck_build(c: &mut Criterion) {
	let mut group = c.benchmark_group("fracaddcheck/build");

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
						FieldBuffer::from_view_in(&alloc, num_buffer.as_view()),
						FieldBuffer::from_view_in(&alloc, den_buffer.as_view()),
					)
				},
				|(witness_num, witness_den)| {
					FracAddCircuit::<_, P>::build(
						k,
						&alloc,
						Fraction::new(witness_num, witness_den),
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
			let num_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let den_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
			let pool = BufferPool::new();
			let alloc = &pool;

			// Build the prover once, then clone it per iteration (untimed setup).
			let (prover, sums) = FracAddCircuit::build(
				k,
				&alloc,
				Fraction::new(
					FieldBuffer::<P, _>::from_values_in(&alloc, &num_scalars),
					FieldBuffer::<P, _>::from_values_in(&alloc, &den_scalars),
				),
			);
			let sum_num_eval = sums.num.evaluate(&[]);
			let sum_den_eval = sums.den.evaluate(&[]);
			let claim = FracAddEvalClaim {
				num_eval: sum_num_eval,
				den_eval: sum_den_eval,
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

criterion_group!(fracaddcheck, bench_fracaddcheck_build, bench_fracaddcheck_prove);
criterion_main!(fracaddcheck);
