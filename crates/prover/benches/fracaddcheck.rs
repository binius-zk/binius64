// Copyright 2025-2026 The Binius Developers

use binius_compute::BufferPool;
use binius_field::{FieldOps, arch::OptimalPackedB128};
use binius_ip::fracaddcheck::FracAddEvalClaim;
use binius_ip_prover::fracaddcheck::{
	FracAddCheckProver, batch_prove_unequal_depths, fraction::Fraction,
};
use binius_math::{
	FieldBuffer,
	multilinear::evaluate::evaluate,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;

/// Tree depths for the batched benchmark, shaped like a logUp* instance: one tree per looker or
/// table, and one label per shape.
const BATCH_SHAPES: &[(&str, &[usize])] = &[
	("mixed_depths=18,16,14,12", &[18, 16, 14, 12]),
	("equal_depths=18,18,18,18", &[18, 18, 18, 18]),
	("mixed_depths=20,18,16,14,12,10,8,6", &[20, 18, 16, 14, 12, 10, 8, 6]),
];

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
						FieldBuffer::clone_from_slice(&alloc, num_buffer.to_ref()),
						FieldBuffer::clone_from_slice(&alloc, den_buffer.to_ref()),
					)
				},
				|(witness_num, witness_den)| {
					FracAddCheckProver::<_, P>::new(
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
			let (prover, sums) = FracAddCheckProver::new(
				k,
				&alloc,
				Fraction::new(
					FieldBuffer::<P, _>::from_values_in(&alloc, &num_scalars),
					FieldBuffer::<P, _>::from_values_in(&alloc, &den_scalars),
				),
			);
			let sum_num_eval = evaluate(&sums.num, &[]);
			let sum_den_eval = evaluate(&sums.den, &[]);
			let claim = FracAddEvalClaim {
				num_eval: sum_num_eval,
				den_eval: sum_den_eval,
				point: vec![],
			};

			// A transcript shared across iterations grows without bound, so each iteration gets a
			// fresh one and every sample proves the same amount of work.
			b.iter_batched(
				|| (prover.clone(), claim.clone(), ProverTranscript::new(StdChallenger::default())),
				|(prover, claim, mut transcript)| prover.prove(claim, &mut transcript),
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

/// Benchmarks the batched driver logUp* actually calls, over trees of mixed and equal depths.
fn bench_fracaddcheck_batch_unequal_depths(c: &mut Criterion) {
	let mut group = c.benchmark_group("fracaddcheck/batch_prove_unequal_depths");

	for &(label, depths) in BATCH_SHAPES {
		// Every leaf of every tree is one hypercube vertex the batch reduces.
		let total_leaves = depths.iter().map(|&depth| 1u64 << depth).sum();
		group.throughput(Throughput::Elements(total_leaves));
		group.bench_function(label, |b| {
			let mut rng = rand::rng();
			let pool = BufferPool::new();
			let alloc = &pool;

			// The selector variables index the trees, as logUp*'s top circuit hands them over.
			let k = log2_ceil_usize(depths.len());
			let selector_point = random_scalars::<F>(&mut rng, k);

			// Build every tree once, then clone the batch per iteration (untimed setup).
			let (provers, roots): (Vec<_>, Vec<_>) = depths
				.iter()
				.map(|&depth| {
					let num_scalars = random_scalars::<F>(&mut rng, 1 << depth);
					let den_scalars = random_scalars::<F>(&mut rng, 1 << depth);
					let (prover, sums) = FracAddCheckProver::new(
						depth,
						&alloc,
						Fraction::new(
							FieldBuffer::<P, _>::from_values_in(&alloc, &num_scalars),
							FieldBuffer::<P, _>::from_values_in(&alloc, &den_scalars),
						),
					);
					(prover, sums.as_ref().map(|buffer| buffer.get(0)))
				})
				.unzip();

			b.iter_batched(
				|| {
					(
						provers.clone(),
						roots.clone(),
						selector_point.clone(),
						ProverTranscript::new(StdChallenger::default()),
					)
				},
				|(provers, roots, selector_point, mut transcript)| {
					batch_prove_unequal_depths(provers, roots, selector_point, &mut transcript)
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(
	fracaddcheck,
	bench_fracaddcheck_new,
	bench_fracaddcheck_prove,
	bench_fracaddcheck_batch_unequal_depths
);
criterion_main!(fracaddcheck);
