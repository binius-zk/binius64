// Copyright 2025 Irreducible Inc.

use binius_field::{Field, FieldOps, PackedField, Random, arch::OptimalPackedB128};
use binius_ip_prover::sumcheck::bivariate_product::{
	compute_round_evals_wide_par, compute_round_evals_wide_seq,
};
use binius_math::{
	inner_product::inner_product_par,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_prover::protocols::sumcheck::{
	bivariate_product::BivariateProductSumcheckProver, prove_single, prove_single_mlecheck,
	quadratic_mle::QuadraticMleCheckProver,
};
use binius_transcript::ProverTranscript;
use binius_utils::rayon::prelude::*;
use binius_verifier::config::StdChallenger;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, prelude::StdRng};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;

fn _assert_scalar_is_field()
where
	F: Field,
{
}

fn bench_sumcheck_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("sumcheck/bivariate_product");

	// Test different sizes of multilinear polynomials
	for n_vars in [12, 16, 20] {
		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			// Setup phase - prepare the multilinears and compute the sum
			let mut rng = StdRng::seed_from_u64(0);
			let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
			let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);
			let sum = inner_product_par(&multilinear_a, &multilinear_b);

			let mut transcript = ProverTranscript::new(StdChallenger::default());

			// Benchmark only the proving phase
			b.iter_batched(
				|| [multilinear_a.clone(), multilinear_b.clone()],
				|multilinears| {
					let prover = BivariateProductSumcheckProver::new(multilinears, sum).unwrap();
					prove_single(prover, &mut transcript).unwrap()
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

fn bench_mlecheck_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("mlecheck");

	let mut rng = StdRng::seed_from_u64(0);

	// Test different sizes of multilinear polynomials
	for n_vars in [12, 16, 20] {
		// Consider each element to be one hypercube vertex.
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("A*B/n_vars={n_vars}"), |b| {
			let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
			let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);

			let eval_point = random_scalars(&mut rng, n_vars);
			let eval_claim = inner_product_par(&multilinear_a, &multilinear_b);

			let mut transcript = ProverTranscript::new(StdChallenger::default());

			// Benchmark only the proving phase
			b.iter_batched(
				|| [multilinear_a.clone(), multilinear_b.clone()],
				|multilinears| {
					let prover = QuadraticMleCheckProver::new(
						multilinears,
						|[a, b]| a * b,
						|[a, b]| a * b,
						eval_point.clone(),
						eval_claim,
					)
					.unwrap();

					prove_single_mlecheck(prover, &mut transcript).unwrap()
				},
				BatchSize::SmallInput,
			);
		});

		// Benchmark mul gate composition: a * b - c
		group.bench_function(format!("A*B-C/n_vars={n_vars}"), |b| {
			let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
			let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);
			let multilinear_c = random_field_buffer::<P>(&mut rng, n_vars);

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
					let prover = QuadraticMleCheckProver::new(
						multilinears,
						|[a, b, c]| a * b - c,
						|[a, b, _c]| a * b,
						eval_point.clone(),
						eval_claim,
					)
					.unwrap();

					prove_single_mlecheck(prover, &mut transcript).unwrap()
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

/// Isolates a single sumcheck round of `BivariateProductSumcheckProver::execute` for both
/// dispatch variants (parallel vs. sequential widening), parameterized by the packed
/// half-length `log_half = n_vars_remaining - 1`. Used to pick the `PAR_THRESHOLD_LOG_HALF`
/// gate in `bivariate_product.rs`: the crossover is the smallest `log_half` at which
/// `wide_par` begins to beat `wide_seq`.
///
/// At half-length `log_half` each of the four input buffers has `2^log_half` packed elements.
fn bench_bivariate_round(c: &mut Criterion) {
	let mut group = c.benchmark_group("bivariate_round");

	for log_half in 0usize..=20 {
		let mut rng = StdRng::seed_from_u64(0xb1);
		let n_packed = 1usize << log_half;
		let mk = |rng: &mut StdRng| -> Vec<P> {
			(0..n_packed).map(|_| P::random(&mut *rng)).collect()
		};
		let a_0 = mk(&mut rng);
		let a_1 = mk(&mut rng);
		let b_0 = mk(&mut rng);
		let b_1 = mk(&mut rng);

		// Count each call as processing `2 * 2^log_half` packed multiplications.
		group.throughput(Throughput::Elements(2 * n_packed as u64));

		group.bench_function(format!("wide_par/log_half={log_half}"), |b| {
			b.iter(|| {
				let out = compute_round_evals_wide_par::<F, P>(
					black_box(&a_0),
					black_box(&a_1),
					black_box(&b_0),
					black_box(&b_1),
				);
				black_box(out);
			});
		});

		group.bench_function(format!("wide_seq/log_half={log_half}"), |b| {
			b.iter(|| {
				let out = compute_round_evals_wide_seq::<F, P>(
					black_box(&a_0),
					black_box(&a_1),
					black_box(&b_0),
					black_box(&b_1),
				);
				black_box(out);
			});
		});
	}

	group.finish();
}

criterion_group!(sumcheck, bench_sumcheck_prove, bench_mlecheck_prove, bench_bivariate_round);
criterion_main!(sumcheck);
