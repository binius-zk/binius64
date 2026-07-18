// Copyright 2026 The Binius Developers

//! Benchmarks the store-backed round-evaluator provers on a single bivariate product `a * b`:
//! a [`SharedSumcheckProver`] driving one [`BivariateProductEvaluator`] (the plain sum claim over
//! the hypercube), and a [`SharedMleCheckProver`] driving one [`QuadraticMleEvaluator`] (the
//! evaluation claim on the product's multilinear extension at a random point).

use binius_field::{FieldOps, PackedField, arch::OptimalPackedB128};
use binius_ip_prover::sumcheck::{
	self,
	bivariate_product_evaluator::BivariateProductEvaluator,
	mle_store::MleStore,
	quadratic_mle_evaluator::QuadraticMleEvaluator,
	round_evaluator::{SharedMleCheckProver, SharedSumcheckProver},
};
use binius_math::{
	FieldBuffer,
	inner_product::inner_product_par,
	multilinear::evaluate::evaluate_inplace,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;
type StdChallenger = HasherChallenger<sha2::Sha256>;

// The bivariate product composition `a * b`. Its only term is the degree-2 product, so the same
// function serves as the Karatsuba infinity composition.
fn product<Pf: PackedField>([a, b]: [Pf; 2]) -> Pf {
	a * b
}

// The evaluation of the product `a * b`'s multilinear extension at `eval_point`, the MLE-check
// claim.
fn product_eval_claim(a: &FieldBuffer<P>, b: &FieldBuffer<P>, eval_point: &[F]) -> F {
	let n_vars = eval_point.len();
	let packed_len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
	let product_vals: Vec<P> = (0..packed_len)
		.map(|i| a.as_ref()[i] * b.as_ref()[i])
		.collect();
	evaluate_inplace(FieldBuffer::new(n_vars, product_vals), eval_point)
}

// Proves one bivariate product `a * b` as a plain sum claim: a `SharedSumcheckProver` with a single
// `BivariateProductEvaluator` over the two store columns.
fn bench_shared_sumcheck_bivariate_product(c: &mut Criterion) {
	let mut group = c.benchmark_group("bivariate_product/shared_sumcheck");
	let mut rng = rand::rng();

	for n_vars in [12, 16, 20] {
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let a = random_field_buffer::<P>(&mut rng, n_vars);
			let b_multilinear = random_field_buffer::<P>(&mut rng, n_vars);
			// The plain sum claim is the sum of `a * b` over the hypercube.
			let sum_claim = inner_product_par(&a, &b_multilinear);
			let transcript = ProverTranscript::new(StdChallenger::default());

			b.iter_batched(
				|| (transcript.clone(), a.clone(), b_multilinear.clone()),
				|(mut transcript, a, b_multilinear)| {
					let mut store = MleStore::new(n_vars);
					let cols = [a, b_multilinear].map(|col| store.push_owned(col));
					let evaluator = BivariateProductEvaluator::new(cols);
					let prover = SharedSumcheckProver::new(store, [(sum_claim, evaluator)]);

					sumcheck::prove_single(prover, &mut transcript)
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

// Proves one bivariate product `a * b` as an MLE-check claim: a `SharedMleCheckProver` with a
// single `QuadraticMleEvaluator` for the quadratic composition `a * b` over the two store columns.
fn bench_shared_mlecheck_bivariate_product(c: &mut Criterion) {
	let mut group = c.benchmark_group("bivariate_product/shared_mlecheck");
	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in [12, 16, 20] {
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |b| {
			let a = random_field_buffer::<P>(&mut rng, n_vars);
			let b_multilinear = random_field_buffer::<P>(&mut rng, n_vars);
			let eval_point = random_scalars::<F>(&mut rng, n_vars);
			let eval_claim = product_eval_claim(&a, &b_multilinear, &eval_point);
			let transcript = ProverTranscript::new(StdChallenger::default());

			b.iter_batched(
				|| (transcript.clone(), a.clone(), b_multilinear.clone(), eval_point.clone()),
				|(mut transcript, a, b_multilinear, eval_point)| {
					let mut store = MleStore::new(n_vars);
					let cols = [a, b_multilinear].map(|col| store.push_owned(col));
					// A single-claim evaluator reading the store's columns and one eq tracker
					// registered for the claim's evaluation point.
					let eq_tracker = store.register_eq_tracker(&eval_point);
					let evaluator =
						QuadraticMleEvaluator::new(cols, eq_tracker, product::<P>, product::<P>);
					let prover =
						SharedMleCheckProver::new(store, [(eval_claim, evaluator)], eval_point);

					sumcheck::prove_single_mlecheck(prover, &mut transcript)
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(
	bivariate_product,
	bench_shared_sumcheck_bivariate_product,
	bench_shared_mlecheck_bivariate_product
);
criterion_main!(bivariate_product);
