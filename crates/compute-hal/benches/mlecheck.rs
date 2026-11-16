// Copyright 2026 The Binius Developers

//! Gruen MLE-check of `a * b - c`: the real `SharedMleCheckProver` against the HAL driver.

use binius_compute::BufferPool;
use binius_compute_hal::{Composition, Compute, Elem, SimdCompute, sumcheck::prove_mlecheck};
use binius_field::{FieldOps, PackedField, arch::OptimalPackedB128};
use binius_ip_prover::sumcheck::{
	self, mle_store::MleStore, quadratic_mle_evaluator::QuadraticMleEvaluator,
	round_evaluator::SharedMleCheckProver,
};
use binius_math::{
	FieldBuffer,
	multilinear::evaluate::evaluate_inplace,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;
type StdChallenger = HasherChallenger<sha2::Sha256>;

fn and_constraint<Pf: PackedField>([a, b, c]: [Pf; 3]) -> Pf {
	a * b - c
}

fn and_constraint_infinity<Pf: PackedField>([a, b, _c]: [Pf; 3]) -> Pf {
	a * b
}

struct AndC;

impl Composition<F> for AndC {
	const N_IN: usize = 3;
	const DEGREE: usize = 2;

	fn eval<T: Elem<F>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone() - x[2].clone()
	}
	fn eval_infinity<T: Elem<F>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone()
	}
}

fn bench_mlecheck(criterion: &mut Criterion) {
	let mut group = criterion.benchmark_group("mlecheck/and_constraint");
	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in [16, 20, 22] {
		group.throughput(Throughput::Elements(1 << n_vars));
		let [a, b, c] = [0, 1, 2].map(|_| random_field_buffer::<P>(&mut rng, n_vars));
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let composite = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.map(|i| and_constraint([a.as_ref()[i], b.as_ref()[i], c.as_ref()[i]]))
			.collect::<Vec<_>>();
		let eval_claim = evaluate_inplace(FieldBuffer::new(n_vars, composite), &eval_point);
		let transcript = ProverTranscript::new(StdChallenger::default());

		let pool = BufferPool::new();
		let alloc = &pool;
		group.bench_function(format!("ip_prover/n_vars={n_vars}"), |bench| {
			bench.iter_batched(
				|| {
					(
						transcript.clone(),
						[&a, &b, &c].map(|col| FieldBuffer::from_view_in(&alloc, col.as_view())),
						eval_point.clone(),
					)
				},
				|(mut transcript, cols, eval_point)| {
					let mut store = MleStore::new(n_vars, &alloc);
					let cols = cols.map(|col| store.push_owned(col));
					let evaluator = QuadraticMleEvaluator::new(
						cols,
						and_constraint::<P>,
						and_constraint_infinity::<P>,
					);
					let prover =
						SharedMleCheckProver::new(store, [(eval_claim, evaluator)], eval_point);
					sumcheck::prove_single_mlecheck(prover, &mut transcript)
				},
				BatchSize::PerIteration,
			);
		});

		let compute = SimdCompute::<P>::new();
		group.bench_function(format!("hal_simd/n_vars={n_vars}"), |bench| {
			bench.iter_batched(
				|| {
					(
						transcript.clone(),
						[&a, &b, &c]
							.map(|col| {
								compute
									.copy_field_from_host(&col.iter_scalars().collect::<Vec<_>>())
							})
							.into(),
					)
				},
				|(mut transcript, cols)| {
					prove_mlecheck(&compute, &AndC, &eval_point, cols, eval_claim, &mut transcript)
				},
				BatchSize::PerIteration,
			);
		});
	}

	group.finish();
}

criterion_group!(mlecheck, bench_mlecheck);
criterion_main!(mlecheck);
