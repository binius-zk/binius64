// Copyright 2025-2026 The Binius Developers

use std::array;

use binius_field::{Field, FieldOps, PackedField, arch::OptimalPackedB128};
use binius_ip::mlecheck;
use binius_ip_prover::sumcheck::{
	common::MleCheckProver,
	mle_store::MleStore,
	quadratic_mle_evaluator::QuadraticMleEvaluator,
	round_evaluator::{RoundEvaluator, SharedMleCheckProver},
};
use binius_math::{
	FieldBuffer,
	multilinear::evaluate::evaluate_inplace,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

type P = OptimalPackedB128;
type F = <P as FieldOps>::Scalar;

const N: usize = 3;
const M: usize = 2;

fn comp_0<Pf: PackedField>([a, b, c]: [Pf; N]) -> Pf {
	a * b - c
}

fn comp_0_inf<Pf: PackedField>([a, b, _c]: [Pf; N]) -> Pf {
	a * b
}

fn comp_1<Pf: PackedField>([a, b, c]: [Pf; N]) -> Pf {
	(a + b) * c
}

fn comp_1_inf<Pf: PackedField>([a, b, c]: [Pf; N]) -> Pf {
	(a + b) * c
}

fn eval_claims<Ff, Pf>(multilinears: &[FieldBuffer<Pf>; N], eval_point: &[Ff]) -> [Ff; M]
where
	Ff: Field,
	Pf: PackedField<Scalar = Ff>,
{
	let n_vars = eval_point.len();
	let packed_len = 1 << n_vars.saturating_sub(Pf::LOG_WIDTH);
	array::from_fn(|claim_idx| {
		let mut composite_vals = Vec::with_capacity(packed_len);
		for i in 0..packed_len {
			let evals = array::from_fn(|j| multilinears[j].as_ref()[i]);
			let composed = match claim_idx {
				0 => comp_0(evals),
				1 => comp_1(evals),
				_ => unreachable!("M is fixed to 2"),
			};
			composite_vals.push(composed);
		}
		let composite_buffer = FieldBuffer::new(n_vars, composite_vals);
		evaluate_inplace(composite_buffer, eval_point)
	})
}

fn prove_batch_mlecheck<Ff, Challenger_, Prover>(
	mut prover: Prover,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Vec<Ff>
where
	Ff: Field,
	Challenger_: Challenger,
	Prover: MleCheckProver<Ff>,
{
	let n_vars = prover.n_vars();

	for _ in 0..n_vars {
		let round_coeffs_vec = prover.execute();
		for round_coeffs in round_coeffs_vec {
			transcript
				.message()
				.write_slice(mlecheck::RoundProof::truncate(round_coeffs).coeffs());
		}

		let challenge = transcript.sample();
		prover.fold(challenge);
	}

	prover.finish()
}

fn bench_batch_quadratic_mlecheck_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("mlecheck/batch_quadratic");
	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in [12, 16, 20] {
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}/claims={M}"), |b| {
			let multilinears: [FieldBuffer<P>; N] =
				array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));
			let eval_point = random_scalars::<F>(&mut rng, n_vars);
			let eval_claims = eval_claims::<F, P>(&multilinears, &eval_point);

			let mut transcript = ProverTranscript::new(StdChallenger::default());

			b.iter_batched(
				|| (multilinears.clone(), eval_point.clone()),
				|(multilinears, eval_point)| {
					let mut store = MleStore::new(eval_point.len());
					let cols = multilinears.map(|multilinear| store.push_owned(multilinear));
					// One single-claim evaluator per composition, sharing the store's columns and
					// one eq tracker registered for the shared point.
					let eq_tracker = store.register_eq_tracker(&eval_point);
					let evaluator_0 =
						QuadraticMleEvaluator::new(cols, eq_tracker, comp_0::<P>, comp_0_inf::<P>);
					let evaluator_1 =
						QuadraticMleEvaluator::new(cols, eq_tracker, comp_1::<P>, comp_1_inf::<P>);
					let claims_with_evaluators: [(F, Box<dyn RoundEvaluator<F, P>>); 2] = [
						(eval_claims[0], Box::new(evaluator_0)),
						(eval_claims[1], Box::new(evaluator_1)),
					];
					let prover =
						SharedMleCheckProver::new(store, claims_with_evaluators, eval_point);

					prove_batch_mlecheck(prover, &mut transcript)
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(batch_quadratic_mle, bench_batch_quadratic_mlecheck_prove);
criterion_main!(batch_quadratic_mle);
