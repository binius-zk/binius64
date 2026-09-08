// Copyright 2026 The Binius Developers

//! Benchmarks the MLE-check prover on the three-column quadratic composition `a * b - c`.
//!
//! That composition is the bit-AND constraint, the widest quadratic the crate proves.

use binius_compute::BufferPool;
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

// A bit-AND constraint vanishes exactly where the product of the two operands equals the result.
fn and_constraint<Pf: PackedField>([a, b, c]: [Pf; 3]) -> Pf {
	a * b - c
}

// The degree-2 part of that constraint, which is what the Karatsuba point at infinity reads.
fn and_constraint_infinity<Pf: PackedField>([a, b, _c]: [Pf; 3]) -> Pf {
	a * b
}

// The claim the MLE check opens: the composition's multilinear extension at the evaluation point.
fn and_constraint_eval_claim(
	a: &FieldBuffer<P>,
	b: &FieldBuffer<P>,
	c: &FieldBuffer<P>,
	eval_point: &[F],
) -> F {
	let n_vars = eval_point.len();
	// The composition is elementwise, so it applies a packed word at a time.
	let packed_len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
	let composite = (0..packed_len)
		.map(|i| and_constraint([a.as_ref()[i], b.as_ref()[i], c.as_ref()[i]]))
		.collect::<Vec<_>>();
	evaluate_inplace(FieldBuffer::new(n_vars, composite), eval_point)
}

// Proves one bit-AND constraint as an MLE-check claim: a shared MLE-check prover driving a single
// three-column quadratic evaluator.
fn bench_quadratic_mlecheck_and_constraint(criterion: &mut Criterion) {
	let mut group = criterion.benchmark_group("quadratic_mlecheck/and_constraint");
	let mut rng = StdRng::seed_from_u64(0);

	// 2^22 is the size the bit-AND reduction runs at on a million-byte message.
	for n_vars in [16, 20, 22] {
		group.throughput(Throughput::Elements(1 << n_vars));
		group.bench_function(format!("n_vars={n_vars}"), |bench| {
			// Three unrelated columns: the constraint does not hold, which the prover never checks
			// and which does not change the work it does.
			let a_buffer = random_field_buffer::<P>(&mut rng, n_vars);
			let b_buffer = random_field_buffer::<P>(&mut rng, n_vars);
			let c_buffer = random_field_buffer::<P>(&mut rng, n_vars);

			let eval_point = random_scalars::<F>(&mut rng, n_vars);
			let eval_claim =
				and_constraint_eval_claim(&a_buffer, &b_buffer, &c_buffer, &eval_point);
			let transcript = ProverTranscript::new(StdChallenger::default());

			let pool = BufferPool::new();
			let alloc = &pool;

			bench.iter_batched(
				|| {
					// The prover folds its columns in place, so each run needs its own copies.
					(
						transcript.clone(),
						FieldBuffer::from_view_in(&alloc, a_buffer.as_view()),
						FieldBuffer::from_view_in(&alloc, b_buffer.as_view()),
						FieldBuffer::from_view_in(&alloc, c_buffer.as_view()),
						eval_point.clone(),
					)
				},
				|(mut transcript, a, b, c, eval_point)| {
					let mut store = MleStore::new(n_vars, &alloc);
					let cols = [a, b, c].map(|col| store.push_owned(col));
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
	}

	group.finish();
}

criterion_group!(quadratic_mlecheck, bench_quadratic_mlecheck_and_constraint);
criterion_main!(quadratic_mlecheck);
