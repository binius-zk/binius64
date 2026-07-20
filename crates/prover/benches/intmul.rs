// Copyright 2025-2026 The Binius Developers
use binius_core::word::Word;
use binius_field::{BinaryField128bGhash, Field, PackedBinaryGhash1x128b};
use binius_hash::StdHashSuite;
use binius_iop::{
	basefold::compiler::BaseFoldVerifierCompiler, channel::OracleSpec, fri::MinProofSizeStrategy,
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::basefold::compiler::BaseFoldProverCompiler;
use binius_ip_prover::{
	prodcheck::ProdcheckProver,
	sumcheck::{
		batch::batch_prove,
		selector_mle::{Claim, SelectorMlecheckProver},
	},
};
use binius_math::{
	multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
	ntt::{NeighborsLastSingleThread, domain_context::GenericPreExpanded},
	test_utils::random_scalars,
};
use binius_prover::protocols::intmul::{
	prove::IntMulProver,
	witness::{Witness, compute_b_leaves, power_table},
};
use binius_transcript::ProverTranscript;
use binius_utils::rayon::ThreadPoolBuilder;
use binius_verifier::{
	config::StdChallenger,
	protocols::intmul::common::{LIMB_BITS, frobenius_twist},
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

type P = PackedBinaryGhash1x128b;
type F = BinaryField128bGhash;

/// Number of exponents is `2^LOG_NUM`.
const LOG_NUM: usize = 14;

/// The Reed-Solomon rate of the standard prover configuration.
const LOG_INV_RATE: usize = 1;
/// One FRI test query keeps the parameters cheap; the benchmarks measure commitment cost, not
/// opening soundness.
const N_TEST_QUERIES: usize = 1;

/// Builds a BaseFold prover compiler for the prove benchmarks.
///
/// The prove benchmarks run over a real BaseFold channel so that oracle transmission (NTT
/// encoding and Merkle tree construction) is measured; a naive channel would serialize
/// oracles for free. The final batched opening in `finish` is not measured, since the full
/// system amortizes it into the single opening shared with the witness trace.
fn basefold_compiler() -> BaseFoldProverCompiler<P, NeighborsLastSingleThread<GenericPreExpanded<F>>>
{
	let verifier_compiler = BaseFoldVerifierCompiler::new(
		BinaryMerkleTreeScheme::<F, StdHashSuite>::new(),
		vec![OracleSpec::new(LIMB_BITS)],
		LOG_INV_RATE,
		N_TEST_QUERIES,
		&MinProofSizeStrategy,
	);
	let domain_context =
		GenericPreExpanded::generate_from_subspace(verifier_compiler.max_subspace());
	let ntt = NeighborsLastSingleThread::new(domain_context);
	BaseFoldProverCompiler::from_verifier_compiler(&verifier_compiler, ntt)
}

fn generate_test_data(log_num: usize) -> (Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>) {
	let num_exponents = 1 << log_num;
	let mut rng = StdRng::seed_from_u64(0);

	let mut a = Vec::with_capacity(num_exponents);
	let mut b = Vec::with_capacity(num_exponents);
	let mut c_lo = Vec::with_capacity(num_exponents);
	let mut c_hi = Vec::with_capacity(num_exponents);

	for _ in 0..num_exponents {
		let a_i = rng.random_range(1..u64::MAX);
		let b_i = rng.random_range(1..u64::MAX);

		let full_result = (a_i as u128) * (b_i as u128);

		let c_lo_i = full_result as u64;
		let c_hi_i = (full_result >> 64) as u64;

		a.push(Word::from_u64(a_i));
		b.push(Word::from_u64(b_i));
		c_lo.push(Word::from_u64(c_lo_i));
		c_hi.push(Word::from_u64(c_hi_i));
	}

	(a, b, c_lo, c_hi)
}

fn bench_intmul_prove(c: &mut Criterion) {
	ThreadPoolBuilder::new().num_threads(1).build_global().ok();

	let mut group = c.benchmark_group("intmul/prove");
	group.sample_size(10);
	group.throughput(Throughput::Elements(1));

	let num_exponents = 1 << LOG_NUM;
	let (a, b, c_lo, c_hi) = generate_test_data(LOG_NUM);

	group.bench_with_input(
		BenchmarkId::new("witness", num_exponents),
		&num_exponents,
		|bencher, _| bencher.iter(|| Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap()),
	);

	// prove
	let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();
	let compiler = basefold_compiler();

	group.bench_with_input(
		BenchmarkId::new("prove", num_exponents),
		&witness,
		|bencher, witness| {
			bencher.iter_batched_ref(
				|| {
					let channel = compiler
						.create_channel_without_zk_from_transcript::<StdHashSuite, StdChallenger, _>(
							ProverTranscript::default(),
						);
					(Some(witness.clone()), channel)
				},
				|(witness, channel)| {
					let mut intmul_prover = IntMulProver::new(0, channel);
					intmul_prover.prove(witness.take().expect("set in setup"));
				},
				BatchSize::SmallInput,
			)
		},
	);

	// execute + prove
	group.bench_with_input(
		BenchmarkId::new("combined", num_exponents),
		&num_exponents,
		|bencher, _| {
			bencher.iter_batched_ref(
				|| {
					compiler
						.create_channel_without_zk_from_transcript::<StdHashSuite, StdChallenger, _>(
							ProverTranscript::default(),
						)
				},
				|channel| {
					let mut intmul_prover = IntMulProver::new(0, channel);
					let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();
					intmul_prover.prove(witness);
				},
				BatchSize::SmallInput,
			)
		},
	);

	group.finish();
}

fn bench_intmul_phases(c: &mut Criterion) {
	let mut group = c.benchmark_group("intmul/phases");
	group.sample_size(10);
	group.throughput(Throughput::Elements(1 << LOG_NUM));

	let (a, b, c_lo, c_hi) = generate_test_data(LOG_NUM);
	let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();

	// The proving phases are sequential and stateful: each consumes the outputs of the previous
	// one. Rather than re-deriving every predecessor inside each phase's per-iteration setup, we
	// advance the protocol once here to capture each phase's inputs; the setup closures below then
	// only clone the datastructures a phase consumes by value. The specific transcript challenges
	// do not affect the work a phase performs, so a throwaway transcript (and a random initial
	// evaluation point) is sufficient.
	let n_vars = witness.b_root.log_len();
	let initial_eval_point = random_scalars::<F>(&mut rand::rng(), n_vars);
	let exp_eval = evaluate(&witness.b_root, &initial_eval_point);

	let phase1 = {
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover = IntMulProver::<P, _>::new(0, &mut transcript);
		prover.phase1(&initial_eval_point, witness.b_prodcheck.clone(), &witness.b_leaves, exp_eval)
	};
	let phase2 = frobenius_twist(Word::LOG_BITS, &phase1.eval_point, &phase1.b_leaves_evals);
	let phase3 = {
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover = IntMulProver::<P, _>::new(0, &mut transcript);
		prover.phase3(
			&phase2.twisted_eval_points,
			&phase2.twisted_evals,
			witness.a_root.clone(),
			witness.b_exponents,
			[witness.c_lo_root.clone(), witness.c_hi_root.clone()],
			&initial_eval_point,
			exp_eval,
		)
	};
	let phase4 = {
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover = IntMulProver::<P, _>::new(0, &mut transcript);
		prover.phase4(
			&phase3.eval_point,
			(phase3.gpow_a_eval, witness.a_prodcheck.clone()),
			(phase3.gpow_c_lo_eval, witness.c_lo_prodcheck.clone()),
			(phase3.gpow_c_hi_eval, witness.c_hi_prodcheck.clone()),
			[
				witness.a_exponents,
				witness.c_lo_exponents,
				witness.c_hi_exponents,
			],
			&witness.tables,
		)
	};

	group.bench_function("phase1", |bencher| {
		bencher.iter_batched(
			|| witness.b_prodcheck.clone(),
			|b_prodcheck| {
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let mut prover = IntMulProver::<P, _>::new(0, &mut transcript);
				prover.phase1(&initial_eval_point, b_prodcheck, &witness.b_leaves, exp_eval)
			},
			BatchSize::SmallInput,
		);
	});

	// The Frobenius twist consumes nothing by value, so no per-iteration clone is needed.
	group.bench_function("phase2_frobenius_twist", |bencher| {
		bencher
			.iter(|| frobenius_twist(Word::LOG_BITS, &phase1.eval_point, &phase1.b_leaves_evals));
	});

	group.bench_function("phase3", |bencher| {
		bencher.iter_batched(
			|| (witness.a_root.clone(), [witness.c_lo_root.clone(), witness.c_hi_root.clone()]),
			|(a_root, c_lo_hi_roots)| {
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let mut prover = IntMulProver::<P, _>::new(0, &mut transcript);
				prover.phase3(
					&phase2.twisted_eval_points,
					&phase2.twisted_evals,
					a_root,
					witness.b_exponents,
					c_lo_hi_roots,
					&initial_eval_point,
					exp_eval,
				)
			},
			BatchSize::SmallInput,
		);
	});

	group.bench_function("phase4", |bencher| {
		bencher.iter_batched(
			|| {
				(
					witness.a_prodcheck.clone(),
					witness.c_lo_prodcheck.clone(),
					witness.c_hi_prodcheck.clone(),
				)
			},
			|(a_prodcheck, c_lo_prodcheck, c_hi_prodcheck)| {
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let mut prover = IntMulProver::<P, _>::new(0, &mut transcript);
				prover.phase4(
					&phase3.eval_point,
					(phase3.gpow_a_eval, a_prodcheck),
					(phase3.gpow_c_lo_eval, c_lo_prodcheck),
					(phase3.gpow_c_hi_eval, c_hi_prodcheck),
					[
						witness.a_exponents,
						witness.c_lo_exponents,
						witness.c_hi_exponents,
					],
					&witness.tables,
				)
			},
			BatchSize::SmallInput,
		);
	});

	let compiler = basefold_compiler();
	group.bench_function("phase5", |bencher| {
		bencher.iter_batched_ref(
			|| {
				compiler
					.create_channel_without_zk_from_transcript::<StdHashSuite, StdChallenger, _>(
						ProverTranscript::default(),
					)
			},
			|channel| {
				let mut prover = IntMulProver::<P, _>::new(0, channel);
				prover.phase5(
					&phase4,
					witness.b_exponents,
					&phase3.eval_point,
					&phase3.r_ib,
					phase3.b_recomb,
					witness.a_exponents,
					witness.c_lo_exponents,
					witness.c_hi_exponents,
					&witness.tables[0],
				)
			},
			BatchSize::SmallInput,
		);
	});

	group.finish();
}

fn bench_intmul_components(c: &mut Criterion) {
	let mut group = c.benchmark_group("intmul/components");
	group.sample_size(10);
	group.throughput(Throughput::Elements(1 << LOG_NUM));

	let (a, b, c_lo, c_hi) = generate_test_data(LOG_NUM);
	let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();

	// Building one twisted power table (the `a` / `c_lo` / `c_hi` limb columns read 2·N_LIMBS of
	// these).
	let g = F::MULTIPLICATIVE_GENERATOR;
	group.bench_function("power_table", |bencher| {
		bencher.iter(|| power_table::<F, P>(LIMB_BITS, g));
	});

	// Computing the leaves of the variable-base exponentiation tree (`a` root as base, `b` as
	// exponents).
	group.bench_function("b_leaves", |bencher| {
		bencher.iter(|| compute_b_leaves::<F, P>(&witness.a_root, witness.b_exponents));
	});

	// Computing a product tree over the leaves.
	group.bench_function("product_tree", |bencher| {
		bencher.iter_batched(
			|| witness.b_leaves.clone(),
			|b_leaves| ProdcheckProver::<P>::new(Word::LOG_BITS, b_leaves),
			BatchSize::SmallInput,
		);
	});

	// The selector sumcheck in phase 3 (constructing and proving the `SelectorMlecheckProver`).
	// Its input claims come from the phase 2 (Frobenius twist) output, which we derive once here so
	// the per-iteration setup only clones what `SelectorMlecheckProver::new` consumes. `Word` is
	// `repr(transparent)` over `u64`, so the exponent bitmasks reinterpret the slice in place.
	let b_bitmasks: &[u64] = bytemuck::cast_slice(witness.b_exponents);
	let n_vars = witness.b_root.log_len();
	let initial_eval_point = random_scalars::<F>(&mut rand::rng(), n_vars);
	let exp_eval = evaluate(&witness.b_root, &initial_eval_point);
	let phase1 = {
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover = IntMulProver::<P, _>::new(0, &mut transcript);
		prover.phase1(&initial_eval_point, witness.b_prodcheck.clone(), &witness.b_leaves, exp_eval)
	};
	let phase2 = frobenius_twist(Word::LOG_BITS, &phase1.eval_point, &phase1.b_leaves_evals);

	group.bench_function("selector_sumcheck", |bencher| {
		let mut rng = rand::rng();
		bencher.iter_batched(
			|| {
				let claims: Vec<Claim<F>> = phase2
					.twisted_eval_points
					.iter()
					.zip(&phase2.twisted_evals)
					.map(|(point, &value)| Claim {
						point: point.clone(),
						value,
					})
					.collect();
				let gamma = random_scalars::<F>(&mut rng, Word::LOG_BITS);
				let eq_weights = eq_ind_partial_eval_scalars::<F>(&gamma);
				(witness.a_root.clone(), claims, eq_weights)
			},
			|(a_root, claims, eq_weights)| {
				let selector_prover =
					SelectorMlecheckProver::new(a_root, claims, b_bitmasks, eq_weights, 0);
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				batch_prove(vec![selector_prover], &mut transcript)
			},
			BatchSize::SmallInput,
		);
	});

	group.finish();
}

criterion_group!(intmul_benches, bench_intmul_prove, bench_intmul_phases, bench_intmul_components);
criterion_main!(intmul_benches);
