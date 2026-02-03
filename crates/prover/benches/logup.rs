// Copyright 2025-2026 The Binius Developers

use std::array;

use binius_field::{PackedField, arch::OptimalPackedB128};
use binius_iop::channel::OracleSpec;
use binius_iop::naive_channel::NaiveVerifierChannel;
use binius_iop_prover::naive_channel::NaiveProverChannel;
use binius_math::{
	FieldBuffer,
	multilinear::evaluate::evaluate,
	test_utils::{random_field_buffer, random_scalars},
};
use binius_prover::protocols::logup::{LogUp, helper::generate_lookup_values};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{config::StdChallenger, protocols::logup as logup_verifier};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

type P = OptimalPackedB128;
type F = <P as PackedField>::Scalar;

const N_TABLES: usize = 2;
const N_LOOKUPS: usize = 6;
const N_MLES: usize = 2 * N_TABLES;

struct LogUpBenchCase {
	indexes: [Vec<usize>; N_LOOKUPS],
	table_ids: [usize; N_LOOKUPS],
	eval_point: Vec<F>,
	lookup_evals: [F; N_LOOKUPS],
	tables: [FieldBuffer<P>; N_TABLES],
}

fn logup_oracle_specs(table_log_len: usize) -> Vec<OracleSpec> {
	vec![OracleSpec {
		log_msg_len: table_log_len + log2_ceil_usize(N_TABLES),
		is_zk: false,
	}]
}

fn build_case(log_len: usize) -> LogUpBenchCase {
	let eq_log_len = log_len;
	let table_log_len = log_len;
	let lookup_len = 1usize << eq_log_len;
	let table_len = 1usize << table_log_len;

	let mut rng = StdRng::seed_from_u64(0x5eed_u64 + log_len as u64);
	let tables: [FieldBuffer<P>; N_TABLES] =
		array::from_fn(|_| random_field_buffer::<P>(&mut rng, table_log_len));

	let indexes: [Vec<usize>; N_LOOKUPS] = array::from_fn(|_| {
		(0..lookup_len)
			.map(|_| rng.random_range(0..table_len))
			.collect::<Vec<_>>()
	});
	let table_ids: [usize; N_LOOKUPS] = array::from_fn(|i| i % N_TABLES);
	let eval_point = random_scalars::<F>(&mut rng, eq_log_len);

	let lookup_evals: [F; N_LOOKUPS] = array::from_fn(|lookup_idx| {
		let table = &tables[table_ids[lookup_idx]];
		let lookup_values = generate_lookup_values::<P, F>(indexes[lookup_idx].as_slice(), table);
		evaluate(&lookup_values, &eval_point)
	});

	LogUpBenchCase {
		indexes,
		table_ids,
		eval_point,
		lookup_evals,
		tables,
	}
}

fn bench_logup_new(c: &mut Criterion) {
	let mut group = c.benchmark_group("logup/new");

	for log_len in [12usize, 16, 20] {
		let lookup_len = 1usize << log_len;
		group.throughput(Throughput::Elements((lookup_len * N_LOOKUPS) as u64));

		let case = build_case(log_len);
		let oracle_specs = logup_oracle_specs(log_len);

		group.bench_function(format!("log_len={log_len}"), |b| {
			let indexes_ref: [&[usize]; N_LOOKUPS] = array::from_fn(|i| case.indexes[i].as_slice());

			b.iter_batched(
				|| case.tables.clone(),
				|tables| {
					let mut transcript = ProverTranscript::new(StdChallenger::default());
					let mut channel =
						NaiveProverChannel::<F, P, _>::new(&mut transcript, oracle_specs.clone());
					LogUp::<P, _, N_TABLES, N_LOOKUPS>::new(
						indexes_ref,
						case.table_ids,
						&case.eval_point,
						case.lookup_evals,
						tables,
						&mut channel,
					);
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

fn bench_logup_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("logup/prove");

	for log_len in [12usize, 16, 20] {
		let lookup_len = 1usize << log_len;
		group.throughput(Throughput::Elements((lookup_len * N_LOOKUPS) as u64));

		let case = build_case(log_len);
		let oracle_specs = logup_oracle_specs(log_len);

		group.bench_function(format!("log_len={log_len}"), |b| {
			let indexes_ref: [&[usize]; N_LOOKUPS] = array::from_fn(|i| case.indexes[i].as_slice());

			b.iter_batched(
				|| case.tables.clone(),
				|tables| {
					let mut transcript = ProverTranscript::new(StdChallenger::default());
					let mut channel = NaiveProverChannel::<F, P, _>::new(
						&mut transcript,
						oracle_specs.clone(),
					);
					let logup = LogUp::<P, _, N_TABLES, N_LOOKUPS>::new(
						indexes_ref,
						case.table_ids,
						&case.eval_point,
						case.lookup_evals,
						tables,
						&mut channel,
					);
					logup.prove_lookup::<N_MLES>(&mut channel).unwrap();
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

fn bench_logup_prove_pushforward(c: &mut Criterion) {
	let mut group = c.benchmark_group("logup/prove_pushforward");

	for log_len in [12usize, 16, 20] {
		let lookup_len = 1usize << log_len;
		group.throughput(Throughput::Elements((lookup_len * N_LOOKUPS) as u64));

		let case = build_case(log_len);
		let oracle_specs = logup_oracle_specs(log_len);

		group.bench_function(format!("log_len={log_len}"), |b| {
			let indexes_ref: [&[usize]; N_LOOKUPS] = array::from_fn(|i| case.indexes[i].as_slice());

			b.iter_batched(
				|| case.tables.clone(),
				|tables| {
					let mut transcript = ProverTranscript::new(StdChallenger::default());
					let mut channel = NaiveProverChannel::<F, P, _>::new(
						&mut transcript,
						oracle_specs.clone(),
					);
					let logup = LogUp::<P, _, N_TABLES, N_LOOKUPS>::new(
						indexes_ref,
						case.table_ids,
						&case.eval_point,
						case.lookup_evals,
						tables,
						&mut channel,
					);
					logup.prove_pushforward::<N_MLES>(&mut channel).unwrap();
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

fn bench_logup_prove_log_sum(c: &mut Criterion) {
	let mut group = c.benchmark_group("logup/prove_log_sum");

	for log_len in [12usize, 16, 20] {
		let lookup_len = 1usize << log_len;
		group.throughput(Throughput::Elements((lookup_len * N_LOOKUPS) as u64));

		let case = build_case(log_len);
		let oracle_specs = logup_oracle_specs(log_len);

		group.bench_function(format!("log_len={log_len}"), |b| {
			let indexes_ref: [&[usize]; N_LOOKUPS] = array::from_fn(|i| case.indexes[i].as_slice());

			b.iter_batched(
				|| case.tables.clone(),
				|tables| {
					let mut transcript = ProverTranscript::new(StdChallenger::default());
					let mut channel = NaiveProverChannel::<F, P, _>::new(
						&mut transcript,
						oracle_specs.clone(),
					);
					let logup = LogUp::<P, _, N_TABLES, N_LOOKUPS>::new(
						indexes_ref,
						case.table_ids,
						&case.eval_point,
						case.lookup_evals,
						tables,
						&mut channel,
					);
					logup
						.prove_pushforward::<N_MLES>(&mut channel)
						.unwrap();
					logup.prove_log_sum(&mut channel).unwrap();
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

fn bench_logup_verify(c: &mut Criterion) {
	let mut group = c.benchmark_group("logup/verify");

	for log_len in [12usize, 16, 20] {
		let eq_log_len = log_len;
		let table_log_len = log_len;
		let lookup_len = 1usize << eq_log_len;
		group.throughput(Throughput::Elements((lookup_len * N_LOOKUPS) as u64));

		let case = build_case(log_len);
		let oracle_specs = logup_oracle_specs(log_len);

		group.bench_function(format!("log_len={log_len}"), |b| {
			let indexes_ref: [&[usize]; N_LOOKUPS] = array::from_fn(|i| case.indexes[i].as_slice());

			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			let mut prover_channel =
				NaiveProverChannel::<F, P, _>::new(&mut prover_transcript, oracle_specs.clone());
			let logup = LogUp::<P, _, N_TABLES, N_LOOKUPS>::new(
				indexes_ref,
				case.table_ids,
				&case.eval_point,
				case.lookup_evals,
				case.tables.clone(),
				&mut prover_channel,
			);
			logup
				.prove_lookup::<N_MLES>(&mut prover_channel)
				.unwrap();

			drop(prover_channel);
			let proof = prover_transcript.finalize();
			let verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);

			b.iter(|| {
				let mut transcript = verifier_transcript.clone();
				let mut verifier_channel =
					NaiveVerifierChannel::<F, _>::new(&mut transcript, &oracle_specs);
				logup_verifier::verify_lookup::<F, _, N_TABLES>(
					eq_log_len,
					table_log_len,
					&case.eval_point,
					&case.lookup_evals,
					&case.table_ids,
					&mut verifier_channel,
				)
				.unwrap();
			});
		});
	}

	group.finish();
}

criterion_group!(
	logup,
	bench_logup_new,
	bench_logup_prove,
	bench_logup_prove_pushforward,
	bench_logup_prove_log_sum,
	bench_logup_verify
);
criterion_main!(logup);
