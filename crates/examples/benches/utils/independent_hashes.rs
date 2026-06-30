// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Shared runner for independent hash primitive benchmarks.

use std::{env, marker::PhantomData, time::Duration};

use binius_examples::{
	ExampleCircuit,
	circuits::independent_hashes::{Instance, PrimitiveParams},
};
use binius_verifier::{config::StdChallenger, transcript::ProverTranscript};
use criterion::{BenchmarkId, Criterion, Throughput};
use peakmem_alloc::PeakMemAllocTrait;

use super::{
	config::{DEFAULT_HASH_LOG_INV_RATE, DEFAULT_INDEPENDENT_NUM_PRIMITIVES},
	reporting::print_benchmark_header,
	runner::{
		ConstraintSystemBenchmarkContext, ExampleBenchmark, run_cs_benchmark_with_extra_groups,
	},
};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum IndependentPrimitive {
	Sha256,
	Blake3,
	Keccak,
}

impl IndependentPrimitive {
	const fn label(self) -> &'static str {
		match self {
			Self::Sha256 => "SHA-256 compressions",
			Self::Blake3 => "BLAKE3 compressions",
			Self::Keccak => "Keccak-f[1600] permutations",
		}
	}

	const fn group_prefix(self) -> &'static str {
		match self {
			Self::Sha256 => "sha256_compressions",
			Self::Blake3 => "blake3_compressions",
			Self::Keccak => "keccak_permutations",
		}
	}

	const fn count_env_key(self) -> &'static str {
		match self {
			Self::Sha256 | Self::Blake3 => "N_COMPRESSIONS",
			Self::Keccak => "N_PERMUTATIONS",
		}
	}
}

struct IndependentHashBenchmark<E> {
	primitive: IndependentPrimitive,
	num_primitives: usize,
	log_inv_rate: usize,
	_marker: PhantomData<E>,
}

impl<E> IndependentHashBenchmark<E> {
	fn new(primitive: IndependentPrimitive) -> Self {
		let num_primitives = env_usize(primitive.count_env_key())
			.or_else(|| env_usize("N_PRIMITIVES"))
			.unwrap_or(DEFAULT_INDEPENDENT_NUM_PRIMITIVES);
		let log_inv_rate = env_usize("LOG_INV_RATE").unwrap_or(DEFAULT_HASH_LOG_INV_RATE);

		Self {
			primitive,
			num_primitives,
			log_inv_rate,
			_marker: PhantomData,
		}
	}
}

impl<E> ExampleBenchmark for IndependentHashBenchmark<E>
where
	E: ExampleCircuit<Params = PrimitiveParams, Instance = Instance>,
{
	type Params = PrimitiveParams;
	type Instance = Instance;
	type Example = E;

	fn create_params(&self) -> Self::Params {
		PrimitiveParams {
			num_primitives: self.num_primitives,
		}
	}

	fn create_instance(&self) -> Self::Instance {
		Instance { seed: None }
	}

	fn bench_name(&self) -> String {
		format!("n_{}", self.num_primitives)
	}

	fn throughput(&self) -> Throughput {
		Throughput::Elements(self.num_primitives as u64)
	}

	fn proof_description(&self) -> String {
		format!("{} {}", self.num_primitives, self.primitive.label())
	}

	fn log_inv_rate(&self) -> usize {
		self.log_inv_rate
	}

	fn print_params(&self) {
		let params = vec![
			("Primitive".to_string(), self.primitive.label().to_string()),
			("Count".to_string(), self.num_primitives.to_string()),
			("Log inverse rate".to_string(), self.log_inv_rate.to_string()),
		];
		print_benchmark_header("Independent hash primitives", &params);
	}
}

pub fn run_independent_hash_benchmark<E>(
	c: &mut Criterion,
	primitive: IndependentPrimitive,
	peak_alloc: &impl PeakMemAllocTrait,
) where
	E: ExampleCircuit<Params = PrimitiveParams, Instance = Instance>,
{
	let benchmark = IndependentHashBenchmark::<E>::new(primitive);
	run_cs_benchmark_with_extra_groups(
		c,
		benchmark,
		primitive.group_prefix(),
		peak_alloc,
		bench_witness_generation_and_proving,
	);
}

fn bench_witness_generation_and_proving<B>(
	c: &mut Criterion,
	ctx: ConstraintSystemBenchmarkContext<'_, B>,
) where
	B: ExampleBenchmark,
{
	let mut group =
		c.benchmark_group(format!("{}_witness_generation_and_proving", ctx.group_prefix));
	group.throughput(ctx.benchmark.throughput());
	group.sample_size(10);
	group.warm_up_time(Duration::from_secs(2));

	// Flock's headline proof path includes witness generation, so this group reports the same
	// end-to-end unit of work for cross-prover comparisons.
	group.bench_function(BenchmarkId::from_parameter(ctx.bench_name), |b| {
		b.iter(|| {
			let mut filler = ctx.circuit.new_witness_filler();
			ctx.example
				.populate_witness(ctx.instance.clone(), &mut filler)
				.unwrap();
			ctx.circuit.populate_wire_witness(&mut filler).unwrap();
			let witness = filler.into_value_vec();
			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			ctx.prover.prove(witness, &mut prover_transcript).unwrap();
			prover_transcript
		})
	});

	group.finish();
}

fn env_usize(key: &str) -> Option<usize> {
	env::var(key).ok().and_then(|s| s.parse::<usize>().ok())
}
