// Copyright 2026 The Binius Developers

mod utils;

use std::alloc::System;

use binius_examples::circuits::{
	sha3_512::Sha3_512Example,
	utils::{HasherInstance, HasherParams},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakMemAlloc;
use utils::{ExampleBenchmark, HashBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage
#[global_allocator]
static SHA3_512_PEAK_ALLOC: PeakMemAlloc<System> = PeakMemAlloc::new(System);

struct Sha3_512Benchmark {
	config: HashBenchConfig,
}

impl Sha3_512Benchmark {
	fn new() -> Self {
		let config = HashBenchConfig::from_env();
		Self { config }
	}
}

impl ExampleBenchmark for Sha3_512Benchmark {
	type Params = HasherParams;
	type Instance = HasherInstance;
	type Example = Sha3_512Example;

	fn create_params(&self) -> Self::Params {
		HasherParams {
			message_len: Some(self.config.max_bytes),
			max_message_len: None,
		}
	}

	fn create_instance(&self) -> Self::Instance {
		HasherInstance {
			random_message: false,
			random_message_len: None,
			message: None,
		}
	}

	fn bench_name(&self) -> String {
		format!("message_bytes_{}", self.config.max_bytes)
	}

	fn throughput(&self) -> Throughput {
		Throughput::Bytes(self.config.max_bytes as u64)
	}

	fn proof_description(&self) -> String {
		format!("{} bytes message", self.config.max_bytes)
	}

	fn log_inv_rate(&self) -> usize {
		self.config.log_inv_rate
	}

	fn print_params(&self) {
		const SHA3_512_RATE: usize = 72;
		let n_permutations = self.config.max_bytes.div_ceil(SHA3_512_RATE);
		let params = vec![
			("Message size".to_string(), format!("{} bytes", self.config.max_bytes)),
			(
				"Permutations required".to_string(),
				format!(
					"{} (for {} bytes at {} bytes/permutation)",
					n_permutations, self.config.max_bytes, SHA3_512_RATE
				),
			),
			("Log inverse rate".to_string(), self.config.log_inv_rate.to_string()),
		];
		print_benchmark_header("SHA3-512", &params);
	}
}

fn bench_sha3_512(c: &mut Criterion) {
	let benchmark = Sha3_512Benchmark::new();
	run_cs_benchmark(c, &benchmark, "sha3_512", &SHA3_512_PEAK_ALLOC);
}

criterion_group!(sha3_512, bench_sha3_512);
criterion_main!(sha3_512);
