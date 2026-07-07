// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

mod utils;

use std::alloc::System;

use binius_examples::circuits::{
	blake2b::Blake2bExample,
	utils::{HasherInstance, HasherParams},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakMemAlloc;
use utils::{ExampleBenchmark, HashBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage

#[global_allocator]
static BLAKE2B_PEAK_ALLOC: PeakMemAlloc<System> = PeakMemAlloc::new(System);

struct Blake2bBenchmark {
	config: HashBenchConfig,
}

impl Blake2bBenchmark {
	fn new() -> Self {
		let config = HashBenchConfig::from_env();
		Self { config }
	}
}

impl ExampleBenchmark for Blake2bBenchmark {
	type Params = HasherParams;
	type Instance = HasherInstance;
	type Example = Blake2bExample;

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
		let blocks = self.config.max_bytes.div_ceil(128);
		let params_list = vec![
			(
				"Circuit capacity".to_string(),
				format!("{} bytes ({} blocks × 128 bytes/block)", self.config.max_bytes, blocks),
			),
			(
				"Message length".to_string(),
				format!("{} bytes (using full capacity)", self.config.max_bytes),
			),
			("Log inverse rate".to_string(), self.config.log_inv_rate.to_string()),
		];
		print_benchmark_header("Blake2b", &params_list);
	}
}

fn bench_blake2b_hash(c: &mut Criterion) {
	let benchmark = Blake2bBenchmark::new();
	run_cs_benchmark(c, benchmark, "blake2b", &BLAKE2B_PEAK_ALLOC);
}

criterion_group!(benches, bench_blake2b_hash);
criterion_main!(benches);
