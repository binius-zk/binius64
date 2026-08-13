// Copyright 2026 The Binius Developers

mod utils;

use std::alloc::System;

use binius_examples::circuits::hashsign::{HashBasedSigExample, Instance, Params};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakMemAlloc;
use utils::{ExampleBenchmark, SignBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage
#[global_allocator]
static HASHSIGN_PEAK_ALLOC: PeakMemAlloc<System> = PeakMemAlloc::new(System);

struct HashSignBenchmark {
	config: SignBenchConfig,
}

impl HashSignBenchmark {
	fn new() -> Self {
		Self {
			config: SignBenchConfig::from_env(4), // default: 4 signatures
		}
	}
}

impl ExampleBenchmark for HashSignBenchmark {
	type Params = Params;
	type Instance = Instance;
	type Example = HashBasedSigExample;

	fn create_params(&self) -> Self::Params {
		Params {
			num_signers: self.config.n_signatures,
		}
	}

	fn create_instance(&self) -> Self::Instance {
		Instance {}
	}

	fn bench_name(&self) -> String {
		format!("sig_{}", self.config.n_signatures)
	}

	fn throughput(&self) -> Throughput {
		Throughput::Elements(self.config.n_signatures as u64)
	}

	fn proof_description(&self) -> String {
		format!("{} signatures", self.config.n_signatures)
	}

	fn log_inv_rate(&self) -> usize {
		self.config.log_inv_rate
	}

	fn print_params(&self) {
		let params_list = vec![
			("Signatures".to_string(), self.config.n_signatures.to_string()),
			("Message size".to_string(), "32 bytes (fixed)".to_string()),
			("Log inverse rate".to_string(), self.config.log_inv_rate.to_string()),
		];
		print_benchmark_header("Hashsign", &params_list);
	}
}

fn bench_hashsign(c: &mut Criterion) {
	let benchmark = HashSignBenchmark::new();
	run_cs_benchmark(c, &benchmark, "hashsign", &HASHSIGN_PEAK_ALLOC);
}

criterion_group!(hashsign, bench_hashsign);
criterion_main!(hashsign);
