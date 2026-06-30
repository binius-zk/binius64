// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Independent SHA-256 compression benchmark.

#[path = "utils/config.rs"]
mod config;
#[path = "utils/independent_hashes.rs"]
mod independent_hashes;
#[path = "utils/reporting.rs"]
mod reporting;
#[path = "utils/runner.rs"]
mod runner;

use std::alloc::System;

use binius_examples::circuits::independent_hashes::IndependentSha256Compressions;
use criterion::{Criterion, criterion_group, criterion_main};
use independent_hashes::{IndependentPrimitive, run_independent_hash_benchmark};
use peakmem_alloc::PeakMemAlloc;

#[global_allocator]
static SHA256_COMPRESSIONS_PEAK_ALLOC: PeakMemAlloc<System> = PeakMemAlloc::new(System);

fn bench_sha256_compressions(c: &mut Criterion) {
	run_independent_hash_benchmark::<IndependentSha256Compressions>(
		c,
		IndependentPrimitive::Sha256,
		&SHA256_COMPRESSIONS_PEAK_ALLOC,
	);
}

criterion_group!(sha256_compressions, bench_sha256_compressions);
criterion_main!(sha256_compressions);
