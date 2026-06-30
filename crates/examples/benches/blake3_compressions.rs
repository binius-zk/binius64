// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Independent BLAKE3 compression benchmark.

#[path = "utils/config.rs"]
mod config;
#[path = "utils/independent_hashes.rs"]
mod independent_hashes;
#[path = "utils/reporting.rs"]
mod reporting;
#[path = "utils/runner.rs"]
mod runner;

use std::alloc::System;

use binius_examples::circuits::independent_hashes::IndependentBlake3Compressions;
use criterion::{Criterion, criterion_group, criterion_main};
use independent_hashes::{IndependentPrimitive, run_independent_hash_benchmark};
use peakmem_alloc::PeakMemAlloc;

#[global_allocator]
static BLAKE3_COMPRESSIONS_PEAK_ALLOC: PeakMemAlloc<System> = PeakMemAlloc::new(System);

fn bench_blake3_compressions(c: &mut Criterion) {
	run_independent_hash_benchmark::<IndependentBlake3Compressions>(
		c,
		IndependentPrimitive::Blake3,
		&BLAKE3_COMPRESSIONS_PEAK_ALLOC,
	);
}

criterion_group!(blake3_compressions, bench_blake3_compressions);
criterion_main!(blake3_compressions);
