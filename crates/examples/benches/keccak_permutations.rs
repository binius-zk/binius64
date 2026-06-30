// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Independent Keccak-f[1600] permutation benchmark.

#[path = "utils/config.rs"]
mod config;
#[path = "utils/independent_hashes.rs"]
mod independent_hashes;
#[path = "utils/reporting.rs"]
mod reporting;
#[path = "utils/runner.rs"]
mod runner;

use std::alloc::System;

use binius_examples::circuits::independent_hashes::IndependentKeccakPermutations;
use criterion::{Criterion, criterion_group, criterion_main};
use independent_hashes::{IndependentPrimitive, run_independent_hash_benchmark};
use peakmem_alloc::PeakMemAlloc;

#[global_allocator]
static KECCAK_PERMUTATIONS_PEAK_ALLOC: PeakMemAlloc<System> = PeakMemAlloc::new(System);

fn bench_keccak_permutations(c: &mut Criterion) {
	run_independent_hash_benchmark::<IndependentKeccakPermutations>(
		c,
		IndependentPrimitive::Keccak,
		&KECCAK_PERMUTATIONS_PEAK_ALLOC,
	);
}

criterion_group!(keccak_permutations, bench_keccak_permutations);
criterion_main!(keccak_permutations);
