// Copyright 2026 The Binius Developers

//! Benchmarks comparing rayon parallelization strategies when the per-iteration work is very small.
//!
//! All three variants compute the same quantity — the sum of `n` `u64`s, one drawn from a
//! [`StdRng`] seeded per index by [`par_rand`]'s deterministic scheme — but differ in how the work
//! is handed to rayon:
//!
//! 1. **baseline**: `par_rand(..).sum()` — a plain `(0..n).into_par_iter().map(..)`, letting rayon
//!    split at its default granularity (down to single elements).
//! 2. **chunked_fold_reduce**: parallelize over `0..n / chunk_size` and, in each parallel task, run
//!    a sequential fold over `chunk_size` indices (`fold_with` + `reduce`). The chunking is
//!    explicit in the iteration structure.
//! 3. **with_min_len**: the baseline iterator with `.with_min_len(chunk_size)`, which asks rayon
//!    not to split work below `chunk_size` consecutive elements.
//!
//! Variants 2 and 3 both coarsen rayon's task granularity to `chunk_size`; the benchmark measures
//! whether that matters (and which spelling is faster) versus the fine-grained baseline.

use std::ops::Div;

use binius_utils::{rand::par_rand, rayon::prelude::*};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

/// Number of `u64`s summed, `2^LOG_N`.
const LOG_N: usize = 12;
/// Minimum rayon task granularity for variants 2 and 3, `2^LOG_CHUNK` elements.
const LOG_CHUNK: usize = 14;

/// Variant 2: parallelize over chunks, folding `chunk_size` indices sequentially in each task.
///
/// Replicates [`par_rand`]'s per-index seeding so the summed values match the other variants for a
/// given base seed. Requires `chunk_size` to divide `n`.
fn par_rand_chunked_sum<InnerR>(n: usize, chunk_size: usize, mut rng: impl Rng) -> u64
where
	InnerR: Rng + SeedableRng,
	InnerR::Seed: Send + Sync,
{
	let mut base_seed = <InnerR as SeedableRng>::Seed::default();
	rng.fill_bytes(base_seed.as_mut());

	(0..n.div_ceil(chunk_size))
		.into_par_iter()
		.fold_with(0u64, |acc, chunk| {
			let mut sum = acc;
			for i in (chunk * chunk_size)..((chunk + 1) * chunk_size).min(n) {
				let mut seed = base_seed.clone();
				let seed_bytes = seed.as_mut();
				let index_bytes = i.to_le_bytes();
				for (seed_byte, &index_byte) in seed_bytes.iter_mut().zip(index_bytes.iter()) {
					*seed_byte ^= index_byte;
				}

				sum = sum.wrapping_add(InnerR::from_seed(seed).next_u64());
			}
			sum
		})
		.reduce(|| 0u64, |a, b| a.wrapping_add(b))
}

fn bench_par_rand(c: &mut Criterion) {
	let n = 1 << LOG_N;
	let chunk_size = 1 << LOG_CHUNK;

	let mut group = c.benchmark_group("par_rand");
	group.throughput(Throughput::Elements(n as u64));

	group.bench_function("baseline", |b| {
		b.iter(|| {
			par_rand::<SmallRng, _, _>(n, rand::rng(), |mut rng| rng.next_u64())
				.reduce(|| 0, |lhs, rhs| lhs.wrapping_add(rhs))
		})
	});

	group.bench_function(format!("chunked_fold_reduce/chunk=2^{LOG_CHUNK}"), |b| {
		b.iter(|| par_rand_chunked_sum::<SmallRng>(n, chunk_size, rand::rng()))
	});

	group.bench_function(format!("with_min_len/chunk=2^{LOG_CHUNK}"), |b| {
		b.iter(|| {
			par_rand::<SmallRng, _, _>(n, rand::rng(), |mut rng| rng.next_u64())
				.with_min_len(chunk_size)
				.reduce(|| 0, |lhs, rhs| lhs.wrapping_add(rhs))
		})
	});

	group.finish();
}

criterion_group!(benches, bench_par_rand);
criterion_main!(benches);
