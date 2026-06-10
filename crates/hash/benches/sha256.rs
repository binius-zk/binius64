// Copyright 2026 The Binius Developers

use std::hint::black_box;

use binius_field::{BinaryField128bGhash as B128, Random};
use binius_hash::{ParallelDigest, ParallelDigestAdapter, StdDigest};
use binius_utils::rayon::{prelude::*, slice::ParallelSlice};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use digest::{Digest, Output};
use rand::{Rng, rng};
use sha2::{Sha256, block_api::compress256};

const DATA_LEN: usize = 1 << 20; // 1 MiB
const N_ELEMS: usize = DATA_LEN / std::mem::size_of::<B128>();
const BATCH_SIZES: [usize; 5] = [1, 2, 4, 8, 16];

/// SHA-256 initial hash values (used as the starting state for a raw compression).
const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Hashes a flat 1 MiB buffer with SHA-256.
fn bench_sha256(c: &mut Criterion) {
	let mut data = vec![0u8; DATA_LEN];
	rng().fill_bytes(&mut data);

	let mut group = c.benchmark_group("sha256");
	group.throughput(Throughput::Bytes(DATA_LEN as u64));
	group.bench_function("hash_1mib", |b| {
		b.iter(|| <StdDigest as Digest>::digest(black_box(&data)));
	});
	group.finish();
}

/// Measures the raw `compress256` block function with no hasher setup, padding, or finalization.
///
/// - `amortized_per_block`: one `compress256` call over many blocks (steady-state per-block cost).
/// - `single_block`: one `compress256` call over a single block (per-block cost + any per-call
///   overhead in `compress256` itself).
fn bench_compress(c: &mut Criterion) {
	const N_BLOCKS: usize = 1 << 14;
	let blocks: Vec<[u8; 64]> = vec![[0u8; 64]; N_BLOCKS];

	let mut group = c.benchmark_group("sha256_compress");

	group.throughput(Throughput::Elements(N_BLOCKS as u64));
	group.bench_function("amortized_per_block", |b| {
		b.iter(|| {
			let mut state = IV;
			compress256(&mut state, black_box(&blocks));
			state
		});
	});

	group.throughput(Throughput::Elements(1));
	group.bench_function("single_block", |b| {
		b.iter(|| {
			let mut state = IV;
			compress256(&mut state, black_box(&blocks[..1]));
			state
		});
	});

	group.finish();
}

/// Benchmarks [`ParallelDigestAdapter`] over 1 MiB of `B128` elements, varying the number of
/// elements folded into each leaf digest (`batch_size`). This isolates the leaf-hashing step that
/// dominates `binary_merkle_tree::build`. The input data size is fixed at 1 MiB, so a larger batch
/// size means fewer, larger leaves (fewer SHA-256 init/finalize calls).
fn bench_digest(c: &mut Criterion) {
	let mut rng = rng();
	let elements: Vec<B128> = (0..N_ELEMS).map(|_| B128::random(&mut rng)).collect();

	let adapter = ParallelDigestAdapter::<Sha256>::new();
	let mut group = c.benchmark_group("sha256_parallel_digest");
	group.throughput(Throughput::Bytes(DATA_LEN as u64));
	for &batch_size in &BATCH_SIZES {
		let n_leaves = N_ELEMS / batch_size;
		// Allocate the output buffer once per batch size so the measurement isolates hashing.
		let mut digests: Vec<Output<Sha256>> = Vec::with_capacity(n_leaves);
		group.bench_with_input(BenchmarkId::from_parameter(batch_size), &batch_size, |b, &bs| {
			b.iter(|| {
				let out = &mut digests.spare_capacity_mut()[..n_leaves];
				adapter.digest(
					black_box(elements.as_slice())
						.par_chunks(bs)
						.map(|chunk| chunk.iter().copied()),
					out,
				);
			});
		});
	}
	group.finish();
}

criterion_group!(benches, bench_sha256, bench_compress, bench_digest);
criterion_main!(benches);
