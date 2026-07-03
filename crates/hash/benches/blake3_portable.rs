// Copyright 2026 The Binius Developers

//! Compares Blake3 leaf-hashing on a fixed 1 MiB of field-element leaves:
//! - the generic per-leaf scalar adapter (the baseline on `main`),
//! - the portable auto-vectorized kernel at several lane widths.

use std::hint::black_box;

use binius_field::{BinaryField128bGhash as B128, Random};
use binius_hash::{
	ParallelDigest, ParallelDigestAdapter, blake3_portable::PortableBlake3ParallelDigest,
};
use binius_utils::rayon::{prelude::*, slice::ParallelSlice};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use digest::Output;
use rand::rng;

/// Total input hashed each iteration, fixed so throughput is comparable across leaf sizes.
const DATA_LEN: usize = 1 << 20; // 1 MiB
/// Number of 16-byte field elements in the input.
const N_ELEMS: usize = DATA_LEN / std::mem::size_of::<B128>();

/// Leaf sizes measured, in 16-byte field elements: byte lengths 16, 32, 48, 64, 128, ..., 1024.
///
/// Sizes span sub-block (16, 32, 48), whole-block, and multi-block leaves within one chunk.
/// All take the portable batch path, including the sub-block sizes the crate kernel could not.
const BATCH_SIZES: [usize; 8] = [1, 2, 3, 4, 8, 16, 32, 64];

/// Hashes the 1 MiB pool folded into `batch_size`-element leaves with the given parallel digest.
fn run<D: ParallelDigest<Digest = blake3::Hasher>>(
	digest: &D,
	elements: &[B128],
	batch_size: usize,
	out: &mut [core::mem::MaybeUninit<Output<blake3::Hasher>>],
) {
	digest.digest_with_const_len(
		batch_size,
		black_box(elements)
			.par_chunks(batch_size)
			.map(|chunk| chunk.iter().copied()),
		out,
	);
}

fn bench_portable(c: &mut Criterion) {
	// One fixed pool of random field elements, reused for every batch size and path.
	let mut rng = rng();
	let elements: Vec<B128> = (0..N_ELEMS).map(|_| B128::random(&mut rng)).collect();

	// The candidate leaf digests: the scalar baseline and the portable kernel at three widths.
	let adapter = ParallelDigestAdapter::<blake3::Hasher>::new();
	let portable4 = PortableBlake3ParallelDigest::<4>::new();
	let portable8 = PortableBlake3ParallelDigest::<8>::new();
	let portable16 = PortableBlake3ParallelDigest::<16>::new();

	let mut group = c.benchmark_group("blake3_portable");
	// Account throughput against the fixed 1 MiB input, so larger leaves don't inflate the number.
	group.throughput(Throughput::Bytes(DATA_LEN as u64));

	for &batch_size in &BATCH_SIZES {
		// Output buffer allocated once per batch size, so the measurement excludes allocation.
		let n_leaves = N_ELEMS / batch_size;
		let mut digests: Vec<Output<blake3::Hasher>> = Vec::with_capacity(n_leaves);

		group.bench_with_input(BenchmarkId::new("adapter", batch_size), &batch_size, |b, &bs| {
			b.iter(|| run(&adapter, &elements, bs, &mut digests.spare_capacity_mut()[..n_leaves]));
		});
		group.bench_with_input(
			BenchmarkId::new("portable_4", batch_size),
			&batch_size,
			|b, &bs| {
				b.iter(|| {
					run(&portable4, &elements, bs, &mut digests.spare_capacity_mut()[..n_leaves])
				});
			},
		);
		group.bench_with_input(
			BenchmarkId::new("portable_8", batch_size),
			&batch_size,
			|b, &bs| {
				b.iter(|| {
					run(&portable8, &elements, bs, &mut digests.spare_capacity_mut()[..n_leaves])
				});
			},
		);
		group.bench_with_input(
			BenchmarkId::new("portable_16", batch_size),
			&batch_size,
			|b, &bs| {
				b.iter(|| {
					run(&portable16, &elements, bs, &mut digests.spare_capacity_mut()[..n_leaves])
				});
			},
		);
	}
	group.finish();
}

criterion_group!(benches, bench_portable);
criterion_main!(benches);
