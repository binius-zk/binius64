// Copyright 2026 The Binius Developers

use std::hint::black_box;

use binius_field::{BinaryField128bGhash as B128, Random};
use binius_hash::{ParallelDigest, ParallelDigestAdapter, blake3::Blake3ParallelDigest};
use binius_utils::rayon::{prelude::*, slice::ParallelSlice};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use digest::Output;
use rand::rng;

/// Total input hashed each iteration, fixed so throughput is comparable across leaf sizes.
const DATA_LEN: usize = 1 << 20; // 1 MiB
/// Number of 16-byte field elements in the input.
const N_ELEMS: usize = DATA_LEN / std::mem::size_of::<B128>();

/// Leaf sizes measured, in 16-byte field elements, giving leaf byte lengths 16, 32, 64, ..., 1024.
///
/// - 4 elements and up are a multiple of the 64-byte block: the fully-SIMD fast path.
/// - 1 and 2 elements are sub-block, where no multi-lane kernel exists.
/// - The two sub-block sizes are expected to stay at parity between the paths.
const BATCH_SIZES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// Compares the SIMD leaf digest against the generic per-leaf adapter for Merkle leaf hashing.
///
/// The input is a fixed 1 MiB of field elements, folded into leaves of `batch_size` elements each.
/// Both paths share the input chunking, output buffer, and throughput accounting.
/// The measured difference therefore isolates the SIMD batching from everything else.
fn bench_leaf_hashing(c: &mut Criterion) {
	// One fixed pool of random field elements, reused for every batch size.
	let mut rng = rng();
	let elements: Vec<B128> = (0..N_ELEMS).map(|_| B128::random(&mut rng)).collect();

	// Baseline: the generic adapter that hashes one leaf at a time.
	let adapter = ParallelDigestAdapter::<blake3::Hasher>::new();
	// Candidate: the SIMD batch leaf digest under test.
	let multi = Blake3ParallelDigest::new();

	let mut group = c.benchmark_group("blake3_parallel_digest");
	// Account throughput against the fixed 1 MiB input, so larger leaves don't inflate the number.
	group.throughput(Throughput::Bytes(DATA_LEN as u64));

	for &batch_size in &BATCH_SIZES {
		// Folding the input into `batch_size`-element leaves yields this many leaves.
		let n_leaves = N_ELEMS / batch_size;
		// Output buffer allocated once per batch size, so the measurement excludes allocation.
		let mut digests: Vec<Output<blake3::Hasher>> = Vec::with_capacity(n_leaves);

		group.bench_with_input(BenchmarkId::new("adapter", batch_size), &batch_size, |b, &bs| {
			b.iter(|| {
				// Reuse the pre-allocated output slots for this run.
				let out = &mut digests.spare_capacity_mut()[..n_leaves];
				// Split the element pool into leaves.
				// Hash each leaf with the adapter.
				adapter.digest(
					black_box(elements.as_slice())
						.par_chunks(bs)
						.map(|chunk| chunk.iter().copied()),
					out,
				);
			});
		});
		group.bench_with_input(
			BenchmarkId::new("multidigest", batch_size),
			&batch_size,
			|b, &bs| {
				b.iter(|| {
					let out = &mut digests.spare_capacity_mut()[..n_leaves];
					// Mirror how the Merkle tree builder hashes leaves: a fixed-length batch.
					// Sub-block leaves route to the adapter.
					// Larger leaves take the SIMD path.
					multi.digest_with_const_len(
						bs,
						black_box(elements.as_slice())
							.par_chunks(bs)
							.map(|chunk| chunk.iter().copied()),
						out,
					);
				});
			},
		);
	}
	group.finish();
}

criterion_group!(benches, bench_leaf_hashing);
criterion_main!(benches);
