// Copyright 2026 The Binius Developers

use std::hint::black_box;

use binius_field::{Ghash128b as B128, Random};
use binius_hash::{ParallelDigest, ParallelDigestAdapter, ParallelSha256Digest, StdDigest};
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
/// dominates binary Merkle tree construction. The input data size is fixed at 1 MiB, so a larger
/// batch size means fewer, larger leaves (fewer SHA-256 init/finalize calls).
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

/// Compares the specialized [`ParallelSha256Digest`] against the generic [`ParallelDigestAdapter`]
/// for the case the BINIUS-75 evaluation targets: leaves of 2 `B128` elements (32 bytes), which fit
/// in a single SHA-256 block. Both paths are identical except for the hashing call — same input
/// chunking, same pre-allocated output buffer, same throughput accounting — so the measured
/// difference isolates the per-leaf padding/`update`/`finalize` bookkeeping the specialization
/// removes.
fn bench_const_leaves(c: &mut Criterion) {
	const BATCH_SIZE: usize = 2;

	let mut rng = rng();
	let elements: Vec<B128> = (0..N_ELEMS).map(|_| B128::random(&mut rng)).collect();
	let n_leaves = N_ELEMS / BATCH_SIZE;

	let adapter = ParallelDigestAdapter::<Sha256>::new();
	let specialized = ParallelSha256Digest::new();

	let mut group = c.benchmark_group("sha256_const_leaves");
	group.throughput(Throughput::Bytes(DATA_LEN as u64));

	let mut digests: Vec<Output<Sha256>> = Vec::with_capacity(n_leaves);
	group.bench_function("unspecialized", |b| {
		b.iter(|| {
			let out = &mut digests.spare_capacity_mut()[..n_leaves];
			adapter.digest(
				black_box(elements.as_slice())
					.par_chunks(BATCH_SIZE)
					.map(|chunk| chunk.iter().copied()),
				out,
			);
		});
	});
	group.bench_function("specialized", |b| {
		b.iter(|| {
			let out = &mut digests.spare_capacity_mut()[..n_leaves];
			specialized.digest_with_const_len(
				BATCH_SIZE,
				black_box(elements.as_slice())
					.par_chunks(BATCH_SIZE)
					.map(|chunk| chunk.iter().copied()),
				out,
			);
		});
	});
	group.finish();
}

/// Compresses one wide Merkle tree layer: pairs of child digests into parent digests.
///
/// Compares the per-node scalar path against the batched multi-lane path.
/// Both run over the same rayon pool.
/// So the delta isolates how fully each path occupies the SHA pipeline.
///
/// The per-node arm spawns a rayon task per node, so at this width it measures scheduling
/// as much as hashing. `sha256_kernel` is the arm to read for the kernel itself.
fn bench_merkle_compress(c: &mut Criterion) {
	use std::mem::MaybeUninit;

	use binius_hash::{
		ParallelCompressionAdaptor, ParallelPseudoCompression, ParallelSha256Compression,
		sha256::Sha256Compression,
	};

	// One layer of 2^15 parent nodes, fed by 2^16 child digests.
	const N_NODES: usize = 1 << 15;
	let mut rng = rng();
	let inputs: Vec<Output<Sha256>> = (0..2 * N_NODES)
		.map(|_| {
			let mut digest = Output::<Sha256>::default();
			rng.fill_bytes(&mut digest);
			digest
		})
		.collect();
	let mut out: Vec<MaybeUninit<Output<Sha256>>> = Vec::with_capacity(N_NODES);
	// Every slot is fully written by every compression call below.
	unsafe { out.set_len(N_NODES) };

	let mut group = c.benchmark_group("sha256_merkle_compress");
	group.throughput(Throughput::Elements(N_NODES as u64));

	// The per-node reference: each node compresses through one scalar block call.
	let per_node = ParallelCompressionAdaptor::new(Sha256Compression::default());
	group.bench_function(BenchmarkId::new("per_node", N_NODES), |b| {
		b.iter(|| per_node.parallel_compress(black_box(&inputs), &mut out));
	});

	// The batched path: `LANES` independent compressions in flight per kernel call.
	let interleaved = ParallelSha256Compression::default();
	group.bench_function(BenchmarkId::new("batched", N_NODES), |b| {
		b.iter(|| interleaved.parallel_compress(black_box(&inputs), &mut out));
	});
	group.finish();
}

/// Compares the batched multi-lane kernel against the per-block `sha2` call it replaces.
///
/// Single-threaded and cache-resident, so the measurement isolates the kernel from rayon
/// scheduling and memory traffic. Every arm hashes the same 4096 independent single-block
/// messages, which is the shape both Merkle stages produce: fixed-size leaves and node pairs.
///
/// The baseline calls `compress256` once per block, as every call site did before. That is
/// not a straw man: restarting from the IV each call leaves the blocks independent, so the
/// out-of-order engine already overlaps them. Handing `sha2` the whole buffer in one call is
/// slower still, because a single multi-block call is one serial chain.
fn bench_kernel(c: &mut Criterion) {
	use binius_hash::sha256::portable::{LANES, compress256_multi, compress256_multi_portable};

	const N_BLOCKS: usize = 4096;
	const IV: [u32; 8] = [
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
		0x5be0cd19,
	];

	let mut rng = rng();
	let blocks: Vec<[u8; 64]> = (0..N_BLOCKS)
		.map(|_| {
			let mut b = [0u8; 64];
			rng.fill_bytes(&mut b);
			b
		})
		.collect();

	let mut group = c.benchmark_group("sha256_kernel");
	group.throughput(Throughput::Elements(N_BLOCKS as u64));

	// The baseline: one `compress256` call per block, from the IV.
	group.bench_function("sha2_per_block", |b| {
		b.iter(|| {
			for block in black_box(&blocks) {
				let mut state = IV;
				compress256(&mut state, std::slice::from_ref(block));
				black_box(state);
			}
		});
	});

	// The portable round loops, at the width this target tunes to. Where a vector transpose
	// exists this arm still uses it, so the delta to `dispatched` is the rounds alone.
	group.bench_function(BenchmarkId::new("lane_loops", LANES), |b| {
		b.iter(|| {
			for chunk in black_box(&blocks).chunks_exact(LANES) {
				let mut states = [IV; LANES];
				let batch: &[[u8; 64]; LANES] =
					chunk.try_into().expect("chunks_exact yields LANES");
				compress256_multi_portable(&mut states, batch);
				black_box(states);
			}
		});
	});

	// The dispatched kernel, which is whichever hand-written path this target compiled in.
	group.bench_function(BenchmarkId::new("dispatched", LANES), |b| {
		b.iter(|| {
			for chunk in black_box(&blocks).chunks_exact(LANES) {
				let mut states = [IV; LANES];
				let batch: &[[u8; 64]; LANES] =
					chunk.try_into().expect("chunks_exact yields LANES");
				compress256_multi(&mut states, batch);
				black_box(states);
			}
		});
	});

	// The 16-lane square, the only width the AVX-512 shuffle network covers.
	// Skipped when it is already the tuned width, since the arm above covers it.
	if LANES != 16 {
		group.bench_function(BenchmarkId::new("dispatched", 16), |b| {
			b.iter(|| {
				for chunk in black_box(&blocks).chunks_exact(16) {
					let mut states = [IV; 16];
					let batch: &[[u8; 64]; 16] = chunk.try_into().expect("chunks_exact yields 16");
					compress256_multi(&mut states, batch);
					black_box(states);
				}
			});
		});
	}

	group.finish();
}

criterion_group!(
	benches,
	bench_kernel,
	bench_sha256,
	bench_compress,
	bench_digest,
	bench_const_leaves,
	bench_merkle_compress
);
criterion_main!(benches);
