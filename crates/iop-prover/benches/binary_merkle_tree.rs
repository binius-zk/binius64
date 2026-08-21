// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::{BufferPool, GlobalAllocator};
use binius_field::{BinaryField128bGhash as B128, PackedField, arch::OptimalPackedB128};
use binius_hash::{
	binary_merkle_tree::HashSuite, blake3::Blake3HashSuite, sha256::Sha256HashSuite,
};
use binius_iop_prover::{
	commit_pipeline::encode_and_commit_pipelined,
	merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver},
};
use binius_math::{
	ntt::{NeighborsLastMultiThread, domain_context::GaoMateerOnTheFly},
	reed_solomon::ReedSolomonCode,
	test_utils::random_field_buffer,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const LOG_LEAVES: usize = 17;
const LOG_ELEMS_IN_LEAF: usize = 4;

/// Benchmarks committing a [`FieldBuffer<P>`] of `2^(LOG_LEAVES + LOG_ELEMS_IN_LEAF)` B128 scalars.
///
/// The committed scalar sequence is identical for every `P`.
/// Only two things change with the packing width:
///
/// * the source buffer's allocation and alignment;
/// * how a leaf of `2^LOG_ELEMS_IN_LEAF` scalars maps onto packed words.
///
/// The second point is the surface under measurement:
///
/// ```text
/// P narrower than a leaf  ->  a leaf spans several whole words
/// P wider than a leaf     ->  a word splits into several leaves
/// ```
///
/// `OptimalPackedB128` widens past `1x128b` only under `RUSTFLAGS="-C target-cpu=native"`.
/// On a baseline target it collapses to `1x128b`.
/// Both cases then measure the same path.
fn bench_binary_merkle_tree<P, H>(c: &mut Criterion, hash_name: &str, packing_name: impl AsRef<str>)
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
{
	let merkle_prover = BinaryMerkleTreeProver::<B128, H>::new();
	let mut rng = rand::rng();
	let buffer = random_field_buffer::<P>(&mut rng, LOG_LEAVES + LOG_ELEMS_IN_LEAF);

	let mut group = c.benchmark_group(format!("slow/merkle_tree/{hash_name}"));
	group.throughput(Throughput::Bytes(
		((1 << (LOG_LEAVES + LOG_ELEMS_IN_LEAF)) * std::mem::size_of::<B128>()) as u64,
	));
	group.sample_size(10);
	let case = format!(
		"{LOG_LEAVES} log leaves size {}xB128 leaf {}",
		1 << LOG_ELEMS_IN_LEAF,
		packing_name.as_ref()
	);
	group.bench_function(&case, |b| {
		b.iter(|| merkle_prover.commit_field_buffer(buffer.to_ref(), LOG_ELEMS_IN_LEAF));
	});

	// The same commitment out of a pool the prover holds across iterations, which is how a real
	// prover uses one. Benching both here rather than against a baseline build keeps the
	// comparison inside a single binary, where the two arms meet identical machine conditions.
	let pool = BufferPool::new();
	let pooled_prover = BinaryMerkleTreeProver::<B128, H, _>::with_allocator(&pool);
	group.bench_function(format!("{case} pooled"), |b| {
		b.iter(|| pooled_prover.commit_field_buffer(buffer.to_ref(), LOG_ELEMS_IN_LEAF));
	});
	group.finish()
}

fn bench_sha256_merkle_tree(c: &mut Criterion) {
	bench_binary_merkle_tree::<B128, Sha256HashSuite>(c, "SHA-256", "128b");
	bench_binary_merkle_tree::<OptimalPackedB128, Sha256HashSuite>(
		c,
		"SHA-256",
		format!("{}x128b", OptimalPackedB128::WIDTH),
	);
}

fn bench_blake3_merkle_tree(c: &mut Criterion) {
	bench_binary_merkle_tree::<B128, Blake3HashSuite>(c, "Blake3", "128b");
	bench_binary_merkle_tree::<OptimalPackedB128, Blake3HashSuite>(
		c,
		"Blake3",
		format!("{}x128b", OptimalPackedB128::WIDTH),
	);
}

/// Base-2 log of the message dimension the RS-encode+commit bench encodes from.
///
/// Matches `LOG_LEAVES + LOG_ELEMS_IN_LEAF` at `LOG_INV_RATE = 1`, so the resulting codeword is
/// the same `2^(LOG_LEAVES + LOG_ELEMS_IN_LEAF)`-scalar shape [`bench_binary_merkle_tree`] commits.
const LOG_DIM: usize = LOG_LEAVES + LOG_ELEMS_IN_LEAF - 1;
const LOG_INV_RATE: usize = 1;

/// Benchmarks the full commit phase — RS-encode then Merkle-commit — two ways.
///
/// "sequential" runs [`ReedSolomonCode::encode_batch`] to completion, then commits the finished
/// codeword, exactly as every prover call site does today.
/// "pipelined" runs [`encode_and_commit_pipelined`], which hashes each independent NTT chunk's
/// leaves the instant that chunk finishes, instead of waiting for the whole codeword.
/// Both arms are timed in one binary, back to back, so they meet identical machine conditions.
fn bench_commit_pipeline(c: &mut Criterion) {
	let mut rng = rand::rng();
	let message = random_field_buffer::<OptimalPackedB128>(&mut rng, LOG_DIM);

	let rs_code = ReedSolomonCode::<B128>::new(LOG_DIM, LOG_INV_RATE);
	let domain_context = GaoMateerOnTheFly::<B128>::generate(rs_code.log_len());
	let log_num_shares =
		binius_utils::checked_arithmetics::log2_ceil_usize(binius_utils::rayon::current_num_threads());
	let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

	let mut group = c.benchmark_group("slow/commit_pipeline/SHA-256");
	group.throughput(Throughput::Bytes(
		((1 << (LOG_LEAVES + LOG_ELEMS_IN_LEAF)) * std::mem::size_of::<B128>()) as u64,
	));
	group.sample_size(10);

	group.bench_function("sequential", |b| {
		b.iter(|| {
			let codeword = rs_code.encode_batch(&ntt, message.to_ref(), 0, &GlobalAllocator);
			BinaryMerkleTreeProver::<B128, Sha256HashSuite>::new()
				.commit_field_buffer(codeword.to_ref(), LOG_ELEMS_IN_LEAF)
		});
	});

	group.bench_function("pipelined", |b| {
		b.iter(|| {
			encode_and_commit_pipelined::<B128, OptimalPackedB128, _, Sha256HashSuite, _, _>(
				&rs_code,
				&ntt,
				message.to_ref(),
				0,
				LOG_ELEMS_IN_LEAF,
				&GlobalAllocator,
			)
		});
	});

	group.finish();
}

criterion_group!(
	binary_merkle_tree,
	bench_sha256_merkle_tree,
	bench_blake3_merkle_tree,
	bench_commit_pipeline
);
criterion_main!(binary_merkle_tree);
