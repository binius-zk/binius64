// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{BinaryField128bGhash as B128, PackedField, arch::OptimalPackedB128};
use binius_hash::{binary_merkle_tree::HashSuite, sha256::Sha256HashSuite};
use binius_iop_prover::merkle_tree::{commit_field_buffer, prover::BinaryMerkleTreeProver};
use binius_math::test_utils::random_field_buffer;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const LOG_LEAVES: usize = 17;
const LOG_ELEMS_IN_LEAF: usize = 4;

/// Benchmarks committing a [`FieldBuffer<P>`] of `2^(LOG_LEAVES + LOG_ELEMS_IN_LEAF)` B128 scalars.
///
/// The committed scalar sequence is identical for every `P`. What changes with the packing width
/// is (a) the source buffer's allocation and alignment, and (b) which branch of
/// [`commit_field_buffer`] runs and at what granularity it parallelizes — for a leaf of
/// `1 << LOG_ELEMS_IN_LEAF` scalars, packings narrower than the leaf take the "big chunks" path
/// while wider packings take the "small chunks" path. That is the surface we want to measure.
///
/// Note: `OptimalPackedB128` only widens past `1x128b` when built with
/// `RUSTFLAGS="-C target-cpu=native"` (or an explicit SIMD `target-feature`); on a baseline target
/// it collapses to `1x128b` and the two cases become identical.
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
	group.bench_function(
		format!(
			"{LOG_LEAVES} log leaves size {}xB128 leaf {}",
			1 << LOG_ELEMS_IN_LEAF,
			packing_name.as_ref()
		),
		|b| {
			b.iter(|| commit_field_buffer(&merkle_prover, buffer.to_ref(), LOG_ELEMS_IN_LEAF));
		},
	);
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

criterion_group!(binary_merkle_tree, bench_sha256_merkle_tree);
criterion_main!(binary_merkle_tree);
