// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter::repeat_with;

use binius_field::Random;
use binius_hash::{
	binary_merkle_tree::HashSuite, sha256::Sha256HashSuite, vision::VisionHashSuite,
};
use binius_prover::merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver};
use binius_verifier::config::B128;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const LOG_ELEMS: usize = 17;
const LOG_ELEMS_IN_LEAF: usize = 4;

type F = B128;

fn bench_binary_merkle_tree<H: HashSuite>(c: &mut Criterion, hash_name: &str) {
	let merkle_prover = BinaryMerkleTreeProver::<F, H>::new();
	let mut rng = rand::rng();
	let data = repeat_with(|| F::random(&mut rng))
		.take(1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF))
		.collect::<Vec<_>>();
	let mut group = c.benchmark_group(format!("slow/merkle_tree/{hash_name}"));
	group.throughput(Throughput::Bytes(
		((1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF)) * std::mem::size_of::<F>()) as u64,
	));
	group.sample_size(10);
	group.bench_function(
		format!("{} log elems size {}xB64 leaf", LOG_ELEMS, 1 << LOG_ELEMS_IN_LEAF),
		|b| {
			b.iter(|| merkle_prover.commit(&data, 1 << LOG_ELEMS_IN_LEAF));
		},
	);
	group.finish()
}

fn bench_sha256_merkle_tree(c: &mut Criterion) {
	bench_binary_merkle_tree::<Sha256HashSuite>(c, "SHA-256");
}

fn bench_vision_merkle_tree(c: &mut Criterion) {
	bench_binary_merkle_tree::<VisionHashSuite>(c, "Vision");
}

criterion_group!(binary_merkle_tree, bench_sha256_merkle_tree, bench_vision_merkle_tree);
criterion_main!(binary_merkle_tree);
