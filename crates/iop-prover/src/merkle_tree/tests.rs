// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use core::slice;

use binius_compute::BufferPool;
use binius_field::{
	BinaryField128bGhash as B128, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::merkle_tree::MerkleTreeScheme;
use binius_math::{FieldBuffer, test_utils::random_scalars};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use rand::prelude::*;

use crate::merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver};

type StdChallenger = HasherChallenger<StdDigest>;

#[test]
fn test_binary_merkle_vcs_commit_prove_open_correctly() {
	let mut rng = StdRng::seed_from_u64(0);

	let mr_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();

	let data = random_scalars::<B128>(&mut rng, 16);
	let (commitment, tree) = mr_prover.commit(&data, 1);

	assert_eq!(commitment.root, tree.root());

	for (i, value) in data.iter().enumerate() {
		let mut proof_writer = ProverTranscript::new(StdChallenger::default());
		mr_prover.prove_opening(&tree, 0, i, &mut proof_writer.message());

		let mut proof_reader = proof_writer.into_verifier();
		mr_prover
			.scheme()
			.verify_opening(
				i,
				slice::from_ref(value),
				0,
				4,
				&[commitment.root],
				&mut proof_reader.message(),
			)
			.unwrap();
	}
}

#[test]
fn test_binary_merkle_vcs_commit_layer_prove_open_correctly() {
	let mut rng = StdRng::seed_from_u64(0);

	let mr_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();

	let data = random_scalars::<B128>(&mut rng, 32);
	let (commitment, tree) = mr_prover.commit(&data, 1);

	assert_eq!(commitment.root, tree.root());
	for layer_depth in 0..5 {
		let layer = mr_prover.layer(&tree, layer_depth);
		mr_prover
			.scheme()
			.verify_layer(&commitment.root, layer_depth, layer)
			.unwrap();
		for (i, value) in data.iter().enumerate() {
			let mut proof_writer = ProverTranscript::new(StdChallenger::default());
			mr_prover.prove_opening(&tree, layer_depth, i, &mut proof_writer.message());

			let mut proof_reader = proof_writer.into_verifier();
			mr_prover
				.scheme()
				.verify_opening(
					i,
					slice::from_ref(value),
					layer_depth,
					5,
					layer,
					&mut proof_reader.message(),
				)
				.unwrap();
		}
	}
}

#[test]
fn test_binary_merkle_vcs_verify_vector() {
	let mut rng = StdRng::seed_from_u64(0);

	let mt_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();

	let data = random_scalars::<B128>(&mut rng, 4);
	let (commitment, _) = mt_prover.commit(&data, 1);

	mt_prover
		.scheme()
		.verify_vector(&commitment.root, &data, 1)
		.unwrap();
}

#[test]
fn test_binary_merkle_vcs_batch_size() {
	let mut rng = StdRng::seed_from_u64(0);

	let mt_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();

	let data = random_scalars::<B128>(&mut rng, 32);
	let batch_size = 4;
	let (commitment, tree) = mt_prover.commit(&data, batch_size);

	assert_eq!(commitment.root, tree.root());

	// Test openings with batch_size > 1
	for i in 0..8 {
		let mut proof_writer = ProverTranscript::new(StdChallenger::default());
		mt_prover.prove_opening(&tree, 0, i, &mut proof_writer.message());

		let mut proof_reader = proof_writer.into_verifier();
		let values = &data[i * batch_size..(i + 1) * batch_size];
		mt_prover
			.scheme()
			.verify_opening(i, values, 0, 3, &[commitment.root], &mut proof_reader.message())
			.unwrap();
	}
}

fn check_commit_field_buffer_matches_commit<P: PackedField<Scalar = B128>>() {
	let mut rng = StdRng::seed_from_u64(0);
	let prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();

	// Invariant: a packed buffer commits to the tree its flat scalar sequence commits to.
	//
	//     reference : flat scalar slice  ->  leaves, packing never involved
	//     under test: packed buffer      ->  the same leaves, read out of packed words
	//
	// Sweep: buffers of 1 to 32 scalars, and every leaf size up to the buffer.
	// Both ranges cross the packing width, so each branch of the chunker is exercised.
	for log_len in 0..=5 {
		let scalars = random_scalars::<B128>(&mut rng, 1 << log_len);
		let buffer = FieldBuffer::<P, _>::from_values(&scalars);

		for log_leaf_len in 0..=log_len {
			let (reference, _) = prover.commit(&scalars, 1 << log_leaf_len);
			let (packed, _) = prover.commit_field_buffer(buffer.to_ref(), log_leaf_len);

			assert_eq!(packed.root, reference.root, "log_len {log_len}, leaf {log_leaf_len}");
			assert_eq!(packed.depth, reference.depth, "log_len {log_len}, leaf {log_leaf_len}");
		}
	}
}

#[test]
fn test_commit_field_buffer_matches_commit() {
	// Two packing widths, so a leaf of 2 scalars lands on either side of a word boundary.
	check_commit_field_buffer_matches_commit::<PackedBinaryGhash2x128b>();
	check_commit_field_buffer_matches_commit::<PackedBinaryGhash4x128b>();
}

#[test]
#[should_panic(expected = "precondition")]
fn test_commit_field_buffer_rejects_oversized_leaf() {
	let mut rng = StdRng::seed_from_u64(0);
	let prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();

	// Fixture state: 4 scalars in the buffer, 2^3 = 8 scalars asked for per leaf.
	//
	// A leaf wider than the buffer would need half a leaf to hold it.
	// No leaf count fits, so the split is rejected up front.
	let buffer = FieldBuffer::<PackedBinaryGhash4x128b, _>::from_values(&random_scalars::<B128>(
		&mut rng, 4,
	));
	let _ = prover.commit_field_buffer(buffer.to_ref(), 3);
}

#[test]
fn a_pooled_prover_commits_the_same_tree_as_the_global_one() {
	// Invariant: the allocator backing the nodes is not observable in what gets committed.
	// Pooling is a memory strategy, so every digest the two provers produce must agree.
	//
	// Fixture state: 64 scalars in leaves of 2, committed twice — once on the global heap, once
	// out of a `BufferPool`.
	let mut rng = StdRng::seed_from_u64(0);
	let data = random_scalars::<B128>(&mut rng, 64);

	let global = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();
	let (global_commitment, global_tree) = global.commit(&data, 2);

	let pool = BufferPool::new();
	let pooled = BinaryMerkleTreeProver::<B128, StdHashSuite, _>::with_allocator(&pool);
	let (pooled_commitment, pooled_tree) = pooled.commit(&data, 2);

	assert_eq!(pooled_commitment.root, global_commitment.root);
	assert_eq!(pooled_commitment.depth, global_commitment.depth);
	// Every node, not just the root: a layer or branch read off the pooled tree must match too.
	assert_eq!(&*pooled_tree.inner_nodes, &*global_tree.inner_nodes);

	// The pool is reused across commitments, which is the whole point of holding one.
	// A second commitment through the same pool draws recycled blocks and must still agree.
	drop(pooled_tree);
	let (again, _) = pooled.commit(&data, 2);
	assert_eq!(again.root, global_commitment.root);
}
