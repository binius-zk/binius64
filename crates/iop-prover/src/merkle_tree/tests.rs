// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use core::slice;

use binius_field::{
	BinaryField128bGhash as B128, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::merkle_tree::MerkleTreeScheme;
use binius_math::{FieldBuffer, test_utils::random_scalars};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
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

	let mut proof_reader = VerifierTranscript::new(StdChallenger::default(), Vec::new());
	let data = random_scalars::<B128>(&mut rng, 4);
	let (commitment, _) = mt_prover.commit(&data, 1);

	mt_prover
		.scheme()
		.verify_vector(&commitment.root, &data, 1, &mut proof_reader.decommitment())
		.unwrap();
}

#[test]
fn test_binary_merkle_vcs_hiding_commit_prove_open() {
	let mut rng = StdRng::seed_from_u64(0);

	let salt_len = 2;
	let mt_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::hiding(&mut rng, salt_len);

	let data = random_scalars::<B128>(&mut rng, 16);
	let (commitment, tree) = mt_prover.commit(&data, 1);

	assert_eq!(commitment.root, tree.root());

	// Test that we can prove openings with salt
	for (i, value) in data.iter().enumerate() {
		let mut proof_writer = ProverTranscript::new(StdChallenger::default());
		mt_prover.prove_opening(&tree, 0, i, &mut proof_writer.message());

		let mut proof_reader = proof_writer.into_verifier();
		mt_prover
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
fn test_binary_merkle_vcs_hiding_verify_vector() {
	let mut rng = StdRng::seed_from_u64(0);

	let salt_len = 3;
	let mt_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::hiding(&mut rng, salt_len);

	let data = random_scalars::<B128>(&mut rng, 8);
	let (commitment, tree) = mt_prover.commit(&data, 1);

	// Create a proof transcript with salt values
	let mut proof_writer = ProverTranscript::new(StdChallenger::default());
	// Write all salt values to the transcript
	for i in 0..data.len() {
		let salt = tree.get_salt(i);
		proof_writer.message().write_slice(salt);
	}

	let mut proof_reader = proof_writer.into_verifier();
	mt_prover
		.scheme()
		.verify_vector(&commitment.root, &data, 1, &mut proof_reader.message())
		.unwrap();
}

#[test]
fn test_binary_merkle_vcs_hiding_prove_open_against_layer() {
	let mut rng = StdRng::seed_from_u64(0);

	let salt_len = 2;
	let mt_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::hiding(&mut rng, salt_len);

	let data = random_scalars::<B128>(&mut rng, 32);
	let (_, tree) = mt_prover.commit(&data, 1);

	// Openings against an internal layer must write the salt of the opened leaf, not of the
	// layer-relative index.
	for layer_depth in 1..5 {
		let layer = mt_prover.layer(&tree, layer_depth);
		for (i, value) in data.iter().enumerate() {
			let mut proof_writer = ProverTranscript::new(StdChallenger::default());
			mt_prover.prove_opening(&tree, layer_depth, i, &mut proof_writer.message());

			let mut proof_reader = proof_writer.into_verifier();
			mt_prover
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
fn test_binary_merkle_vcs_hiding_batch_size() {
	let mut rng = StdRng::seed_from_u64(0);

	let salt_len = 1;
	let mt_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::hiding(&mut rng, salt_len);

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

/// Commits the same scalars through both entry points and requires identical trees.
///
/// [`MerkleTreeProver::commit`] reads a flat scalar slice and never sees the packing.
/// It is therefore an independent reference for the packed path.
fn check_commit_field_buffer_matches_commit<P: PackedField<Scalar = B128>>() {
	let mut rng = StdRng::seed_from_u64(0);
	let prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();

	// Sweep both sides of the packing width: a buffer may be narrower than one word, and a leaf
	// may be narrower, equal, or wider than one word.
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
	// Two widths so that every leaf size lands on both sides of P::LOG_WIDTH across the sweep.
	check_commit_field_buffer_matches_commit::<PackedBinaryGhash2x128b>();
	check_commit_field_buffer_matches_commit::<PackedBinaryGhash4x128b>();
}

#[test]
#[should_panic(expected = "precondition")]
fn test_commit_field_buffer_rejects_oversized_leaf() {
	let mut rng = StdRng::seed_from_u64(0);
	let prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();

	// A leaf wider than the whole buffer has no valid leaf count.
	let buffer = FieldBuffer::<PackedBinaryGhash4x128b, _>::from_values(&random_scalars::<B128>(
		&mut rng, 4,
	));
	let _ = prover.commit_field_buffer(buffer.to_ref(), 3);
}
