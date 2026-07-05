// Copyright 2026 The Binius Developers

use binius_field::{BinaryField128bGhash as B128, PackedBinaryGhash2x128b};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::{
	merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_math::{FieldBuffer, test_utils::random_scalars};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use rand::prelude::*;

use super::{MerkleIPProverChannel, ProverMerkleTranscriptChannel};

type StdChallenger = HasherChallenger<StdDigest>;
type P = PackedBinaryGhash2x128b;
type VerifierChannel<T> = VerifierMerkleTranscriptChannel<T, StdChallenger, B128, StdHashSuite>;
type ProverChannel<T> = ProverMerkleTranscriptChannel<T, StdChallenger, B128, StdHashSuite>;

const LOG_LEN: usize = 8;
const LOG_LEAF_SIZE: usize = 2;
const LEAF_SIZE: usize = 1 << LOG_LEAF_SIZE;
const DEPTH: usize = LOG_LEN - LOG_LEAF_SIZE;
const N_QUERIES: usize = 5;

fn sample_indices(channel: &mut impl MerkleIPProverChannel<B128>) -> Vec<usize> {
	(0..N_QUERIES).map(|_| channel.sample_bits(DEPTH)).collect()
}

#[test]
fn test_merkle_channel_roundtrip() {
	let mut rng = StdRng::seed_from_u64(0);

	let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
	let data = FieldBuffer::<P, _>::from_values(&scalars);

	// Prover side: commit, sample query indices, open them, then send the vector in full.
	let mut prover_channel = ProverChannel::new(ProverTranscript::new(StdChallenger::default()));
	let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
	let indices = sample_indices(&mut prover_channel);
	prover_channel.send_openings(&commitment, data.to_ref(), &indices);
	prover_channel.send_committed_vector(&commitment, data.to_ref());

	// Verifier side: mirror the interaction and check the opened values against the data.
	let transcript = prover_channel.into_transcript().into_verifier();
	let mut verifier_channel = VerifierChannel::new(transcript);
	let commitment = verifier_channel
		.recv_merkle_commitment(LEAF_SIZE, DEPTH)
		.unwrap();
	let verifier_indices = (0..N_QUERIES)
		.map(|_| verifier_channel.sample_bits(DEPTH))
		.collect::<Vec<_>>();
	assert_eq!(verifier_indices, indices);

	let values = verifier_channel
		.recv_openings(&commitment, &indices)
		.unwrap();
	assert_eq!(values.len(), N_QUERIES * LEAF_SIZE);
	for (chunk, &index) in values.chunks(LEAF_SIZE).zip(&indices) {
		assert_eq!(chunk, &scalars[index * LEAF_SIZE..(index + 1) * LEAF_SIZE]);
	}

	let vector = verifier_channel.recv_committed_vector(&commitment).unwrap();
	assert_eq!(vector, scalars);

	verifier_channel.into_transcript().finalize().unwrap();
}

#[test]
fn test_merkle_channel_roundtrip_hiding() {
	let mut rng = StdRng::seed_from_u64(0);
	let salt_len = 2;

	let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
	let data = FieldBuffer::<P, _>::from_values(&scalars);

	let mut prover_channel =
		ProverChannel::hiding(ProverTranscript::new(StdChallenger::default()), &mut rng, salt_len);
	let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
	let indices = sample_indices(&mut prover_channel);
	prover_channel.send_openings(&commitment, data.to_ref(), &indices);
	prover_channel.send_committed_vector(&commitment, data.to_ref());

	let transcript = prover_channel.into_transcript().into_verifier();
	let mut verifier_channel =
		VerifierChannel::with_scheme(transcript, BinaryMerkleTreeScheme::hiding(salt_len));
	let commitment = verifier_channel
		.recv_merkle_commitment(LEAF_SIZE, DEPTH)
		.unwrap();
	let verifier_indices = (0..N_QUERIES)
		.map(|_| verifier_channel.sample_bits(DEPTH))
		.collect::<Vec<_>>();
	assert_eq!(verifier_indices, indices);

	let values = verifier_channel
		.recv_openings(&commitment, &indices)
		.unwrap();
	for (chunk, &index) in values.chunks(LEAF_SIZE).zip(&indices) {
		assert_eq!(chunk, &scalars[index * LEAF_SIZE..(index + 1) * LEAF_SIZE]);
	}

	let vector = verifier_channel.recv_committed_vector(&commitment).unwrap();
	assert_eq!(vector, scalars);

	verifier_channel.into_transcript().finalize().unwrap();
}

#[test]
fn test_merkle_channel_borrowed_transcript() {
	let mut rng = StdRng::seed_from_u64(0);

	let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
	let data = FieldBuffer::<P, _>::from_values(&scalars);

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	{
		let mut prover_channel = ProverChannel::new(&mut prover_transcript);
		let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
		let indices = sample_indices(&mut prover_channel);
		prover_channel.send_openings(&commitment, data.to_ref(), &indices);
	}

	let mut verifier_transcript = prover_transcript.into_verifier();
	{
		let mut verifier_channel = VerifierChannel::new(&mut verifier_transcript);
		let commitment = verifier_channel
			.recv_merkle_commitment(LEAF_SIZE, DEPTH)
			.unwrap();
		let indices = (0..N_QUERIES)
			.map(|_| verifier_channel.sample_bits(DEPTH))
			.collect::<Vec<_>>();
		let values = verifier_channel
			.recv_openings(&commitment, &indices)
			.unwrap();
		for (chunk, &index) in values.chunks(LEAF_SIZE).zip(&indices) {
			assert_eq!(chunk, &scalars[index * LEAF_SIZE..(index + 1) * LEAF_SIZE]);
		}
	}
	verifier_transcript.finalize().unwrap();
}

#[test]
fn test_merkle_channel_rejects_openings_at_wrong_index() {
	let mut rng = StdRng::seed_from_u64(0);

	let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
	let data = FieldBuffer::<P, _>::from_values(&scalars);

	let mut prover_channel = ProverChannel::new(ProverTranscript::new(StdChallenger::default()));
	let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
	let indices = sample_indices(&mut prover_channel);
	prover_channel.send_openings(&commitment, data.to_ref(), &indices);

	let transcript = prover_channel.into_transcript().into_verifier();
	let mut verifier_channel = VerifierChannel::new(transcript);
	let commitment = verifier_channel
		.recv_merkle_commitment(LEAF_SIZE, DEPTH)
		.unwrap();
	let _ = (0..N_QUERIES)
		.map(|_| verifier_channel.sample_bits(DEPTH))
		.collect::<Vec<_>>();

	// Requesting openings at indices other than the ones the prover opened must fail.
	let wrong_indices = indices.iter().map(|&index| index ^ 1).collect::<Vec<_>>();
	assert!(
		verifier_channel
			.recv_openings(&commitment, &wrong_indices)
			.is_err()
	);

	// Drop the transcript without finalizing; the tampered read left it misaligned.
	let _ = verifier_channel.into_transcript();
}

#[test]
fn test_merkle_channel_rejects_wrong_root() {
	let mut rng = StdRng::seed_from_u64(0);

	let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
	let data = FieldBuffer::<P, _>::from_values(&scalars);
	let other_scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
	let other_data = FieldBuffer::<P, _>::from_values(&other_scalars);

	// Commit one buffer but open the other, so the openings do not match the commitment.
	let mut prover_channel = ProverChannel::new(ProverTranscript::new(StdChallenger::default()));
	let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
	let other_commitment = prover_channel.send_merkle_commitment(other_data.to_ref(), LEAF_SIZE);
	let indices = sample_indices(&mut prover_channel);
	prover_channel.send_openings(&other_commitment, other_data.to_ref(), &indices);
	let _ = commitment;

	let transcript = prover_channel.into_transcript().into_verifier();
	let mut verifier_channel = VerifierChannel::new(transcript);
	let commitment = verifier_channel
		.recv_merkle_commitment(LEAF_SIZE, DEPTH)
		.unwrap();
	let _other_commitment = verifier_channel
		.recv_merkle_commitment(LEAF_SIZE, DEPTH)
		.unwrap();
	let indices = (0..N_QUERIES)
		.map(|_| verifier_channel.sample_bits(DEPTH))
		.collect::<Vec<_>>();

	// The openings on the tape are bound to `other_commitment`, so verifying them against
	// `commitment` must fail.
	assert!(
		verifier_channel
			.recv_openings(&commitment, &indices)
			.is_err()
	);

	let _ = verifier_channel.into_transcript();
}
