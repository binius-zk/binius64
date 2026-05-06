// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_utils::SerializationError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Length of the input vector is incorrect, expected {expected}")]
	IncorrectVectorLen { expected: usize },
	#[error("binary Merkle tree error: {0}")]
	BinaryMerkleTree(#[from] binius_hash::binary_merkle_tree::Error),
	#[error("Failed to serialize leaf element: {0}")]
	Serialization(SerializationError),
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the length of the vector does not match the committed length")]
	IncorrectVectorLength,
	#[error("the shape of the proof is incorrect")]
	IncorrectProofShape,
	#[error("the proof is invalid")]
	InvalidProof,
}
