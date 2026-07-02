// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_transcript::TranscriptError;
use binius_utils::SerializationError;

#[derive(Debug, thiserror::Error)]
pub enum MerkleTreeError {
	#[error("Failed to serialize leaf element: {0}")]
	Serialization(SerializationError),
	#[error("transcript error: {0}")]
	Transcript(#[from] TranscriptError),
	#[error("verification failure: {0}")]
	Verification(#[from] MerkleTreeVerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum MerkleTreeVerificationError {
	#[error("the proof is invalid")]
	InvalidProof,
}
