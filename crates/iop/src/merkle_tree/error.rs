// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_utils::SerializationError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Failed to serialize leaf element: {0}")]
	Serialization(SerializationError),
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the proof is invalid")]
	InvalidProof,
}
