// Copyright 2024-2025 Irreducible Inc.

use binius_transcript::TranscriptError;

use super::batch::BatchFriError;
use crate::merkle_tree::{MerkleTreeError, MerkleTreeVerificationError};

#[derive(Debug, thiserror::Error)]
pub enum FriError {
	#[error("Merkle tree error: {0}")]
	MerkleError(MerkleTreeError),
	#[error("Reed-Solomon encoding error: {0}")]
	Verification(#[from] FriVerificationError),
	#[error("transcript error: {0}")]
	TranscriptError(#[from] TranscriptError),
}

impl From<MerkleTreeError> for FriError {
	fn from(err: MerkleTreeError) -> Self {
		match err {
			MerkleTreeError::Verification(err) => Self::Verification(err.into()),
			_ => Self::MerkleError(err),
		}
	}
}

impl From<BatchFriError> for FriError {
	fn from(err: BatchFriError) -> Self {
		match err {
			BatchFriError::Merkle(err) => err.into(),
			BatchFriError::Transcript(err) => Self::TranscriptError(err),
			BatchFriError::ClaimMismatch { index } => FriVerificationError::IncorrectFold {
				query_round: 0,
				index,
			}
			.into(),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum FriVerificationError {
	#[error("incorrect codeword folding in query round {query_round} at index {index}")]
	IncorrectFold { query_round: usize, index: usize },
	#[error("the size of the query proof is incorrect, expected {expected}")]
	IncorrectQueryProofLength { expected: usize },
	#[error(
		"the number of values in round {round} of the query proof is incorrect, expected {coset_size}"
	)]
	IncorrectQueryProofValuesLength { round: usize, coset_size: usize },
	#[error("The dimension-1 codeword must contain the same values")]
	IncorrectDegree,
	#[error("Merkle tree error: {0}")]
	MerkleError(#[from] MerkleTreeVerificationError),
}
