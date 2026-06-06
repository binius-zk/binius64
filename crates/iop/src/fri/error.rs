// Copyright 2024-2025 Irreducible Inc.

use super::batch;
use crate::merkle_tree;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Merkle tree error: {0}")]
	MerkleError(merkle_tree::Error),
	#[error("Reed-Solomon encoding error: {0}")]
	Verification(#[from] VerificationError),
	#[error("transcript error: {0}")]
	TranscriptError(#[from] binius_transcript::Error),
}

impl From<merkle_tree::Error> for Error {
	fn from(err: merkle_tree::Error) -> Self {
		match err {
			merkle_tree::Error::Verification(err) => Self::Verification(err.into()),
			_ => Self::MerkleError(err),
		}
	}
}

impl From<batch::Error> for Error {
	fn from(err: batch::Error) -> Self {
		match err {
			batch::Error::Merkle(err) => err.into(),
			batch::Error::Transcript(err) => Self::TranscriptError(err),
			batch::Error::ClaimMismatch { index } => VerificationError::IncorrectFold {
				query_round: 0,
				index,
			}
			.into(),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
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
	MerkleError(#[from] merkle_tree::VerificationError),
}
