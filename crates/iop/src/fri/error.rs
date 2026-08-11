// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::batch;
use crate::{merkle_channel, merkle_tree};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Merkle channel error: {0}")]
	Channel(merkle_channel::Error),
	#[error("Reed-Solomon encoding error: {0}")]
	Verification(#[from] VerificationError),
}

impl From<binius_ip::channel::Error> for Error {
	fn from(err: binius_ip::channel::Error) -> Self {
		Self::Channel(err.into())
	}
}

impl From<merkle_channel::Error> for Error {
	fn from(err: merkle_channel::Error) -> Self {
		match err {
			merkle_channel::Error::MerkleTree(merkle_tree::Error::Verification(err)) => {
				Self::Verification(err.into())
			}
			_ => Self::Channel(err),
		}
	}
}

impl From<batch::Error> for Error {
	fn from(err: batch::Error) -> Self {
		match err {
			batch::Error::Channel(err) => err.into(),
			batch::Error::IPChannel(err) => Self::Channel(err.into()),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("Merkle tree error: {0}")]
	MerkleError(#[from] merkle_tree::VerificationError),
}
