// Copyright 2026 The Binius Developers

use crate::{fri, merkle_channel};

/// Anything that can go wrong verifying a WHIR opening.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// The Merkle channel could not deliver a commitment or an opening.
	#[error("Merkle channel: {0}")]
	Channel(#[from] merkle_channel::Error),
	/// The interactive-proof channel could not deliver a message.
	#[error("IP channel: {0}")]
	IPChannel(#[from] binius_ip::channel::Error),
	/// A committed value did not match what the transcript bound it to.
	#[error("verification: {0}")]
	Verification(#[from] VerificationError),
	/// The prover did not pay the proof of work the parameters fix, or sent no nonce at all.
	#[error("proof of work: {0}")]
	ProofOfWork(#[from] binius_transcript::Error),
}

/// A prover message that was well-formed but wrong.
///
/// Separated from [`Error`] so a caller can tell a malformed transcript from a rejected proof.
#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	/// A Merkle opening did not match its commitment.
	#[error("Merkle tree: {0}")]
	MerkleTree(#[from] crate::merkle_tree::VerificationError),
}

impl From<fri::batch::Error> for Error {
	fn from(err: fri::batch::Error) -> Self {
		match err {
			fri::batch::Error::Channel(err) => err.into(),
			fri::batch::Error::IPChannel(err) => Self::IPChannel(err),
		}
	}
}
