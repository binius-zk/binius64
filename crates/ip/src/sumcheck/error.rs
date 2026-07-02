// Copyright 2025 Irreducible Inc.

use binius_transcript::TranscriptError;

use crate::channel;

#[derive(Debug, thiserror::Error)]
pub enum SumcheckError {
	#[error("transcript error: {0}")]
	Transcript(#[source] TranscriptError),
	#[error("verification error: {0}")]
	Verification(#[from] SumcheckVerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum SumcheckVerificationError {
	#[error("transcript is empty")]
	TranscriptIsEmpty,
	#[error("invalid assertion: value is not zero")]
	InvalidAssert,
}

impl From<TranscriptError> for SumcheckError {
	fn from(err: TranscriptError) -> Self {
		match err {
			TranscriptError::NotEnoughBytes => SumcheckVerificationError::TranscriptIsEmpty.into(),
			_ => SumcheckError::Transcript(err),
		}
	}
}

impl From<channel::IPChannelError> for SumcheckError {
	fn from(err: channel::IPChannelError) -> Self {
		match err {
			channel::IPChannelError::ProofEmpty => {
				SumcheckVerificationError::TranscriptIsEmpty.into()
			}
			channel::IPChannelError::InvalidAssert => {
				SumcheckVerificationError::InvalidAssert.into()
			}
		}
	}
}
