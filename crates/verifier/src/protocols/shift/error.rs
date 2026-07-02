// Copyright 2025 Irreducible Inc.

use binius_ip::{channel::IPChannelError, sumcheck::SumcheckError};
use binius_transcript::TranscriptError;

#[derive(thiserror::Error, Debug)]
pub enum ShiftError {
	#[error("transcript error")]
	Transcript(#[from] TranscriptError),
	#[error("channel error")]
	Channel(#[from] IPChannelError),
	#[error("sumcheck error")]
	Sumcheck(#[from] SumcheckError),
	#[error("verification failure")]
	VerificationFailure,
}
