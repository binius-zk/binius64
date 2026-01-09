// Copyright 2025 Irreducible Inc.

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("transcript error")]
	Transcript(#[from] binius_transcript::Error),
	#[error("sumcheck error")]
	Sumcheck(#[from] crate::protocols::sumcheck::Error),
	#[error("verification failure")]
	VerificationFailure,
}
