// Copyright 2025 Irreducible Inc.

use binius_ip_prover::{prodcheck::ProdcheckError, sumcheck::SumcheckError};
use binius_transcript::TranscriptError;

#[derive(thiserror::Error, Debug)]
pub enum IntMulError {
	#[error("Exponent length should be a power of two")]
	ExponentsPowerOfTwoLengthRequired,
	#[error("All exponent slices must have the same length")]
	ExponentLengthMismatch,
	#[error("transcript error")]
	Transcript(#[from] TranscriptError),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("prodcheck error: {0}")]
	Prodcheck(#[from] ProdcheckError),
}
