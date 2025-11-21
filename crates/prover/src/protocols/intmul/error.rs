// Copyright 2025 Irreducible Inc.

use crate::protocols::sumcheck::Error as SumcheckError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("Exponent length should be a power of two")]
	ExponentsPowerOfTwoLengthRequired,
	#[error("All exponent slices must have the same length")]
	ExponentLengthMismatch,
	#[error("transcript error")]
	Transcript(#[from] binius_transcript::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("math error: {0}")]
	Math(#[from] binius_math::Error),
}
