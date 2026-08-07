// Copyright 2025 Irreducible Inc.

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("Exponent length should be a power of two")]
	ExponentsPowerOfTwoLengthRequired,
	#[error("All exponent slices must have the same length")]
	ExponentLengthMismatch,
	#[error("transcript error")]
	Transcript(#[from] binius_transcript::Error),
}
