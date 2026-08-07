// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("All exponent slices must have the same length")]
	ExponentLengthMismatch,
	#[error("transcript error")]
	Transcript(#[from] binius_transcript::Error),
}
