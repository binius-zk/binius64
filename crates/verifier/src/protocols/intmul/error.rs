// Copyright 2025 Irreducible Inc.

use crate::protocols::sumcheck::Error as SumcheckError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("math error: {0}")]
	Math(#[from] binius_math::Error),
	#[error("transcript error")]
	Transcript(#[from] binius_transcript::Error),
	#[error("sumcheck verify error")]
	SumcheckVerify(#[from] SumcheckError),
	#[error("composition claim mismatch")]
	CompositionClaimMismatch,
}
