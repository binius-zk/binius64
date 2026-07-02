// Copyright 2026 The Binius Developers

//! LogupStarError types for logUp* verification.

use crate::{fracaddcheck::FracAddCheckError, sumcheck::SumcheckError};

#[derive(Debug, thiserror::Error)]
pub enum LogupStarError {
	#[error("fractional-addition check error: {0}")]
	FracAddCheck(#[from] FracAddCheckError),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("verification error: {0}")]
	Verification(#[from] LogupStarVerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum LogupStarVerificationError {
	#[error("the two lookup fractional sums are not equal")]
	LookupSumMismatch,
	#[error("the eq_r multilinear evaluation is incorrect")]
	IncorrectXEvaluation,
	#[error("the batched final layer evaluation is incorrect")]
	FinalLayerMismatch,
	#[error("the proof is truncated or empty")]
	TranscriptIsEmpty,
}
