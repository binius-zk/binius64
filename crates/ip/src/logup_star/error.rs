// Copyright 2026 The Binius Developers

//! Error types for logUp* verification.

use crate::{fracaddcheck, sumcheck};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("fractional-addition check error: {0}")]
	FracAddCheck(#[from] fracaddcheck::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the two lookup fractional sums are not equal")]
	LookupSumMismatch,
	#[error("the eq_r multilinear evaluation is incorrect")]
	IncorrectXEvaluation,
	#[error("the index evaluations do not combine to the leaf denominator")]
	IncorrectIndexEvaluation,
	#[error("the table-side leaf denominator is not the transparent c - J")]
	IncorrectTableDenominator,
	#[error("the pushforward reduction evaluation is incorrect")]
	PushforwardMismatch,
	#[error("the proof is truncated or empty")]
	TranscriptIsEmpty,
}
