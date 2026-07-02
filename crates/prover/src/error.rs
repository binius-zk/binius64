// Copyright 2025 Irreducible Inc.

use binius_iop_prover::basefold::BaseFoldError;
use binius_ip_prover::sumcheck::SumcheckError;
use binius_transcript::TranscriptError;

use crate::protocols::{intmul::IntMulError, shift::ShiftError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("invalid argument {arg}: {msg}")]
	ArgumentError { arg: String, msg: String },
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("basefold error: {0}")]
	Basefold(#[from] BaseFoldError),
	#[error("transcript error: {0}")]
	Transcript(#[from] TranscriptError),
	#[error("integer multiplication error: {0}")]
	IntMul(#[from] IntMulError),
	#[error("shift reduction error: {0}")]
	ShiftReduction(#[from] ShiftError),
}
