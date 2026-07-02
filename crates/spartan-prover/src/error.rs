// Copyright 2025 Irreducible Inc.

use binius_iop_prover::basefold::BaseFoldError;
use binius_ip_prover::sumcheck::SumcheckError;
use binius_transcript::TranscriptError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("invalid argument {arg}: {msg}")]
	ArgumentError { arg: String, msg: String },
	#[error("basefold error: {0}")]
	Basefold(#[from] BaseFoldError),
	#[error("transcript error: {0}")]
	Transcript(#[from] TranscriptError),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
}
