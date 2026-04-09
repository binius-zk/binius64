// Copyright 2025 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("invalid argument {arg}: {msg}")]
	ArgumentError { arg: String, msg: String },
	#[error("FRI error: {0}")]
	Fri(#[from] binius_iop_prover::fri::Error),
	#[error("basefold error: {0}")]
	Basefold(#[from] binius_iop_prover::basefold::Error),
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] binius_ip_prover::sumcheck::Error),
}
