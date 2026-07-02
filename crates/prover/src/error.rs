// Copyright 2025 Irreducible Inc.

use crate::protocols::intmul;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("invalid argument {arg}: {msg}")]
	ArgumentError { arg: String, msg: String },
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("integer multiplication error: {0}")]
	IntMul(#[from] intmul::Error),
}
