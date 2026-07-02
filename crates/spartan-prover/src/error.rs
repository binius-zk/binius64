// Copyright 2025 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("invalid argument {arg}: {msg}")]
	ArgumentError { arg: String, msg: String },
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
}
