// Copyright 2025 Irreducible Inc.

use binius_ip::{channel::IPChannelError, prodcheck::ProdcheckError, sumcheck::SumcheckError};
use binius_transcript::TranscriptError;

#[derive(thiserror::Error, Debug)]
pub enum IntMulError {
	#[error("transcript error")]
	Transcript(#[from] TranscriptError),
	#[error("channel error")]
	Channel(#[from] IPChannelError),
	#[error("sumcheck verify error")]
	SumcheckVerify(#[from] SumcheckError),
	#[error("prodcheck verify error")]
	ProdcheckVerify(#[from] ProdcheckError),
}
