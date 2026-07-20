// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::fri;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("FRI: {0}")]
	FRI(#[source] fri::Error),
	#[error("IP channel: {0}")]
	IPChannel(#[from] binius_ip::channel::Error),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("FRI: {0}")]
	FRI(#[from] fri::VerificationError),
}

impl From<fri::Error> for Error {
	fn from(err: fri::Error) -> Self {
		match err {
			fri::Error::Verification(err) => Error::Verification(err.into()),
			_ => Error::FRI(err),
		}
	}
}
