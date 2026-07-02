// Copyright 2025 Irreducible Inc.

use binius_ip_prover::sumcheck::SumcheckError;

#[derive(thiserror::Error, Debug)]
pub enum ShiftError {
	#[error("sumcheck error: {0}")]
	SumcheckError(#[from] SumcheckError),
}
