// Copyright 2025 Irreducible Inc.

use binius_ip_prover::sumcheck::Error as SumcheckError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	SumcheckError(#[from] SumcheckError),
}
