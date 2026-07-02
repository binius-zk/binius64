// Copyright 2025 Irreducible Inc.

use crate::protocols::sumcheck::SumcheckError;

#[derive(thiserror::Error, Debug)]
pub enum ShiftError {
	#[error("sumcheck error: {0}")]
	SumcheckError(#[from] SumcheckError),
}
