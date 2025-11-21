// Copyright 2025 Irreducible Inc.

use binius_math::Error as MathError;

use crate::protocols::sumcheck::Error as SumcheckError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	SumcheckError(#[from] SumcheckError),
	#[error("math error: {0}")]
	MathError(#[from] MathError),
}
