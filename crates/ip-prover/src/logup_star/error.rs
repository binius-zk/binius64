// Copyright 2026 The Binius Developers

//! Error types for logUp* proving.

use crate::{fracaddcheck::FracAddCheckError, sumcheck::SumcheckError};

/// An error raised while proving a logUp* reduction.
///
/// It wraps the sub-protocol failures and the input-validation failures of the index column.
#[derive(Debug, thiserror::Error)]
pub enum LogupStarError {
	#[error("fractional-addition check error: {0}")]
	FracAddCheck(#[from] FracAddCheckError),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error(
		"the index column has {got} entries but {expected} were expected for {n_vars} variables"
	)]
	IndexLengthMismatch {
		got: usize,
		expected: usize,
		n_vars: usize,
	},
}
