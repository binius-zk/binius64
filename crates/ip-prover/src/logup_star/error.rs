// Copyright 2026 The Binius Developers

//! Error types for logUp* proving.

use crate::{fracaddcheck, sumcheck};

/// An error raised while proving a logUp* reduction.
///
/// It wraps the sub-protocol failures and the input-validation failures of the index column.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("fractional-addition check error: {0}")]
	FracAddCheck(#[from] fracaddcheck::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error(
		"the index column has {got} entries but {expected} were expected for {n_vars} variables"
	)]
	IndexLengthMismatch {
		got: usize,
		expected: usize,
		n_vars: usize,
	},
}
