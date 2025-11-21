// Copyright 2025 Irreducible Inc.

use binius_core::ConstraintSystemError;

use crate::{
	fri, pcs,
	protocols::{intmul, shift, sumcheck},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("FRI error: {0}")]
	FRI(#[from] fri::Error),
	#[error("NTT error: {0}")]
	PCS(#[from] pcs::Error),
	#[error("IntMul error: {0}")]
	IntMul(#[from] intmul::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("Math error: {0}")]
	Math(#[from] binius_math::Error),
	#[error("incorrect public inputs length: expected {expected}, got {actual}")]
	IncorrectPublicInputLength { expected: usize, actual: usize },
	#[error("constraint system error: {0}")]
	ConstraintSystem(#[from] ConstraintSystemError),
	#[error("invalid proof: {0}")]
	Verification(#[from] VerificationError),
	#[error("shift reduction error: {0}")]
	ShiftReduction(#[from] shift::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("public input check failed")]
	PublicInputCheckFailed,
	#[error("AND-reduction MLE check failed, polynomial evals don't match sumcheck claim")]
	AndReductionMLECheckFailed,
}
