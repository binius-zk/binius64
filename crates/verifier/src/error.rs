// Copyright 2025 Irreducible Inc.

use binius_core::ConstraintSystemError;
use binius_iop::{channel::IOPChannelError, fri::FriError};
use binius_ip::{channel::IPChannelError, sumcheck::SumcheckError};
use binius_transcript::TranscriptError;

use crate::{
	protocols::{intmul::IntMulError, shift::ShiftError},
	ring_switch::RingSwitchError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("transcript error: {0}")]
	Transcript(#[from] TranscriptError),
	#[error("channel error: {0}")]
	Channel(#[from] IPChannelError),
	#[error("IOP channel error: {0}")]
	IOPChannel(#[from] IOPChannelError),
	#[error("FRI error: {0}")]
	FRI(#[from] FriError),
	#[error("ring switch error: {0}")]
	RingSwitch(#[from] RingSwitchError),
	#[error("IntMul error: {0}")]
	IntMul(#[from] IntMulError),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("incorrect public inputs length: expected {expected}, got {actual}")]
	IncorrectPublicInputLength { expected: usize, actual: usize },
	#[error("constraint system error: {0}")]
	ConstraintSystem(#[from] ConstraintSystemError),
	#[error("invalid proof: {0}")]
	Verification(#[from] VerificationError),
	#[error("shift reduction error: {0}")]
	ShiftReduction(#[from] ShiftError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("public input check failed")]
	PublicInputCheckFailed,
	#[error("final evaluation check of sumcheck and FRI reductions failed")]
	EvaluationInconsistency,
}
