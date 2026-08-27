// Copyright 2025 Irreducible Inc.

use binius_core::ConstraintSystemError;
use binius_iop::channel::Error as IOPChannelError;
use binius_ip::{channel::Error as ChannelError, sumcheck};

use crate::{
	fri,
	protocols::{intmul, shift},
	ring_switch,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("channel error: {0}")]
	Channel(#[from] ChannelError),
	#[error("IOP channel error: {0}")]
	IOPChannel(#[from] IOPChannelError),
	#[error("FRI error: {0}")]
	FRI(#[from] fri::Error),
	#[error("ring switch error: {0}")]
	RingSwitch(#[from] ring_switch::Error),
	#[error("IntMul error: {0}")]
	IntMul(#[from] intmul::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("incorrect public inputs length: expected {expected}, got {actual}")]
	IncorrectPublicInputLength { expected: usize, actual: usize },
	#[error("constraint system error: {0}")]
	ConstraintSystem(#[from] ConstraintSystemError),
	#[error("shift reduction error: {0}")]
	ShiftReduction(#[from] shift::Error),
	#[error(
		"no WHIR rate ladder over a 2^{log_msg_len} message at inverse rate 2^{log_inv_rate} \
		 reaches {security_bits} bits"
	)]
	NoWHIRLadder {
		log_msg_len: usize,
		log_inv_rate: usize,
		security_bits: usize,
	},
}
