// Copyright 2025-2026 The Binius Developers

//! Reduction from fractional-addition layers to a multilinear evaluation claim.
//!
//! Each layer represents combining siblings with the fractional-addition rule:
//! (a0 / b0) + (a1 / b1) = (a0 * b1 + a1 * b0) / (b0 * b1).

use binius_field::Field;
use binius_math::line::extrapolate_line;
use binius_transcript::TranscriptError;

use crate::{
	channel::{IPChannelError, IPVerifierChannel},
	sumcheck::{self, BatchSumcheckOutput, SumcheckError, SumcheckVerificationError},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FracAddEvalClaim<F> {
	/// The evaluation of the numerator and denominator multilinears.
	pub num_eval: F,
	pub den_eval: F,
	/// The evaluation point.
	pub point: Vec<F>,
}

pub fn verify<F, C>(
	k: usize,
	claim: FracAddEvalClaim<C::Elem>,
	channel: &mut C,
) -> Result<FracAddEvalClaim<C::Elem>, FracAddCheckError>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	if k == 0 {
		return Ok(claim);
	}

	let FracAddEvalClaim {
		num_eval,
		den_eval,
		point,
	} = claim;

	let evals = [num_eval, den_eval];

	// Reduce numerator and denominator sum claims to evaluations at a challenge point.
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify_mle(&point, 2, &evals, channel)?;

	// Read evaluations of numerator/denominator halves at the reduced point.
	let [num_0, num_1, den_0, den_1] = channel.recv_array()?;

	// Sumcheck binds variables high-to-low; reverse to low-to-high for point evaluation.
	challenges.reverse();
	let reduced_eval_point = challenges;

	let numerator_eval = num_0.clone() * den_1.clone() + num_1.clone() * den_0.clone();
	let denominator_eval = den_0.clone() * den_1.clone();
	let batched_eval = numerator_eval + denominator_eval * batch_coeff;

	channel.assert_zero(batched_eval - eval)?;

	// Reduce evaluations of the two halves to a single evaluation at the next point.
	let r = channel.sample();
	let next_num = extrapolate_line(num_0, num_1, r.clone());
	let next_den = extrapolate_line(den_0, den_1, r.clone());

	let mut next_point = reduced_eval_point;
	next_point.push(r);

	verify(
		k - 1,
		FracAddEvalClaim {
			num_eval: next_num,
			den_eval: next_den,
			point: next_point,
		},
		channel,
	)
}

#[derive(Debug, thiserror::Error)]
pub enum FracAddCheckError {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[source] SumcheckError),
	#[error("transcript error: {0}")]
	Transcript(#[source] TranscriptError),
	#[error("verification error: {0}")]
	Verification(#[from] FracAddCheckVerificationError),
}

impl From<SumcheckError> for FracAddCheckError {
	fn from(err: SumcheckError) -> Self {
		match err {
			SumcheckError::Verification(err) => FracAddCheckVerificationError::Sumcheck(err).into(),
			_ => FracAddCheckError::Sumcheck(err),
		}
	}
}

impl From<TranscriptError> for FracAddCheckError {
	fn from(err: TranscriptError) -> Self {
		match err {
			TranscriptError::NotEnoughBytes => {
				FracAddCheckVerificationError::TranscriptIsEmpty.into()
			}
			_ => FracAddCheckError::Transcript(err),
		}
	}
}

impl From<IPChannelError> for FracAddCheckError {
	fn from(err: IPChannelError) -> Self {
		match err {
			IPChannelError::ProofEmpty => FracAddCheckVerificationError::TranscriptIsEmpty.into(),
			IPChannelError::InvalidAssert => FracAddCheckVerificationError::InvalidAssert.into(),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum FracAddCheckVerificationError {
	#[error("sumcheck: {0}")]
	Sumcheck(#[from] SumcheckVerificationError),
	#[error("incorrect layer fraction sum evaluation: {round}")]
	IncorrectLayerFractionSumEvaluation { round: usize },
	#[error("incorrect round evaluation: {round}")]
	IncorrectRoundEvaluation { round: usize },
	#[error("transcript is empty")]
	TranscriptIsEmpty,
	#[error("invalid assertion: value is not zero")]
	InvalidAssert,
}
