// Copyright 2025-2026 The Binius Developers

//! Reduction from fractional-addition layers to a multilinear evaluation claim.
//!
//! Each layer represents combining siblings with the fractional-addition rule:
//! (a0 / b0) + (a1 / b1) = (a0 * b1 + a1 * b0) / (b0 * b1).

use binius_field::Field;
use binius_math::line::extrapolate_line_packed;
use binius_transcript::{
	Error as TranscriptError, VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::sumcheck::{self, BatchSumcheckOutput};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FracAddEvalClaim<F: Field> {
	/// The evaluation of the numerator and denominator multilinears.
	pub num_eval: F,
	pub den_eval: F,
	/// The evaluation point.
	pub point: Vec<F>,
}

pub fn verify<F: Field, Challenger_: Challenger>(
	k: usize,
	claim: FracAddEvalClaim<F>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<FracAddEvalClaim<F>, Error> {
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
	} = sumcheck::batch_verify_mle(&point, 2, &evals, transcript)?;

	// Read evaluations of numerator/denominator halves at the reduced point.
	let [num_0, num_1, den_0, den_1] = transcript.message().read()?;

	// Sumcheck binds variables high-to-low; reverse to low-to-high for point evaluation.
	challenges.reverse();
	let reduced_eval_point = challenges;

	let numerator_eval = num_0 * den_1 + num_1 * den_0;
	let denominator_eval = den_0 * den_1;
	let batched_eval = numerator_eval + denominator_eval * batch_coeff;

	if batched_eval != eval {
		return Err(VerificationError::IncorrectLayerFractionSumEvaluation { round: k }.into());
	}

	// Reduce evaluations of the two halves to a single evaluation at the next point.
	let r = transcript.sample();
	let next_num = extrapolate_line_packed(num_0, num_1, r);
	let next_den = extrapolate_line_packed(den_0, den_1, r);

	let mut next_point = reduced_eval_point;
	next_point.push(r);

	verify(
		k - 1,
		FracAddEvalClaim {
			num_eval: next_num,
			den_eval: next_den,
			point: next_point,
		},
		transcript,
	)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[source] sumcheck::Error),
	#[error("transcript error: {0}")]
	Transcript(#[source] TranscriptError),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

impl From<sumcheck::Error> for Error {
	fn from(err: sumcheck::Error) -> Self {
		match err {
			sumcheck::Error::Verification(err) => VerificationError::Sumcheck(err).into(),
			_ => Error::Sumcheck(err),
		}
	}
}

impl From<TranscriptError> for Error {
	fn from(err: TranscriptError) -> Self {
		match err {
			TranscriptError::NotEnoughBytes => VerificationError::TranscriptIsEmpty.into(),
			_ => Error::Transcript(err),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("sumcheck: {0}")]
	Sumcheck(#[from] sumcheck::VerificationError),
	#[error("incorrect layer fraction sum evaluation: {round}")]
	IncorrectLayerFractionSumEvaluation { round: usize },
	#[error("incorrect round evaluation: {round}")]
	IncorrectRoundEvaluation { round: usize },
	#[error("transcript is empty")]
	TranscriptIsEmpty,
}
