// Copyright 2025 The Binius Developers

//! Reduction from fractional-addition layers to a multilinear evaluation claim.
//!
//! Each layer represents combining siblings with the fractional-addition rule:
//! (a0 / b0) + (a1 / b1) = (a0 * b1 + a1 * b0) / (b0 * b1).

use binius_field::Field;
use binius_math::{line::extrapolate_line_packed, multilinear::eq::eq_ind};
use binius_transcript::{
	Error as TranscriptError, VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::{
	prodcheck::MultilinearEvalClaim,
	sumcheck::{self, BatchSumcheckOutput},
};

pub fn verify<F: Field, Challenger_: Challenger>(
	k: usize,
	claim: (MultilinearEvalClaim<F>, MultilinearEvalClaim<F>),
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<(MultilinearEvalClaim<F>, MultilinearEvalClaim<F>), Error> {
	if k == 0 {
		return Ok(claim);
	}

	let (num_claim, den_claim) = claim;
	assert_eq!(
		num_claim.point, den_claim.point,
		"fractional claims must share the evaluation point"
	);

	let n_vars = num_claim.point.len();
	let sums = [num_claim.eval, den_claim.eval];

	// Reduce numerator and denominator sum claims to evaluations at a challenge point.
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify(n_vars, 3, &sums, transcript)?;

	// Read evaluations of numerator/denominator halves at the reduced point.
	let [num_0, den_0, num_1, den_1] = transcript.message().read()?;

	// Sumcheck binds variables high-to-low; reverse to low-to-high for point evaluation.
	challenges.reverse();
	let reduced_eval_point = challenges;

	let eq_eval = eq_ind(&num_claim.point, &reduced_eval_point);
	let numerator_eval = (num_0 * den_1 + num_1 * den_0) * eq_eval;
	let denominator_eval = (den_0 * den_1) * eq_eval;
	let batched_eval = numerator_eval + denominator_eval * batch_coeff;

	if batched_eval != eval {
		return Err(VerificationError::IncorrectRoundEvaluation { round: k }.into());
	}

	// Reduce evaluations of the two halves to a single evaluation at the next point.
	let r = transcript.sample();
	let next_num = extrapolate_line_packed(num_0, num_1, r);
	let next_den = extrapolate_line_packed(den_0, den_1, r);

	let mut next_point = reduced_eval_point;
	next_point.push(r);

	verify(
		k - 1,
		(
			MultilinearEvalClaim {
				eval: next_num,
				point: next_point.clone(),
			},
			MultilinearEvalClaim {
				eval: next_den,
				point: next_point,
			},
		),
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
	#[error("incorrect round evaluation: {round}")]
	IncorrectRoundEvaluation { round: usize },
	#[error("transcript is empty")]
	TranscriptIsEmpty,
}
