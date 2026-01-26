// Copyright 2025-2026 The Binius Developers

//! Reduction from fractional-addition layers to a multilinear evaluation claim.
//!
//! Each layer represents combining siblings with the fractional-addition rule:
//! (a0 / b0) + (a1 / b1) = (a0 * b1 + a1 * b0) / (b0 * b1).

use binius_field::Field;
use binius_math::{line::extrapolate_line_packed, univariate::evaluate_univariate};
use binius_transcript::{
	Error as TranscriptError, VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::{DeserializeBytes, SerializeBytes, SerializationError};
use bytes::{Buf, BufMut};

use crate::sumcheck::{self, BatchSumcheckOutput};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FracAddEvalClaim<F: Field> {
	/// The evaluation of the numerator and denominator multilinears.
	pub num_eval: F,
	pub den_eval: F,
	/// The evaluation point.
	pub point: Vec<F>,
}

impl<F: Field> SerializeBytes for FracAddEvalClaim<F> {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		SerializeBytes::serialize(&self.num_eval, &mut write_buf)?;
		SerializeBytes::serialize(&self.den_eval, &mut write_buf)?;
		SerializeBytes::serialize(&self.point, &mut write_buf)?;
		Ok(())
	}
}

impl<F: Field> DeserializeBytes for FracAddEvalClaim<F> {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let num_eval = DeserializeBytes::deserialize(&mut read_buf)?;
		let den_eval = DeserializeBytes::deserialize(&mut read_buf)?;
		let point = DeserializeBytes::deserialize(&mut read_buf)?;
		Ok(Self {
			num_eval,
			den_eval,
			point,
		})
	}
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

pub fn verify_batch<F: Field, Challenger_: Challenger>(
	k: usize,
	claims: Vec<FracAddEvalClaim<F>>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<Vec<FracAddEvalClaim<F>>, Error> {
	if k == 0 || claims.is_empty() {
		return Ok(claims);
	}

	let mut claims = claims;

	for round in (0..k).rev() {
		let eval_point = claims[0].point.clone();
		if !claims.iter().all(|claim| claim.point == eval_point) {
			return Err(VerificationError::BatchPointMismatch.into());
		}

		let evals = claims
			.iter()
			.flat_map(|claim| [claim.num_eval, claim.den_eval])
			.collect::<Vec<_>>();

		let BatchSumcheckOutput {
			batch_coeff,
			eval,
			mut challenges,
		} = sumcheck::batch_verify_mle(&eval_point, 2, &evals, transcript)?;

		let mut layer_evals = Vec::with_capacity(claims.len());
		let mut composed_evals = Vec::with_capacity(claims.len() * 2);

		for _ in 0..claims.len() {
			let [num_0, num_1, den_0, den_1] = transcript.message().read()?;
			let numerator_eval = num_0 * den_1 + num_1 * den_0;
			let denominator_eval = den_0 * den_1;
			composed_evals.push(numerator_eval);
			composed_evals.push(denominator_eval);
			layer_evals.push((num_0, num_1, den_0, den_1));
		}

		let expected_eval = evaluate_univariate(&composed_evals, batch_coeff);
		if expected_eval != eval {
			return Err(VerificationError::IncorrectLayerFractionSumEvaluation { round }.into());
		}

		// Sumcheck binds variables high-to-low; reverse to low-to-high for point evaluation.
		challenges.reverse();
		let mut reduced_eval_point = challenges;

		let r = transcript.sample();
		reduced_eval_point.push(r);

		let mut next_claims = Vec::with_capacity(claims.len());
		for (num_0, num_1, den_0, den_1) in layer_evals {
			let next_num = extrapolate_line_packed(num_0, num_1, r);
			let next_den = extrapolate_line_packed(den_0, den_1, r);
			next_claims.push(FracAddEvalClaim {
				num_eval: next_num,
				den_eval: next_den,
				point: reduced_eval_point.clone(),
			});
		}

		claims = next_claims;
	}

	Ok(claims)
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
	#[error("batch claims must share the evaluation point")]
	BatchPointMismatch,
	#[error("transcript is empty")]
	TranscriptIsEmpty,
}
