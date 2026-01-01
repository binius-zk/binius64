// Copyright 2025-2026 The Binius Developers

//! Reduction from the products over the sumcubes of a multilinear to a multilinear evaluation.
//!
//! The reduction input is a multilinear $f(Z_0, \ldots, Z_{k-1}, X_0, \ldots, X_{n-1})$. The
//! product polynomial is the multilinear
//!
//! $$
//! p(X_0, \ldots, X_{n-1}) = \sum_{x \in B_n} \text{eq}(x ; X) \prod_{z \in B_k} f(z, x)
//! $$
//!
//! This protocol is a GKR-based protocol with $k$ sumcheck invocations. We define a sequence of
//! multilinears $p_0, \ldots, p_k$, where $p_k = f$ and for all $i < k$:
//!
//! $$
//! p_i(Z_0, \ldots, Z_{i-1}, X_0, \ldots, X_{n-1}) = \sum_{x \in B_n} \sum_{z \in B_i} \text{eq}(x
//! ; X) \text{eq}(z ; Z) p_{i+1}(z, 0, x) p_{i+1}(z, 1, x) $$

use binius_field::Field;
use binius_math::line::extrapolate_line_packed;
use binius_transcript::{
	Error as TranscriptError, VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::{
	mlecheck,
	sumcheck::{self, SumcheckOutput},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearEvalClaim<F: Field> {
	/// The evaluation of the multilinear.
	pub eval: F,
	/// The evaluation point.
	pub point: Vec<F>,
}

pub fn verify<F: Field, Challenger_: Challenger>(
	k: usize,
	claim: MultilinearEvalClaim<F>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<MultilinearEvalClaim<F>, Error> {
	if k == 0 {
		return Ok(claim);
	}

	let MultilinearEvalClaim { eval, point } = claim;

	// Reduce p_i evaluation to two evaluations of p_{i+1}.
	let SumcheckOutput { eval, challenges } = mlecheck::verify(&point, 2, eval, transcript)?;

	// Read evaluations of p_{i+1)(0, \ldots) and p_{i+1}(1, \ldots).
	let [eval_0, eval_1] = transcript.message().read()?;

	if eval_0 * eval_1 != eval {
		return Err(VerificationError::IncorrectRoundEvaluation { round: k }.into());
	}

	// Reduce evaluations of p_{i+1}(0, \ldots) and p_{i+1}(1, \ldots) to single eval at
	// p_{i+1}(r, \ldots).
	let r = transcript.sample();

	let next_eval = extrapolate_line_packed(eval_0, eval_1, r);

	let mut next_point = challenges;
	next_point.reverse();
	next_point.push(r);

	verify(
		k - 1,
		MultilinearEvalClaim {
			eval: next_eval,
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
	#[error("incorrect round evaluation: {round}")]
	IncorrectRoundEvaluation { round: usize },
	#[error("transcript is empty")]
	TranscriptIsEmpty,
}
