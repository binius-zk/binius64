// Copyright 2025 Irreducible Inc.

//! Polynomial commitment scheme for binary field multilinears using BaseFold.
//!
//! This is a PCS that directly commits binary field multilinear polynomials using FRI and opens
//! them at evaluation points using the BaseFold protocol from [DP24] Section 4.
//!
//! [DP24]: <https://eprint.iacr.org/2024/504>

use binius_field::BinaryField;
use binius_transcript::{Error as TranscriptError, VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::DeserializeBytes;
use binius_verifier::{fri::FRIParams, merkle_tree::MerkleTreeScheme, protocols::basefold};

/// Verifies a PCS opening of a committed binary field multilinear polynomial at a given point.
///
/// See module documentation for protocol description.
///
/// ## Arguments
///
/// * `transcript` - the transcript of the prover's proof
/// * `evaluation_claim` - the claimed evaluation of the multilinear at eval_point
/// * `eval_point` - the evaluation point (in F^n_vars)
/// * `codeword_commitment` - VCS commitment to the codeword
/// * `fri_params` - the FRI parameters
/// * `merkle_scheme` - the merkle tree scheme
pub fn verify<F, MTScheme, Challenger_>(
	transcript: &mut VerifierTranscript<Challenger_>,
	evaluation_claim: F,
	eval_point: &[F],
	codeword_commitment: MTScheme::Digest,
	fri_params: &FRIParams<F>,
	merkle_scheme: &MTScheme,
) -> Result<(), Error>
where
	F: BinaryField,
	Challenger_: Challenger,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
{
	let basefold::ReducedOutput {
		final_fri_value,
		final_sumcheck_value,
		challenges,
	} = basefold::verify(
		fri_params,
		merkle_scheme,
		codeword_commitment,
		evaluation_claim,
		transcript,
	)?;

	// Verify consistency between sumcheck and FRI final values
	if !basefold::sumcheck_fri_consistency(
		final_fri_value,
		final_sumcheck_value,
		eval_point,
		challenges,
	) {
		return Err(VerificationError::EvaluationInconsistency.into());
	}

	Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("transcript error: {0}")]
	Transcript(#[source] TranscriptError),
	#[error("basefold error: {0}")]
	Basefold(#[source] basefold::Error),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("final evaluation check of sumcheck and FRI reductions failed")]
	EvaluationInconsistency,
	#[error("basefold: {0}")]
	Basefold(#[from] basefold::VerificationError),
	#[error("proof tape is empty")]
	EmptyProof,
}

impl From<TranscriptError> for Error {
	fn from(err: TranscriptError) -> Self {
		match err {
			TranscriptError::NotEnoughBytes => VerificationError::EmptyProof.into(),
			_ => Error::Transcript(err),
		}
	}
}

impl From<basefold::Error> for Error {
	fn from(err: basefold::Error) -> Self {
		match err {
			basefold::Error::Verification(err) => VerificationError::Basefold(err).into(),
			_ => Error::Basefold(err),
		}
	}
}
