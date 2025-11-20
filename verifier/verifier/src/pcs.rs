// Copyright 2025 Irreducible Inc.

//! The multilinear polynomial commitment scheme for 1-bit polynomials from [DP24] (FRI-Binius).
//!
//! This is the verifier protocol for the small-field polynomial commitment scheme (PCS) from
//! [DP24], Section 5. This implementation is specialized for the case of polynomials over GF(2).
//! The PCS combines ring-switching and the BaseFold protocol to commit packed multilinears and
//! open the evaluation of the unpacked polynomial at a target point from the extension field.
//!
//! [DP24]: <https://eprint.iacr.org/2024/504>

use binius_field::{BinaryField, ExtensionField, Field, PackedField};
use binius_math::{
	field_buffer::FieldBuffer,
	multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
	tensor_algebra::TensorAlgebra,
};
use binius_transcript::{
	Error as TranscriptError, VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::DeserializeBytes;

use crate::{
	config::B1, fri::FRIParams, merkle_tree::MerkleTreeScheme, protocols::basefold,
	ring_switch::verifier::eval_rs_eq,
};

/// Verifies a PCS opening of a committed polynomial at a given point.
///
/// See module documentation for protocol description.
///
/// ## Arguments
///
/// * `transcript` - the transcript of the prover's proof
/// * `evaluation_claim` - the evaluation claim of the prover
/// * `eval_point` - the evaluation point of the prover
/// * `codeword_commitment` - VCS commitment to the codeword
/// * `fri_params` - the FRI parameters
/// * `vcs` - the vector commitment scheme
pub fn verify<F, MTScheme, Challenger_>(
	transcript: &mut VerifierTranscript<Challenger_>,
	evaluation_claim: F,
	eval_point: &[F],
	codeword_commitment: MTScheme::Digest,
	fri_params: &FRIParams<F>,
	merkle_scheme: &MTScheme,
) -> Result<(), Error>
where
	F: Field + BinaryField + PackedField<Scalar = F>,
	Challenger_: Challenger,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
{
	let packing_degree = <F as ExtensionField<B1>>::LOG_DEGREE;

	let s_hat_v = FieldBuffer::from_values(
		&transcript
			.message()
			.read_scalar_slice::<F>(1 << packing_degree)?,
	)
	.unwrap();

	// check valid partial eval
	let (eval_point_low, _) = eval_point.split_at(packing_degree);
	let computed_claim = evaluate::<F, F, _>(&s_hat_v, eval_point_low).unwrap();
	if evaluation_claim != computed_claim {
		return Err(VerificationError::EvaluationClaimMismatch.into());
	}

	// basis decompose/recombine s_hat_v across opposite dimension
	let s_hat_u = FieldBuffer::from_values(
		&<TensorAlgebra<B1, F>>::new(s_hat_v.as_ref().to_vec())
			.transpose()
			.elems,
	)
	.expect("tensor algebra has 2^packing_degree elements");

	let batching_scalars = transcript.sample_vec(packing_degree);

	let verifier_computed_sumcheck_claim = evaluate::<F, F, _>(&s_hat_u, &batching_scalars).expect(
		"s_hat_u has 2^packing_degree elements; batching_scalars has packing_degree elements",
	);

	let basefold::ReducedOutput {
		final_fri_value,
		final_sumcheck_value,
		mut challenges,
	} = basefold::verify(
		fri_params,
		merkle_scheme,
		codeword_commitment,
		verifier_computed_sumcheck_claim,
		transcript,
	)?;

	let (_, eval_point_high) = eval_point.split_at(packing_degree);

	// Reverse challenges to get the reduced evaluation point.
	challenges.reverse();

	let rs_eq_at_basefold_challenges = eval_rs_eq::<F>(
		eval_point_high,
		&challenges,
		eq_ind_partial_eval(&batching_scalars).as_ref(),
	);

	if final_sumcheck_value != final_fri_value * rs_eq_at_basefold_challenges {
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
	#[error("evaluation claim verification failed")]
	EvaluationClaimMismatch,
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
