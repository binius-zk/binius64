// Copyright 2025-2026 The Binius Developers

//! Verifier for the LogUp indexed lookup protocol.

use binius_field::Field;
use binius_math::univariate::evaluate_univariate;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::{
	fracaddcheck::{self, FracAddEvalClaim},
	sumcheck::{self, BatchSumcheckOutput},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogUpLookupClaims<F: Field> {
	pub table_id: usize,
	pub pushforward_eval: F,
	pub table_eval: F,
	pub eq_frac_claim: FracAddEvalClaim<F>,
	pub push_frac_claim: FracAddEvalClaim<F>,
}

/// Verify the LogUp lookup batch.
///
/// `eq_log_len` is the log-length of the lookup indices (eq-kernel), and `table_log_len`
/// is the log-length of the tables/pushforwards. `lookup_evals` must be aligned with the
/// lookup ordering of `lookup_claims`.
pub fn verify_lookup<F: Field, Challenger_: Challenger>(
	eq_log_len: usize,
	table_log_len: usize,
	lookup_evals: &[F],
	lookup_claims: &[LogUpLookupClaims<F>],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<(), Error> {
	let [_fingerprint_scalar, _shift_scalar]: [F; 2] = transcript.sample_array();

	if lookup_claims.len() != lookup_evals.len() {
		return Err(VerificationError::LookupEvalCountMismatch {
			claims: lookup_claims.len(),
			evals: lookup_evals.len(),
		}
		.into());
	}

	if lookup_claims.is_empty() {
		return Ok(());
	}

	let n_lookups = lookup_claims.len();
	let n_tables = lookup_claims
		.iter()
		.map(|claim| claim.table_id)
		.max()
		.map(|max_id| max_id + 1)
		.unwrap_or(0);

	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		challenges: _,
	} = sumcheck::batch_verify(table_log_len, 2, lookup_evals, transcript)?;

	let pushforward_evals = read_scalar_slice(transcript, n_lookups)?;
	let table_evals = read_scalar_slice(transcript, n_tables)?;

	for (index, claim) in lookup_claims.iter().enumerate() {
		if claim.pushforward_eval != pushforward_evals[index] {
			return Err(VerificationError::PushforwardEvalMismatch { index }.into());
		}

		if claim.table_id >= n_tables {
			return Err(VerificationError::TableIdOutOfRange {
				table_id: claim.table_id,
			}
			.into());
		}

		if claim.table_eval != table_evals[claim.table_id] {
			return Err(VerificationError::TableEvalMismatch { index }.into());
		}
	}

	let expected_terms = lookup_claims
		.iter()
		.enumerate()
		.map(|(index, claim)| pushforward_evals[index] * table_evals[claim.table_id])
		.collect::<Vec<_>>();
	let expected_eval = evaluate_univariate(&expected_terms, batch_coeff);
	if expected_eval != eval {
		return Err(VerificationError::PushforwardCompositionMismatch.into());
	}

	for (index, claim) in lookup_claims.iter().enumerate() {
		let eq = &claim.eq_frac_claim;
		let push = &claim.push_frac_claim;
		if eq.num_eval * push.den_eval != push.num_eval * eq.den_eval {
			return Err(VerificationError::LogSumClaimMismatch { index }.into());
		}
	}

	let eq_claims = lookup_claims
		.iter()
		.map(|claim| claim.eq_frac_claim.clone())
		.collect::<Vec<_>>();
	let push_claims = lookup_claims
		.iter()
		.map(|claim| claim.push_frac_claim.clone())
		.collect::<Vec<_>>();

	fracaddcheck::verify_batch(eq_log_len, eq_claims, transcript)?;
	fracaddcheck::verify_batch(eq_log_len, push_claims, transcript)?;

	Ok(())
}

fn read_scalar_slice<F: Field, C: Challenger>(
	transcript: &mut VerifierTranscript<C>,
	len: usize,
) -> Result<Vec<F>, Error> {
	Ok(transcript.message().read_scalar_slice::<F>(len)?)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("frac-add check error: {0}")]
	FracAddCheck(#[from] fracaddcheck::Error),
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("lookup eval count mismatch: claims {claims}, evals {evals}")]
	LookupEvalCountMismatch { claims: usize, evals: usize },
	#[error("table id out of range: {table_id}")]
	TableIdOutOfRange { table_id: usize },
	#[error("pushforward eval mismatch at lookup {index}")]
	PushforwardEvalMismatch { index: usize },
	#[error("table eval mismatch at lookup {index}")]
	TableEvalMismatch { index: usize },
	#[error("pushforward composition claim mismatch")]
	PushforwardCompositionMismatch,
	#[error("log-sum claim mismatch at lookup {index}")]
	LogSumClaimMismatch { index: usize },
}
