// Copyright 2025-2026 The Binius Developers

//! Verifier for the LogUp indexed lookup protocol.

use binius_field::Field;
use binius_math::{multilinear::eq::eq_ind, univariate::evaluate_univariate};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::{
	fracaddcheck::{self, FracAddEvalClaim},
	prodcheck::MultilinearEvalClaim,
	sumcheck::{self, BatchSumcheckOutput},
};

/// Per-lookup claims emitted by the prover and serialized into the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogUpLookupClaims<F: Field> {
	/// Lookup table identifier used to pick the corresponding table evaluation.
	pub table_id: usize,
	/// Evaluation of the pushforward multilinear at the verifier's challenge point.
	pub pushforward_eval: F,
	/// Evaluation of the table multilinear at the verifier's challenge point.
	pub table_eval: F,
	/// Fractional-addition claim for the eq-kernel numerator/denominator tree.
	pub eq_frac_claim: FracAddEvalClaim<F>,
	/// Fractional-addition claim for the pushforward numerator/denominator tree.
	pub push_frac_claim: FracAddEvalClaim<F>,
}

/// Evaluation claims produced by the LogUp verifier for external checking.
///
/// These claims are the reduced multilinear evaluations that remain after the
/// verifier checks the LogUp sub-protocols. They can be passed into other
/// verifiers that own the commitments for the corresponding multilinears.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogUpEvalClaims<F: Field> {
	/// Fingerprinted index evaluations from the eq-kernel frac-add reductions.
	pub index_claims: Vec<MultilinearEvalClaim<F>>,
	/// Pushforward evaluations at the pushforward sumcheck point.
	pub pushforward_sumcheck_claims: Vec<MultilinearEvalClaim<F>>,
	/// Table evaluations at the pushforward sumcheck point.
	pub table_sumcheck_claims: Vec<MultilinearEvalClaim<F>>,
	/// Pushforward evaluations from the pushforward frac-add reductions.
	pub pushforward_frac_claims: Vec<MultilinearEvalClaim<F>>,
}

/// Verify the LogUp lookup batch and return evaluation claims for external checks.
///
/// `eq_log_len` is the log-length of the lookup indices (eq-kernel), and `table_log_len`
/// is the log-length of the tables/pushforwards. `eval_point` is the point used to
/// instantiate the eq-kernel. `lookup_evals` must be aligned with the lookup ordering
/// encoded in the transcript.
pub fn verify_lookup<F: Field, Challenger_: Challenger>(
	eq_log_len: usize,
	table_log_len: usize,
	eval_point: &[F],
	lookup_evals: &[F],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<LogUpEvalClaims<F>, Error> {
	// Match the prover's Fiat-Shamir sampling performed during `LogUp::new`.
	let [fingerprint_scalar, shift_scalar]: [F; 2] = transcript.sample_array();

	if lookup_evals.is_empty() {
		return Ok(LogUpEvalClaims {
			index_claims: Vec::new(),
			pushforward_sumcheck_claims: Vec::new(),
			table_sumcheck_claims: Vec::new(),
			pushforward_frac_claims: Vec::new(),
		});
	}
	if eval_point.len() != eq_log_len {
		return Err(VerificationError::EvalPointLengthMismatch {
			expected: eq_log_len,
			actual: eval_point.len(),
		}
		.into());
	}

	let n_lookups = lookup_evals.len();
	let table_ids: Vec<usize> = transcript.message().read_vec(n_lookups)?;
	let max_table_id = table_ids.iter().copied().max().unwrap_or(0);
	let n_tables = max_table_id
		.checked_add(1)
		.ok_or(VerificationError::TableIdOutOfRange {
			table_id: max_table_id,
		})?;

	// TODO: Add comment linking to paper logic.
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify(table_log_len, 2, lookup_evals, transcript)?;
	challenges.reverse();
	let sumcheck_point = challenges;

	let pushforward_evals = read_scalar_slice(transcript, n_lookups)?;
	let table_evals = read_scalar_slice(transcript, n_tables)?;

	// Recompute the batched quadratic composition at the verifier's point.
	let expected_terms = table_ids
		.iter()
		.enumerate()
		.map(|(index, &table_id)| pushforward_evals[index] * table_evals[table_id])
		.collect::<Vec<_>>();
	let expected_eval = evaluate_univariate(&expected_terms, batch_coeff);
	if expected_eval != eval {
		return Err(VerificationError::PushforwardCompositionMismatch.into());
	}

	// Each lookup must satisfy the log-sum equality of fraction claims.
	let eq_claims = read_frac_claims(transcript, n_lookups)?;
	let push_claims = read_frac_claims(transcript, n_lookups)?;

	for (index, (eq, push)) in eq_claims.iter().zip(push_claims.iter()).enumerate() {
		if eq.num_eval * push.den_eval != push.num_eval * eq.den_eval {
			return Err(VerificationError::LogSumClaimMismatch { index }.into());
		}
	}

	// Drive batched fractional-addition checks for the two reduction trees.
	// These return the reduced evaluation claims for the final layer.
	let eq_claims = fracaddcheck::verify_batch(eq_log_len, eq_claims, transcript)?;
	let push_claims = fracaddcheck::verify_batch(eq_log_len, push_claims, transcript)?;

	if let Some(first_claim) = eq_claims.first() {
		if first_claim.point.len() != eval_point.len() {
			return Err(VerificationError::EqKernelPointLengthMismatch {
				expected: eval_point.len(),
				actual: first_claim.point.len(),
			}
			.into());
		}

		// The eq-kernel numerator must equal eq_ind(eval_point, reduced_point).
		let expected_num_eval = eq_ind(eval_point, &first_claim.point);
		for (index, claim) in eq_claims.iter().enumerate() {
			if claim.num_eval != expected_num_eval {
				return Err(VerificationError::EqKernelNumeratorMismatch { index }.into());
			}
		}
	}
	if let Some(first_claim) = push_claims.first() {
		if first_claim.point.len() != eval_point.len() {
			return Err(VerificationError::PushforwardPointLengthMismatch {
				expected: eval_point.len(),
				actual: first_claim.point.len(),
			}
			.into());
		}

		// Common denominator is shift + sum_i fingerprint_scalar^i * r_i.
		let mut expected_den_eval = shift_scalar;
		let mut power = F::ONE;
		for &coord in &first_claim.point {
			expected_den_eval += power * coord;
			power *= fingerprint_scalar;
		}

		for (index, claim) in push_claims.iter().enumerate() {
			if claim.den_eval != expected_den_eval {
				return Err(VerificationError::CommonDenominatorMismatch { index }.into());
			}
		}
	}

	let pushforward_sumcheck_claims = pushforward_evals
		.into_iter()
		.map(|eval| MultilinearEvalClaim {
			eval,
			point: sumcheck_point.clone(),
		})
		.collect();
	let table_sumcheck_claims = table_evals
		.into_iter()
		.map(|eval| MultilinearEvalClaim {
			eval,
			point: sumcheck_point.clone(),
		})
		.collect();
	let index_claims = eq_claims
		.into_iter()
		.map(|claim| MultilinearEvalClaim {
			// Denominator tracks the fingerprinted index MLE.
			eval: claim.den_eval,
			point: claim.point,
		})
		.collect();
	let pushforward_frac_claims = push_claims
		.into_iter()
		.map(|claim| MultilinearEvalClaim {
			// Numerator tracks the pushforward MLE.
			eval: claim.num_eval,
			point: claim.point,
		})
		.collect();

	Ok(LogUpEvalClaims {
		index_claims,
		pushforward_sumcheck_claims,
		table_sumcheck_claims,
		pushforward_frac_claims,
	})
}

/// Reads a fixed-length scalar slice from the transcript.
fn read_scalar_slice<F: Field, C: Challenger>(
	transcript: &mut VerifierTranscript<C>,
	len: usize,
) -> Result<Vec<F>, Error> {
	Ok(transcript.message().read_scalar_slice::<F>(len)?)
}

fn read_frac_claims<F: Field, C: Challenger>(
	transcript: &mut VerifierTranscript<C>,
	len: usize,
) -> Result<Vec<FracAddEvalClaim<F>>, Error> {
	Ok(transcript.message().read_vec(len)?)
}

/// Errors returned by the LogUp verifier.
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

/// Verification-specific failures for LogUp.
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
	#[error("eval point length mismatch: expected {expected}, got {actual}")]
	EvalPointLengthMismatch { expected: usize, actual: usize },
	#[error("eq-kernel claim point length mismatch: expected {expected}, got {actual}")]
	EqKernelPointLengthMismatch { expected: usize, actual: usize },
	#[error("eq-kernel numerator mismatch at lookup {index}")]
	EqKernelNumeratorMismatch { index: usize },
	#[error("pushforward claim point length mismatch: expected {expected}, got {actual}")]
	PushforwardPointLengthMismatch { expected: usize, actual: usize },
	#[error("common denominator mismatch at lookup {index}")]
	CommonDenominatorMismatch { index: usize },
}
