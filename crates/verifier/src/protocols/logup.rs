// Copyright 2025-2026 The Binius Developers

//! Verifier for the LogUp indexed lookup protocol.

use std::iter::zip;

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct PushforwardVerificationOutput<F: Field> {
	table_ids: Vec<usize>,
	sumcheck_point: Vec<F>,
	pushforward_evals: Vec<F>,
	table_evals: Vec<F>,
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
	assert_eq!(eval_point.len(), eq_log_len, "eval_point length must match eq_log_len");
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
	let PushforwardVerificationOutput {
		sumcheck_point,
		pushforward_evals,
		table_evals,
		..
	} = verify_pushforward(table_log_len, lookup_evals, transcript)?;
	let (eq_claims, push_claims) = verify_log_sum(
		eq_log_len,
		eval_point,
		fingerprint_scalar,
		shift_scalar,
		lookup_evals.len(),
		transcript,
	)?;

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

/// Verifies the LogUp* ([Lev25] section 2 & 3) pullback/lookup values to pushforward" reduction for
/// a batch of lookups.
///
/// The key identity is the reduction of the evaluation of the "pullback" (which is simply the
/// lookup values) multilinear referred $I^*T(r)$ with a table-sized inner product $\langle T,
/// I_*eq_r \rangle$, avoiding a commitment to the pullback. Thus we understand a batch of lookup
/// claims is a collection of pushforwards, tables and evaluation claims on the corresponding
/// pullbacks. We read a vector of table_ids as advice from the prover in order to decide which
/// table the pushforward corresponds to, as many may share the same table.
///
/// This routine:
/// 1. Reads the per-lookup `table_id[i]` from the transcript.
/// 2. Verifies a batched degree-2 sumcheck over the `table_log_len`-variate boolean hypercube for
///    the claimed sums `lookup_evals` (with batching coefficient `beta`).
/// 3. Reads pushforward and table evaluations at the sumcheck point r'.
/// 4. Computes `e[i] = pushforward[i] * table_evals[table_id[i]]`, and essentially checks that the
///    final batch sumcheck eval is equal to a random linear combination of these claims.
/// 5. Returns `r'` along with the evaluation values for downstream [`MultilinearEvalClaim`]
///    openings.
///
/// [Lev25]: <https://eprint.iacr.org/2025/946>
fn verify_pushforward<F: Field, C: Challenger>(
	table_log_len: usize,
	lookup_evals: &[F],
	transcript: &mut VerifierTranscript<C>,
) -> Result<PushforwardVerificationOutput<F>, Error> {
	let n_lookups = lookup_evals.len();
	let table_ids: Vec<usize> = transcript.message().read_vec(n_lookups)?;
	let max_table_id = table_ids.iter().copied().max().unwrap_or(0);
	let n_tables = max_table_id
		.checked_add(1)
		.ok_or(PushforwardError::TableIdOutOfRange {
			table_id: max_table_id,
		})?;

	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify(table_log_len, 2, lookup_evals, transcript)?;
	challenges.reverse();
	let sumcheck_point = challenges;

	// The first n_lookups many final multilinear evals corresponds to the pushforwards, and the
	// next n_tables many correspond to the tables.
	let pushforward_evals = read_scalar_slice(transcript, n_lookups)?;
	let table_evals = read_scalar_slice(transcript, n_tables)?;

	// Recompute the batched quadratic composition at the verifier's point by multiplying the
	// pushforward eval with the table eval of the table it looked up into.
	let expected_terms = table_ids
		.iter()
		.zip(pushforward_evals.iter())
		.map(|(&table_id, &push_eval)| push_eval * table_evals[table_id])
		.collect::<Vec<_>>();
	let expected_eval = evaluate_univariate(&expected_terms, batch_coeff);
	if expected_eval != eval {
		return Err(PushforwardError::CompositionMismatch.into());
	}

	Ok(PushforwardVerificationOutput {
		table_ids,
		sumcheck_point,
		pushforward_evals,
		table_evals,
	})
}

fn verify_log_sum<F: Field, C: Challenger>(
	eq_log_len: usize,
	eval_point: &[F],
	fingerprint_scalar: F,
	shift_scalar: F,
	n_lookups: usize,
	transcript: &mut VerifierTranscript<C>,
) -> Result<(Vec<FracAddEvalClaim<F>>, Vec<FracAddEvalClaim<F>>), Error> {
	// Each lookup must satisfy the log-sum equality of fraction claims.
	let eq_claims = read_frac_claims(transcript, n_lookups)?;
	let push_claims = read_frac_claims(transcript, n_lookups)?;

	zip(eq_claims.iter(), push_claims.iter())
		.enumerate()
		.try_for_each(|(index, (eq, push))| {
			if eq.num_eval * push.den_eval != push.num_eval * eq.den_eval {
				return Err(LogSumError::LogSumClaimMismatch { index });
			}
			Ok(())
		})?;

	// Drive batched fractional-addition checks for the two reduction trees.
	// These return the reduced evaluation claims for the final layer.
	let eq_claims = fracaddcheck::verify_batch(eq_log_len, eq_claims, transcript)?;
	let push_claims = fracaddcheck::verify_batch(eq_log_len, push_claims, transcript)?;

	if let Some(first_claim) = eq_claims.first() {
		if first_claim.point.len() != eval_point.len() {
			return Err(LogSumError::EqKernelPointLengthMismatch {
				expected: eval_point.len(),
				actual: first_claim.point.len(),
			}
			.into());
		}

		// The eq-kernel numerator must equal eq_ind(eval_point, reduced_point).
		let expected_num_eval = eq_ind(eval_point, &first_claim.point);
		eq_claims
			.iter()
			.enumerate()
			.try_for_each(|(index, claim)| {
				if claim.num_eval != expected_num_eval {
					return Err(LogSumError::EqKernelNumeratorMismatch { index });
				}
				Ok(())
			})?;
	}
	if let Some(first_claim) = push_claims.first() {
		if first_claim.point.len() != eval_point.len() {
			return Err(LogSumError::PushforwardPointLengthMismatch {
				expected: eval_point.len(),
				actual: first_claim.point.len(),
			}
			.into());
		}

		let expected_den_eval =
			common_denominator_eval(&first_claim.point, fingerprint_scalar, shift_scalar);

		push_claims
			.iter()
			.enumerate()
			.try_for_each(|(index, claim)| {
				if claim.den_eval != expected_den_eval {
					return Err(LogSumError::CommonDenominatorMismatch { index });
				}
				Ok(())
			})?;
	}

	Ok((eq_claims, push_claims))
}

fn common_denominator_eval<F: Field>(point: &[F], fingerprint_scalar: F, shift_scalar: F) -> F {
	let (acc, _) = point
		.iter()
		.fold((shift_scalar, F::ONE), |(acc, power), &coord| {
			(acc + power * coord, power * fingerprint_scalar)
		});
	acc
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
	#[error("pushforward protocol error: {0}")]
	Pushforward(#[from] PushforwardError),
	#[error("log-sum protocol error: {0}")]
	LogSum(#[from] LogSumError),
}

/// Pushforward-specific failures for LogUp.
#[derive(Debug, thiserror::Error)]
pub enum PushforwardError {
	#[error("table id out of range: {table_id}")]
	TableIdOutOfRange { table_id: usize },
	#[error("pushforward composition claim mismatch")]
	CompositionMismatch,
}

/// Log-sum-specific failures for LogUp.
#[derive(Debug, thiserror::Error)]
pub enum LogSumError {
	#[error("log-sum claim mismatch at lookup {index}")]
	LogSumClaimMismatch { index: usize },
	#[error("eq-kernel claim point length mismatch: expected {expected}, got {actual}")]
	EqKernelPointLengthMismatch { expected: usize, actual: usize },
	#[error("eq-kernel numerator mismatch at lookup {index}")]
	EqKernelNumeratorMismatch { index: usize },
	#[error("pushforward claim point length mismatch: expected {expected}, got {actual}")]
	PushforwardPointLengthMismatch { expected: usize, actual: usize },
	#[error("common denominator mismatch at lookup {index}")]
	CommonDenominatorMismatch { index: usize },
}
