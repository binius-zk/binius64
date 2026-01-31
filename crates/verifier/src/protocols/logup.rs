// Copyright 2025-2026 The Binius Developers

//! Verifier for the LogUp indexed lookup protocol.

use std::iter::zip;

use binius_field::Field;
use binius_ip::channel::{self, IPVerifierChannel};
use binius_math::{multilinear::eq::eq_ind, univariate::evaluate_univariate};

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
pub fn verify_lookup<F: Field>(
	eq_log_len: usize,
	table_log_len: usize,
	eval_point: &[F],
	lookup_evals: &[F],
	table_ids: &[usize],
	channel: &mut impl IPVerifierChannel<F>,
) -> Result<LogUpEvalClaims<F>, Error> {
	assert_eq!(eval_point.len(), eq_log_len, "eval_point length must match eq_log_len");
	// Match the prover's Fiat-Shamir sampling performed during `LogUp::new`.
	let [fingerprint_scalar, shift_scalar]: [F; 2] = channel.sample_array();

	assert!(table_ids.len() == lookup_evals.len());

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
	} = verify_pushforward(table_log_len, lookup_evals, table_ids, channel)?;
	let (eq_claims, push_claims) = verify_log_sum(
		eq_log_len,
		eval_point,
		fingerprint_scalar,
		shift_scalar,
		lookup_evals.len(),
		channel,
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
fn read_scalar_slice<F: Field>(
	transcript: &mut impl IPVerifierChannel<F>,
	len: usize,
) -> Result<Vec<F>, Error> {
	Ok(transcript
		.recv_many(len)
		.expect("Received values should be of expected length or there should be a panic."))
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
fn verify_pushforward<F: Field>(
	table_log_len: usize,
	lookup_evals: &[F],
	table_ids: &[usize],
	channel: &mut impl IPVerifierChannel<F>,
) -> Result<PushforwardVerificationOutput<F>, Error> {
	let n_lookups = lookup_evals.len();

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
	} = sumcheck::batch_verify(table_log_len, 2, lookup_evals, channel)?;
	challenges.reverse();
	let sumcheck_point = challenges;

	// The first n_lookups many final multilinear evals corresponds to the pushforwards, and the
	// next n_tables many correspond to the tables.
	let pushforward_evals = read_scalar_slice(channel, n_lookups)?;
	let table_evals = read_scalar_slice(channel, n_tables)?;

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
		sumcheck_point,
		pushforward_evals,
		table_evals,
	})
}

/// Verifies the logarithmic sum claims as part of Logup*. Assumes lookups are of the same length
/// and tables are of the same length and returns the
type LogSumOutput<F> = Vec<FracAddEvalClaim<F>>;
fn verify_log_sum<F: Field>(
	eq_log_len: usize,
	eval_point: &[F],
	fingerprint_scalar: F,
	shift_scalar: F,
	n_lookups: usize,
	transcript: &mut impl IPVerifierChannel<F>,
) -> Result<(LogSumOutput<F>, LogSumOutput<F>), Error> {
	// Read combined log-sum claims from the transcript and convert to frac-add claims.
	let (eq_claims, push_claims): (Vec<_>, Vec<_>) = (0..n_lookups)
		.map(|_| match transcript.recv_array() {
			Ok([eq_num_eval, eq_den_eval, push_num_eval, push_den_eval]) => Ok((
				FracAddEvalClaim {
					num_eval: eq_num_eval,
					den_eval: eq_den_eval,
					point: Vec::new(),
				},
				FracAddEvalClaim {
					num_eval: push_num_eval,
					den_eval: push_den_eval,
					point: Vec::new(),
				},
			)),
			Err(err) => Err(err),
		})
		.collect::<Result<(Vec<_>, Vec<_>), _>>()?;
	// Convert LogSumClaims into pairs of FracAddEvalClaims.

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
		assert!(first_claim.point.len() == eval_point.len());

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
		assert!(first_claim.point.len() == eval_point.len());

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
	let enum_eval = evaluate_univariate(point, fingerprint_scalar);
	shift_scalar + enum_eval
}

/// Errors returned by the LogUp verifier.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("frac-add check error: {0}")]
	FracAddCheck(#[from] fracaddcheck::Error),
	#[error("transcript error: {0}")]
	Channel(#[from] channel::Error),
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
