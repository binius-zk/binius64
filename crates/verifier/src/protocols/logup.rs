// Copyright 2025-2026 The Binius Developers

//! Verifier for the batched LogUp* indexed lookup protocol.
//!
//! Logup* describes the single-claim reduction:
//! `I^*T(r)` is checked by introducing `Y = I_*eq_r`, then proving
//! (1) `I^*T(r) = <T, Y>` and (2) `Y` is a correct pushforward via log-sum.
//!
//! This file verifies the batched form used by the prover:
//! - many lookup instances are folded into one claim per table with random lookup-slot weights;
//! - each table gets one concatenated pushforward instance;
//! - sub-protocol outputs are further folded across the table axis.
//!
//! The verifier mirrors that structure and returns only the reduced
//! multilinear opening claims that must be checked against external commitments.

use std::iter::zip;

use binius_field::Field;
use binius_iop::channel::IOPVerifierChannel;
use binius_ip::{
	channel::{self, IPVerifierChannel},
	sumcheck::SumcheckOutput,
};
use binius_math::{
	multilinear::eq::{eq_ind, eq_ind_partial_eval},
	univariate::evaluate_univariate,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::Itertools;

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
pub struct LogUpEvalClaims<F: Field, Oracle: Clone> {
	/// Fingerprinted index evaluations from the eq-kernel frac-add reductions.
	pub index_claims: Vec<MultilinearEvalClaim<F>>,
	/// Table evaluations at the pushforward sumcheck point.
	pub table_sumcheck_claims: Vec<MultilinearEvalClaim<F>>,

	pub pushforward_oracle: Oracle,
	pub pushforward_eval_claim: MultilinearEvalClaim<F>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PushforwardVerificationOutput<F: Field> {
	sumcheck_point: Vec<F>,
	pushforward_evals: Vec<F>,
	table_evals: Vec<F>,
}

/// Verify the LogUp lookup batch and return evaluation claims for external checks.
///
/// Relative to the single-claim story from  Logup*, this verifies the
/// fully batched pipeline emitted by the prover:
/// 1. Reconstruct batching context from the transcript (`recv_oracle`, `sample_array`,
///    `batch_lookup_evals`), yielding one batched lookup claim per table and the extended eq-kernel
///    point `[batch_prefix || r]`.
/// 2. [`verify_pushforward`] checks the batched `<table_t, pushforward_t>` identities (one per
///    table), i.e. the batched analogue of `I^*T(r) = <T, I_*eq_r>`.
/// 3. [`verify_log_sum`] checks that each pushforward is consistent with the concatenated
///    fingerprinted indices and the batched eq-kernel.
/// 4. The verifier samples fresh table-axis batching randomness and a reduction scalar, then
///    verifies the final bivariate reduction sumcheck that folds pushforward evaluations coming
///    from steps (2) and (3) into one opening.
/// 5. Returns reduced claims for external opening checks: table evaluations, fingerprinted index
///    evaluations, and one pushforward evaluation claim bound to `pushforward_oracle`.
///
/// `eq_log_len` is the lookup-domain log size, `table_log_len` is the
/// table/pushforward log size, and `lookup_evals`/`table_ids` must match the
/// lookup ordering used by the prover.
pub fn verify_lookup<F: Field, Channel: IOPVerifierChannel<F>, const N_TABLES: usize>(
	eq_log_len: usize,
	table_log_len: usize,
	eval_point: &[F],
	lookup_evals: &[F],
	table_ids: &[usize],
	channel: &mut Channel,
) -> Result<LogUpEvalClaims<F, Channel::Oracle>, Error> {
	// Basic shape checks for transcript-consistent inputs.
	assert!(table_ids.len() == lookup_evals.len());
	assert!(!lookup_evals.is_empty());
	assert_eq!(eval_point.len(), eq_log_len, "eval_point length must match eq_log_len");
	// Match the prover's Fiat-Shamir sampling performed during `LogUp::new`.
	let grouped_evals = zip(lookup_evals.into_iter().copied(), table_ids.into_iter().copied())
		.into_group_map_by(|&(_, id)| id);

	assert!(
		grouped_evals.iter().map(|(_, vals)| vals.len()).all_equal(),
		"There must be an equal number of lookups into each table"
	);
	// Observe the pushforward oracle before sampling, then draw log-sum batching scalars.
	let pushforward_oracle = channel.recv_oracle().unwrap();
	let [fingerprint_scalar, shift_scalar]: [F; 2] = channel.sample_array();

	// Fold per-lookup evaluations into one claim per table and extend the eq-kernel point.
	let (batched_evals, extended_eval_point): ([F; N_TABLES], Vec<F>) =
		batch_lookup_evals(&lookup_evals, eval_point, &table_ids, channel);

	// Check <table_t, pushforward_t> identities for each table in batched form.
	let PushforwardVerificationOutput {
		sumcheck_point,
		pushforward_evals,
		table_evals,
	} = verify_pushforward::<_, N_TABLES>(table_log_len, &batched_evals, channel)?;

	// Check the two log-sum trees and recover reduced eq/push claims.
	let (index_log_sum_claims, pushforward_log_sum_claims) = verify_log_sum::<_, N_TABLES>(
		eq_log_len,
		&extended_eval_point,
		fingerprint_scalar,
		shift_scalar,
		channel,
	)?;

	// Fold the pushforward and log-sum outputs into one reduced opening claim.
	let (pf_claim, challenges) = verify_reduction::<_, N_TABLES>(
		&sumcheck_point,
		&pushforward_evals,
		&pushforward_log_sum_claims,
		channel,
	)?;

	// Package reduced table openings at the pushforward sumcheck point.
	let table_sumcheck_claims = table_evals
		.into_iter()
		.map(|eval| MultilinearEvalClaim {
			eval,
			point: sumcheck_point.clone(),
		})
		.collect();
	let index_claims = index_log_sum_claims
		.into_iter()
		.map(|claim| MultilinearEvalClaim {
			// Denominator tracks the fingerprinted index MLE.
			eval: claim.den_eval,
			point: claim.point,
		})
		.collect();

	// Final reduced opening for the pushforward multilinear.
	let pushforward_eval_claim = MultilinearEvalClaim {
		eval: pf_claim,
		point: challenges,
	};

	Ok(LogUpEvalClaims {
		index_claims,
		table_sumcheck_claims,
		pushforward_oracle,
		pushforward_eval_claim,
	})
}

/// Verifies the final reduction sumcheck that combines:
/// - pushforward evaluations from `verify_pushforward`, and
/// - pushforward numerators from `verify_log_sum`,
/// into one reduced pushforward opening claim.
fn verify_reduction<F: Field, const N_TABLES: usize>(
	sumcheck_point: &[F],
	pushforward_evals: &[F],
	pushforward_log_sum_claims: &[FracAddEvalClaim<F>],
	channel: &mut impl IPVerifierChannel<F>,
) -> Result<(F, Vec<F>), Error> {
	let log_sum_eval_point = pushforward_log_sum_claims
		.first()
		.map(|claim| &claim.point)
		.ok_or(ReductionError::MissingPushClaim)?;

	// Calculate the number of variables to extend evaluation points.
	let batch_vars = log2_ceil_usize(N_TABLES);
	let batch_prefix = channel.sample_many(batch_vars);

	let batch_weights = eq_ind_partial_eval::<F>(&batch_prefix);

	let pushforward_batch_eval: F = zip(batch_weights.iter_scalars(), pushforward_evals.iter())
		.map(|(weight, eval)| weight * eval)
		.sum();

	// The push claims have the pushforward evaluation claim in their numerator evaluation.
	let log_sum_batch_eval: F =
		zip(batch_weights.iter_scalars(), pushforward_log_sum_claims.iter())
			.map(|(weight, &FracAddEvalClaim { num_eval, .. })| weight * num_eval)
			.sum();

	let reduction_scalar = channel.sample();

	let extended_pushforward_eval_point = [sumcheck_point.to_vec(), batch_prefix.clone()].concat();
	let extended_log_sum_eval_point = [log_sum_eval_point.clone(), batch_prefix.clone()].concat();

	let SumcheckOutput {
		eval,
		mut challenges,
	} = sumcheck::verify(
		extended_pushforward_eval_point.len(),
		2,
		pushforward_batch_eval + log_sum_batch_eval * reduction_scalar,
		channel,
	)?;

	let [pf_claim, reduction_buffer] = channel.recv_array()?;

	if pf_claim * reduction_buffer != eval {
		return Err(ReductionError::OpeningMismatch.into());
	}

	challenges.reverse();
	let reduction_buffer_eval = eq_ind(&extended_pushforward_eval_point, &challenges)
		+ eq_ind(&extended_log_sum_eval_point, &challenges) * reduction_scalar;

	if reduction_buffer != reduction_buffer_eval {
		return Err(ReductionError::BufferMismatch.into());
	}

	Ok((pf_claim, challenges))
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

/// Verifies the batched pushforward/table reduction for one claim per table.
///
/// The checked identity is the batched form of `I^*T(r) = <T, I_*eq_r>`:
/// each `lookup_evals[t]` is claimed to equal `<table_t, pushforward_t>`.
/// The verifier:
/// 1. verifies a degree-2 batch sumcheck over `table_log_len` variables;
/// 2. receives final-point evaluations for all `pushforward_t` and `table_t`;
/// 3. recomputes the expected batched composition `pushforward_t(r') * table_t(r')` and checks it
///    matches the sumcheck eval;
/// 4. returns `r'` plus those evaluations for later cross-protocol reduction.
///
/// [Lev25]: <https://eprint.iacr.org/2025/946>
fn verify_pushforward<F: Field, const N_TABLES: usize>(
	table_log_len: usize,
	lookup_evals: &[F],
	channel: &mut impl IPVerifierChannel<F>,
) -> Result<PushforwardVerificationOutput<F>, Error> {
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify(table_log_len, 2, lookup_evals, channel)?;
	challenges.reverse();
	let sumcheck_point = challenges;

	// Final multilinear evaluations are emitted in prover order:
	// [pushforward_0..pushforward_{N_TABLES-1}, table_0..table_{N_TABLES-1}].
	let pushforward_evals = read_scalar_slice(channel, N_TABLES)?;
	let table_evals = read_scalar_slice(channel, N_TABLES)?;

	// Recompute the batched quadratic composition at the verifier's point:
	// one product per table claim.
	let expected_terms = zip(pushforward_evals.iter(), table_evals.iter())
		.map(|(&push_eval, &table_eval)| push_eval * table_eval)
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

/// Verifies the two batched log-sum trees used by LogUp*.
///
/// For each table, transcript inputs carry top-layer fraction claims for:
/// - eq-kernel numerator / fingerprinted-index denominator,
/// - pushforward numerator / common-enumeration denominator.
///
/// This function checks their initial consistency relation, runs batched
/// `fracaddcheck` reductions, then enforces the semantic endpoint checks:
/// - all eq numerators equal `eq_ind(eval_point, reduced_point)`,
/// - all pushforward denominators equal the shared enumeration fingerprint evaluated at the same
///   reduced point.
type LogSumOutput<F> = Vec<FracAddEvalClaim<F>>;
fn verify_log_sum<F: Field, const N_TABLES: usize>(
	eq_log_len: usize,
	eval_point: &[F],
	fingerprint_scalar: F,
	shift_scalar: F,
	transcript: &mut impl IPVerifierChannel<F>,
) -> Result<(LogSumOutput<F>, LogSumOutput<F>), Error> {
	// Read combined log-sum claims from the transcript and convert to frac-add claims.
	let (index_claims, pushforward_claims): (Vec<_>, Vec<_>) = (0..N_TABLES)
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

	zip(index_claims.iter(), pushforward_claims.iter())
		.enumerate()
		.try_for_each(|(index, (index_claim, pushforward_claim))| {
			if index_claim.num_eval * pushforward_claim.den_eval
				!= pushforward_claim.num_eval * index_claim.den_eval
			{
				return Err(LogSumError::LogSumClaimMismatch { index });
			}
			Ok(())
		})?;

	// Drive batched fractional-addition checks for the two reduction trees.
	// These return the reduced evaluation claims for the final layer.
	let index_claims = fracaddcheck::verify_batch(eq_log_len, index_claims, transcript)?;
	let pushforward_claims =
		fracaddcheck::verify_batch(eq_log_len, pushforward_claims, transcript)?;

	if let Some(first_claim) = index_claims.first() {
		assert!(first_claim.point.len() == eval_point.len());

		// The eq-kernel numerator must equal eq_ind(eval_point, reduced_point).
		let expected_num_eval = eq_ind(eval_point, &first_claim.point);
		index_claims
			.iter()
			.enumerate()
			.try_for_each(|(index, claim)| {
				if claim.num_eval != expected_num_eval {
					return Err(LogSumError::EqKernelNumeratorMismatch { index });
				}
				Ok(())
			})?;
	}
	if let Some(first_claim) = pushforward_claims.first() {
		assert!(first_claim.point.len() == eval_point.len());

		let expected_den_eval =
			common_denominator_eval(&first_claim.point, fingerprint_scalar, shift_scalar);

		pushforward_claims
			.iter()
			.enumerate()
			.try_for_each(|(index, claim)| {
				if claim.den_eval != expected_den_eval {
					return Err(LogSumError::CommonDenominatorMismatch { index });
				}
				Ok(())
			})?;
	}

	Ok((index_claims, pushforward_claims))
}

fn common_denominator_eval<F: Field>(point: &[F], fingerprint_scalar: F, shift_scalar: F) -> F {
	let enum_eval = evaluate_univariate(point, fingerprint_scalar);
	shift_scalar + enum_eval
}

pub fn batch_lookup_evals<F: Field, const N_TABLES: usize>(
	lookup_evals: &[F],
	eval_point: &[F],
	table_ids: &[usize],
	channel: &mut impl IPVerifierChannel<F>,
) -> ([F; N_TABLES], Vec<F>) {
	let grouped_evals = zip(lookup_evals.into_iter().copied(), table_ids.into_iter().copied())
		.into_group_map_by(|&(_, id)| id);

	// We assume each table has an equal number of lookups. This mainly serves to simplify the
	// structure of the various sumchecks in the protocol. A possible future todo would be to remove
	// this assumption.
	assert!(
		grouped_evals.iter().map(|(_, vals)| vals.len()).all_equal(),
		"There must be an equal number of lookups into each table"
	);

	let batch_log_len = log2_ceil_usize(grouped_evals[&0].len().next_power_of_two());

	let batching_prefix = channel.sample_many(batch_log_len);

	let batch_weights = eq_ind_partial_eval::<F>(&batching_prefix);

	let mut batched_evals = [F::ZERO; N_TABLES];

	// Iterate over the lookup evals per table and batch them using the batch_weights,
	for (table_id, vals) in grouped_evals {
		let (evals, _): (Vec<_>, Vec<_>) = vals.into_iter().unzip();
		batched_evals[table_id] = zip(evals, batch_weights.iter_scalars())
			.map(|(eval, weight)| eval * weight)
			.sum();
	}

	let extended_eval_point = [batching_prefix, eval_point.to_vec()].concat();

	(batched_evals, extended_eval_point)
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
	#[error("reduction protocol error: {0}")]
	Reduction(#[from] ReductionError),
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

/// Final reduction-specific failures for LogUp.
#[derive(Debug, thiserror::Error)]
pub enum ReductionError {
	#[error("missing push claim for final reduction")]
	MissingPushClaim,
	#[error("reduction opening claim mismatch")]
	OpeningMismatch,
	#[error("reduction buffer evaluation mismatch")]
	BufferMismatch,
}
