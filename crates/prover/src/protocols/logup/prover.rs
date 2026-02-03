// Copyright 2025-2026 The Binius Developers
use std::{array, iter::zip};

use binius_field::{Field, PackedField};
use binius_iop_prover::channel::IOPProverChannel;
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{ProveSingleOutput, prove_single},
};
use binius_math::FieldBuffer;
use binius_verifier::protocols::{
	fracaddcheck::FracAddEvalClaim,
	logup::{LogUpEvalClaims, LogUpLookupClaims},
	prodcheck::MultilinearEvalClaim,
};
use rand::seq::index;

use crate::protocols::{
	fracaddcheck,
	logup::{LogUp, helper::generate_pushforward},
	sumcheck::Error as SumcheckError,
};

/// Errors that can arise while proving the LogUp lookup batch.
#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("fractional-addition check error: {0}")]
	FracAddCheck(#[from] fracaddcheck::Error),
}

impl<
	P: PackedField<Scalar = F>,
	Channel: IOPProverChannel<P>,
	F: Field,
	const N_TABLES: usize,
	const N_LOOKUPS: usize,
> LogUp<P, Channel, N_TABLES, N_LOOKUPS>
{
	/// Runs the full LogUp proving flow and returns per-lookup claims.
	///
	/// The output is ordered to match the lookup ordering in the `LogUp` instance.
	///
	/// # Preconditions
	/// * `N_MLES == N_TABLES + N_LOOKUPS`.
	pub fn prove_lookup<const N_MLES: usize>(
		&self,
		channel: &mut Channel,
	) -> Result<LogUpEvalClaims<F, Channel::Oracle>, Error> {
		assert!(N_MLES == 2 * N_TABLES);
		// Reduce lookup evaluations to pushforward/table evaluations.
		let pushforward_claims = self.prove_pushforward::<N_MLES>(channel)?;
		// Prove log-sum consistency for eq-kernel and pushforward trees.
		let (eq_claims, push_claims) = self.prove_log_sum(channel)?;

		// Reduce pushforward and log-sum evaluations via bivariate product sumcheck.
		let ProveSingleOutput {
			multilinear_evals: reduction_sumcheck_evals,
			challenges: reduction_sumcheck_point,
		} = reduce_pushforward_logsum::<P, F, N_TABLES>(
			&pushforward_claims,
			&push_claims,
			&self.push_forwards,
			channel,
		)?;

		let PushforwardEvalClaims {
			challenges,
			table_evals,
			..
		} = pushforward_claims;

		let table_sumcheck_claims = table_evals
			.iter()
			.map(|&eval| MultilinearEvalClaim {
				eval,
				point: challenges.clone(),
			})
			.collect::<Vec<_>>();

		let index_claims = eq_claims
			.iter()
			.map(
				|&FracAddEvalClaim {
				     den_eval: shifted_index_eval,
				     ref point,
				     ..
				 }| MultilinearEvalClaim {
					eval: shifted_index_eval - self.shift_scalar,
					point: point.clone(),
				},
			)
			.collect::<Vec<_>>();

		let pushforward_eval_claim = MultilinearEvalClaim {
			eval: reduction_sumcheck_evals[0],
			point: reduction_sumcheck_point,
		};

		Ok(LogUpEvalClaims {
			index_claims: index_claims,
			table_sumcheck_claims: table_sumcheck_claims,
			pushforward_oracle: self.batch_pushforward_oracle.clone(),
			pushforward_eval_claim: pushforward_eval_claim,
		})
	}
}

/// Builds pushforward tables for each lookup batch.
///
/// Each output table accumulates `eq_kernel` values at the indices referenced by the lookup.
pub fn build_pushforwards<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize>(
	indexes: &[&[usize]; N_LOOKUPS],
	table_ids: &[usize; N_LOOKUPS],
	eq_kernel: &FieldBuffer<P>,
	tables: &[FieldBuffer<P>; N_TABLES],
) -> [FieldBuffer<P>; N_LOOKUPS] {
	array::from_fn(|i| {
		let (indices, table_id) = (indexes[i], table_ids[i]);
		generate_pushforward(indices, eq_kernel, tables[table_id].len())
	})
}

/// Builds pushforward tables for each lookup batch.
///
/// Each output table accumulates `eq_kernel` values at the indices referenced by the lookup.
pub fn build_pushforwards_from_concat_indexes<P: PackedField, const N_TABLES: usize>(
	concat_indexes: &[Vec<usize>; N_TABLES],
	tables: &[FieldBuffer<P>; N_TABLES],
	eq_kernel: &FieldBuffer<P>,
) -> [FieldBuffer<P>; N_TABLES] {
	array::from_fn(|i| generate_pushforward(&concat_indexes[i], eq_kernel, tables[i].len()))
}

use crate::protocols::logup::pushforward::PushforwardEvalClaims;
use binius_ip_prover::sumcheck::bivariate_product::BivariateProductSumcheckProver;
use binius_math::multilinear::eq::eq_ind_partial_eval;
use binius_utils::checked_arithmetics::log2_ceil_usize;

/// Reduces pushforward evaluation claims throughout the protocol to a single point.
///
///
/// # Arguments
/// * `pushforward_claims` - Evaluation claims from the pushforward sumcheck
/// * `push_claims` - Fractional addition claims from the log-sum protocol
/// * `push_forwards` - The pushforward MLE data
/// * `channel` - Prover channel for sampling and communication
///
/// # Returns
/// Returns `Ok(())` if the reduction succeeds, or a `SumcheckError` on failure.
pub fn reduce_pushforward_logsum<P, F, const N_TABLES: usize>(
	pushforward_claims: &PushforwardEvalClaims<F>,
	push_claims: &[FracAddEvalClaim<F>],
	push_forwards: &[FieldBuffer<P>],
	channel: &mut impl IPProverChannel<F>,
) -> Result<ProveSingleOutput<F>, SumcheckError>
where
	P: PackedField<Scalar = F>,
	F: Field,
{
	let pushforward_eval_point = &pushforward_claims.challenges;
	let log_sum_eval_point = &push_claims[0].point;

	// Calculate the number of variables to extend evaluation points.
	let batch_vars = log2_ceil_usize(N_TABLES);
	let batch_prefix = channel.sample_many(batch_vars);

	let batch_weights = eq_ind_partial_eval::<P>(&batch_prefix);

	let pushforward_batch_eval: F =
		zip(batch_weights.iter_scalars(), pushforward_claims.pushforward_evals.iter())
			.map(|(weight, eval)| weight * eval)
			.sum();

	// The push claims have the pushforward evaluation claim in their numerator evaluation.
	let log_sum_batch_eval: F = zip(batch_weights.iter_scalars(), push_claims.iter())
		.map(|(weight, &FracAddEvalClaim { num_eval, .. })| weight * num_eval)
		.sum();

	let reduction_scalar = channel.sample();

	let extended_pushforward_eval_point =
		[pushforward_eval_point.clone(), batch_prefix.clone()].concat();

	let extended_log_sum_eval_point = [log_sum_eval_point.clone(), batch_prefix.clone()].concat();

	let batch_next_pow_2 = 1 << (batch_vars + pushforward_eval_point.len());
	let mut batch_pushforward: Vec<F> = push_forwards
		.iter()
		.flat_map(|push_forward| push_forward.iter_scalars())
		.collect();

	batch_pushforward.resize_with(batch_next_pow_2, || F::ZERO);

	let batch_pushforward = FieldBuffer::from_values(&batch_pushforward);

	let pf_eq_ind = eq_ind_partial_eval::<P>(&extended_pushforward_eval_point);
	let ls_eq_ind = eq_ind_partial_eval::<P>(&extended_log_sum_eval_point);
	let lin_comb = FieldBuffer::<P>::from_values(
		&zip(pf_eq_ind.iter_scalars(), ls_eq_ind.iter_scalars())
			.map(|(pf, ls)| pf + ls * reduction_scalar)
			.collect::<Vec<_>>(),
	);

	let reduction_prover = BivariateProductSumcheckProver::new(
		[batch_pushforward.clone(), lin_comb],
		pushforward_batch_eval + log_sum_batch_eval * reduction_scalar,
	)?;

	let output = prove_single(reduction_prover, channel)?;

	channel.send_many(&output.multilinear_evals);

	Ok(output)
}
