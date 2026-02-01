// Copyright 2025-2026 The Binius Developers
use std::array;

use binius_field::{Field, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::FieldBuffer;
use binius_verifier::protocols::logup::LogUpLookupClaims;

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

impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	/// Runs the full LogUp proving flow and returns per-lookup claims.
	///
	/// The output is ordered to match the lookup ordering in the `LogUp` instance.
	///
	/// # Preconditions
	/// * `N_MLES == N_TABLES + N_LOOKUPS`.
	pub fn prove_lookup<const N_MLES: usize>(
		&self,
		channel: &mut impl IPProverChannel<F>,
	) -> Result<Vec<LogUpLookupClaims<F>>, Error> {
		assert!(N_MLES == 2 * N_TABLES);
		// Reduce lookup evaluations to pushforward/table evaluations.
		let pushforward_claims = self.prove_pushforward::<N_MLES>(channel)?;
		// Prove log-sum consistency for eq-kernel and pushforward trees.
		let (eq_claims, push_claims) = self.prove_log_sum(channel)?;

		assert_eq!(pushforward_claims.pushforward_evals.len(), N_TABLES);
		assert_eq!(eq_claims.len(), N_TABLES);
		assert_eq!(push_claims.len(), N_TABLES);

		let mut claims = Vec::with_capacity(N_TABLES);
		// Pair each lookup batch with its table id and fractional claims.
		for (lookup_idx, (eq_frac_claim, push_frac_claim)) in
			eq_claims.into_iter().zip(push_claims).enumerate()
		{
			let table_id = self.table_ids[lookup_idx];
			let table_eval = pushforward_claims.table_evals[table_id];
			claims.push(LogUpLookupClaims {
				table_id,
				pushforward_eval: pushforward_claims.pushforward_evals[lookup_idx],
				table_eval,
				eq_frac_claim,
				push_frac_claim,
			});
		}

		Ok(claims)
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
