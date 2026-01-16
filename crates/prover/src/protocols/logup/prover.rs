use binius_field::{Field, PackedField};
use binius_math::FieldBuffer;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_verifier::protocols::logup::LogUpLookupClaims;
use std::array;

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
	pub fn prove_lookup<Challenger_: Challenger, const N_MLES: usize>(
		&self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<Vec<LogUpLookupClaims<F>>, Error> {
		assert!(N_MLES == N_TABLES + N_LOOKUPS);

		// Reduce lookup evaluations to pushforward/table evaluations.
		let pushforward_claims = self.prove_pushforward::<Challenger_, N_MLES>(transcript)?;
		// Prove log-sum consistency for eq-kernel and pushforward trees.
		let (eq_claims, push_claims) = self.prove_log_sum(transcript)?;

		assert_eq!(pushforward_claims.pushforward_evals.len(), N_LOOKUPS);
		assert_eq!(eq_claims.len(), N_LOOKUPS);
		assert_eq!(push_claims.len(), N_LOOKUPS);

		let mut claims = Vec::with_capacity(N_LOOKUPS);
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
