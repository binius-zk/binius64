// Copyright 2025-2026 The Binius Developers

//! LogUp prover helpers and sub-protocols.
//!
//! This module groups the logic for:
//! - building pushforward tables from indexed lookups,
//! - running the log-sum reduction via fractional-addition trees,
//! - and emitting the evaluation claims consumed by the verifier.
pub mod helper;
pub mod log_sum;
pub mod prover;
pub mod pushforward;
#[cfg(test)]
mod tests;

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, multilinear::eq};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::logup::{
	helper::{
		batch_lookup_evals, concatenate_indices, generate_index_fingerprints,
		generate_index_fingerprints_new,
	},
	prover::{build_pushforwards, build_pushforwards_from_concat_indexes},
};
/// Prover state for the LogUp indexed lookup argument.
///
/// The instance aggregates multiple lookup batches that may target different
/// tables. Each lookup batch is represented by its index fingerprints,
/// pushforward, and claimed lookup evaluation.
pub struct LogUp<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize> {
	/// Fingerprinted indices for each lookup batch.
	fingerprinted_indexes: [FieldBuffer<P>; N_TABLES],
	/// Table selector for each lookup batch.
	table_ids: [usize; N_LOOKUPS],
	/// Pushforward tables built from the fingerprinted indices.
	push_forwards: [FieldBuffer<P>; N_TABLES],
	/// Lookup tables used by the batch.
	tables: [FieldBuffer<P>; N_TABLES],
	/// Verifier evaluation point for lookup MLEs.
	/// Equality-indicator expansion at `eval_point`.
	eq_kernel: FieldBuffer<P>,
	/// Claimed lookup values at `eval_point`.
	batched_evals: [P::Scalar; N_TABLES],

	extended_eval_point: Vec<P::Scalar>,
	/// Fiat-Shamir scalar used for fingerprint hashing.
	pub fingerprint_scalar: P::Scalar,
	/// Fiat-Shamir shift applied in fingerprint hashing.
	pub shift_scalar: P::Scalar,
}

/// Builder for LogUp prover state.
///
/// We assume the bits for each index have been committed as separate MLEs.
impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	/// Creates a LogUp instance for batched indexed lookup claims.
	///
	/// This samples the fingerprint/shift scalars, constructs the eq-kernel at the
	/// verifier point, and builds both pushforwards and fingerprinted indices so
	/// that the proving sub-protocols can run without re-deriving shared state.
	pub fn new<Challenger_: Challenger>(
		indexes: [&[usize]; N_LOOKUPS],
		table_ids: [usize; N_LOOKUPS],
		eval_point: &[P::Scalar],
		lookup_evals: [F; N_LOOKUPS],
		tables: [FieldBuffer<P>; N_TABLES],
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Self {
		assert!(N_TABLES > 0 && N_LOOKUPS > 0);

		let (batched_evals, extended_eval_point) =
			batch_lookup_evals(&lookup_evals, eval_point, &table_ids, transcript);

		let eq_kernel = eq::eq_ind_partial_eval::<P>(&extended_eval_point);
		let concat_indices: [Vec<usize>; N_TABLES] = concatenate_indices(indexes, &table_ids);

		let push_forwards =
			build_pushforwards_from_concat_indexes(&concat_indices, &tables, &eq_kernel);
		let max_log_len = tables
			.iter()
			.map(|table| table.log_len())
			.max()
			.expect("There will be atleast 1 table");
		// Fiat-Shamir scalar used to hash index bits into field elements.
		let [fingerprint_scalar, shift_scalar] = transcript.sample_array();
		let fingerprinted_indexes = generate_index_fingerprints_new(
			indexes,
			&table_ids,
			fingerprint_scalar,
			shift_scalar,
			max_log_len,
		);

		LogUp {
			fingerprinted_indexes,
			table_ids,
			push_forwards,
			tables,
			eq_kernel,
			fingerprint_scalar,
			shift_scalar,
			batched_evals,
			extended_eval_point,
		}
	}
}
