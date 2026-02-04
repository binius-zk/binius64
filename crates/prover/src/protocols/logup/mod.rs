// Copyright 2025-2026 The Binius Developers

//! LogUp prover helpers and sub-protocols.
//!
//! `docs/logup.md` explains the single-lookup claim. For one lookup instance
//! with index map `I`, table `T`, and claim `e = I^*T(r)`, the prover:
//! 1. sets `Y = I_*eq_r`,
//! 2. proves `e = <T, Y>`,
//! 3. proves `Y` is the correct pushforward via the log-sum identity.
//!
//! This module implements the batched analogue of that flow.
//! Instead of proving each lookup instance independently, we fold many
//! instances into one claim per table:
//! - `batch_lookup_evals` samples a batching prefix and forms per-table random linear combinations
//!   of lookup evaluations;
//! - `concatenate_indices` rewrites many `I_{t,k}` maps (lookup `k` into table `t`) into one
//!   concatenated map `I_t`;
//! - `build_pushforwards_from_concat_indexes` constructs one pushforward per table, i.e. `Y_t =
//!   (I_t)_*eq_[batch_prefix || r]`.
//!
//! The protocol is therefore "single-lookup LogUp*" applied to each table's
//! concatenated instance, plus random linear batching over lookup slots. The
//! current implementation assumes each table receives the same number of
//! lookup instances, which keeps the batch dimensions aligned.
pub mod helper;
pub mod log_sum;
pub mod prover;
pub mod pushforward;
#[cfg(test)]
mod tests;

use std::{iter::zip, ops::Deref};

use binius_field::{Field, PackedField};
use binius_iop_prover::channel::IOPProverChannel;
use binius_math::{FieldBuffer, multilinear::eq};
use itertools::Itertools;

use crate::protocols::logup::{
	helper::{
		batch_lookup_evals, batch_pushforwards, concatenate_and_fingerprint_indexes,
		concatenate_indices,
	},
	prover::build_pushforwards_from_concat_indexes,
};
/// Prover state for the LogUp indexed lookup argument.
///
/// The instance aggregates multiple lookup batches that may target different
/// tables. Each lookup batch is represented by its index fingerprints,
/// pushforward, and claimed lookup evaluation.
pub struct LogUp<P: PackedField, Channel: IOPProverChannel<P>, const N_TABLES: usize> {
	/// Fingerprinted indices for each lookup batch.
	fingerprinted_indexes: [FieldBuffer<P>; N_TABLES],
	/// Pushforward tables built from the fingerprinted indices.
	push_forwards: [FieldBuffer<P>; N_TABLES],
	/// Lookup tables used by the batch.
	tables: [FieldBuffer<P>; N_TABLES],
	/// Verifier evaluation point for lookup MLEs.
	/// Equality-indicator expansion at `eval_point`.
	eq_kernel: FieldBuffer<P>,
	/// Claimed lookup values at `eval_point`.
	batched_evals: [P::Scalar; N_TABLES],

	batch_pushforward_oracle: Channel::Oracle,
	/// Fiat-Shamir scalar used for fingerprint hashing.
	pub fingerprint_scalar: P::Scalar,
	/// Fiat-Shamir shift applied in fingerprint hashing.
	pub shift_scalar: P::Scalar,
}

/// Builder for LogUp prover state.
///
/// We assume the bits for each index have been committed as separate MLEs.
impl<P: PackedField<Scalar = F>, Channel: IOPProverChannel<P>, F: Field, const N_TABLES: usize>
	LogUp<P, Channel, N_TABLES>
{
	/// Creates a LogUp instance for batched indexed lookup claims.
	///
	/// Relative to the single-claim picture in LogUp*, this constructor
	/// performs the batching rewrite:
	/// - sample lookup-slot batching randomness and fold many lookup values into
	///   `batched_evals[table_id]` via `batch_lookup_evals`;
	/// - extend the verifier point from `r` to `[batch_prefix || r]`, then build `eq_kernel =
	///   eq_[batch_prefix || r]`;
	/// - concatenate indices per table and build one pushforward per table, so each table now has a
	///   single `Y_t = (I_t)_*eq_[batch_prefix || r]`;
	/// - commit one packed oracle containing all pushforwards, and sample the fingerprint/shift
	///   scalars used by the log-sum checks.
	///
	/// The resulting state is exactly what the proving phases consume:
	/// batched `<pushforward, table>` reductions and batched log-sum reductions.
	pub fn new<Index: Deref<Target = [usize]>>(
		indexes: &[Index],
		table_ids: &[usize],
		eval_point: &[P::Scalar],
		lookup_evals: &[F],
		tables: [FieldBuffer<P>; N_TABLES],
		transcript: &mut Channel,
	) -> Self {
		assert!(indexes.len() == table_ids.len() && indexes.len() == indexes.len());

		let grouped_evals = zip(lookup_evals.iter().copied(), table_ids.iter().copied())
			.into_group_map_by(|&(_, id)| id);

		assert!(
			grouped_evals.values().map(|vals| vals.len()).all_equal(),
			"There must be an equal number of lookups into each table"
		);

		let (batched_evals, extended_eval_point) =
			batch_lookup_evals(lookup_evals, eval_point, table_ids, transcript);

		let eq_kernel = eq::eq_ind_partial_eval::<P>(&extended_eval_point);
		let concat_indices: [Vec<usize>; N_TABLES] = concatenate_indices(indexes, table_ids);

		let push_forwards =
			build_pushforwards_from_concat_indexes(&concat_indices, &tables, &eq_kernel);

		let batch_pushforward = batch_pushforwards(&push_forwards);

		let batch_pushforward_oracle = transcript.send_oracle(batch_pushforward.to_ref());

		let max_log_len = tables
			.iter()
			.map(|table| table.log_len())
			.max()
			.expect("There will be atleast 1 table");
		// Fiat-Shamir scalar used to hash index bits into field elements.
		let [fingerprint_scalar, shift_scalar] = transcript.sample_array();
		let fingerprinted_indexes = concatenate_and_fingerprint_indexes(
			indexes,
			table_ids,
			fingerprint_scalar,
			shift_scalar,
			max_log_len,
		);

		LogUp {
			fingerprinted_indexes,
			push_forwards,
			tables,
			eq_kernel,
			fingerprint_scalar,
			shift_scalar,
			batched_evals,
			batch_pushforward_oracle,
		}
	}
}
