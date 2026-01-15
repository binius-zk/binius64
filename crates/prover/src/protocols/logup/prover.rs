use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, line::extrapolate_line_packed, multilinear::eq};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::prodcheck::MultilinearEvalClaim;
use itertools::Itertools;
use std::{array, iter::chain};

use crate::protocols::fracaddcheck::FracAddCheckProver;
use crate::protocols::sumcheck::{
	Error as SumcheckError, batch::BatchSumcheckOutput,
	batch_quadratic::BatchQuadraticSumcheckProver,
};
use crate::protocols::{
	logup::helper::{generate_index_fingerprints, generate_pushforward},
	sumcheck::batch::{batch_prove_and_write_evals, batch_prove_mle_and_write_evals},
};

/// Prover state for the LogUp indexed lookup argument.
///
/// The instance aggregates multiple lookup batches that may target different
/// tables. Each lookup batch is represented by its index fingerprints,
/// pushforward, and claimed lookup evaluation.
pub struct LogUp<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize> {
	pub(super) fingerprinted_indexes: [FieldBuffer<P>; N_LOOKUPS],
	pub(super) table_ids: [usize; N_LOOKUPS],
	pub(super) push_forwards: [FieldBuffer<P>; N_LOOKUPS],
	pub(super) tables: [FieldBuffer<P>; N_TABLES],
	pub(super) eval_point: Vec<P::Scalar>,
	pub(super) eq_kernel: FieldBuffer<P>,
	pub(super) lookup_evals: [P::Scalar; N_LOOKUPS],
	pub(super) fingerprint_scalar: P::Scalar,
	pub(super) shift_scalar: P::Scalar,
}

/// Builder for LogUp prover state.
///
/// We assume the bits for each index have been committed as separate MLEs.
impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	/// Creates a LogUp instance for batched indexed lookup claims.
	pub fn new<Challenger_: Challenger>(
		indexes: [&[usize]; N_LOOKUPS],
		table_ids: [usize; N_LOOKUPS],
		eval_point: &[P::Scalar],
		lookup_evals: [F; N_LOOKUPS],
		tables: [FieldBuffer<P>; N_TABLES],
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Self {
		assert!(N_TABLES > 0 && N_LOOKUPS > 0);
		let eq_kernel = eq::eq_ind_partial_eval::<P>(eval_point);
		let push_forwards = build_pushforwards(&indexes, &table_ids, &eq_kernel, &tables);
		let max_log_len = tables
			.iter()
			.map(|table| table.log_len())
			.max()
			.expect("There will be atleast 1 table");
		// Fiat-Shamir scalar used to hash index bits into field elements.
		let [fingerprint_scalar, shift_scalar] = transcript.sample_array();
		let indexes =
			generate_index_fingerprints(indexes, fingerprint_scalar, shift_scalar, max_log_len);

		LogUp {
			fingerprinted_indexes: indexes,
			table_ids,
			push_forwards,
			tables,
			eval_point: eval_point.to_vec(),
			eq_kernel,
			fingerprint_scalar,
			shift_scalar,
			lookup_evals,
		}
	}
}

/// Builds pushforward tables for each lookup batch.
fn build_pushforwards<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize>(
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
