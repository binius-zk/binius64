//! LogUp prover helpers and sub-protocols.
//!
//! This module groups the logic for constructing pushforwards, running the
//! log-sum reduction, and batching fractional-addition checks.
pub mod helper;
pub mod log_sum;
pub mod prover;
pub mod pushforward;

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, line::extrapolate_line_packed, multilinear::eq};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::prodcheck::MultilinearEvalClaim;
use itertools::Itertools;
use std::{array, iter::chain};

use crate::protocols::{
	fracaddcheck::FracAddCheckProver, logup::helper::generate_index_fingerprints,
};
use crate::protocols::{
	logup::prover::build_pushforwards,
	sumcheck::{
		Error as SumcheckError, batch::BatchSumcheckOutput,
		batch_quadratic::BatchQuadraticSumcheckProver,
	},
};
/// Prover state for the LogUp indexed lookup argument.
///
/// The instance aggregates multiple lookup batches that may target different
/// tables. Each lookup batch is represented by its index fingerprints,
/// pushforward, and claimed lookup evaluation.
pub struct LogUp<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize> {
	fingerprinted_indexes: [FieldBuffer<P>; N_LOOKUPS],
	table_ids: [usize; N_LOOKUPS],
	push_forwards: [FieldBuffer<P>; N_LOOKUPS],
	tables: [FieldBuffer<P>; N_TABLES],
	eval_point: Vec<P::Scalar>,
	eq_kernel: FieldBuffer<P>,
	lookup_evals: [P::Scalar; N_LOOKUPS],
	fingerprint_scalar: P::Scalar,
	shift_scalar: P::Scalar,
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
