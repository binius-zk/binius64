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

/// Builds pushforward tables for each lookup batch.
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
