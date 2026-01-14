use std::array;

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldSlice, multilinear::eq};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::logup::helper::{generate_index_fingerprints, generate_pushforward};

/// This struct enscapsulates logic required by the prover for the LogUp* indexed lookup arguement.
/// It operates in the batch mode by default. Supports N_LOOKUPS into N_TABLES.
pub struct LogUp<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize> {
	indexes: [FieldBuffer<P>; N_LOOKUPS],
	table_ids: [usize; N_LOOKUPS],
	lookup_values: [FieldBuffer<P>; N_LOOKUPS],
	push_forwards: [FieldBuffer<P>; N_LOOKUPS],
	tables: [FieldBuffer<P>; N_TABLES],
	eval_point: Vec<P::Scalar>,
	fingerprint_scalar: P::Scalar,
}

/// We assume the bits for each index has been committed as a separate MLE over the base field.
impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	pub fn new<Challenger_: Challenger>(
		indexes: [&[usize]; N_LOOKUPS],
		table_ids: [usize; N_LOOKUPS],
		lookup_values: [FieldBuffer<P>; N_LOOKUPS],
		eval_point: &[P::Scalar],
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
		let fingerprint_scalar = transcript.sample();
		let indexes = generate_index_fingerprints(indexes, fingerprint_scalar, max_log_len);

		LogUp {
			indexes,
			table_ids,
			lookup_values,
			push_forwards,
			tables,
			eval_point: eval_point.to_vec(),
			fingerprint_scalar,
		}
	}

	/// Proves the outer instance, which reduces the evaluation claim on the lookup values, to that on the pushforward.
	pub fn prove_pushforward() {}

	/// Proves the inner instance which is reminiscient of logup gkr, using a binary tree of fractional additions.
	pub fn prove_log_sum() {}
}

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
