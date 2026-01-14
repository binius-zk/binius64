use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, inner_product::inner_product_buffers, multilinear::eq};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use itertools::Itertools;
use std::{array, iter::chain};

use crate::protocols::sumcheck::{
	Error as SumcheckError, ProveSingleOutput, batch::BatchSumcheckOutput,
	batch_quadratic::BatchQuadraticSumcheckProver, prove_single,
};
use crate::protocols::{
	logup::helper::{generate_index_fingerprints, generate_pushforward},
	sumcheck::batch::batch_prove_and_write_evals,
};

/// This struct enscapsulates logic required by the prover for the LogUp* indexed lookup arguement.
/// It operates in the batch mode by default. Supports N_LOOKUPS into N_TABLES.
pub struct LogUp<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize> {
	indexes: [FieldBuffer<P>; N_LOOKUPS],
	table_ids: [usize; N_LOOKUPS],
	push_forwards: [FieldBuffer<P>; N_LOOKUPS],
	tables: [FieldBuffer<P>; N_TABLES],
	eval_point: Vec<P::Scalar>,
	lookup_evals: [P::Scalar; N_LOOKUPS],
	fingerprint_scalar: P::Scalar,
}

#[derive(Debug, Clone)]
pub struct PushforwardEvalClaims<F: Field> {
	pub challenges: Vec<F>,
	pub pushforward_evals: Vec<F>,
	pub table_evals: Vec<F>,
}

/// We assume the bits for each index has been committed as a separate MLE over the base field.
impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
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
		let fingerprint_scalar = transcript.sample();
		let indexes = generate_index_fingerprints(indexes, fingerprint_scalar, max_log_len);

		LogUp {
			indexes,
			table_ids,
			push_forwards,
			tables,
			eval_point: eval_point.to_vec(),
			fingerprint_scalar,
			lookup_evals,
		}
	}

	/// Proves the outer instance, which reduces the evaluation claim on the lookup values, to that on the pushforward.
	pub fn prove_pushforward<Challenger_: Challenger, const N_MLES: usize>(
		&self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<PushforwardEvalClaims<F>, SumcheckError> {
		assert_eq!(N_TABLES + N_LOOKUPS, N_MLES);
		let prover = make_pushforward_sumcheck_prover::<P, F, N_TABLES, N_LOOKUPS, N_MLES>(
			&self.table_ids,
			&self.tables,
			&self.push_forwards,
			self.lookup_evals,
		)?;

		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = batch_prove_and_write_evals(vec![prover], transcript)?;

		let (pushforward_evals, table_evals) = multilinear_evals[0].split_at(N_LOOKUPS);

		Ok(PushforwardEvalClaims {
			challenges,
			pushforward_evals: pushforward_evals.to_vec(),
			table_evals: table_evals.to_vec(),
		})
	}

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

fn make_pushforward_sumcheck_prover<
	P: PackedField<Scalar = F>,
	F: Field,
	const N_TABLES: usize,
	const N_LOOKUPS: usize,
	const N_MLES: usize,
>(
	table_ids: &[usize; N_LOOKUPS],
	tables: &[FieldBuffer<P>; N_TABLES],
	push_forwards: &[FieldBuffer<P>; N_LOOKUPS],
	lookup_evals: [F; N_LOOKUPS],
) -> Result<
	BatchQuadraticSumcheckProver<
		P,
		impl Fn([P; N_MLES], &mut [P; N_LOOKUPS]),
		impl Fn([P; N_MLES], &mut [P; N_LOOKUPS]),
		N_MLES,
		N_LOOKUPS,
	>,
	SumcheckError,
> {
	assert!(N_TABLES + N_LOOKUPS == N_MLES);
	let mles: [FieldBuffer<P>; N_MLES] =
		chain(push_forwards.iter().cloned(), tables.iter().cloned())
			.collect_array()
			.expect("N_TABLES + N_LOOKUPS == N_MLES");

	let pushforward_composition = |mle_evals: [P; N_MLES], comp_evals: &mut [P; N_LOOKUPS]| {
		let (pushforwards, tables) = mle_evals.split_at(N_LOOKUPS);
		for i in 0..N_LOOKUPS {
			comp_evals[i] = pushforwards[i] * tables[table_ids[i]]
		}
	};
	BatchQuadraticSumcheckProver::new(
		mles,
		pushforward_composition,
		pushforward_composition,
		lookup_evals,
	)
}
