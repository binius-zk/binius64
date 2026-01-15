use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, line::extrapolate_line_packed, multilinear::eq};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::prodcheck::MultilinearEvalClaim;
use itertools::Itertools;
use std::{array, iter::chain};

use crate::protocols::sumcheck::{
	Error as SumcheckError, batch::BatchSumcheckOutput,
	batch_quadratic::BatchQuadraticSumcheckProver,
};
use crate::protocols::{fracaddcheck::FracAddCheckProver, logup::prover::LogUp};
use crate::protocols::{
	logup::helper::{generate_index_fingerprints, generate_pushforward},
	sumcheck::batch::{batch_prove_and_write_evals, batch_prove_mle_and_write_evals},
};

/// Output of the pushforward sumcheck, grouped by lookup and table claims.
#[derive(Debug, Clone)]
pub struct PushforwardEvalClaims<F: Field> {
	pub challenges: Vec<F>,
	pub pushforward_evals: Vec<F>,
	pub table_evals: Vec<F>,
}

impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	/// Proves the outer instance, reducing lookup value claims to pushforward claims.
	pub fn prove_pushforward<
		Challenger_: Challenger,
		// N_MLES is the total number of MLEs involved, this is precisely N_LOOKUPS + N_TABLES.
		const N_MLES: usize,
	>(
		&self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<PushforwardEvalClaims<F>, SumcheckError> {
		// TODO: Remove implicit assumption of equal table size.
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
}

/// Constructs the sumcheck prover for the pushforward relation.
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
		// Enforce pushforward[i] * table[table_id[i]] at each lookup slot.
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
