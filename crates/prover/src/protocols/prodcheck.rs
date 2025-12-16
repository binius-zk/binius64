// Copyright 2025 The Binius Developers

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, line::extrapolate_line_packed};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::prodcheck::MultilinearEvalClaim;

use crate::protocols::sumcheck::{
	Error as SumcheckError, ProveSingleOutput, bivariate_product_mle, prove_single_mlecheck,
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
}

pub fn prove<F, P, Challenger_>(
	k: usize,
	claim: MultilinearEvalClaim<F>,
	mut witness: FieldBuffer<P>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<MultilinearEvalClaim<F>, Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Challenger_: Challenger,
{
	assert_eq!(witness.log_len(), k + claim.point.len()); // precondition

	if k == 0 {
		return Ok(claim);
	}

	let mut split = witness
		.split_half_mut()
		.expect("witness.log_len() >= k; k > 0; => witness.log_len() > 0");
	let (half_0, half_1) = split.halves();

	let next_layer_evals = (half_0.as_ref(), half_1.as_ref())
		.into_par_iter()
		.map(|(v0, v1)| *v0 * *v1)
		.collect();
	let next_layer = FieldBuffer::new(k - 1 + claim.point.len(), next_layer_evals)
		.expect("half of witness length");

	let MultilinearEvalClaim { eval, point } = prove(k - 1, claim, next_layer, transcript)?;

	let prover = bivariate_product_mle::new([half_0, half_1], &point, eval)?;
	let ProveSingleOutput {
		multilinear_evals,
		challenges,
	} = prove_single_mlecheck(prover, transcript)?;

	let [eval_0, eval_1] = multilinear_evals
		.try_into()
		.expect("prover has two multilinears");

	transcript.message().write(&[eval_0, eval_1]);

	let r = transcript.sample();
	let next_eval = extrapolate_line_packed(eval_0, eval_0, r);

	let mut next_point = challenges;
	next_point.reverse();
	next_point.push(r);

	Ok(MultilinearEvalClaim {
		eval: next_eval,
		point: next_point,
	})
}
