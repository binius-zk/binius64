// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, Field};
use binius_math::{
	multilinear::eq::{eq_ind, eq_ind_partial_eval, eq_one_var},
	univariate::evaluate_univariate,
};
use binius_spartan_frontend::constraint_system::{
	ConstraintSystemPadded, MulConstraint, WitnessIndex,
};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::DeserializeBytes;
use binius_verifier::{
	fri::FRIParams,
	merkle_tree::MerkleTreeScheme,
	protocols::{basefold, sumcheck},
};

#[derive(Debug)]
pub struct Output<F> {
	pub lambda: F,
	pub batch_coeff: F,
	pub r_y: Vec<F>,
	pub eval: F,
	pub witness_eval: F,
}

pub fn verify<F, MTScheme, Challenger_>(
	fri_params: &FRIParams<F>,
	merkle_scheme: &MTScheme,
	codeword_commitment: MTScheme::Digest,
	eval_claims: &[F],
	public_eval: F,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<Output<F>, Error>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	// \lambda is the batching challenge for the constraint operands
	let lambda = transcript.sample();

	// Coefficient for batching the public input check with the wiring check.
	let batch_coeff = transcript.sample();

	// Batch together the witness public input consistency claim (`public_eval`) with the
	// constraint operand evaluation claims (`eval_claims`).
	let batched_claim = evaluate_univariate(eval_claims, lambda) + batch_coeff * public_eval;

	let basefold::ReducedOutput {
		final_fri_value: witness_eval,
		final_sumcheck_value: eval,
		challenges: mut r_y,
	} = basefold::verify_zk(
		fri_params,
		merkle_scheme,
		codeword_commitment,
		batched_claim,
		transcript,
	)?;

	r_y.reverse();

	// The challenges include the batch_challenge as the last element after reversal.
	// For the wiring check, we only need the sumcheck challenges (first n elements).
	r_y.pop();

	Ok(Output {
		lambda,
		batch_coeff,
		r_y,
		eval,
		witness_eval,
	})
}

pub fn check_eval<F: Field>(
	constraint_system: &ConstraintSystemPadded,
	r_public: &[F],
	r_x: &[F],
	output: &Output<F>,
) -> Result<(), Error> {
	let Output {
		lambda,
		batch_coeff,
		r_y,
		eval,
		witness_eval,
	} = output;

	assert!(r_public.len() <= r_y.len());

	let wiring_eval = evaluate_wiring_mle(constraint_system.mul_constraints(), *lambda, r_x, r_y);

	// Evaluate eq(r_public || ZERO, r_y)
	let (r_y_head, r_y_tail) = r_y.split_at(r_public.len());
	let eq_head = eq_ind(r_public, r_y_head);
	let eq_public = r_y_tail
		.iter()
		.fold(eq_head, |eval, &r_x_i| eval * eq_one_var(r_x_i, F::ZERO));

	if *eval != (wiring_eval + *batch_coeff * eq_public) * *witness_eval {
		return Err(Error::SumcheckComposition);
	}

	Ok(())
}

pub fn evaluate_wiring_mle<F: Field>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	lambda: F,
	r_x: &[F],
	r_y: &[F],
) -> F {
	let mut acc = [F::ZERO; 3];

	let r_x_tensor = eq_ind_partial_eval::<F>(r_x);
	let r_y_tensor = eq_ind_partial_eval::<F>(r_y);
	for (&r_x_tensor_i, MulConstraint { a, b, c }) in
		iter::zip(r_x_tensor.as_ref(), mul_constraints)
	{
		for (dst, operand) in iter::zip(&mut acc, [a, b, c]) {
			let r_y_tensor_sum = operand
				.wires()
				.iter()
				.map(|j| r_y_tensor[j.0 as usize])
				.sum::<F>();
			*dst += r_x_tensor_i * r_y_tensor_sum;
		}
	}

	evaluate_univariate(&acc, lambda)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("BaseFold error: {0}")]
	BaseFold(#[from] basefold::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("sumcheck composition check failed")]
	SumcheckComposition,
}
