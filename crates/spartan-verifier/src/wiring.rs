// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::Field;
use binius_ip::{MultilinearRationalEvalClaim, channel::IPVerifierChannel};
use binius_math::{
	multilinear::eq::{eq_ind, eq_ind_partial_eval, eq_one_var},
	univariate::evaluate_univariate,
};
use binius_spartan_frontend::constraint_system::{
	ConstraintSystemPadded, MulConstraint, WitnessIndex,
};
use binius_verifier::protocols::{basefold, sumcheck};

/// Claim components from the wiring check computation via IOP channel.
#[derive(Debug, Clone)]
pub struct WiringClaim<F: Field> {
	/// Batching challenge for constraint operands.
	pub lambda: F,
	/// Coefficient for batching public input check with wiring check.
	pub batch_coeff: F,
	/// The batched sum of all claims.
	pub batched_sum: F,
}

/// Computes the wiring claim using an IOP channel interface.
///
/// Samples the batching challenges and computes the batched claim from the
/// evaluation claims and public input evaluation.
pub fn compute_claim<F: Field>(
	_constraint_system: &ConstraintSystemPadded,
	_r_public: &[F],
	eval_claims: &[F],
	public_eval: F,
	channel: &mut impl IPVerifierChannel<F>,
) -> WiringClaim<F> {
	// \lambda is the batching challenge for the constraint operands
	let lambda = channel.sample();

	// Coefficient for batching the public input check with the wiring check.
	let batch_coeff = channel.sample();

	// Batch together the witness public input consistency claim with the
	// constraint operand evaluation claims.
	let batched_sum = evaluate_univariate(eval_claims, lambda) + batch_coeff * public_eval;

	WiringClaim {
		lambda,
		batch_coeff,
		batched_sum,
	}
}

/// Checks the wiring evaluation claim from a multilinear rational evaluation claim.
///
/// This works with the `MultilinearRationalEvalClaim` type returned by the IOP channel's
/// `finish` method.
pub fn check_eval_claim<F: Field>(
	constraint_system: &ConstraintSystemPadded,
	r_public: &[F],
	r_x: &[F],
	claim: &MultilinearRationalEvalClaim<F>,
	lambda: F,
	batch_coeff: F,
) -> Result<(), Error> {
	// claim.point is in low-to-high order with batch_challenge at the end.
	// Remove the batch_challenge to get r_y.
	let r_y: Vec<F> = claim.point[..claim.point.len() - 1].to_vec();

	assert!(r_public.len() <= r_y.len());

	let wiring_eval = evaluate_wiring_mle(constraint_system.mul_constraints(), lambda, r_x, &r_y);

	// Evaluate eq(r_public || ZERO, r_y)
	let (r_y_head, r_y_tail) = r_y.split_at(r_public.len());
	let eq_head = eq_ind(r_public, r_y_head);
	let eq_public = r_y_tail
		.iter()
		.fold(eq_head, |eval, &r_y_i| eval * eq_one_var(r_y_i, F::ZERO));

	let expected = wiring_eval + batch_coeff * eq_public;
	if claim.eval_numerator != expected * claim.eval_denominator {
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
