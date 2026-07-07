// Copyright 2025 Irreducible Inc.

//! Batched BitAnd shift reduction verifier for the M4 data-parallel proof system.
//!
//! It reduces the batched BitAnd operand claim to one witness evaluation.
//! It then checks that evaluation against the monster multilinear.
//!
//! The instance challenge `r_kappa` does not appear here.
//! The prover folded it into the witness, so the verifier only sees the claim at `(r_j, r_y)`.
//! The caller appends `r_kappa` when it opens the batch commitment.
//!
//! This handles BitAnd only; IntMul is out of scope for the initial M4 batch.

use binius_core::constraint_system::{AndConstraint, ConstraintSystem};
use binius_field::{BinaryField, field::FieldOps};
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{SumcheckOutput, verify as verify_sumcheck},
};
use binius_math::{
	BinarySubspace,
	multilinear::eq::eq_ind_partial_eval_scalars,
	univariate::{evaluate_univariate, lagrange_evals_scalars},
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::Itertools;

use super::{
	BITAND_ARITY, OperatorData, error::Error, evaluate_h_op,
	evaluate_monster_multilinear_for_operation,
};
use crate::config::{LOG_WORD_SIZE_BITS, LOG_WORDS_PER_ELEM};

/// Output of the batched shift reduction verification.
///
/// The evaluation point of the committed batch witness is `(r_j, r_y, r_kappa)`.
/// This output carries `(r_j, r_y)`; the caller supplies `r_kappa` from the batched reduction.
#[derive(Debug)]
pub struct BatchVerifyOutput<F> {
	/// Challenge point for the bit index within a word, of length `LOG_WORD_SIZE_BITS`.
	pub r_j: Vec<F>,
	/// Challenge point for the word index within one instance.
	pub r_y: Vec<F>,
	/// The claimed evaluation of the committed batch witness at `(r_j, r_y, r_kappa)`.
	pub witness_eval: F,
}

/// Verifies the batched BitAnd shift reduction and checks the reduced claim.
///
/// # Protocol
///
/// - Sample the operand-batching coefficient for `(a, b, c)`.
/// - Verify the phase-1 sumcheck over the bit index and shift amount, then split into `r_j`, `r_s`.
/// - Verify the phase-2 sumcheck over the per-instance word index, giving `r_y`.
/// - Read the witness evaluation and check `eval == witness_eval * monster_eval`.
///
/// # Arguments
///
/// - `constraint_system`: the per-instance constraint system, shared by every instance.
/// - `bitand_data`: the BitAnd operand claim, with the local constraint challenge `r_x` and the
///   operand evaluations.
/// - `subspace`: the Lagrange basis subspace over the word bits.
/// - `r_zhat_prime`: the univariate bit challenge from the BitAnd reduction.
/// - `channel`: the verifier channel.
///
/// # Errors
///
/// Returns an error if either sumcheck fails or the final monster identity does not hold.
///
/// # Panics
///
/// Assumes a non-zero-knowledge channel.
/// The monster evaluation is computed directly over the channel element type.
/// M4 commits without zero-knowledge, so this holds.
pub fn verify_batch<F, C>(
	constraint_system: &ConstraintSystem,
	bitand_data: &OperatorData<C::Elem, BITAND_ARITY>,
	subspace: &BinarySubspace<F>,
	r_zhat_prime: C::Elem,
	channel: &mut C,
) -> Result<BatchVerifyOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	// Operand-batching coefficient for (a, b, c). Only BitAnd is reduced, so one lambda suffices.
	let bitand_lambda = channel.sample();

	// The batched operand claim is lambda * (a + b*lambda + c*lambda^2).
	// The extra lambda scaling lets it compose additively with other batched claims.
	let eval =
		bitand_lambda.clone() * evaluate_univariate(&bitand_data.evals, bitand_lambda.clone());

	// Phase 1: sumcheck over (j, s), degree 2.
	let SumcheckOutput {
		eval: gamma,
		challenges: mut r_jr_s,
	} = verify_sumcheck(LOG_WORD_SIZE_BITS * 2, 2, eval, channel)?;

	r_jr_s.reverse();
	// r_j is the low bit-index half; r_s is the high shift-amount half.
	let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
	let r_j = r_jr_s;

	// The witness folds as a power-of-two multilinear.
	// So the word index rounds the committed word count up, floored at one field element.
	// The committed count is the flat public-plus-hidden length, matching the prover's key count.
	let log_word_count =
		log2_ceil_usize(constraint_system.value_vec_layout.combined_len()).max(LOG_WORDS_PER_ELEM);

	// Phase 2: sumcheck over the per-instance word index, degree 2.
	let SumcheckOutput {
		eval,
		challenges: mut r_y,
	} = verify_sumcheck(log_word_count, 2, gamma, channel)?;

	r_y.reverse();

	// The prover sends the witness evaluation; the verifier checks it against the monster below.
	let witness_eval = channel.recv_one()?;

	// The monster evaluation is a public function of the constraint system and the challenges.
	// It carries no batch structure: the instance dimension already folded into `witness_eval`.
	let monster_eval = {
		let r_y_tensor = eq_ind_partial_eval_scalars(&r_y);
		let l_tilde = lagrange_evals_scalars(subspace, r_zhat_prime);
		let h_op_evals = evaluate_h_op(&l_tilde, &r_j, &r_s);

		let (a, b, c) = constraint_system
			.and_constraints
			.iter()
			.map(|AndConstraint { a, b, c }| (a, b, c))
			.multiunzip();
		evaluate_monster_multilinear_for_operation::<F, C::Elem>(
			&[a, b, c],
			&bitand_data.r_x_prime,
			bitand_lambda,
			&r_s,
			&r_y_tensor,
			&h_op_evals,
		)
		.expect("evaluate_monster_multilinear_for_operation has no fallible path")
	};

	// Closing check: the reduced claim equals the witness evaluation times the monster evaluation.
	channel.assert_zero(witness_eval.clone() * monster_eval - eval)?;

	Ok(BatchVerifyOutput {
		r_j,
		r_y,
		witness_eval,
	})
}
