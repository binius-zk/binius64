// Copyright 2026 The Binius Developers

//! The verifier for the M4 constraint reduction: the AND-check followed by the shift reduction.
//!
//! This mirrors the prover's composition on the same transcript:
//!
//! 1. The AND-check verifies `A & B == C` over all rows, yielding operand claims at a row point.
//! 2. That point's low coordinates are the constraint index `r_x`, its high coordinates `r_rho`.
//! 3. The shift reduction reduces the operand claims to one evaluation of the folded witness.
//! 4. The public-input consistency check ties in the shared constants.
//!
//! The output is that witness claim together with `r_rho`.
//! Binding the claim to the committed trace oracle is a later step, not done here.

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, Field};
use binius_ip::channel::IPVerifierChannel;
use binius_math::BinarySubspace;
use binius_utils::checked_arithmetics::checked_log_2;
use binius_verifier::{
	Error,
	config::B128,
	protocols::{
		bitand::AndCheckOutput,
		shift::{self, OperatorData, VerifyOutput},
	},
};

use crate::verify_bitand_reduction;

/// The verifier's output of the M4 constraint reduction.
pub struct ReductionVerifierOutput {
	/// The instance-fold challenge: the high coordinates of the AND-check row point.
	pub r_rho: Vec<B128>,
	/// The reduced claim about the instance-folded witness, from the shift reduction.
	pub shift: VerifyOutput<B128>,
}

/// Verifies the AND-check and shift reduction over the batch on one transcript.
///
/// # Arguments
///
/// - `cs`: the prepared single-instance constraint system shared by every instance.
/// - `log_instances`: base-2 logarithm of the instance count.
/// - `channel`: the verifier channel reading messages and redrawing Fiat-Shamir challenges.
///
/// # Errors
///
/// Returns an error if the AND-check, the shift reduction, or the public-input check fails.
pub fn verify_reduction<Channel>(
	cs: &ConstraintSystem,
	log_instances: usize,
	channel: &mut Channel,
) -> Result<ReductionVerifierOutput, Error>
where
	Channel: IPVerifierChannel<B128, Elem = B128>,
{
	// AND-check over all `K * n_and` rows.
	let log_n_and = checked_log_2(cs.and_constraints.len());
	let AndCheckOutput {
		a_eval,
		b_eval,
		c_eval,
		z_challenge,
		eval_point,
	} = verify_bitand_reduction(log_instances + log_n_and, channel)?;

	// The row point is `r_x || r_rho`: the constraint index low, the instance index high.
	let (r_x, r_rho) = eval_point.split_at(log_n_and);

	// Reduce the operand claims to one witness evaluation.
	// No MUL constraints here, so the intmul claim is a zero claim at an empty point.
	let bitand = OperatorData::new(r_x.to_vec(), [a_eval, b_eval, c_eval]);
	let intmul = OperatorData::new(Vec::new(), [B128::ZERO; 4]);
	let shift = shift::verify::<B128, _>(cs, &bitand, &intmul, channel)?;

	// Tie in the shared constants through the public-input consistency check.
	let domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS).isomorphic::<B128>();
	shift::check_eval::<B128, _>(
		cs,
		&cs.constants,
		&bitand,
		&intmul,
		&domain,
		z_challenge,
		&shift,
		channel,
	)?;

	Ok(ReductionVerifierOutput {
		r_rho: r_rho.to_vec(),
		shift,
	})
}
