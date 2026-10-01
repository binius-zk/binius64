// Copyright 2026 The Binius Developers

//! The shift reduction's operand evaluation claims, all at one constraint point.
//!
//! The reduction closes four constraint families in one proof: ZERO, AND, IMUL and BMUL.
//!
//! Their operands form one flat run of columns, in [`OPERATION_ARITIES`] order.
//! Each column arrives with its evaluation, claimed at the one constraint point `r_x` with its
//! matrix padded with empty rows. Those claims then travel together through both phases of the
//! reduction, batched on one operand axis.

use std::iter;

use binius_core::constraint_system::ConstraintSystem;
use binius_field::Field;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	inner_product::inner_product,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{
	protocols::{rerand::RerandOutput, zero},
	reduction::{BINMUL_ARITY, INTMUL_ARITY, OPERATION_ARITIES, ZERO_ARITY, padding_scales},
};

/// The operand evaluation claims of every operation, as the shift reduction receives them.
#[derive(Debug, Clone)]
pub struct OperandClaims<F: Field> {
	/// The constraint point, as long as the widest operation's constraint count.
	///
	/// Every operation is claimed at the whole point, its matrix padded with empty rows; see
	/// [`padding_scales`].
	pub r_x: Vec<F>,
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// One evaluation per operand column, the four operations' runs in [`OPERATION_ARITIES`]
	/// order.
	pub evals: Vec<F>,
}

impl<F: Field> OperandClaims<F> {
	/// Assembles the claims from the output of the BitAnd sumcheck.
	///
	/// The sumcheck's point is `r_rho || r_x_star`, instance index low, constraint index high.
	/// `r_x_star` spans the widest AND, IMUL and BMUL set. The constraint point `r_x` extends it
	/// through [`zero::reduction_point`] when the ZERO set is wider still, drawing the extra
	/// challenges from `sample`.
	///
	/// The sumcheck claims each operation at its own prefix of `r_x`. Its padding factor lifts the
	/// claim to its padded matrix at the whole point.
	///
	/// An operation the constraint system does not use gets zero evaluations.
	///
	/// # Arguments
	///
	/// - `cs`: the constraint system, whose row counts pick each operation's padding factor.
	/// - `log_instances`: the instance variables at the low end of the point.
	/// - `z_challenge`: the univariate challenge, shared by every operation.
	/// - `rerand`: the sumcheck's output, with the IntMul operand evaluations before the BinMul
	///   ones.
	/// - `sample`: draws the constraint point's extra challenges from the transcript.
	pub fn from_rerand(
		cs: &ConstraintSystem,
		log_instances: usize,
		z_challenge: F,
		rerand: &RerandOutput<F>,
		sample: impl FnMut() -> F,
	) -> Self {
		let r_x_star = &rerand.eval_point[log_instances..];
		let log_n_zero = cs.log_zero_constraints().unwrap_or(0);
		let r_x = zero::reduction_point(r_x_star, r_x_star.len().max(log_n_zero), sample);

		let [_, bitand_scale, intmul_scale, binmul_scale] = padding_scales(cs, &r_x);
		let mut operand_evals = rerand.operand_evals.iter().copied();
		// An absent operation's run is zeros; a present one takes its evaluations off the front.
		let mut run = |present: bool, arity: usize, scale: F| {
			if present {
				operand_evals
					.by_ref()
					.take(arity)
					.map(|eval| eval * scale)
					.collect()
			} else {
				vec![F::ZERO; arity]
			}
		};
		let intmul = run(cs.n_imul_constraints() > 0, INTMUL_ARITY, intmul_scale);
		let binmul = run(cs.n_bmul_constraints() > 0, BINMUL_ARITY, binmul_scale);

		// The BitAnd check has no skip branch: an empty AND set reduces over one zero row.
		let bitand = rerand.bitand_evals.map(|eval| eval * bitand_scale);
		let evals = iter::repeat_n(F::ZERO, ZERO_ARITY)
			.chain(bitand)
			.chain(intmul)
			.chain(binmul)
			.collect::<Vec<_>>();
		debug_assert_eq!(evals.len(), OPERATION_ARITIES.iter().sum::<usize>());

		Self {
			r_x,
			r_zhat_prime: z_challenge,
			evals,
		}
	}

	/// Draws the batching challenges and folds their weights into the claims.
	///
	/// The claim of column `m` is weighted by the equality indicator of the operand axis at `m`.
	/// The axis is padded to a cube of `log2_ceil(evals.len())` challenges; the columns past the
	/// last claim name nothing. The verifier draws the same challenges at the same place, so the
	/// count is protocol, not detail.
	///
	/// # Arguments
	///
	/// - `channel`: the transcript the batching challenges are drawn from.
	pub fn prepare(self, channel: &mut impl IPProverChannel<F>) -> PreparedOperandClaims<F> {
		let operand_batch_challenges = channel.sample_many(log2_ceil_usize(self.evals.len()));
		let operand_weights = eq_ind_partial_eval_scalars(&operand_batch_challenges);
		let batched_eval = inner_product(
			self.evals.iter().copied(),
			operand_weights[..self.evals.len()].iter().copied(),
		);

		PreparedOperandClaims {
			batched_eval,
			r_zhat_prime: self.r_zhat_prime,
			r_x_tensor: eq_ind_partial_eval::<F>(&self.r_x).into_inner(),
			operand_weights,
		}
	}
}

/// The claims with their batching weights folded in, as both proving phases read them.
///
/// The constraint table is shared by every key, so it is built once here.
#[derive(Debug, Clone)]
pub struct PreparedOperandClaims<F: Field> {
	/// The operand claims collapsed into the single value the reduction proves:
	///
	/// ```text
	/// batched_eval = sum_m evals[m] * operand_weights[m]
	/// ```
	///
	/// This is the claim phase 1 hands its sumcheck, and it is the same value the verifier
	/// computes from the operand evaluation claims before running its own.
	pub batched_eval: F,
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// The constraint table: the equality indicator of `r_x`, one weight per row of the padded
	/// operation matrices, shared by every operation.
	pub r_x_tensor: Vec<F>,
	/// The weight of each operand column, padded to a power of two.
	///
	/// A key reads it at the column its constraint index names.
	pub operand_weights: Vec<F>,
}

#[cfg(test)]
mod tests {
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::{B128, StdChallenger};

	use super::*;

	#[test]
	fn prepare_draws_one_challenge_per_operand_axis_variable() {
		// Invariant: the operand axis takes `log2_ceil(n)` challenges, drawn first, and its
		// weights batch the claims.
		//
		// The verifier draws in that order; any other weights the claims by a different tensor.
		//
		// Two transcripts from the same seed hand out the same sequence, so drawing the axis by
		// hand from one pins what `prepare` must have drawn from the other.
		let n = OPERATION_ARITIES.iter().sum::<usize>();
		let evals = (1..=n as u128).map(B128::new).collect::<Vec<_>>();

		let mut expected = ProverTranscript::<StdChallenger>::default();
		let operand_weights =
			eq_ind_partial_eval_scalars::<B128>(&expected.sample_many(log2_ceil_usize(n)));

		let mut channel = ProverTranscript::<StdChallenger>::default();
		let prepared = OperandClaims {
			r_x: Vec::new(),
			r_zhat_prime: B128::ZERO,
			evals: evals.clone(),
		}
		.prepare(&mut channel);

		// The two stay in lockstep only if `prepare` drew exactly those and no more.
		assert_eq!(
			IPProverChannel::<B128>::sample(&mut channel),
			IPProverChannel::<B128>::sample(&mut expected)
		);

		assert_eq!(prepared.operand_weights, operand_weights);
		assert_eq!(prepared.operand_weights.len(), 16);
		assert_eq!(
			prepared.batched_eval,
			inner_product(evals, operand_weights[..n].iter().copied())
		);
		// The constraint point is empty, so the constraint table is the single weight one.
		assert_eq!(prepared.r_x_tensor, [B128::ONE]);
	}
}
