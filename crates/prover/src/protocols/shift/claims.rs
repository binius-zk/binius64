// Copyright 2026 The Binius Developers

//! The shift reduction's operand evaluation claims, all at prefixes of one constraint point.
//!
//! The reduction closes four constraint families in one proof: ZERO, AND, IMUL and BMUL.
//!
//! Each family arrives with its own operand evaluations.
//! Each is claimed at the prefix of the one constraint point `r_x` its rows span.
//! Those claims then travel together through both phases of the reduction.
//!
//! Each family's evaluations carry its operation's arity in their type, so no two share one.
//! Putting a BMUL claim in the IMUL field is a type error, not a wrong proof.
//!
//! The batched form erases that arity, because a shift key picks its operation at run time.
//! There the operation is named by indexing instead: `prepared[key.operation]`.

use std::{array, iter, ops::Index};

use binius_core::constraint_system::ConstraintSystem;
use binius_field::Field;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	inner_product::inner_product,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};
use binius_verifier::protocols::{
	rerand::RerandOutput,
	shift::{
		BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, LOG_MAX_ARITY, LOG_OPERATION_COUNT,
		OPERATION_COUNT, ZERO_ARITY, constraint_tables, log_constraints,
	},
	zero,
};

use super::Operation;

/// The operand evaluation claims of every operation, as the shift reduction receives them.
///
/// The four eval arrays have four distinct types, one per arity, so none can stand in for another.
#[derive(Debug, Clone)]
pub struct OperatorClaims<F: Field> {
	/// The unified constraint point, as long as the widest operation's constraint count.
	pub r_x: Vec<F>,
	/// Each operation's constraint-variable count, in `[zero, bitand, intmul, binmul]` order.
	///
	/// Operation `z` is claimed at `r_x[..log_constraints[z]]`; see [`log_constraints`].
	pub log_constraints: [usize; OPERATION_COUNT],
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// The evaluation of the ZERO constraints' operand, `VAL == 0`.
	pub zero: [F; ZERO_ARITY],
	/// The evaluations of the AND constraints' operands, `A & B ^ C == 0`.
	pub bitand: [F; BITAND_ARITY],
	/// The evaluations of the IMUL constraints' operands, `A * B == (HI << 64) | LO`.
	pub intmul: [F; INTMUL_ARITY],
	/// The evaluations of the BMUL constraints' operands, `A * B == C` in the GHASH field.
	pub binmul: [F; BINMUL_ARITY],
}

impl<F: Field> OperatorClaims<F> {
	/// Assembles the claims from the output of the BitAnd sumcheck.
	///
	/// The sumcheck's point is `r_rho || r_x_star`, instance index low, constraint index high.
	/// `r_x_star` spans the widest AND, IMUL and BMUL set. The constraint point `r_x` extends it
	/// through [`zero::reduction_point`] when the ZERO set is wider still, drawing the extra
	/// challenges from `sample`.
	///
	/// An operation the constraint system does not use gets zero evaluations.
	///
	/// # Arguments
	///
	/// - `cs`: the constraint system, whose row counts pick each operation's prefix.
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

		let mut evals = iter::chain(rerand.bitand_evals, rerand.operand_evals.iter().copied());
		// The BitAnd check has no skip branch: an empty AND set reduces over one zero row.
		let bitand = operand_evals(true, &mut evals);
		let intmul = operand_evals(cs.n_imul_constraints() > 0, &mut evals);
		let binmul = operand_evals(cs.n_bmul_constraints() > 0, &mut evals);
		Self {
			r_x,
			log_constraints: log_constraints(cs),
			r_zhat_prime: z_challenge,
			zero: [F::ZERO],
			bitand,
			intmul,
			binmul,
		}
	}

	/// Draws the two batching challenge vectors and folds their weights into the claims.
	///
	/// An operation holds one claim per operand, and the reduction proves them all at once.
	/// The claim of operand `m` of operation `op` is weighted by the product of two equality
	/// indicators, one per axis:
	///
	/// ```text
	/// eq(operation_batch_challenges, op) * eq(operand_batch_challenges, m)
	/// ```
	///
	/// The operand axis is shared by the four operations, and padded to a cube: an operation of
	/// arity below `1 << LOG_MAX_ARITY` reads a prefix of the same weights, and the slots above
	/// its arity name no claim.
	///
	/// The challenges are drawn operation axis first, and the operation weights are indexed in
	/// the order `[Zero, BitwiseAnd, IntegerMul, BinMul]`. The verifier does both the same way, so
	/// these orders are protocol, not detail.
	///
	/// The arities are erased on the way out, since both proving phases dispatch on a shift key.
	///
	/// # Arguments
	///
	/// - `channel`: the transcript the batching challenges are drawn from.
	pub fn prepare(self, channel: &mut impl IPProverChannel<F>) -> PreparedOperatorClaims<F> {
		let operation_batch_challenges = channel.sample_many(LOG_OPERATION_COUNT);
		let operand_batch_challenges = channel.sample_many(LOG_MAX_ARITY);

		let operation_weights = eq_ind_partial_eval_scalars(&operation_batch_challenges);
		let operand_weights = eq_ind_partial_eval_scalars(&operand_batch_challenges);

		// Only the leading `ARITY` operand weights name a claim; `inner_product` pairs the two
		// sequences exactly, so the shared tail is cut here.
		let operation_evals =
			[&self.zero[..], &self.bitand, &self.intmul, &self.binmul].map(|evals| {
				inner_product(evals.iter().copied(), operand_weights[..evals.len()].iter().copied())
			});
		let batched_eval = inner_product(operation_weights.iter().copied(), operation_evals);

		// Every operation is claimed at a prefix of `r_x`, so one expansion of it truncates to
		// each operation's table. The operation's weight reaches every term of the operation, so
		// it is folded into the table.
		let mut weighted_r_x_tensors = constraint_tables(
			eq_ind_partial_eval::<F>(&self.r_x).into_inner(),
			self.log_constraints,
		);
		for (tensor, &weight) in iter::zip(&mut weighted_r_x_tensors, &operation_weights) {
			tensor
				.par_iter_mut()
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|entry| *entry *= weight);
		}

		PreparedOperatorClaims {
			batched_eval,
			r_zhat_prime: self.r_zhat_prime,
			weighted_r_x_tensors,
			operand_weights,
		}
	}
}

/// An operation's operand evaluations, or zeros when it is absent.
///
/// A present operation takes its `ARITY` evaluations off the front of `evals`.
fn operand_evals<F: Field, const ARITY: usize>(
	present: bool,
	evals: &mut impl Iterator<Item = F>,
) -> [F; ARITY] {
	if present {
		array::from_fn(|_| {
			evals
				.next()
				.expect("the sumcheck returns one evaluation per operand column")
		})
	} else {
		[F::ZERO; ARITY]
	}
}

/// The claims with their batching weights folded in, as both proving phases read them.
///
/// Each operation's constraint table is shared by every key of the operation, so it is built once
/// here. Indexing by an [`Operation`] reads it.
#[derive(Debug, Clone)]
pub struct PreparedOperatorClaims<F: Field> {
	/// The operand claims collapsed into the single value the reduction proves:
	///
	/// ```text
	/// batched_eval = sum_op operation_weights[op] * sum_m evals_op[m] * operand_weights[m]
	/// ```
	///
	/// This is the claim phase 1 hands its sumcheck, and it is the same value the verifier
	/// computes from the operand evaluation claims before running its own.
	pub batched_eval: F,
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// Each operation's constraint table, in `[zero, bitand, intmul, binmul]` order: the equality
	/// indicator of its prefix of `r_x`, one weight per constraint, scaled by the operation's own
	/// batching weight.
	pub weighted_r_x_tensors: [Vec<F>; OPERATION_COUNT],
	/// The weight of each operand position, `1 << LOG_MAX_ARITY` entries.
	///
	/// The operand axis is shared by the four operations, so this table is too: a key reads it at
	/// the operand position its constraint index names, whatever operation the key belongs to.
	pub operand_weights: Vec<F>,
}

impl<F: Field> Index<Operation> for PreparedOperatorClaims<F> {
	type Output = [F];

	/// The operation's weighted constraint table.
	fn index(&self, operation: Operation) -> &[F] {
		let index = match operation {
			Operation::Zero => 0,
			Operation::BitwiseAnd => 1,
			Operation::IntegerMul => 2,
			Operation::BinMul => 3,
		};
		&self.weighted_r_x_tensors[index]
	}
}

#[cfg(test)]
mod tests {
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::{B128, StdChallenger};

	use super::*;

	// Zero claims at the empty constraint point.
	fn zero_claims() -> OperatorClaims<B128> {
		OperatorClaims {
			r_x: Vec::new(),
			log_constraints: [0; OPERATION_COUNT],
			r_zhat_prime: B128::ZERO,
			zero: [B128::ZERO; ZERO_ARITY],
			bitand: [B128::ZERO; BITAND_ARITY],
			intmul: [B128::ZERO; INTMUL_ARITY],
			binmul: [B128::ZERO; BINMUL_ARITY],
		}
	}

	#[test]
	fn prepare_shares_one_operand_axis_across_the_four_operations() {
		// Invariant: the operand axis is a cube wide enough for every arity, and the four
		// operations read the same one.
		//
		// The widest arity is BMUL's six, so a cube narrower than that would leave two of its
		// operand claims unweighted.
		let prepared = zero_claims().prepare(&mut ProverTranscript::<StdChallenger>::default());

		assert_eq!(prepared.operand_weights.len(), 1 << LOG_MAX_ARITY);
		for arity in [ZERO_ARITY, BITAND_ARITY, INTMUL_ARITY, BINMUL_ARITY] {
			assert!(arity <= prepared.operand_weights.len());
		}
	}

	#[test]
	fn prepare_draws_the_two_axes_in_transcript_order() {
		// Invariant: the operation axis is drawn before the operand axis, and each axis takes as
		// many challenges as its width.
		//
		// The verifier draws in that order; any other weights the claims by a different tensor.
		//
		// Two transcripts from the same seed hand out the same sequence, so drawing the axes by
		// hand from one pins what `prepare` must have drawn from the other. The axes have
		// different widths, so drawing them in the other order splits that sequence differently
		// and the expansions below disagree.
		let mut expected = ProverTranscript::<StdChallenger>::default();
		let operation_weights =
			eq_ind_partial_eval_scalars(&expected.sample_many(LOG_OPERATION_COUNT));
		let operand_weights = eq_ind_partial_eval_scalars(&expected.sample_many(LOG_MAX_ARITY));

		let mut channel = ProverTranscript::<StdChallenger>::default();
		let prepared = zero_claims().prepare(&mut channel);

		assert_eq!(prepared.operand_weights, operand_weights);

		// The two stay in lockstep only if `prepare` drew exactly those five and no more.
		assert_eq!(
			IPProverChannel::<B128>::sample(&mut channel),
			IPProverChannel::<B128>::sample(&mut expected)
		);

		// A table scaled by its operation's weight reproduces that weight, since the constraint
		// point is empty.
		for (operation, weight) in [
			(Operation::Zero, operation_weights[0]),
			(Operation::BitwiseAnd, operation_weights[1]),
			(Operation::IntegerMul, operation_weights[2]),
			(Operation::BinMul, operation_weights[3]),
		] {
			assert_eq!(prepared[operation], [weight]);
		}
	}
}
