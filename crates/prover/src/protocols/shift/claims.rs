// Copyright 2026 The Binius Developers

//! The shift reduction's operand evaluation claims, one per operation.
//!
//! The reduction closes four constraint families in one proof: ZERO, AND, IMUL and BMUL.
//!
//! Each family arrives with its own operand evaluation claim.
//! Those four claims then travel together through both phases of the reduction.
//!
//! Passed as four positional arguments, they are easy to hand over in the wrong order.
//! Bundling them names each claim where it is built.
//!
//! Each claim also carries its operation's arity in its type, so no two share one.
//! Putting a BMUL claim in the IMUL field is a type error, not a wrong proof.
//!
//! The batched form erases that arity, because a shift key picks its operation at run time.
//! There the operation is named by indexing instead: `prepared[key.operation]`.

use std::{array, iter, ops::Index};

use binius_core::constraint_system::ConstraintSystem;
use binius_field::Field;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::multilinear::eq::eq_ind_partial_eval_scalars;
use binius_verifier::protocols::{
	rerand::RerandOutput,
	shift::{
		BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, LOG_MAX_ARITY, LOG_OPERATION_COUNT, ZERO_ARITY,
	},
	zero,
};

use super::{Operation, OperatorData, PreparedOperatorData};

/// The operand evaluation claim of every operation, as the shift reduction receives them.
///
/// The four fields have four distinct types, one per arity, so none can stand in for another.
#[derive(Debug, Clone)]
pub struct OperatorClaims<F: Field> {
	/// The claim for the ZERO constraints, `VAL == 0`.
	pub zero: OperatorData<F, ZERO_ARITY>,
	/// The claim for the AND constraints, `A & B ^ C == 0`.
	pub bitand: OperatorData<F, BITAND_ARITY>,
	/// The claim for the IMUL constraints, `A * B == (HI << 64) | LO`.
	pub intmul: OperatorData<F, INTMUL_ARITY>,
	/// The claim for the BMUL constraints, `A * B == C` in the GHASH field.
	pub binmul: OperatorData<F, BINMUL_ARITY>,
}

impl<F: Field> OperatorClaims<F> {
	/// Assembles the four claims from the output of the BitAnd sumcheck.
	///
	/// The sumcheck's point is `r_rho || r_x_star`, instance index low, constraint index high.
	/// Each operation is claimed at the prefix of `r_x_star` its rows span, so all four share
	/// `r_rho`. The Zero point comes from [`zero::reduction_point`], which draws any extra
	/// challenges from `sample`.
	///
	/// An operation the constraint system does not use gets a zero claim.
	///
	/// # Arguments
	///
	/// - `cs`: the constraint system, whose row counts pick each operation's prefix.
	/// - `log_instances`: the instance variables at the low end of the point.
	/// - `z_challenge`: the univariate challenge, shared by every operation.
	/// - `rerand`: the sumcheck's output, with the IntMul operand evaluations before the BinMul
	///   ones.
	/// - `sample`: draws the Zero point's extra challenges from the transcript.
	pub fn from_rerand(
		cs: &ConstraintSystem,
		log_instances: usize,
		z_challenge: F,
		rerand: &RerandOutput<F>,
		sample: impl FnMut() -> F,
	) -> Self {
		let r_x_star = &rerand.eval_point[log_instances..];
		let mut evals = iter::chain(rerand.bitand_evals, rerand.operand_evals.iter().copied());
		// The BitAnd check has no skip branch: an empty AND set reduces over one zero row.
		let log_n_and = cs.log_and_constraints().unwrap_or(0);
		let bitand = operation_claim(Some(log_n_and), r_x_star, z_challenge, &mut evals);
		let intmul = operation_claim(cs.log_imul_constraints(), r_x_star, z_challenge, &mut evals);
		let binmul = operation_claim(cs.log_bmul_constraints(), r_x_star, z_challenge, &mut evals);
		let log_n_zero = cs.log_zero_constraints().unwrap_or(0);
		let zero = OperatorData {
			evals: [F::ZERO],
			r_zhat_prime: z_challenge,
			r_x_prime: zero::reduction_point(r_x_star, log_n_zero, sample),
		};
		Self {
			zero,
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
	/// Two operations' weights are distinct entries of one equality tensor, so their batched
	/// values sum directly, with no further scaling to separate them.
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

		PreparedOperatorClaims {
			zero: PreparedOperatorData::new(self.zero, operation_weights[0], &operand_weights),
			bitand: PreparedOperatorData::new(self.bitand, operation_weights[1], &operand_weights),
			intmul: PreparedOperatorData::new(self.intmul, operation_weights[2], &operand_weights),
			binmul: PreparedOperatorData::new(self.binmul, operation_weights[3], &operand_weights),
			operand_weights,
		}
	}
}

/// An operation's claim at its prefix of `r_x_star`, or a zero claim when it is absent.
///
/// A present operation takes its `ARITY` evaluations off the front of `evals`.
fn operation_claim<F: Field, const ARITY: usize>(
	log_n_constraints: Option<usize>,
	r_x_star: &[F],
	z_challenge: F,
	evals: &mut impl Iterator<Item = F>,
) -> OperatorData<F, ARITY> {
	log_n_constraints.map_or_else(
		|| OperatorData::zero_claim(z_challenge),
		|log_n| OperatorData {
			evals: array::from_fn(|_| {
				evals
					.next()
					.expect("the sumcheck returns one evaluation per operand column")
			}),
			r_zhat_prime: z_challenge,
			r_x_prime: r_x_star[..log_n].to_vec(),
		},
	)
}

/// The claims with their batching weights folded in, as both proving phases read them.
///
/// Each entry also carries the tensor expansion of its constraint point.
/// That expansion is shared by every key of the operation, so it is built once here.
#[derive(Debug, Clone)]
pub struct PreparedOperatorClaims<F: Field> {
	/// The prepared claim for the ZERO constraints.
	pub zero: PreparedOperatorData<F>,
	/// The prepared claim for the AND constraints.
	pub bitand: PreparedOperatorData<F>,
	/// The prepared claim for the IMUL constraints.
	pub intmul: PreparedOperatorData<F>,
	/// The prepared claim for the BMUL constraints.
	pub binmul: PreparedOperatorData<F>,
	/// The weight of each operand position, `1 << LOG_MAX_ARITY` entries.
	///
	/// The operand axis is shared by the four operations, so this table is too: a key reads it at
	/// the operand position its constraint index names, whatever operation the key belongs to.
	pub operand_weights: Vec<F>,
}

impl<F: Field> PreparedOperatorClaims<F> {
	/// The four operations' claims collapsed into the single value the reduction proves.
	///
	/// Each operation's batched evaluation already carries its own weight on the operation axis,
	/// so the four sum directly with no further scaling.
	///
	/// This is the claim phase 1 hands its sumcheck, and it is the same value the verifier
	/// computes from the operand evaluation claims before running its own.
	pub fn batched_eval(&self) -> F {
		self.zero.batched_eval
			+ self.bitand.batched_eval
			+ self.intmul.batched_eval
			+ self.binmul.batched_eval
	}
}

impl<F: Field> Index<Operation> for PreparedOperatorClaims<F> {
	type Output = PreparedOperatorData<F>;

	fn index(&self, operation: Operation) -> &PreparedOperatorData<F> {
		match operation {
			Operation::Zero => &self.zero,
			Operation::BitwiseAnd => &self.bitand,
			Operation::IntegerMul => &self.intmul,
			Operation::BinMul => &self.binmul,
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::{B128, StdChallenger};

	use super::*;

	// A zero claim per operation, each at the arity its field's type fixes.
	fn zero_claims() -> OperatorClaims<B128> {
		OperatorClaims {
			zero: OperatorData::zero_claim(B128::ZERO),
			bitand: OperatorData::zero_claim(B128::ZERO),
			intmul: OperatorData::zero_claim(B128::ZERO),
			binmul: OperatorData::zero_claim(B128::ZERO),
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

		// A claim scaled by its operation's weight reproduces that weight, since the operand
		// claims here are all zero and the constraint point is empty.
		for (operation, weight) in [
			(Operation::Zero, operation_weights[0]),
			(Operation::BitwiseAnd, operation_weights[1]),
			(Operation::IntegerMul, operation_weights[2]),
			(Operation::BinMul, operation_weights[3]),
		] {
			assert_eq!(prepared[operation].weighted_r_x_prime_tensor.as_ref(), &[weight]);
		}
	}
}
