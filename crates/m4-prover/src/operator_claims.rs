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

use std::ops::Index;

use binius_field::Field;
use binius_prover::protocols::shift::{Operation, OperatorData, PreparedOperatorData};
use binius_verifier::protocols::shift::{BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, ZERO_ARITY};

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
	/// Draws one batching coefficient per operation and folds it into that operation's claim.
	///
	/// An operation holds one claim per operand, and the reduction proves them all at once.
	/// Batching weights operand `i` by the `i`-th power of the operation's own coefficient.
	///
	/// The powers start at the first, never the zeroth.
	/// So each operation's batched value already carries a random factor of its own.
	/// The four can then be summed directly, with no further scaling to separate them.
	///
	/// The coefficients are drawn in the order `[Zero, BitwiseAnd, IntegerMul, BinMul]`.
	/// The verifier draws in that same order, so this sequence is protocol, not detail.
	///
	/// The arities are erased on the way out, since both proving phases dispatch on a shift key.
	///
	/// # Arguments
	///
	/// - `sample`: draws the next batching coefficient from the transcript.
	pub fn prepare(self, mut sample: impl FnMut() -> F) -> PreparedOperatorClaims<F> {
		PreparedOperatorClaims {
			zero: PreparedOperatorData::new(self.zero, sample()),
			bitand: PreparedOperatorData::new(self.bitand, sample()),
			intmul: PreparedOperatorData::new(self.intmul, sample()),
			binmul: PreparedOperatorData::new(self.binmul, sample()),
		}
	}
}

/// The claims with their batching coefficients folded in, as both proving phases read them.
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
	use binius_verifier::config::B128;

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
	fn prepare_carries_each_operations_arity_into_the_erased_form() {
		// Invariant: erasing the arity keeps it as a length, one lambda power per operand.
		//
		// The four arities are 1, 3, 4 and 6, all distinct.
		// So a claim reached through the wrong index arm shows up as the wrong count.
		let prepared = zero_claims().prepare(|| B128::ONE);

		assert_eq!(prepared[Operation::Zero].lambda_powers.len(), ZERO_ARITY);
		assert_eq!(prepared[Operation::BitwiseAnd].lambda_powers.len(), BITAND_ARITY);
		assert_eq!(prepared[Operation::IntegerMul].lambda_powers.len(), INTMUL_ARITY);
		assert_eq!(prepared[Operation::BinMul].lambda_powers.len(), BINMUL_ARITY);
	}

	#[test]
	fn prepare_draws_one_coefficient_per_operation_in_transcript_order() {
		// Invariant: the coefficients are drawn in the order [Zero, AND, IMUL, BMUL].
		// The verifier draws in that order; any other batches a claim by the wrong coefficient.
		//
		// This stand-in transcript hands out 1, 2, 3, 4, so a coefficient names its draw position.
		let mut draws = 0u128;
		let prepared = zero_claims().prepare(|| {
			draws += 1;
			B128::new(draws)
		});

		assert_eq!(draws, 4, "one draw per operation");

		// Powers start at the first, so a claim's leading power is the coefficient it was given.
		assert_eq!(prepared[Operation::Zero].lambda_powers[0], B128::new(1));
		assert_eq!(prepared[Operation::BitwiseAnd].lambda_powers[0], B128::new(2));
		assert_eq!(prepared[Operation::IntegerMul].lambda_powers[0], B128::new(3));
		assert_eq!(prepared[Operation::BinMul].lambda_powers[0], B128::new(4));
	}
}
