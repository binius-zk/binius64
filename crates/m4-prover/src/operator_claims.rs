// Copyright 2026 The Binius Developers

//! The shift reduction's operand evaluation claims, one per operation.
//!
//! The reduction closes four constraint families in one proof: ZERO, AND, IMUL and BMUL.
//!
//! Each family arrives with its own operand evaluation claim.
//! Those four claims then travel together through both phases of the reduction.
//!
//! Passed as four positional arguments, two same-typed neighbours are interchangeable.
//! The compiler accepts either order, so a swap yields a wrong proof instead of an error.
//!
//! Bundled instead, each claim is named where it is built.
//! It is then reached by indexing on [`Operation`] where it is read.

use std::ops::Index;

use binius_field::Field;
use binius_prover::protocols::shift::{Operation, OperatorData, PreparedOperatorData};

/// The operand evaluation claim of every operation, as the shift reduction receives them.
///
/// A shift key names the operation it belongs to, so its claim is reached by indexing:
///
/// ```text
/// claims[key.operation]
/// ```
#[derive(Debug, Clone)]
pub struct OperatorClaims<F: Field> {
	/// The claim for the ZERO constraints, `VAL == 0`.
	pub zero: OperatorData<F>,
	/// The claim for the AND constraints, `A & B ^ C == 0`.
	pub bitand: OperatorData<F>,
	/// The claim for the IMUL constraints, `A * B == (HI << 64) | LO`.
	pub intmul: OperatorData<F>,
	/// The claim for the BMUL constraints, `A * B == C` in the GHASH field.
	pub binmul: OperatorData<F>,
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

impl<F: Field> Index<Operation> for OperatorClaims<F> {
	type Output = OperatorData<F>;

	fn index(&self, operation: Operation) -> &OperatorData<F> {
		match operation {
			Operation::Zero => &self.zero,
			Operation::BitwiseAnd => &self.bitand,
			Operation::IntegerMul => &self.intmul,
			Operation::BinMul => &self.binmul,
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

	// One claim per operation, each tagged by a distinct operand count.
	// A mis-wired index arm therefore returns a count no assertion accepts.
	fn claims_tagged_by_arity() -> OperatorClaims<B128> {
		OperatorClaims {
			zero: OperatorData::zero_claim(1, B128::ZERO),
			bitand: OperatorData::zero_claim(2, B128::ZERO),
			intmul: OperatorData::zero_claim(3, B128::ZERO),
			binmul: OperatorData::zero_claim(4, B128::ZERO),
		}
	}

	#[test]
	fn each_operation_indexes_its_own_claim() {
		// Invariant: an operation indexes its own claim, never a neighbour's.
		let claims = claims_tagged_by_arity();

		assert_eq!(claims[Operation::Zero].evals.len(), 1);
		assert_eq!(claims[Operation::BitwiseAnd].evals.len(), 2);
		assert_eq!(claims[Operation::IntegerMul].evals.len(), 3);
		assert_eq!(claims[Operation::BinMul].evals.len(), 4);
	}

	#[test]
	fn prepare_draws_one_coefficient_per_operation_in_transcript_order() {
		// Invariant: the coefficients are drawn in the order [Zero, AND, IMUL, BMUL].
		// The verifier draws in that order; any other batches a claim by the wrong coefficient.
		//
		// This stand-in transcript hands out 1, 2, 3, 4, so a coefficient names its draw position.
		let mut draws = 0u128;
		let prepared = claims_tagged_by_arity().prepare(|| {
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
