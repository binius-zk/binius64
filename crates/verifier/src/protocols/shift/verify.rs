// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::{AndConstraint, ConstraintSystem, MulConstraint};
use binius_field::{BinaryField, field::FieldOps};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{BinarySubspace, univariate::evaluate_univariate};
use binius_utils::checked_arithmetics::strict_log_2;
use getset::Getters;
use itertools::Itertools;

use super::{BITAND_ARITY, INTMUL_ARITY, error::Error, evaluate_monster_multilinear_for_operation};
use crate::{
	config::LOG_WORD_SIZE_BITS,
	protocols::sumcheck::{SumcheckOutput, verify as verify_sumcheck},
};

/// Verifier data for an operation with the specified arity.
///
/// Contains the challenge points and evaluation claims needed by the verifier.
/// The verifier receives these values during the protocol and uses them to
/// verify the monster multilinear evaluations.
///
/// # Fields
///
/// - `r_x_prime`: multilinear challenge point from the protocol
/// - `r_zhat_prime`: univariate challenge point
/// - `lambda`: random linear combination coefficient for operand weighting
/// - `evals`: array of evaluation claims, one per operand position
#[derive(Debug, Clone)]
pub struct OperatorData<F, const ARITY: usize> {
	pub r_x_prime: Vec<F>,
	pub r_zhat_prime: F,
	pub evals: [F; ARITY],
}

impl<F: FieldOps, const ARITY: usize> OperatorData<F, ARITY> {
	// Constructs a new operator data instance encoding
	// evaluation claim with univariate challenge `r_zhat_prime`
	// multilinear challenge `r_x_prime`, and evaluations `evals`
	// with one eval for each operand of the operation.
	pub fn new(r_zhat_prime: F, r_x_prime: Vec<F>, evals: [F; ARITY]) -> Self {
		Self {
			r_x_prime,
			r_zhat_prime,
			evals,
		}
	}

	// Batching is scaled by random lambda and therefore this batched
	// evaluation claim can be added to other batched evaluation claims
	// without further random scaling.
	fn batched_eval(&self, lambda: F) -> F {
		let lambda_clone = lambda.clone();
		lambda_clone * evaluate_univariate(&self.evals, lambda)
	}
}

/// Output of the shift reduction verification protocol.
///
/// Contains all the challenge points, evaluation claims, and random coefficients
/// produced during the shift reduction protocol. These values are used for subsequent
/// verification steps including PCS verification.
#[derive(Debug, Getters)]
pub struct VerifyOutput<F> {
	/// Random coefficient for batching AND constraint evaluations.
	bitand_lambda: F,
	/// Random coefficient for batching MUL constraint evaluations.
	intmul_lambda: F,
	/// Challenge point for the bit index variables (length `LOG_WORD_SIZE_BITS`).
	pub r_j: Vec<F>,
	/// Challenge point for the shift variables (length `LOG_WORD_SIZE_BITS`).
	pub r_s: Vec<F>,
	/// Challenge point for the word index variables (length `log_word_count`).
	pub r_y: Vec<F>,
	/// Final evaluation claim from the second sumcheck.
	eval: F,
	/// The claimed witness evaluation at the challenge point.
	#[getset(get = "pub")]
	pub witness_eval: F,
}

impl<F> VerifyOutput<F> {
	/// Returns the challenge point for bit index variables.
	///
	/// This corresponds to the first `LOG_WORD_SIZE_BITS` variables
	/// in the witness encoding, indexing individual bits within words.
	pub fn r_j(&self) -> &[F] {
		&self.r_j
	}

	/// Returns the challenge point for shift variables.
	///
	/// This corresponds to `LOG_WORD_SIZE_BITS` variables encoding
	/// the shift operations in the constraint system.
	pub fn r_s(&self) -> &[F] {
		&self.r_s
	}

	/// Returns the challenge point for word index variables.
	///
	/// This corresponds to `log_word_count` variables indexing
	/// the words in the witness vector.
	pub fn r_y(&self) -> &[F] {
		&self.r_y
	}
}

/// Verifies the shift protocol using a two-phase sumcheck approach.
///
/// # Protocol Overview
/// 1. **Sampling Phase**: Samples random lambda coefficients for batching bitand and intmul
///    evaluation claims across operands.
/// 2. **First Sumcheck**: Verifies the batched evaluation claim over `LOG_WORD_SIZE_BITS * 2`
///    variables
/// 3. **Challenge Splitting**: Splits sumcheck challenges into `r_j` and `r_s` components
/// 4. **Second Sumcheck**: Verifies the gamma claim over `log_word_count` variables
/// 5. **Monster Multilinear Verification**: Checks that the claimed evaluations match expected
///    monster multilinear evaluations for both AND constraints (bitand) and MUL constraints
///    (intmul)
///
/// # Parameters
/// - `constraint_system`: The constraint system containing AND and MUL constraints (consumed)
/// - `bitand_data`: Operator data for bit multiplication operations
/// - `intmul_data`: Operator data for integer multiplication operations
/// - `transcript`: Interactive transcript for challenge sampling and message reading
///
/// # Returns
/// Returns [`VerifyOutput`] containing the final challenges and witness evaluation,
/// or an error if verification fails.
///
/// # Errors
/// - Returns `Error::VerificationFailure` if monster multilinear evaluations don't match expected
///   values
/// - Propagates sumcheck verification errors
pub fn verify<F, C>(
	constraint_system: &ConstraintSystem,
	bitand_data: &OperatorData<C::Elem, BITAND_ARITY>,
	intmul_data: &OperatorData<C::Elem, INTMUL_ARITY>,
	channel: &mut C,
) -> Result<VerifyOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
{
	let bitand_lambda = channel.sample();
	let intmul_lambda = channel.sample();

	let eval = bitand_data.batched_eval(bitand_lambda.clone())
		+ intmul_data.batched_eval(intmul_lambda.clone());

	let SumcheckOutput {
		eval: gamma,
		challenges: mut r_jr_s,
	} = verify_sumcheck(LOG_WORD_SIZE_BITS * 2, 2, eval, channel)?;

	r_jr_s.reverse();
	// Split challenges as `r_j,r_s` where `r_j` is the first `LOG_WORD_SIZE_BITS`
	// variables and `r_s` is the last `LOG_WORD_SIZE_BITS` variables
	// Thus `r_s` are the more significant variables.
	let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
	let r_j = r_jr_s;

	let log_word_count = strict_log_2(constraint_system.value_vec_layout.committed_total_len)
		.expect("constraints preprocessed");

	let SumcheckOutput {
		eval,
		challenges: mut r_y,
	} = verify_sumcheck(log_word_count, 2, gamma, channel)?;

	r_y.reverse();

	let witness_eval = channel.recv_one()?;

	Ok(VerifyOutput {
		bitand_lambda,
		intmul_lambda,
		r_j,
		r_y,
		r_s,
		eval,
		witness_eval,
	})
}

/// Validates the evaluation claims from the shift reduction protocol.
///
/// After the shift reduction protocol completes, this function checks that the
/// prover-provided witness evaluation is consistent with the expected values.
/// It computes the monster multilinear evaluations for both AND and MUL constraints
/// and verifies the final equation relating the witness and monster evaluations.
///
/// # Protocol Details
///
/// The function verifies that:
/// ```text
/// eval = witness_eval * monster_eval
/// ```
///
/// Where `monster_eval` is the sum of evaluations for AND and MUL constraint polynomials.
///
/// # Errors
///
/// - `Error::VerificationFailure` if the evaluation equation doesn't hold
/// - Propagates errors from monster multilinear evaluation
pub fn check_eval<F: BinaryField>(
	constraint_system: &ConstraintSystem,
	bitand_data: &OperatorData<F, BITAND_ARITY>,
	intmul_data: &OperatorData<F, INTMUL_ARITY>,
	subspace: &BinarySubspace<F>,
	output: &VerifyOutput<F>,
) -> Result<(), Error> {
	let VerifyOutput {
		bitand_lambda,
		intmul_lambda,
		eval,
		r_j,
		r_s,
		r_y,
		witness_eval,
	} = output;

	// Compute monster multilinear evaluation
	let monster_eval_for_bitand = {
		let (a, b, c) = constraint_system
			.and_constraints
			.iter()
			.map(|AndConstraint { a, b, c }| (a, b, c))
			.multiunzip();
		evaluate_monster_multilinear_for_operation(
			&[a, b, c],
			bitand_data,
			subspace,
			*bitand_lambda,
			r_j,
			r_s,
			r_y,
		)
	}?;
	let monster_eval_for_intmul = {
		let (a, b, lo, hi) = constraint_system
			.mul_constraints
			.iter()
			.map(|MulConstraint { a, b, hi, lo }| (a, b, lo, hi))
			.multiunzip();
		evaluate_monster_multilinear_for_operation(
			&[a, b, lo, hi],
			intmul_data,
			subspace,
			*intmul_lambda,
			r_j,
			r_s,
			r_y,
		)
	}?;
	let monster_eval = monster_eval_for_bitand + monster_eval_for_intmul;

	// Check if the prover-provided witness value is satisfying.
	//
	// The protocol could compute this witness value instead of reading it from the prover. This
	// would require inverting a random element, however, making the protocol incomplete with
	// negligible probability. As a matter of taste, we read the witness value from the prover.
	let expected_eval = *witness_eval * monster_eval;
	if *eval != expected_eval {
		return Err(Error::VerificationFailure);
	}

	Ok(())
}
