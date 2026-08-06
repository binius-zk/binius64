// Copyright 2025 Irreducible Inc.

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, util::powers};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer, inner_product::inner_product, multilinear::eq::eq_ind_partial_eval,
};
use binius_verifier::protocols::shift::{BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, ZERO_ARITY};

use super::{key_collection::KeyCollection, phase_1::prove_phase_1, phase_2::prove_phase_2};

/// One operation's operand evaluation claims, with the point they are claimed at.
///
/// An operation constrains a fixed number of operands at once, its arity:
///
/// ```text
/// ZERO 1   AND 3   IMUL 4   BMUL 6
/// ```
///
/// Each operand is one column of the witness, and each column carries one evaluation claim.
///
/// The arity is a type parameter, so no two operations share a type.
/// Two operations' claims therefore cannot be passed in each other's place.
///
/// This mirrors [`binius_verifier::protocols::shift::OperatorData`], which is already arity-typed.
///
/// Every operand of an operation is claimed at the same point.
/// So the point is stored once here rather than once per operand.
///
/// That point is oblong: a univariate coordinate over the bit axis, then the constraint index.
#[derive(Debug, Clone)]
pub struct OperatorData<F: Field, const ARITY: usize> {
	/// The claimed evaluation of each operand column, in the operation's operand order.
	pub evals: [F; ARITY],
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// The multilinear challenge over the constraint index.
	pub r_x_prime: Vec<F>,
}

impl<F: Field, const ARITY: usize> OperatorData<F, ARITY> {
	/// The claim of an operation the constraint system does not use.
	///
	/// Every operand evaluates to zero, at the empty constraint point.
	///
	/// That operation's constraint set is empty.
	/// So the shift finds no key naming it, and the claim contributes nothing to the batch.
	///
	/// # Arguments
	///
	/// - `r_zhat_prime`: the univariate challenge, shared by every operation.
	pub const fn zero_claim(r_zhat_prime: F) -> Self {
		Self {
			evals: [F::ZERO; ARITY],
			r_zhat_prime,
			r_x_prime: Vec::new(),
		}
	}
}

/// One operation's claims, with the expansions both proving phases need precomputed.
///
/// Every shift key of the operation reads the same two expansions, so each is built once here:
///
/// - the constraint point, expanded into its equality-indicator tensor;
/// - the batching coefficient, expanded into its powers, one per operand.
///
/// The arity is erased here, unlike in [`OperatorData`].
/// Both phases pick an operation at run time, from a shift key, so all four must share a type.
#[derive(Debug, Clone)]
pub struct PreparedOperatorData<F: Field> {
	/// The claimed evaluation of each operand column, in the operation's operand order.
	pub evals: Vec<F>,
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// The equality-indicator tensor of the constraint point, one weight per constraint.
	pub r_x_prime_tensor: FieldBuffer<F>,
	/// The batching coefficient's powers, one per operand, starting at the first power.
	pub lambda_powers: Vec<F>,
}

impl<F: Field> PreparedOperatorData<F> {
	/// Expands one operation's claims against the batching coefficient drawn for it.
	///
	/// # Arguments
	///
	/// - `operator_data`: the operand claims, and the point they are claimed at.
	/// - `lambda`: the batching coefficient for this operation.
	pub fn new<const ARITY: usize>(operator_data: OperatorData<F, ARITY>, lambda: F) -> Self {
		let OperatorData {
			evals,
			r_zhat_prime,
			r_x_prime,
		} = operator_data;
		let r_x_prime_tensor = eq_ind_partial_eval::<F>(&r_x_prime);
		let lambda_powers = powers(lambda).skip(1).take(ARITY).collect();
		Self {
			evals: evals.to_vec(),
			r_zhat_prime,
			r_x_prime_tensor,
			lambda_powers,
		}
	}

	/// The operand claims collapsed into one value by the batching coefficient.
	///
	/// Operand `i` is weighted by the `i`-th power, and the powers start at the first:
	///
	/// ```text
	/// batched = sum_i evals[i] * lambda^(i + 1)
	/// ```
	///
	/// So the result already carries a random factor unique to this operation.
	/// Two operations' batched values can therefore be summed with no further scaling.
	pub fn batched_eval(&self) -> F {
		inner_product(self.evals.iter().copied(), self.lambda_powers.iter().copied())
	}
}

/// Proves the shift protocol reduction using a two-phase approach.
///
/// This function orchestrates the complete shift protocol proof, reducing bitand and intmul
/// evaluation claims to a single multilinear claim on the witness. The protocol consists
/// of two sequential sumcheck phases that progressively reduce the complexity of the claims.
///
/// # Protocol Overview
/// 1. **Lambda Sampling**: Samples random coefficients for batching operator claims
/// 2. **Phase 1**: Proves batched operator claims over shift variants and operand positions
/// 3. **Phase 2**: Reduces to witness evaluation using monster multilinear polynomial
///
/// # Parameters
/// - `key_collection`: Prover's key collection representing the constraint system
/// - `words`: The witness words (must have power-of-2 length)
/// - `zero_data`: Operator data for the linear (ZERO) constraints
/// - `bitand_data`: Operator data for bit multiplication (AND) constraints
/// - `intmul_data`: Operator data for integer multiplication (IMUL) constraints
/// - `binmul_data`: Operator data for GHASH-field multiplication (BMUL) constraints
/// - `transcript`: The prover's transcript for interactive protocol
///
/// Each of the four carries its own arity, so no two can be passed in the other's place.
///
/// # Returns
/// Returns `SumcheckOutput` containing the final challenges and witness evaluation,
/// or an error if the proof generation fails.
///
/// # Requirements
/// - `words` must have power-of-2 length for efficient multilinear operations
#[allow(clippy::too_many_arguments)]
pub fn prove<F, P, Channel, A>(
	key_collection: &KeyCollection,
	words: &[Word],
	zero_data: OperatorData<F, ZERO_ARITY>,
	bitand_data: OperatorData<F, BITAND_ARITY>,
	intmul_data: OperatorData<F, INTMUL_ARITY>,
	binmul_data: OperatorData<F, BINMUL_ARITY>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
	alloc: &A,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	// Sample lambdas, one for each operator.
	let zero_lambda = channel.sample();
	let bitand_lambda = channel.sample();
	let intmul_lambda = channel.sample();
	let binmul_lambda = channel.sample();

	// Create prepared operator data with sampled lambdas
	let expand_scope = tracing::debug_span!("Expand tensor queries").entered();
	let prepared_zero_data = PreparedOperatorData::new(zero_data, zero_lambda);
	let prepared_bitand_data = PreparedOperatorData::new(bitand_data, bitand_lambda);
	let prepared_intmul_data = PreparedOperatorData::new(intmul_data, intmul_lambda);
	let prepared_binmul_data = PreparedOperatorData::new(binmul_data, binmul_lambda);
	drop(expand_scope);

	// Prove the first phase, receiving a `SumcheckOutput`
	// with challenges made of `r_j` and `r_s`,
	// and eval equal to `gamma` (see paper).
	let phase_1_output = prove_phase_1::<_, P, _, _>(
		key_collection,
		words,
		&prepared_zero_data,
		&prepared_bitand_data,
		&prepared_intmul_data,
		&prepared_binmul_data,
		domain_subspace,
		channel,
		alloc,
	);

	// Prove the second phase, receiving a `SumcheckOutput`
	// with challenges `r_y` and eval the evaluation of
	// the witness at oblong point had by univariate
	// variable `r_j` and multilinear variable `r_y`.
	let SumcheckOutput { challenges, eval } = prove_phase_2::<_, P, _, _>(
		key_collection,
		words,
		&prepared_zero_data,
		&prepared_bitand_data,
		&prepared_intmul_data,
		&prepared_binmul_data,
		domain_subspace,
		phase_1_output,
		channel,
		alloc,
	);

	// Return evaluation claim on the witness.
	SumcheckOutput { challenges, eval }
}
