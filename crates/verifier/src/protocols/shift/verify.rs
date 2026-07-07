// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, iter};

use binius_core::{
	constraint_system::{AndConstraint, ConstraintSystem, MulConstraint, Operand},
	word::Word,
};
use binius_field::{BinaryField, field::FieldOps, util::FieldFn};
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{SumcheckOutput, verify as verify_sumcheck},
};
use binius_math::{
	BinarySubspace,
	line::extrapolate_line,
	multilinear::eq::{
		eq_ind_partial_eval_scalars, eq_ind_zero, eq_one_var, scaled_eq_ind_partial_eval_scalars,
	},
	univariate::{evaluate_univariate, lagrange_evals_scalars},
};
use getset::Getters;
use itertools::Itertools;

use super::{
	BITAND_ARITY, INTMUL_ARITY, SHIFT_VARIANT_COUNT, error::Error, evaluate_h_op,
	evaluate_monster_multilinear_for_operation,
	monster::evaluate_monster_multilinear_for_operation_native,
};
use crate::config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS};

/// Evaluates the bit-level multilinear extension of a word slice at the point `r_j ++ r_y`.
///
/// The multilinear has `LOG_WORD_SIZE_BITS + r_y.len()` variables: the low variables index the
/// bit within a word and the high variables index the word. Words past `words.len()` (up to
/// `2^r_y.len()`) are treated as zero.
///
/// ## Preconditions
///
/// * `r_j` has exactly `LOG_WORD_SIZE_BITS` entries
/// * `words` has at most `2^r_y.len()` entries
pub fn evaluate_words_mle<F, E>(words: &[Word], r_j: &[E], r_y: &[E]) -> E
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
{
	assert_eq!(r_j.len(), LOG_WORD_SIZE_BITS); // precondition
	assert!(words.len() <= 1 << r_y.len()); // precondition

	let r_j_tensor = eq_ind_partial_eval_scalars(r_j);
	let r_y_tensor = eq_ind_partial_eval_scalars(r_y);
	iter::zip(words, r_y_tensor)
		.map(|(word, weight)| {
			let word_eval = (0..WORD_SIZE_BITS)
				.filter(|bit| (word.as_u64() >> bit) & 1 == 1)
				.map(|bit| &r_j_tensor[bit])
				.sum::<E>();
			weight * word_eval
		})
		.sum()
}

/// Verifier data for an operation with the specified arity.
///
/// Contains the challenge points and evaluation claims needed by the verifier.
/// The verifier receives these values during the protocol and uses them to
/// verify the monster multilinear evaluations.
///
/// # Fields
///
/// - `r_x_prime`: multilinear challenge point from the protocol
/// - `evals`: array of evaluation claims, one per operand position
#[derive(Debug, Clone)]
pub struct OperatorData<F, const ARITY: usize> {
	pub r_x_prime: Vec<F>,
	pub evals: [F; ARITY],
}

impl<F: FieldOps, const ARITY: usize> OperatorData<F, ARITY> {
	// Constructs a new operator data instance encoding
	// evaluation claim with multilinear challenge `r_x_prime` and evaluations `evals`
	// (one eval for each operand of the operation).
	pub const fn new(r_x_prime: Vec<F>, evals: [F; ARITY]) -> Self {
		Self { r_x_prime, evals }
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
	/// Challenge point for the word index variables (length `log_witness_words`).
	pub r_y: Vec<F>,
	/// Challenge for the witness's segment selector variable.
	pub r_segment: F,
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

	// The second sumcheck runs over the witness: the public segment in the low half-cube and
	// the hidden segment in the high half-cube, selected by the top word-index variable. This
	// matches the prover, which zero-pads each segment to the half size.
	let log_word_count = constraint_system.value_vec_layout.log_witness_words() + 1;

	let SumcheckOutput {
		eval,
		challenges: mut r_y,
	} = verify_sumcheck(log_word_count, 2, gamma, channel)?;

	r_y.reverse();
	let r_segment = r_y.pop().expect("log_word_count >= 1");

	let witness_eval = channel.recv_one()?;

	Ok(VerifyOutput {
		bitand_lambda,
		intmul_lambda,
		r_j,
		r_y,
		r_segment,
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
/// eval = trace_eval * monster_eval
/// ```
///
/// where `monster_eval` is the sum of evaluations for AND and MUL constraint polynomials, and
/// `trace_eval` is the witness evaluation reconstructed from its two segments:
/// ```text
/// trace_eval = (1 - r_segment) * prod_{k <= i} (1 - r_y_i) * public_eval + r_segment * witness_eval
/// ```
/// The public half is evaluated over the *verifier's* public words, so a prover that used
/// different public values fails this check with high probability; this subsumes the former
/// standalone public input check.
///
/// # Errors
///
/// - `Error::VerificationFailure` if the evaluation equation doesn't hold
/// - Propagates errors from monster multilinear evaluation
#[allow(clippy::too_many_arguments)]
pub fn check_eval<F, C>(
	constraint_system: &ConstraintSystem,
	public: &[Word],
	bitand_data: &OperatorData<C::Elem, BITAND_ARITY>,
	intmul_data: &OperatorData<C::Elem, INTMUL_ARITY>,
	subspace: &BinarySubspace<F>,
	r_zhat_prime: C::Elem,
	output: &VerifyOutput<C::Elem>,
	channel: &mut C,
) -> Result<(), Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	let VerifyOutput {
		bitand_lambda,
		intmul_lambda,
		eval,
		r_j,
		r_s,
		r_y,
		r_segment,
		witness_eval,
	} = output;

	// `monster_eval` is a function of purely public-channel-derived elements
	// (`r_zhat_prime`, `bitand_lambda`, `intmul_lambda`, the operator data's `r_x_prime`
	// vectors, `r_j`, `r_s`, `r_y`) plus the constant `subspace` and `constraint_system`.
	// Trade those Elems for plain field values, run the MLE evaluation in plaintext, and
	// materialize the result as a single inout wire instead of building the entire
	// sub-circuit in wrapper channels.
	let monster_eval = {
		let bitand_r_x_prime_len = bitand_data.r_x_prime.len();
		let intmul_r_x_prime_len = intmul_data.r_x_prime.len();
		let r_j_len = r_j.len();
		let r_s_len = r_s.len();
		let r_y_len = r_y.len();

		let inputs: Vec<C::Elem> = iter::once(r_zhat_prime)
			.chain(iter::once(bitand_lambda.clone()))
			.chain(iter::once(intmul_lambda.clone()))
			.chain(bitand_data.r_x_prime.iter().cloned())
			.chain(intmul_data.r_x_prime.iter().cloned())
			.chain(r_j.iter().cloned())
			.chain(r_s.iter().cloned())
			.chain(r_y.iter().cloned())
			.chain(iter::once(r_segment.clone()))
			.collect();

		let eval_fn = MonsterEvalFn {
			subspace,
			constraint_system,
			bitand_r_x_prime_len,
			intmul_r_x_prime_len,
			r_j_len,
			r_s_len,
			r_y_len,
		};
		channel.compute_public_value(&inputs, eval_fn)
	};

	// The public-half evaluation is a function of the verifier's public words (plaintext) and
	// public-channel-derived challenges, so it is computed the same way as `monster_eval`.
	let log_public_words = constraint_system.value_vec_layout.log_public_words();
	let public_eval = {
		let inputs: Vec<C::Elem> = r_j
			.iter()
			.chain(&r_y[..log_public_words])
			.cloned()
			.collect();
		channel.compute_public_value(&inputs, PublicWordsEvalFn { public })
	};

	// Reconstruct the witness evaluation from its two segments. The public segment is
	// zero-padded up to the hidden segment length, contributing the eq-zero padding factor.
	let padded_public_eval = eq_ind_zero(&r_y[log_public_words..]) * public_eval;
	let trace_eval = extrapolate_line(padded_public_eval, witness_eval.clone(), r_segment.clone());

	// Check if the reconstructed trace value is satisfying.
	//
	// The protocol could compute the committed-half value instead of reading it from the prover.
	// This would require inverting a random element, however, making the protocol incomplete
	// with negligible probability. As a matter of taste, we read the value from the prover.
	let expected_eval = trace_eval * monster_eval;
	channel.assert_zero(expected_eval - eval)?;

	Ok(())
}

/// The bit-level MLE evaluation of the public words, as a [`FieldFn`] over the inputs
/// `r_j.. | r_y_low..`: the word-bit challenges followed by the low `log_public_words`
/// word-index challenges. Computed via the same public-value mechanism as [`MonsterEvalFn`].
struct PublicWordsEvalFn<'a> {
	/// The public words of the value vector.
	public: &'a [Word],
}

impl<F: BinaryField> FieldFn<F> for PublicWordsEvalFn<'_> {
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, vals: &[E]) -> E {
		let (r_j, r_y_low) = vals.split_at(LOG_WORD_SIZE_BITS);
		evaluate_words_mle(self.public, r_j, r_y_low)
	}
}

/// The monster multilinear evaluation, as a [`FieldFn`] over public-channel-derived inputs.
///
/// The inputs are the flat concatenation of these sections, in order:
///
/// ```text
/// r_zhat_prime | bitand_lambda | intmul_lambda | bitand_r_x_prime.. | intmul_r_x_prime.. | r_j.. | r_s.. | r_y..
/// ```
///
/// The stored lengths recover each variable-length section from that flat slice.
struct MonsterEvalFn<'a, F: BinaryField> {
	/// The evaluation subspace for the Lagrange basis over the word bits.
	subspace: &'a BinarySubspace<F>,
	/// The AND and MUL constraints whose monster multilinears are evaluated.
	constraint_system: &'a ConstraintSystem,
	/// Length of the BitAnd operator's `r_x_prime` section.
	bitand_r_x_prime_len: usize,
	/// Length of the IntMul operator's `r_x_prime` section.
	intmul_r_x_prime_len: usize,
	/// Length of the `r_j` section (the low-order word-bit challenges).
	r_j_len: usize,
	/// Length of the `r_s` section (the shift-amount challenges).
	r_s_len: usize,
	/// Length of the `r_y` section (the column challenges).
	r_y_len: usize,
}

impl<F: BinaryField> MonsterEvalFn<'_, F> {
	/// Shared evaluation logic for [`FieldFn::call`] and [`FieldFn::call_native`].
	///
	/// `eval_op` computes one operation's monster evaluation from its operands and the shared
	/// tensors — the expensive part that differs between the generic
	/// ([`evaluate_monster_multilinear_for_operation`]) and native
	/// (`evaluate_monster_multilinear_for_operation_native`) paths. Everything else (the input
	/// splitting and the tensor expansions) is shared.
	fn evaluate<E, G>(&self, vals: &[E], eval_op: G) -> E
	where
		E: FieldOps<Scalar = F> + From<F>,
		G: Fn(&[Vec<&Operand>], &[E], E, &[E; SHIFT_VARIANT_COUNT * WORD_SIZE_BITS], &[E]) -> E,
	{
		// Split the flat input back into its sections, in the order they were concatenated.
		let r_zhat_prime_v = vals[0].clone();
		let bitand_lambda_v = vals[1].clone();
		let intmul_lambda_v = vals[2].clone();
		let mut off = 3;
		let bitand_r_x_prime_v = &vals[off..off + self.bitand_r_x_prime_len];
		off += self.bitand_r_x_prime_len;
		let intmul_r_x_prime_v = &vals[off..off + self.intmul_r_x_prime_len];
		off += self.intmul_r_x_prime_len;
		let r_j_v = &vals[off..off + self.r_j_len];
		off += self.r_j_len;
		let r_s_v = &vals[off..off + self.r_s_len];
		off += self.r_s_len;
		let r_y_v = &vals[off..off + self.r_y_len];
		off += self.r_y_len;
		// `r_segment` is the top word-index coordinate, appended after `r_y`; it selects the public
		// (0) vs hidden (1) segment.
		let r_segment = vals[off].clone();

		// Build the word-index equality tensor over the value vector: public words in the low
		// segment, hidden words in the high segment.
		//
		// Rather than expand the full `(r_y, r_segment)` tensor (which doubles the multiplications
		// and then gets re-indexed), build each segment's portion directly from the shared `r_y`
		// indicator:
		//   * hidden — the `r_y` indicator scaled by `r_segment` (the high-half weight);
		//   * public — the `log_public_words`-length prefix indicator (the public segment occupies
		//     that prefix of the address space) scaled by `(1 - r_segment)` and the eq-zero padding
		//     over the unused `r_y` coordinates — the same `padded_public_eval` factor that
		//     `check_eval` reconstructs the witness evaluation with.
		let layout = &self.constraint_system.value_vec_layout;
		let n_public_words = layout.n_public_words();
		let log_public_words = layout.log_public_words();

		let public_scale =
			eq_one_var(r_segment.clone(), E::zero()) * eq_ind_zero(&r_y_v[log_public_words..]);
		let public_tensor =
			scaled_eq_ind_partial_eval_scalars(&r_y_v[..log_public_words], public_scale);
		let hidden_tensor = scaled_eq_ind_partial_eval_scalars(r_y_v, r_segment);

		// `n_public_words` is a power of two, so the public prefix indicator has exactly that many
		// entries; the hidden portion fills the remainder of the value vector.
		let mut r_y_tensor = Vec::with_capacity(layout.combined_len());
		r_y_tensor.extend_from_slice(&public_tensor);
		r_y_tensor.extend_from_slice(&hidden_tensor[..layout.combined_len() - n_public_words]);
		let l_tilde = lagrange_evals_scalars(self.subspace, r_zhat_prime_v);
		let h_op_evals = evaluate_h_op(&l_tilde, r_j_v, r_s_v);

		// Tensor the shift-selector evaluations with the shift-amount equality indicator once, so
		// both the BitAnd and IntMul monster evaluations share it. Indexed by
		// `variant * WORD_SIZE_BITS + amount`.
		let eq_r_s = eq_ind_partial_eval_scalars(r_s_v);
		let shift_scalars =
			Box::new(array::from_fn::<_, { SHIFT_VARIANT_COUNT * WORD_SIZE_BITS }, _>(|i| {
				h_op_evals[i / WORD_SIZE_BITS].clone() * &eq_r_s[i % WORD_SIZE_BITS]
			}));

		// BitAnd contribution: operands (a, b, c) batched by `bitand_lambda`.
		let bitand_part = {
			let (a, b, c) = self
				.constraint_system
				.and_constraints
				.iter()
				.map(|AndConstraint { a, b, c }| (a, b, c))
				.multiunzip();
			eval_op(&[a, b, c], bitand_r_x_prime_v, bitand_lambda_v, &shift_scalars, &r_y_tensor)
		};
		// IntMul contribution: operands (a, b, lo, hi) batched by `intmul_lambda`.
		let intmul_part = if !self.constraint_system.mul_constraints.is_empty() {
			let (a, b, lo, hi) = self
				.constraint_system
				.mul_constraints
				.iter()
				.map(|MulConstraint { a, b, hi, lo }| (a, b, lo, hi))
				.multiunzip();
			eval_op(
				&[a, b, lo, hi],
				intmul_r_x_prime_v,
				intmul_lambda_v,
				&shift_scalars,
				&r_y_tensor,
			)
		} else {
			E::zero()
		};

		bitand_part + intmul_part
	}
}

impl<F: BinaryField> FieldFn<F> for MonsterEvalFn<'_, F> {
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, vals: &[E]) -> E {
		self.evaluate(vals, evaluate_monster_multilinear_for_operation)
	}

	/// Native fast path: the per-operation evaluation defers `WideMul` reductions (see
	/// `evaluate_monster_multilinear_for_operation_native`).
	fn call_native(&self, vals: &[F]) -> F {
		self.evaluate(vals, evaluate_monster_multilinear_for_operation_native)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Field;
	use binius_math::test_utils::random_scalars;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::config::B128;

	#[test]
	fn test_evaluate_words_mle_matches_naive() {
		let mut rng = StdRng::seed_from_u64(0);
		let log_words = 3;
		// A non-power-of-two word count exercises the implicit zero padding.
		let words = (0..(1 << log_words) - 3)
			.map(|_| Word::from_u64(rng.random::<u64>()))
			.collect::<Vec<_>>();
		let r_j = random_scalars::<B128>(&mut rng, LOG_WORD_SIZE_BITS);
		let r_y = random_scalars::<B128>(&mut rng, log_words);

		// Naive reference: sum the full bit-level eq tensor over every set bit.
		let full_point = [r_j.clone(), r_y.clone()].concat();
		let full_tensor = eq_ind_partial_eval_scalars::<B128>(&full_point);
		let mut expected = B128::ZERO;
		for (word_index, word) in words.iter().enumerate() {
			for bit in 0..WORD_SIZE_BITS {
				if (word.as_u64() >> bit) & 1 == 1 {
					expected += full_tensor[(word_index << LOG_WORD_SIZE_BITS) | bit];
				}
			}
		}

		assert_eq!(evaluate_words_mle::<B128, B128>(&words, &r_j, &r_y), expected);
	}
}
