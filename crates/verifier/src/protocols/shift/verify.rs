// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::word::Word;
use binius_field::{BinaryField, field::FieldOps, util::FieldFn};
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{SumcheckOutput, verify as verify_sumcheck},
};
use binius_math::{
	BinarySubspace,
	inner_product::inner_product,
	line::extrapolate_line,
	multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate_inplace_scalars},
	univariate::EvaluationDomain,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use getset::Getters;
use itertools::chain;

use super::{
	LOG_SHIFT_COUNT, SHIFT_LOG_VARS, WiringInfo, error::Error, shift_ind::evaluate_shift_inds,
};

/// Evaluates the bit-level multilinear extension of a word slice at the point `r_j ++ r_y`.
///
/// The multilinear has `Word::LOG_BITS + r_y.len()` variables: the low variables index the
/// bit within a word and the high variables index the word. Words past `words.len()` (up to
/// `2^r_y.len()`) are treated as zero.
///
/// ## Preconditions
///
/// * `r_j` has exactly `Word::LOG_BITS` entries
/// * `words` has at most `2^r_y.len()` entries
pub fn evaluate_words_mle<F, E>(words: &[Word], r_j: &[E], r_y: &[E]) -> E
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
{
	assert_eq!(r_j.len(), Word::LOG_BITS); // precondition
	assert!(words.len() <= 1 << r_y.len()); // precondition

	let r_j_tensor = eq_ind_partial_eval_scalars(r_j);
	let r_y_tensor = eq_ind_partial_eval_scalars(r_y);
	iter::zip(words, r_y_tensor)
		.map(|(word, weight)| {
			let word_eval = (0..Word::BITS)
				.filter(|bit| (word.as_u64() >> bit) & 1 == 1)
				.map(|bit| &r_j_tensor[bit])
				.sum::<E>();
			weight * word_eval
		})
		.sum()
}

/// Output of the shift reduction verification protocol.
///
/// Contains all the challenge points, evaluation claims, and random coefficients
/// produced during the shift reduction protocol. These values are used for subsequent
/// verification steps including PCS verification.
#[derive(Debug, Getters)]
pub struct VerifyOutput<F> {
	/// The challenges whose equality indicator weights each operand column in the batch (length
	/// `log2_ceil` of the column count).
	operand_batch_challenges: Vec<F>,
	/// Challenge point for the witness bit index (length `Word::LOG_BITS`).
	pub r_j: Vec<F>,
	/// Challenge point for the inner shift's amount variables (length `Word::LOG_BITS`).
	pub r_s_inner: Vec<F>,
	/// Challenge point for the inner shift's variant variables (length
	/// `LOG_SHIFT_VARIANT_COUNT`).
	pub r_v_inner: Vec<F>,
	/// Challenge point for the outer shift's amount variables (length `Word::LOG_BITS`).
	pub r_s_outer: Vec<F>,
	/// Challenge point for the outer shift's variant variables (length
	/// `LOG_SHIFT_VARIANT_COUNT`).
	pub r_v_outer: Vec<F>,
	/// Challenge point for the word index variables (length `log_segment_words`).
	pub r_y: Vec<F>,
	/// Challenge point for the bit index of the intermediate word, where the two shift indicators
	/// meet (length `Word::LOG_BITS`).
	pub r_k: Vec<F>,
	/// Challenge point for the output bit index the oblong weights attach to (length
	/// `Word::LOG_BITS`).
	pub r_i: Vec<F>,
	/// Challenge for the witness's segment selector variable.
	pub r_segment: F,
	/// Final evaluation claim from the sumcheck.
	eval: F,
	/// The claimed witness evaluation at the challenge point.
	#[getset(get = "pub")]
	pub witness_eval: F,
}

impl<F> VerifyOutput<F> {
	/// Returns the challenge point for bit index variables.
	///
	/// This corresponds to the first `Word::LOG_BITS` variables
	/// in the witness encoding, indexing individual bits within words.
	pub fn r_j(&self) -> &[F] {
		&self.r_j
	}

	/// Returns the challenge point for word index variables.
	///
	/// This corresponds to `log_word_count` variables indexing
	/// the words in the witness vector.
	pub fn r_y(&self) -> &[F] {
		&self.r_y
	}
}

/// Verifies the shift protocol with a single sumcheck.
///
/// # Protocol Overview
/// 1. **Sampling Phase**: Samples the challenge vector whose equality indicator batches the operand
///    columns' evaluation claims.
/// 2. **Sumcheck**: Verifies the batched evaluation claim over all `SHIFT_LOG_VARS +
///    log_word_count` variables of the claim, degree 2 in each. A shifted value index names two
///    shifts applied in sequence, and the rounds peel them from the output end inward: the outer
///    shift's variant and amount, then the inner shift's, then the bit position within a word, then
///    the intermediate word's bit index where the two shift indicators meet, then the output bit
///    index, and last the word index — the order the prover's phases need.
/// 3. **Challenge Splitting**: Splits the challenge point into its per-slot shift runs, its three
///    bit-index runs and `r_y`
/// 4. **Monster Multilinear Verification**: Checks that the claim the sumcheck reduced to matches
///    the product of its five factors, for AND constraints (bitand), IMUL constraints (intmul) and
///    BMUL constraints (binmul)
///
/// # Parameters
/// - `log_segment_words`: The word-index variables each value segment spans, the wider of the two
/// - `operand_claims`: One evaluation per operand column, in the flat column order [`WiringInfo`]
///   lays the terms out in. The point they are claimed at is [`check_eval`]'s to read
/// - `channel`: Interactive channel for challenge sampling and message reading
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
	log_segment_words: usize,
	operand_claims: &[C::Elem],
	channel: &mut C,
) -> Result<VerifyOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
{
	// SOUNDNESS: the prover draws the same number of challenges at the same place.
	let operand_batch_challenges = channel.sample_many(log2_ceil_usize(operand_claims.len()));

	// A claim is weighted by the equality indicator of the operand axis at its column. The axis is
	// padded to a cube; the columns past the last claim name nothing.
	let operand_weights = eq_ind_partial_eval_scalars(&operand_batch_challenges);
	let eval = inner_product(
		operand_claims.iter().cloned(),
		operand_weights[..operand_claims.len()].iter().cloned(),
	);

	// The sumcheck runs over the witness as well: the public segment in the low half-cube and
	// the hidden segment in the high half-cube, selected by the top word-index variable. Each
	// half spans the wider of the two segments, which the prover zero-pads the shorter one up
	// to, so a public segment longer than the hidden one draws the extra word-index challenges.
	let log_word_count = log_segment_words + 1;

	let SumcheckOutput {
		eval,
		challenges: mut point,
	} = verify_sumcheck(SHIFT_LOG_VARS + log_word_count, 2, eval, channel)?;

	// Reverse the challenges into the evaluation point, whose coordinates then run in increasing
	// order of significance: the word index, the output bit index, the intermediate bit index, the
	// witness bit index, then the inner shift slot and the outer one. The rounds bind them in the
	// opposite order — the outer shift first, the word index last — which is what admits the
	// prover's phases, and which peels the two shifts from the output end inward.
	point.reverse();
	debug_assert_eq!(point.len(), SHIFT_LOG_VARS + log_word_count);
	// Where each run starts, counting up from the word index. `split_off` cuts from the top, so
	// the runs come off in decreasing significance.
	let bit_indices = log_word_count + Word::LOG_BITS * 3;
	let inner_slot = bit_indices + LOG_SHIFT_COUNT;
	let r_v_outer = point.split_off(inner_slot + Word::LOG_BITS);
	let r_s_outer = point.split_off(inner_slot);
	let r_v_inner = point.split_off(bit_indices + Word::LOG_BITS);
	let r_s_inner = point.split_off(bit_indices);
	let r_j = point.split_off(log_word_count + Word::LOG_BITS * 2);
	let r_k = point.split_off(log_word_count + Word::LOG_BITS);
	let r_i = point.split_off(log_word_count);
	let mut r_y = point;
	let r_segment = r_y.pop().expect("log_word_count >= 1");

	let witness_eval = channel.recv_one()?;

	Ok(VerifyOutput {
		operand_batch_challenges,
		r_j,
		r_y,
		r_segment,
		r_s_inner,
		r_v_inner,
		r_s_outer,
		r_v_outer,
		r_k,
		r_i,
		eval,
		witness_eval,
	})
}

/// Validates the evaluation claims from the shift reduction protocol.
///
/// After the shift reduction protocol completes, this function checks that the
/// prover-provided witness evaluation is consistent with the expected values.
/// It reads the wiring multilinear's evaluation from the prover and verifies the final equation
/// relating the witness and monster evaluations.
///
/// # Protocol Details
///
/// The function verifies that:
/// ```text
/// eval = trace_eval * monster_eval
/// ```
///
/// where `monster_eval` is the prover's claimed wiring evaluation — the AND, IMUL and BMUL
/// constraint polynomials summed — scaled by the sumcheck's two bit-index factors, the Lagrange
/// weights and the interpolated shift indicators, both at `r_i`.
///
/// That claim is not checked here. It comes back as a [`WiringEvalClaim`], holding the function
/// that evaluates the wiring multilinear from public-channel-derived values together with the
/// claimed value it must equal, for the caller to discharge however it opens claims.
///
/// `trace_eval` is the witness evaluation reconstructed from its two segments:
/// ```text
/// trace_eval = (1 - r_segment) * public_eval + r_segment * witness_eval
/// ```
///
/// `public_eval` is the public segment over the shift's whole index space — `r_j` over the bit
/// within a word and all of `r_y` over the word — so it already carries the zero-padding above the
/// segment's own length. Tying it to the public values is the caller's job: the caller reads it
/// from the prover and reduces it onto the packed public segment, so a prover that used different
/// public values fails there rather than here.
///
/// `r_x` is the constraint point every padded operation matrix is claimed at. Only the point enters
/// here; the evaluation claims at it are [`verify`]'s to batch.
///
/// `wiring` is the wiring multilinear the returned claim is about.
///
/// # Panics
///
/// Panics if `wiring` is laid out over a different point than `output` and `r_x` form, since it
/// would then name a different polynomial.
///
/// # Errors
///
/// - `Error::VerificationFailure` if the evaluation equation doesn't hold
/// - Propagates errors from reading the wiring evaluation off the channel
#[allow(clippy::too_many_arguments)]
pub fn check_eval<'a, F, C>(
	wiring: &'a WiringInfo,
	public_eval: C::Elem,
	r_x: &[C::Elem],
	subspace: &BinarySubspace<F>,
	r_zhat_prime: &C::Elem,
	output: &VerifyOutput<C::Elem>,
	channel: &mut C,
) -> Result<WiringEvalClaim<'a, C::Elem>, Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	let VerifyOutput {
		operand_batch_challenges,
		eval,
		r_j,
		r_s_inner,
		r_v_inner,
		r_s_outer,
		r_v_outer,
		r_y,
		r_segment,
		r_k,
		r_i,
		witness_eval,
	} = output;

	// Three of the sumcheck's five factors are the verifier's to evaluate from the bit-index
	// challenges alone: the Lagrange weights of the univariate challenge at `r_i`, and the two
	// shift indicators, one per slot of a term's shift sequence. The indicators chain through the
	// intermediate word — the outer one carries the output bit down to `r_k`, the inner one carries
	// `r_k` down to the witness bit — which is what makes a sequence of two shifts one index entry.
	let l_tilde_eval = evaluate_inplace_scalars(subspace.lagrange_evals(r_zhat_prime), r_i);
	let outer_ind_eval =
		evaluate_inplace_scalars(&mut evaluate_shift_inds(r_i, r_k, r_s_outer)[..], r_v_outer);
	let inner_ind_eval =
		evaluate_inplace_scalars(&mut evaluate_shift_inds(r_k, r_j, r_s_inner)[..], r_v_inner);
	let shift_ind_eval = outer_ind_eval * inner_ind_eval;

	// The wiring multilinear's evaluation comes from the prover, as a claim the verifier could
	// compute for itself. Checking it against the constraint system is left to the caller, which is
	// handed the function that computes it below.
	let wiring_eval = channel.recv_public_claim()?;

	// The three bit-index factors scale every shift scalar of the wiring multilinear; they multiply
	// the claim out here rather than entering the function, which keeps its input free of `r_i` and
	// `r_k`.
	let monster_eval = l_tilde_eval * shift_ind_eval * wiring_eval.clone();

	// The flat input the caller checks the claim with. Every entry is a public-channel-derived
	// element: `r_segment`, then the point the hidden vector is read at, whose bit order is the
	// order `WiringInfo` lays a term's address out in.
	assert_eq!(r_y.len(), wiring.log_segment_words());
	assert_eq!(r_x.len(), wiring.log_constraint_point());
	let inputs: Vec<C::Elem> = chain!(
		iter::once(r_segment),
		operand_batch_challenges,
		r_s_inner,
		r_v_inner,
		r_s_outer,
		r_v_outer,
		r_y,
		r_x,
	)
	.cloned()
	.collect();
	assert_eq!(inputs.len(), 1 + wiring.hidden_segment().log_len());
	let claim = WiringEvalClaim {
		eval_fn: wiring,
		inputs,
		claimed: wiring_eval,
	};

	// Reconstruct the witness evaluation from its two segments.
	let trace_eval = extrapolate_line(public_eval, witness_eval.clone(), r_segment.clone());

	// Check if the reconstructed trace value is satisfying.
	//
	// The protocol could compute the committed-half value instead of reading it from the prover.
	// This would require inverting a random element, however, making the protocol incomplete
	// with negligible probability. As a matter of taste, we read the value from the prover.
	let expected_eval = trace_eval * monster_eval;
	channel.assert_zero(expected_eval - eval)?;

	Ok(claim)
}

/// The prover's wiring multilinear evaluation, with what it takes to check it.
///
/// [`check_eval`] reads the evaluation from the prover and closes the shift reduction with it,
/// leaving this behind: the claimed value, and the function and inputs that recompute it from the
/// wiring matrix. The holder discharges the claim by evaluating the function and requiring the
/// two to agree.
///
/// Both discharges below evaluate the same function; they differ in where. A verifier holding
/// values checks it in the field, and one building a circuit checks it in constraints — which is
/// why the function is kept rather than a value, and why the claimed value sits beside it rather
/// than folded into it.
///
/// Dropping a claim drops a check, so it is `#[must_use]`.
#[must_use]
#[derive(Debug)]
pub struct WiringEvalClaim<'a, E> {
	/// Evaluates the wiring multilinear from `inputs`.
	pub eval_fn: &'a WiringInfo,
	/// The flat input `eval_fn` reads.
	pub inputs: Vec<E>,
	/// The evaluation the prover claims, which `eval_fn` must return.
	pub claimed: E,
}

impl<F: BinaryField> WiringEvalClaim<'_, F> {
	/// Discharges the claim in the field: evaluates the wiring multilinear and compares.
	///
	/// This is the discharge for a verifier holding values rather than wires, and it takes
	/// [`FieldFn::call_native`]'s accelerated path.
	pub fn check_native(self) -> Result<(), Error> {
		if self.eval_fn.call_native(&self.inputs) == self.claimed {
			Ok(())
		} else {
			Err(Error::VerificationFailure)
		}
	}
}

impl<'a, E> WiringEvalClaim<'a, E> {
	/// Exports the claim instead of discharging it.
	///
	/// Discharging evaluates the wiring multilinear, which reads every operand term of the system.
	/// Inside a circuit that cost tracks the inner system.
	/// So a circuit that pays it can never verify a proof of itself.
	///
	/// Exporting hands the claim out whole instead.
	/// It is settled once, natively, where the cost is ordinary.
	///
	/// The claim is short.
	/// Every section of its input is a challenge vector over a padded log-sized index.
	/// So the input length is logarithmic in the constraint count.
	///
	/// # Correctness
	///
	/// Nothing is verified here.
	/// Nothing is verified later either, unless a holder settles the claim.
	/// A dropped claim is an unchecked constraint.
	pub fn defer(self) -> DeferredWiringClaim<E> {
		DeferredWiringClaim {
			inputs: self.inputs,
			claimed: self.claimed,
		}
	}
}

/// A wiring claim that was exported rather than discharged.
///
/// Holding one is owing a check.
///
/// Settling it needs only the [`WiringInfo`] the claim is about, which is public data.
/// So it can run far from the verifier that raised the claim.
///
/// ```text
///   verify -> claim -> export -> travels as public values -> settled natively at the root
/// ```
#[must_use = "a deferred wiring claim that nobody discharges is an unchecked constraint"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredWiringClaim<E> {
	/// The point the wiring multilinear is claimed to be evaluated at.
	pub inputs: Vec<E>,
	/// The evaluation the prover claims.
	pub claimed: E,
}

impl<F: BinaryField> DeferredWiringClaim<F> {
	/// Settles the claim against the wiring matrix it is about.
	///
	/// The matrix must be the one the claim was raised over.
	/// A different one names a different polynomial.
	/// The check would then be meaningless rather than wrong, so the caller owes that pairing.
	///
	/// # Errors
	///
	/// Returns an error when the evaluation disagrees with the claim.
	pub fn check(&self, wiring: &WiringInfo) -> Result<(), Error> {
		if wiring.call_native(&self.inputs) == self.claimed {
			Ok(())
		} else {
			Err(Error::VerificationFailure)
		}
	}
}

impl<E> WiringEvalClaim<'_, E> {
	/// Discharges the claim over `channel`'s elements: evaluates the wiring multilinear there and
	/// asserts it equals the claimed value.
	///
	/// This is the discharge for a channel carrying elements as wires, where the evaluation becomes
	/// a sub-circuit and the comparison an assertion within it. A holder with another way to open a
	/// claim — a sparse-polynomial argument, say — reads the fields instead.
	pub fn check_symbolic<F, C>(self, channel: &mut C) -> Result<(), Error>
	where
		F: BinaryField,
		C: IPVerifierChannel<F, Elem = E>,
		E: FieldOps<Scalar = F> + From<F>,
	{
		let Self {
			eval_fn,
			inputs,
			claimed,
		} = self;
		let wiring_eval = FieldFn::<F>::call::<E>(eval_fn, &inputs);
		channel.assert_zero(wiring_eval - claimed)?;
		Ok(())
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
			.map(|_| Word::from_u64(rng.random()))
			.collect::<Vec<_>>();
		let r_j = random_scalars::<B128>(&mut rng, Word::LOG_BITS);
		let r_y = random_scalars::<B128>(&mut rng, log_words);

		// Naive reference: sum the full bit-level eq tensor over every set bit.
		let full_point = [r_j.clone(), r_y.clone()].concat();
		let full_tensor = eq_ind_partial_eval_scalars(&full_point);
		let mut expected = B128::ZERO;
		for (word_index, word) in words.iter().enumerate() {
			for bit in 0..Word::BITS {
				if (word.as_u64() >> bit) & 1 == 1 {
					expected += full_tensor[(word_index << Word::LOG_BITS) | bit];
				}
			}
		}

		assert_eq!(evaluate_words_mle::<B128, B128>(&words, &r_j, &r_y), expected);
	}
}
