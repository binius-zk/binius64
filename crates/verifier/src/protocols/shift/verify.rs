// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::{
	constraint_system::{ConstraintSystem, InoutSegment},
	word::Word,
};
use binius_field::{BinaryField, field::FieldOps, util::FieldFn};
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{SumcheckOutput, verify as verify_sumcheck},
};
use binius_math::{
	BinarySubspace,
	inner_product::inner_product,
	line::extrapolate_line,
	multilinear::{
		eq::{
			eq_ind_partial_eval_scalars, eq_ind_zero, eq_one_var, scaled_eq_ind_partial_eval,
			scaled_eq_ind_partial_eval_scalars,
		},
		evaluate::evaluate_inplace_scalars,
	},
	univariate::EvaluationDomain,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use getset::Getters;
use itertools::chain;

use super::{
	LOG_SHIFT_COUNT, OperationEvalFn, SHIFT_COUNT, SHIFT_LOG_VARS, WiringWeights, error::Error,
	shift_ind::evaluate_shift_inds,
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

/// Each operation's row-variable count, in `[zero, bitand, intmul, binmul]` order.
///
/// An empty constraint array reads as zero variables.
fn log_rows(cs: &ConstraintSystem) -> [usize; 4] {
	[
		cs.log_zero_constraints(),
		cs.log_and_constraints(),
		cs.log_imul_constraints(),
		cs.log_bmul_constraints(),
	]
	.map(|log_rows| log_rows.unwrap_or(0))
}

/// The length of the constraint point `r_x`: the widest operation's row-variable count.
///
/// Every operation matrix is padded with empty rows up to `2^log_constraint_point(cs)` rows.
pub fn log_constraint_point(cs: &ConstraintSystem) -> usize {
	log_rows(cs).into_iter().max().unwrap_or(0)
}

/// Each operation's padding factor at the constraint point `r_x`, in `[zero, bitand, intmul,
/// binmul]` order.
///
/// The constraint reductions claim an operation of `2^n` rows at the prefix `r_x[..n]`. Its matrix,
/// padded with empty rows up to the length of `r_x`, evaluates at `r_x` to that claim times
/// `eq(0, r_x[n..])`: a padding row carries no operand term. Scaled by its factor, every
/// operation's claim is at the same full point, so one expansion of `r_x` serves them all.
///
/// # Soundness
///
/// A factor is zero only with negligible probability. A nonzero factor scales the claim and the
/// multilinear it is about by the same unit, which is a bijection.
///
/// # Preconditions
///
/// * `r_x` has [`log_constraint_point`] coordinates.
pub fn padding_scales<E: FieldOps>(cs: &ConstraintSystem, r_x: &[E]) -> [E; 4] {
	assert_eq!(r_x.len(), log_constraint_point(cs)); // precondition
	log_rows(cs).map(|log_rows| eq_ind_zero(&r_x[log_rows..]))
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
/// - `constraint_system`: The constraint system containing AND, IMUL and BMUL constraints
/// - `inout`: Which segment holds the inout values, which fixes where the two segments split
/// - `operand_claims`: One evaluation per operand column, in the flat column order [`WiringEvalFn`]
///   reads the constraint system in. The point they are claimed at is [`check_eval`]'s to read
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
	constraint_system: &ConstraintSystem,
	inout: InoutSegment,
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
	let log_word_count = constraint_system.log_segment_words(inout) + 1;

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
/// # Errors
///
/// - `Error::VerificationFailure` if the evaluation equation doesn't hold
/// - Propagates errors from reading the wiring evaluation off the channel
#[allow(clippy::too_many_arguments)]
pub fn check_eval<'a, F, C>(
	constraint_system: &'a ConstraintSystem,
	inout: InoutSegment,
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

	// The function the caller checks the claim with, and the flat input it reads. Every entry is a
	// public-channel-derived element (the batching challenge vector, the constraint point
	// `r_x`, both shift slots' challenges, `r_y`, and `r_segment` last); the constraint system it
	// sums over is fixed.
	let claim = {
		let r_x_len = r_x.len();
		let r_s_len = r_s_inner.len();
		let r_v_len = r_v_inner.len();
		let r_y_len = r_y.len();

		let inputs: Vec<C::Elem> = chain!(
			operand_batch_challenges,
			r_x,
			r_s_inner,
			r_v_inner,
			r_s_outer,
			r_v_outer,
			r_y,
			iter::once(r_segment),
		)
		.cloned()
		.collect();

		let eval_fn = WiringEvalFn::new(
			constraint_system,
			WiringEvalShape {
				inout,
				log_operands: operand_batch_challenges.len(),
				r_x_len,
				r_s_len,
				r_v_len,
				r_y_len,
			},
		);
		WiringEvalClaim {
			eval_fn,
			inputs,
			claimed: wiring_eval,
		}
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
/// constraint system. The holder discharges the claim by evaluating the function and requiring the
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
	pub eval_fn: WiringEvalFn<'a>,
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
	/// Discharging evaluates the wiring multilinear, which walks every constraint of the system.
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
			shape: self.eval_fn.shape(),
			inputs: self.inputs,
			claimed: self.claimed,
		}
	}
}

/// A wiring claim that was exported rather than discharged.
///
/// Holding one is owing a check.
///
/// Settling it needs only the constraint system the claim is about, which is public data.
/// So it can run far from the verifier that raised the claim.
///
/// ```text
///   verify -> claim -> export -> travels as public values -> settled natively at the root
/// ```
#[must_use = "a deferred wiring claim that nobody discharges is an unchecked constraint"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredWiringClaim<E> {
	/// How the flat input below splits back into its sections.
	pub shape: WiringEvalShape,
	/// The point the wiring multilinear is claimed to be evaluated at.
	pub inputs: Vec<E>,
	/// The evaluation the prover claims.
	pub claimed: E,
}

impl<F: BinaryField> DeferredWiringClaim<F> {
	/// Settles the claim against the constraint system it is about.
	///
	/// The system must be the one the claim was raised over.
	/// A different one names a different polynomial.
	/// The check would then be meaningless rather than wrong, so the caller owes that pairing.
	///
	/// # Errors
	///
	/// Returns an error when the evaluation disagrees with the claim.
	pub fn check(&self, constraint_system: &ConstraintSystem) -> Result<(), Error> {
		let eval_fn = WiringEvalFn::new(constraint_system, self.shape);
		if eval_fn.call_native(&self.inputs) == self.claimed {
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
		let wiring_eval = FieldFn::<F>::call::<E>(&eval_fn, &inputs);
		channel.assert_zero(wiring_eval - claimed)?;
		Ok(())
	}
}

/// The wiring multilinear evaluation, as a [`FieldFn`] over public-channel-derived inputs.
///
/// The inputs are the flat concatenation of these sections, in order:
///
/// ```text
/// operand_batch_challenges.. | r_x.. | r_s_inner.. | r_v_inner.. | r_s_outer.. | r_v_outer.. | r_y.. | r_segment
/// ```
///
/// The operand axis runs over the flat operand columns: the four operations' operands in
/// `[zero, bitand, intmul, binmul]` order, each operation's in its own operand order.
///
/// The stored lengths recover each variable-length section from that flat slice. Both shift slots
/// have the same shape, so one pair of lengths covers the two. `r_segment` is the single element
/// after them all: it is the top word-index coordinate, and it has no stored length because it is
/// always one element.
///
/// The bit-index factors every shift scalar is scaled by are left out: they depend on prover
/// messages, so [`check_eval`] multiplies them in outside.
#[derive(Debug)]
pub struct WiringEvalFn<'a> {
	/// The AND, IMUL and BMUL constraints whose monster multilinears are evaluated.
	constraint_system: &'a ConstraintSystem,
	/// How the flat input splits back into its sections.
	shape: WiringEvalShape,
}

/// How a wiring claim's flat input splits back into its sections.
///
/// Every length here is fixed by the constraint system and the reduction over it.
/// None is read off a prover message.
///
/// So one shape covers every proof of one shape, at no per-proof cost.
/// That is what lets a claim be settled away from the run that raised it.
///
/// The fields are private on purpose.
/// A hand-built shape could disagree with the inputs it reads.
/// The mismatch would surface as a wrong evaluation, not an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WiringEvalShape {
	/// Which segment holds the inout values, which fixes where the word-index tensor is cut.
	inout: InoutSegment,
	/// Length of the operand batch section: the base-2 logarithm of the padded operand axis.
	log_operands: usize,
	/// Length of the `r_x` section, the constraint point every padded operation matrix is read at:
	/// [`log_constraint_point`].
	r_x_len: usize,
	/// Length of each `r_s` section (one shift slot's amount challenges).
	r_s_len: usize,
	/// Length of each `r_v` section (one shift slot's variant challenges).
	r_v_len: usize,
	/// Length of the `r_y` section (the column challenges).
	r_y_len: usize,
}

impl<'a> WiringEvalFn<'a> {
	/// Rebuilds the evaluation over a constraint system, reading its input with the given shape.
	///
	/// This is how an exported claim is settled.
	/// The shape travels with the claim.
	/// The system it sums over is public.
	/// So no part of the original run is needed.
	pub const fn new(constraint_system: &'a ConstraintSystem, shape: WiringEvalShape) -> Self {
		Self {
			constraint_system,
			shape,
		}
	}

	/// How this evaluation reads its flat input.
	pub const fn shape(&self) -> WiringEvalShape {
		self.shape
	}
}

/// The weight tables [`WiringEvalFn::wiring_weights`] builds from its flat input slice.
///
/// The four operations read every table alike, so each is built once and lent to all four.
struct WiringInputs<E> {
	/// The equality indicator of `r_x`, one weight per row of the padded operation matrices.
	r_x_tensor: Vec<E>,
	/// The weight of each `(inner shift, operand column)` pair, at
	/// `(inner_shift << log_operands) | column`.
	operand_inner_shift_scalars: Vec<E>,
	/// The base-2 logarithm of the padded operand axis.
	log_operands: usize,
	/// The weight of each spelling the outer shift slot can take.
	outer_shift_scalars: Vec<E>,
	/// The word-index indicator over the public half of the value vector.
	public_tensor: Vec<E>,
	/// The word-index indicator over the hidden half of the value vector.
	hidden_tensor: Vec<E>,
}

impl<E> WiringInputs<E> {
	/// The tables as every operation's evaluation reads them, the value axis cut into one run per
	/// value segment, which is what an operand term's `(segment, index)` pair reads against.
	///
	/// The constants lead the public indicator and the private values trail the hidden one; the
	/// inout values follow whichever indicator they are placed in. The padding words between the
	/// runs are dropped, since no index can name one.
	fn weights(&self, cs: &ConstraintSystem, inout: InoutSegment) -> WiringWeights<'_, E> {
		let value = match inout {
			InoutSegment::Public => [
				&self.public_tensor[..cs.n_const()],
				&self.public_tensor[cs.offset_inout()..cs.offset_inout() + cs.n_inout],
				&self.hidden_tensor[..cs.n_private],
			],
			InoutSegment::Hidden => [
				&self.public_tensor[..cs.n_const()],
				&self.hidden_tensor[..cs.n_inout],
				&self.hidden_tensor[cs.n_inout..cs.n_inout + cs.n_private],
			],
		};
		WiringWeights {
			constraint: &self.r_x_tensor,
			inner_operand: &self.operand_inner_shift_scalars,
			log_operands: self.log_operands,
			outer: &self.outer_shift_scalars,
			value,
		}
	}
}

impl WiringEvalFn<'_> {
	/// Shared setup for [`FieldFn::call`] and [`FieldFn::call_native`]: splits the flat `vals`
	/// slice and builds one weight table per axis of each operation's wiring tensor.
	///
	/// The expansion is supplied by the caller:
	///
	/// - The word-index axis spans the whole trace, so it is the expansion worth threading.
	/// - Only a concrete field element can cross threads, so the generic path stays serial.
	///
	/// It is handed a scale as well as a point, since the word-index tables carry one.
	fn wiring_weights<E: FieldOps>(
		&self,
		vals: &[E],
		scaled_expand: impl Fn(&[E], E) -> Vec<E>,
	) -> WiringInputs<E> {
		// The `r_x` section spans the padded constraint count every operation's evaluation walks.
		debug_assert_eq!(self.shape.r_x_len, log_constraint_point(self.constraint_system));

		// Split the flat input back into its sections, in the order they were concatenated.
		let log_operands = self.shape.log_operands;
		let operand_batch_v = &vals[..log_operands];
		let mut off = log_operands;
		let r_x_v = &vals[off..off + self.shape.r_x_len];
		off += self.shape.r_x_len;
		let r_s_inner_v = &vals[off..off + self.shape.r_s_len];
		off += self.shape.r_s_len;
		let r_v_inner_v = &vals[off..off + self.shape.r_v_len];
		off += self.shape.r_v_len;
		let r_s_outer_v = &vals[off..off + self.shape.r_s_len];
		off += self.shape.r_s_len;
		let r_v_outer_v = &vals[off..off + self.shape.r_v_len];
		off += self.shape.r_v_len;
		let r_y_v = &vals[off..off + self.shape.r_y_len];
		off += self.shape.r_y_len;
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
		let cs = &self.constraint_system;
		let log_public_words = cs.log_public_words(self.shape.inout);

		let public_scale =
			eq_one_var(r_segment.clone(), E::zero()) * eq_ind_zero(&r_y_v[log_public_words..]);
		let public_tensor = scaled_expand(&r_y_v[..log_public_words], public_scale);
		let hidden_tensor = scaled_expand(r_y_v, r_segment);

		// A term's sequence selects itself through an equality indicator over both slots' axes.
		// The weight factorizes, so this is one table per slot rather than one over the whole
		// sequence space.
		//
		// A shift is indexed `variant * Word::BITS + amount`, the amount below the variant, so a
		// slot's table is the expansion of the amount challenges followed by the variant ones.
		// The operand batching weight rides below the shift in the inner table, since the
		// expansion puts the first segment of a point in the low index bits. It already weighs
		// every operand column of every operation, so no operation carries a scale of its own.
		//
		// Both are built once, so the Zero, BitAnd, IntMul and BinMul evaluations share them. The
		// bit-index factors scaling them are left for `check_eval` to multiply in.
		let operand_inner_shift_scalars =
			eq_ind_partial_eval_scalars(&[operand_batch_v, r_s_inner_v, r_v_inner_v].concat());
		let outer_shift_scalars = eq_ind_partial_eval_scalars(&[r_s_outer_v, r_v_outer_v].concat());
		debug_assert_eq!(outer_shift_scalars.len(), SHIFT_COUNT);

		// Every operation matrix is padded to the same rows, so one expansion of `r_x` serves all
		// four as well.
		let r_x_tensor = eq_ind_partial_eval_scalars(r_x_v);

		WiringInputs {
			r_x_tensor,
			operand_inner_shift_scalars,
			log_operands,
			outer_shift_scalars,
			public_tensor,
			hidden_tensor,
		}
	}
}

impl<F: BinaryField> FieldFn<F> for WiringEvalFn<'_> {
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, vals: &[E]) -> E {
		let inputs = self.wiring_weights(vals, scaled_eq_ind_partial_eval_scalars);
		let cs = self.constraint_system;
		let weights = inputs.weights(cs, self.shape.inout);

		// The operations take their flat operand columns in `[zero, bitand, intmul, binmul]` order,
		// the order the reduction hands [`verify`] their claims in.
		let zero = OperationEvalFn::new(&cs.zero_constraints, 0);
		let bitand = OperationEvalFn::new(&cs.and_constraints, zero.end_column());
		let intmul = OperationEvalFn::new(&cs.imul_constraints, bitand.end_column());
		let binmul = OperationEvalFn::new(&cs.bmul_constraints, intmul.end_column());
		debug_assert!(binmul.end_column() <= 1 << self.shape.log_operands);
		zero.call(weights) + bitand.call(weights) + intmul.call(weights) + binmul.call(weights)
	}

	/// Native fast path: each operation's evaluation defers `WideMul` reductions (see
	/// [`OperationEvalFn`]'s `call_native`).
	fn call_native(&self, vals: &[F]) -> F {
		let inputs = self.wiring_weights(vals, |point, scale| {
			// The packed expansion threads the tensor's multiplications.
			// It applies over the base field, which is its own single-element packing.
			scaled_eq_ind_partial_eval::<F>(point, scale).into_inner()
		});
		let cs = self.constraint_system;
		let weights = inputs.weights(cs, self.shape.inout);

		// The operations take their flat operand columns in `[zero, bitand, intmul, binmul]` order,
		// the order the reduction hands [`verify`] their claims in.
		let zero = OperationEvalFn::new(&cs.zero_constraints, 0);
		let bitand = OperationEvalFn::new(&cs.and_constraints, zero.end_column());
		let intmul = OperationEvalFn::new(&cs.imul_constraints, bitand.end_column());
		let binmul = OperationEvalFn::new(&cs.bmul_constraints, intmul.end_column());
		debug_assert!(binmul.end_column() <= 1 << self.shape.log_operands);
		zero.call_native(weights)
			+ bitand.call_native(weights)
			+ intmul.call_native(weights)
			+ binmul.call_native(weights)
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_core::constraint_system::{
		AndConstraint, BmulConstraint, Shift, ShiftedValueIndex, ValueIndex, ZeroConstraint,
	};
	use binius_field::Field;
	use binius_math::test_utils::random_scalars;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::{super::LOG_SHIFT_VARIANT_COUNT, *};
	use crate::config::B128;

	/// A system with constraints in three of the four operations.
	///
	/// IMUL is left empty, so the columns after it only line up if its run is still skipped.
	fn three_operation_system(rng: &mut StdRng) -> ConstraintSystem {
		let n_private = 12usize;
		let term = |rng: &mut StdRng| {
			ShiftedValueIndex::new(
				ValueIndex::private(rng.random_range(0..n_private) as u32),
				[
					Shift::srl(rng.random_range(0..Word::BITS)),
					Shift::srl(rng.random_range(0..Word::BITS)),
				],
			)
		};

		ConstraintSystem {
			constants: vec![Word::ZERO, Word::ONE],
			n_inout: 4,
			n_private,
			// A handful of each, so every present operation's tensor carries entries.
			zero_constraints: (0..5)
				.map(|_| ZeroConstraint(array::from_fn(|_| vec![term(rng)])))
				.collect(),
			and_constraints: (0..7)
				.map(|_| AndConstraint(array::from_fn(|_| vec![term(rng), term(rng)])))
				.collect(),
			imul_constraints: Vec::new(),
			bmul_constraints: (0..3)
				.map(|_| BmulConstraint(array::from_fn(|_| vec![term(rng)])))
				.collect(),
		}
	}

	/// The flat input, cut back into a weight table per axis.
	///
	/// This reverses the layout [`WiringEvalFn`] documents, and it reads nothing but the input.
	struct RebuiltWeights {
		constraint: Vec<B128>,
		inner_operand: Vec<B128>,
		outer: Vec<B128>,
		/// One entry per padded value address: the word index under the segment bit.
		value: Vec<B128>,
	}

	impl RebuiltWeights {
		/// Cuts the input into runs and expands each into the table its axis is read with.
		fn cut(vals: &[B128], shape: WiringEvalShape) -> Self {
			let mut off = 0;
			let mut take = |len: usize| {
				let run = &vals[off..off + len];
				off += len;
				run
			};
			let operand = take(shape.log_operands);
			let r_x = take(shape.r_x_len);
			let r_s_inner = take(shape.r_s_len);
			let r_v_inner = take(shape.r_v_len);
			let r_s_outer = take(shape.r_s_len);
			let r_v_outer = take(shape.r_v_len);
			let r_y = take(shape.r_y_len);
			let r_segment = take(1)[0];
			assert_eq!(off, vals.len(), "the input holds exactly its documented runs");

			// The value axis is the padded address space, so one flat indicator covers it whole.
			//
			// Building it that way rather than as two scaled half-tensors is what holds the
			// padded reading against the cut the evaluation walks.
			let mut value_point = r_y.to_vec();
			value_point.push(r_segment);

			Self {
				constraint: eq_ind_partial_eval_scalars(r_x),
				inner_operand: eq_ind_partial_eval_scalars(
					&[operand, r_s_inner, r_v_inner].concat(),
				),
				outer: eq_ind_partial_eval_scalars(&[r_s_outer, r_v_outer].concat()),
				value: eq_ind_partial_eval_scalars(&value_point),
			}
		}

		/// The tables as one operand term reads them, the value axis in its three runs.
		///
		/// The runs are constants, then inout, then private, sliced out of the one padded space.
		fn weights(
			&self,
			cs: &ConstraintSystem,
			shape: WiringEvalShape,
		) -> WiringWeights<'_, B128> {
			let hidden = 1 << shape.r_y_len;
			let value = match shape.inout {
				InoutSegment::Public => [
					&self.value[..cs.n_const()],
					&self.value[cs.offset_inout()..cs.offset_inout() + cs.n_inout],
					&self.value[hidden..hidden + cs.n_private],
				],
				InoutSegment::Hidden => [
					&self.value[..cs.n_const()],
					&self.value[hidden..hidden + cs.n_inout],
					&self.value[hidden + cs.n_inout..hidden + cs.n_inout + cs.n_private],
				],
			};
			WiringWeights {
				constraint: &self.constraint,
				inner_operand: &self.inner_operand,
				log_operands: shape.log_operands,
				outer: &self.outer,
				value,
			}
		}
	}

	#[test]
	fn the_wiring_evaluation_reads_one_operand_axis_across_the_operations() {
		// Invariant: the evaluation is the wiring tensor contracted against the flat input's
		// tables, each operation's operands at its run of the one operand axis:
		//
		//     zero  column 0 | bitand  1..4 | intmul  4..8 | binmul  8..14
		//
		// The flat entry walk is a different code path from the grouped evaluation, and the
		// columns are named here rather than chained, so agreeing is not a tautology.
		let mut rng = StdRng::seed_from_u64(31);
		let cs = three_operation_system(&mut rng);

		for inout in [InoutSegment::Public, InoutSegment::Hidden] {
			let shape = WiringEvalShape {
				inout,
				log_operands: 4,
				r_x_len: log_constraint_point(&cs),
				r_s_len: Word::LOG_BITS,
				r_v_len: LOG_SHIFT_VARIANT_COUNT,
				r_y_len: cs.log_segment_words(inout),
			};
			let eval_fn = WiringEvalFn::new(&cs, shape);

			// The flat input holds every section the shape describes, and the segment challenge
			// last.
			let n_inputs =
				shape.log_operands
					+ shape.r_x_len + 2 * (shape.r_s_len + shape.r_v_len)
					+ shape.r_y_len + 1;
			let vals = random_scalars::<B128>(&mut rng, n_inputs);

			let whole = FieldFn::<B128>::call::<B128>(&eval_fn, &vals);
			assert_eq!(
				whole,
				FieldFn::<B128>::call_native(&eval_fn, &vals),
				"the generic and native discharges must agree"
			);
			// A vacuous claim would let a broken layout pass unnoticed.
			assert_ne!(whole, B128::ZERO);

			let rebuilt = RebuiltWeights::cut(&vals, shape);
			let weights = rebuilt.weights(&cs, shape);
			let contracted = OperationEvalFn::new(&cs.zero_constraints, 0).contract(weights)
				+ OperationEvalFn::new(&cs.and_constraints, 1).contract(weights)
				+ OperationEvalFn::new(&cs.imul_constraints, 4).contract(weights)
				+ OperationEvalFn::new(&cs.bmul_constraints, 8).contract(weights);
			assert_eq!(contracted, whole, "{inout:?}");
		}
	}

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
