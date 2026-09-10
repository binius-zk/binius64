// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The constraint reduction shared by the single-instance and batched Binius64 verifiers.
//!
//! One circuit, and `2^k` instances of one circuit, reduce the same way. The instance count joins
//! every row count, instance index low, constraint index high. One circuit is the batch with
//! `k = 0`.
//!
//! The BitAnd sumcheck carries the IntMul and BinMul operand claims, each zero-padded on its high
//! end. So it ends at one point, and every operation is claimed at a prefix of it:
//!
//! ```text
//! r_rho      the instance, shared by every operation
//! r_x_star   the constraint; each operation reads the prefix its row count spans
//! ```

use std::iter;

use binius_core::{
	constraint_system::{ConstraintSystem, InoutSegment},
	word::Word,
};
use binius_field::{ExtensionField, FieldOps, Rijndael8b as B8};
use binius_iop::channel::IOPVerifierChannel;
use binius_ip::{
	channel::WordIPVerifierChannel,
	sumcheck::{SumcheckOutput, verify as verify_sumcheck},
};
use binius_math::{
	BinarySubspace,
	multilinear::{eq::eq_ind_zero, evaluate::evaluate_inplace_scalars},
};

use crate::{
	Error,
	config::{B1, B128, LOG_WORDS_PER_ELEM},
	protocols::{
		binmul::{BinMulOutput, verify as verify_binmul_reduction},
		bitand::AndCheckOutput,
		intmul::{IntMulOutput, verify as verify_intmul_reduction},
		rerand::RerandOutput,
		shift::{self, BINMUL_ARITY, INTMUL_ARITY, WiringEvalClaim},
		zero,
	},
	ring_switch::{self, RingSwitchVerifyOutput, eval_rs_eq},
	verify_bitand_reduction,
};

/// What [`reduce_constraints`] leaves for the caller: the claim on the committed trace, and the
/// wiring claim the constraint system is read through.
#[derive(Debug)]
pub struct ReductionOutput<'a, F> {
	/// The instance point every operation is claimed at.
	///
	/// Empty for a single circuit.
	pub r_rho: Vec<F>,
	/// The shift reduction's output, holding the claimed witness evaluation.
	pub shift: shift::VerifyOutput<F>,
	/// The prover's wiring evaluation, still to be tied to the constraint system.
	pub wiring: WiringEvalClaim<'a, F>,
}

impl<F: Clone> ReductionOutput<'_, F> {
	/// The point the committed trace is opened at.
	///
	/// The trace's bit index is `[bit | instance | wire]`, low to high:
	///
	/// ```text
	/// r_j     the bit within a word
	/// r_rho   the instance, empty for a single circuit
	/// r_y     the committed word
	/// ```
	///
	/// Evaluating the instance coordinates at `r_rho` folds the trace over the batch.
	/// That fold is what the reduction's witness claim is about.
	pub fn trace_point(&self) -> Vec<F> {
		[self.shift.r_j(), &self.r_rho, self.shift.r_y()].concat()
	}
}

/// Reduces every constraint of `2^log_instances` instances to one claim on the committed trace.
///
/// The reductions run in this order, and the order is load-bearing:
///
/// ```text
/// IntMul -> BinMul -> BitAnd -> Zero -> shift -> public check
/// ```
///
/// # Arguments
///
/// - `cs`: the single-instance constraint system every instance satisfies.
/// - `log_instances`: the base-2 logarithm of the instance count, 0 for a single circuit.
/// - `inout`: which value segment the inout words sit in.
/// - `public`: the declared public values as the channel carries them, unpadded — the constants,
///   then the inout values.
/// - `channel`: the verifier channel that reads messages and redraws Fiat-Shamir challenges.
///
/// # Errors
///
/// Returns an error if any reduction's sumcheck or final consistency check fails.
///
/// # Soundness
///
/// IntMul and BinMul must run *before* BitAnd.
/// BitAnd draws the univariate challenge that collapses their per-bit operand evaluations.
/// Committing those evaluations first stops a prover choosing them as a function of it.
///
/// Do not reorder these, and keep the same order in the prover.
pub fn reduce_constraints<'a, Channel>(
	cs: &'a ConstraintSystem,
	log_instances: usize,
	inout: InoutSegment,
	public: &[Channel::Word],
	channel: &mut Channel,
) -> Result<ReductionOutput<'a, Channel::Elem>, Error>
where
	Channel: IOPVerifierChannel<B128> + WordIPVerifierChannel<B128>,
	Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
{
	// One base domain shared by the AND-check, the shift, and the operand collapse.
	// The AND-check's univariate-skip domain spans one dimension above the 64-bit word.
	let andcheck_domain = BinarySubspace::<B8>::default()
		.isomorphic::<B128>()
		.reduce_dim(Word::LOG_BITS + 1);
	// The shift domain drops that extra dimension.
	let shift_domain = andcheck_domain.reduce_dim(Word::LOG_BITS);

	// The multiplication columns span every instance's constraints.
	// So each check runs over `log_instances` more row variables than one instance has.
	let intmul_output = match cs.log_imul_constraints() {
		Some(log_n_imul) => {
			let _guard = tracing::info_span!(
				"[phase] Verify IntMul Reduction",
				phase = "verify_intmul_reduction",
				perfetto_category = "phase",
				n_constraints = cs.n_imul_constraints()
			)
			.entered();
			Some(verify_intmul_reduction::<B128, _>(log_instances + log_n_imul, channel)?)
		}
		// An empty IMUL set skips the reduction entirely, reading nothing from the transcript.
		// The prover carries the identical guard, so the two stay in sync.
		None => None,
	};

	let binmul_output = match cs.log_bmul_constraints() {
		Some(log_n_bmul) => {
			let _guard = tracing::info_span!(
				"[phase] Verify BinMul Reduction",
				phase = "verify_binmul_reduction",
				perfetto_category = "phase",
				n_constraints = cs.n_bmul_constraints()
			)
			.entered();
			Some(verify_binmul_reduction::<B128, _>(log_instances + log_n_bmul, channel)?)
		}
		None => None,
	};

	// The BitAnd sumcheck carries the multiplications' per-bit operand claims, IntMul first.
	let operands = [
		intmul_output.as_ref().map(IntMulOutput::operand_claims),
		binmul_output.as_ref().map(BinMulOutput::operand_claims),
	]
	.into_iter()
	.flatten()
	.collect::<Vec<_>>();

	// The BitAnd check has no skip branch: an empty AND set still reduces, over the single all-zero
	// padding row, so `None` is zero constraint variables.
	let log_n_and = cs.log_and_constraints().unwrap_or(0);
	let AndCheckOutput {
		z_challenge,
		rerand: RerandOutput {
			eval_point,
			bitand_evals,
			operand_evals,
		},
	} = {
		let _guard = tracing::info_span!(
			"[phase] Verify BitAnd Reduction",
			phase = "verify_bitand_reduction",
			perfetto_category = "phase",
			n_constraints = cs.n_and_constraints()
		)
		.entered();
		verify_bitand_reduction(log_instances + log_n_and, &andcheck_domain, &operands, channel)?
	};

	// The sumcheck's point is `r_rho || r_x_star`: the instance index low, the constraint index
	// high. `r_x_star` spans the widest AND, IMUL and BMUL set.
	let (r_rho, r_x_star) = eval_point.split_at(log_instances);

	// The Zero reduction reads nothing and runs no sumcheck.
	// A ZERO constraint is linear, so its oblong form vanishing at one unpredictable point
	// certifies it.
	//
	// A ZERO array vanishes identically, so its claim is zero at any point. A ZERO set wider than
	// the rest extends `r_x_star` with fresh challenges, and the prover draws the same extension
	// at the same place. The result is the one constraint point `r_x` every operation is claimed at
	// a prefix of.
	let log_n_zero = cs.log_zero_constraints().unwrap_or(0);
	let r_x_len = r_x_star.len().max(log_n_zero);
	let r_x = zero::reduction_point(r_x_star, r_x_len, || channel.sample());

	// The four operations' claims, in the order the shift reduction batches them.
	let mut operand_evals = operand_evals.into_iter();
	let claims = shift::OperationClaims {
		r_x,
		evals: [
			vec![Channel::Elem::zero()],
			bitand_evals.to_vec(),
			absent_or_claimed::<_, INTMUL_ARITY>(cs.log_imul_constraints(), &mut operand_evals),
			absent_or_claimed::<_, BINMUL_ARITY>(cs.log_bmul_constraints(), &mut operand_evals),
		],
	};

	// Reduce the operand claims to one witness evaluation.
	let shift = {
		let _guard = tracing::info_span!(
			"[phase] Verify Shift Reduction",
			phase = "verify_shift_reduction",
			perfetto_category = "phase"
		)
		.entered();
		shift::verify::<B128, _>(cs, inout, &claims, channel)?
	};

	// Tie in the public values through the public-input consistency check.
	// The reduction reads them over the layout's power-of-two word count.
	// Their count need not be a power of two, so they are passed unpadded.
	let wiring = {
		let _guard = tracing::info_span!(
			"[phase] Verify Public Input",
			phase = "verify_public_input",
			perfetto_category = "phase"
		)
		.entered();
		let public_eval = verify_public_eval(cs, inout, public, &shift, channel)?;
		shift::check_eval::<B128, _>(
			cs,
			inout,
			public_eval,
			&claims.r_x,
			&shift_domain,
			&z_challenge,
			&shift,
			channel,
		)?
	};

	Ok(ReductionOutput {
		r_rho: r_rho.to_vec(),
		shift,
		wiring,
	})
}

/// Reads the public segment's evaluation and reduces it onto the segment's packed form.
///
/// The shift closes over the public segment as a bit matrix: its multilinear is claimed at `r_j`
/// over the bit within a word and the low coordinates of `r_y` over the word index. Evaluating
/// that here would cost the verifier the bits of every public word. So the prover states the
/// evaluation instead, and it is reduced in two steps:
///
/// 1. a ring-switch onto the packed segment — two words to a field element — leaving the claim
///    `sum_x P(x) A(x)` against the ring-switching indicator;
/// 2. a sumcheck over the packed segment's own variables, leaving one evaluation of each factor.
///
/// Nothing here is committed, so the verifier finishes on its own, with field arithmetic and no
/// word bits: it evaluates the packed segment's multilinear from the words it holds, and the
/// indicator from its succinct formula.
///
/// The reduction stands on its own, sharing nothing with the trace's: that one ends in a committed
/// opening, this one in two evaluations the verifier computes from values it already has.
///
/// # Returns
///
/// The public segment over the shift's whole index space, which is what [`shift::check_eval`]
/// reconstructs the trace evaluation from: the claim scaled by the eq-zero factors of the
/// word-index coordinates above the segment's span.
///
/// # Preconditions
///
/// * `r_y` must have at least as many coordinates as the packed segment spans words
fn verify_public_eval<Channel>(
	cs: &ConstraintSystem,
	inout: InoutSegment,
	public: &[Channel::Word],
	shift: &shift::VerifyOutput<Channel::Elem>,
	channel: &mut Channel,
) -> Result<Channel::Elem, Error>
where
	Channel: WordIPVerifierChannel<B128>,
	Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
{
	// The claim is over the packed segment, so it spans whole field elements: a segment shorter
	// than one still spans one, reading the words past its end as zero.
	let log_public_elems = cs
		.log_public_words(inout)
		.saturating_sub(LOG_WORDS_PER_ELEM);
	let log_packed_words = log_public_elems + LOG_WORDS_PER_ELEM;

	// The claimed evaluation's point: the bit within a word, then the words the segment spans.
	let r_y = shift.r_y();
	let eval_point = [shift.r_j(), &r_y[..log_packed_words]].concat();

	let public_eval = channel.recv_one()?;
	let RingSwitchVerifyOutput {
		eq_r_double_prime,
		sumcheck_claim,
	} = ring_switch::verify(public_eval.clone(), &eval_point, channel)?;

	// Reduce the ring-switched claim to one evaluation of each factor.
	let SumcheckOutput {
		eval,
		challenges: mut r_public,
	} = verify_sumcheck(log_public_elems, 2, sumcheck_claim, channel)?;
	r_public.reverse();

	// Close it out against the two factors, both of which the verifier computes: the packed
	// segment's multilinear, over the words it already holds, and the ring-switching indicator.
	let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
	let packed_public = channel
		.pack_words(public)
		.into_iter()
		// The words past the segment's end read as zero, so the elements past its packed length do
		// too, up to the power-of-two span the sumcheck ran over.
		.chain(iter::repeat_with(Channel::Elem::zero))
		.take(1 << log_public_elems)
		.collect::<Vec<_>>();
	let packed_eval = evaluate_inplace_scalars(packed_public, &r_public);
	let rs_eq_eval = eval_rs_eq(&eval_point[log_packing..], &r_public, &eq_r_double_prime);
	channel.assert_zero(packed_eval * rs_eq_eval - eval)?;

	// Extend the claim from the segment's own span to the shift's whole word-index space. Every
	// word above the segment is zero, so each extra coordinate contributes its eq-zero factor.
	Ok(eq_ind_zero(&r_y[log_packed_words..]) * public_eval)
}

/// An operation's operand evaluations, or zeros when it is absent.
///
/// A present operation takes its `ARITY` evaluations off the front of `evals`.
/// An absent operation has an empty constraint set, so the shift finds no key naming it.
/// Its zero claim therefore contributes nothing.
fn absent_or_claimed<F: FieldOps, const ARITY: usize>(
	log_n_constraints: Option<usize>,
	evals: impl Iterator<Item = F>,
) -> Vec<F> {
	match log_n_constraints {
		Some(_) => evals.take(ARITY).collect(),
		None => vec![F::zero(); ARITY],
	}
}
