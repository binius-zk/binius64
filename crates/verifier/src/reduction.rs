// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The constraint reduction shared by the single-instance and batched Binius64 verifiers.
//!
//! One circuit, and `2^k` instances of one circuit, reduce the same way.
//! Which of the two it is enters as [`Instances`], and its count joins every row count.
//!
//! A batch adds one step the monolithic reduction does not have:
//!
//! ```text
//! monolithic   every operand claim already sits at a common point
//! batch        each operation's claims sit at its own instance point
//!              -> one sumcheck transports them onto a shared point
//! ```
//!
//! A batch of one instance still runs that sumcheck, over no rounds.
//! It is the batched protocol either way, because its prover folds a batched witness.

use std::array;

use binius_core::{
	constraint_system::{ConstraintSystem, InoutSegment},
	word::Word,
};
use binius_field::{AESTowerField8b as B8, FieldOps};
use binius_iop::channel::IOPVerifierChannel;
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{BatchSumcheckOutput, batch_verify},
};
use binius_math::{
	BinarySubspace,
	inner_product::inner_product_scalars,
	multilinear::eq::eq_ind,
	univariate::{evaluate_univariate, lagrange_evals_scalars},
};

use crate::{
	Error,
	config::B128,
	protocols::{
		binmul::{BinMulOutput, verify as verify_binmul_reduction},
		bitand::AndCheckOutput,
		intmul::{IntMulOutput, verify as verify_intmul_reduction},
		shift::{self, BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, OperatorData},
		zero,
	},
	verify_bitand_reduction,
};

/// The degree of the instance re-randomization's round polynomials.
///
/// Each operand is a multilinear evaluation, expressed with the quadratic evaluator.
/// Its degree-2 prime polynomial gains one degree from the equality factor, giving 3.
// TODO: a degree-1 multilinear-eval store evaluator would drop this to 2; none exists yet.
const RERAND_DEGREE: usize = 3;

/// The instances a reduction runs over.
///
/// This picks the protocol, not just a count.
/// A batch of one instance is still a batch: its prover folds a batched witness, where the
/// monolithic prover reads packed words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Instances {
	/// One circuit, with no instance axis at all.
	Single,
	/// `2^log_instances` instances of one circuit, sharing its constraint system.
	Batch {
		/// The base-2 logarithm of the instance count.
		log_instances: usize,
	},
}

impl Instances {
	/// The instance variables every reduction's row count includes.
	pub const fn log_count(self) -> usize {
		match self {
			Self::Single => 0,
			Self::Batch { log_instances } => log_instances,
		}
	}

	/// Whether the operand claims need transporting onto a shared instance point.
	///
	/// Only a batch does. A batch of one transports over no rounds rather than skipping the step,
	/// which is what keeps its transcript the same shape at every instance count.
	const fn transports_claims(self) -> bool {
		matches!(self, Self::Batch { .. })
	}
}

/// What [`reduce_constraints`] leaves for the caller to open against the committed trace.
#[derive(Debug)]
pub struct ReductionOutput<F> {
	/// The instance point every operand claim was transported onto.
	///
	/// Empty for the monolithic reduction, which transports nothing.
	pub r_rho: Vec<F>,
	/// The shift reduction's output, holding the claimed witness evaluation.
	pub shift: shift::VerifyOutput<F>,
}

impl<F: Clone> ReductionOutput<F> {
	/// The point the committed trace is opened at.
	///
	/// The trace's bit index is `[bit | instance | wire]`, low to high:
	///
	/// ```text
	/// r_j     the bit within a word
	/// r_rho   the instance, empty for the monolithic reduction
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
/// IntMul -> BinMul -> BitAnd -> Zero -> re-randomization -> shift -> public check
/// ```
///
/// # Arguments
///
/// - `cs`: the single-instance constraint system every instance satisfies.
/// - `instances`: whether this is the monolithic reduction or a batch, and how wide.
/// - `inout`: which value segment the inout words sit in.
/// - `public`: the declared public values, unpadded — the constants, then the inout values.
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
pub fn reduce_constraints<Channel>(
	cs: &ConstraintSystem,
	instances: Instances,
	inout: InoutSegment,
	public: &[Word],
	channel: &mut Channel,
) -> Result<ReductionOutput<Channel::Elem>, Error>
where
	Channel: IOPVerifierChannel<B128>,
	Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
{
	let log_instances = instances.log_count();

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

	// The BitAnd check has no skip branch: an empty AND set still reduces, over the single all-zero
	// padding row, so `None` is zero constraint variables.
	let log_n_and = cs.log_and_constraints().unwrap_or(0);
	let AndCheckOutput {
		a_eval,
		b_eval,
		c_eval,
		z_challenge,
		eval_point,
	} = {
		let _guard = tracing::info_span!(
			"[phase] Verify BitAnd Reduction",
			phase = "verify_bitand_reduction",
			perfetto_category = "phase",
			n_constraints = cs.n_and_constraints()
		)
		.entered();
		verify_bitand_reduction(log_instances + log_n_and, &andcheck_domain, channel)?
	};

	// The AND-check row point is `r_rho_and || r_x_and`: the instance index low, the constraint
	// index high. With one instance the instance half is empty.
	let (r_rho_and, r_x_and) = eval_point.split_at(log_instances);

	// The Zero reduction reads nothing and runs no sumcheck.
	// A ZERO constraint is linear, so its oblong form vanishing at one unpredictable point
	// certifies it.
	//
	// The point is the constraint half of the AND-check's, extended when ZERO has more rows.
	// The prover draws the same extension at the same place.
	//
	// It skips the re-randomization below: a ZERO array vanishes identically, so its fold at any
	// instance point is still zero.
	let log_n_zero = cs.log_zero_constraints().unwrap_or(0);
	let zero_point = zero::reduction_point(r_x_and, log_n_zero, || channel.sample());
	let zero = OperatorData::new(zero_point, [Channel::Elem::zero()]);

	// Collapse the multiplications' per-bit operand claims to the oblong form BitAnd already has.
	// The Lagrange weights fold them at the univariate challenge BitAnd just drew.
	let lagrange = lagrange_evals_scalars::<B128, Channel::Elem>(&shift_domain, &z_challenge);
	let bitand_claim = OperationClaim::new([a_eval, b_eval, c_eval], r_x_and, r_rho_and);
	let intmul_claim =
		intmul_output.map(|output| OperationClaim::from_intmul(output, &lagrange, log_instances));
	let binmul_claim =
		binmul_output.map(|output| OperationClaim::from_binmul(output, &lagrange, log_instances));

	// Transport every operand claim onto one shared instance point.
	// The monolithic reduction has only one point to begin with, and so does a batch with no
	// multiplication constraints.
	let needs_rerandomization =
		instances.transports_claims() && (intmul_claim.is_some() || binmul_claim.is_some());
	let (r_rho, bitand, intmul, binmul) = if needs_rerandomization {
		RerandomizedOperations {
			bitand: bitand_claim,
			intmul: intmul_claim,
			binmul: binmul_claim,
		}
		.verify(channel)?
	} else {
		(
			bitand_claim.r_rho,
			OperatorData::new(bitand_claim.r_x, bitand_claim.operand_claims),
			absent_or_claimed(intmul_claim),
			absent_or_claimed(binmul_claim),
		)
	};

	// Reduce the operand claims to one witness evaluation.
	let shift = {
		let _guard = tracing::info_span!(
			"[phase] Verify Shift Reduction",
			phase = "verify_shift_reduction",
			perfetto_category = "phase"
		)
		.entered();
		shift::verify::<B128, _>(cs, inout, &zero, &bitand, &intmul, &binmul, channel)?
	};

	// Tie in the public values through the public-input consistency check.
	// The shift evaluates them over the layout's power-of-two word count.
	// Their count need not be a power of two, so they are passed unpadded.
	{
		let _guard = tracing::info_span!(
			"[phase] Verify Public Input",
			phase = "verify_public_input",
			perfetto_category = "phase"
		)
		.entered();
		shift::check_eval::<B128, _>(
			cs,
			inout,
			public,
			&zero,
			&bitand,
			&intmul,
			&binmul,
			&shift_domain,
			z_challenge,
			&shift,
			channel,
		)?;
	}

	Ok(ReductionOutput { r_rho, shift })
}

/// An operation's operand data, or a zero claim at an empty point when it is absent.
///
/// An absent operation has an empty constraint set, so the shift finds no key naming it.
/// Its zero claim therefore contributes nothing.
fn absent_or_claimed<F: FieldOps, const ARITY: usize>(
	claim: Option<OperationClaim<F, ARITY>>,
) -> OperatorData<F, ARITY> {
	match claim {
		Some(claim) => OperatorData::new(claim.r_x, claim.operand_claims),
		None => OperatorData::new(Vec::new(), array::from_fn(|_| F::zero())),
	}
}

/// The shared instance point together with every operation's operand data at that point.
type RerandOutput<F> = (
	Vec<F>,
	OperatorData<F, BITAND_ARITY>,
	OperatorData<F, INTMUL_ARITY>,
	OperatorData<F, BINMUL_ARITY>,
);

/// One operation's oblong operand claims and the points they are claimed at.
///
/// Every operation reduces to this shape.
/// The re-randomization transports the claims to the instance point shared by all of them.
///
/// Generic over the channel's element type `F`, so this composes with a channel whose challenges
/// live in an extension of the base field (e.g. a symbolic verifier channel), not only `B128`.
struct OperationClaim<F, const ARITY: usize> {
	/// The oblong operand claim per operand, in operand order.
	operand_claims: [F; ARITY],
	/// The constraint-index point the operands are claimed at.
	r_x: Vec<F>,
	/// The instance-index point the operands are claimed at.
	r_rho: Vec<F>,
}

impl<F: FieldOps, const ARITY: usize> OperationClaim<F, ARITY> {
	/// The operand claims at the constraint point `r_x` and instance point `r_rho`.
	fn new(operand_claims: [F; ARITY], r_x: &[F], r_rho: &[F]) -> Self {
		Self {
			operand_claims,
			r_x: r_x.to_vec(),
			r_rho: r_rho.to_vec(),
		}
	}
}

impl<F: FieldOps> OperationClaim<F, INTMUL_ARITY> {
	/// Builds the IntMul claim by collapsing its per-bit operand claims to oblong claims.
	///
	/// The Lagrange weights fold the per-bit claims at the univariate challenge.
	/// This gives the oblong form the BitAnd claims already have.
	/// The IntMul row point splits into an instance part (low) and a constraint part (high).
	fn from_intmul(intmul_output: IntMulOutput<F>, lagrange: &[F], log_instances: usize) -> Self {
		let IntMulOutput {
			eval_point: r_out_mul,
			a_evals,
			b_evals,
			c_lo_evals,
			c_hi_evals,
		} = intmul_output;
		let oblong = |evals: Vec<F>| inner_product_scalars(evals, lagrange.iter().cloned());
		let (r_rho, r_x) = r_out_mul.split_at(log_instances);
		Self::new(
			[
				oblong(a_evals),
				oblong(b_evals),
				oblong(c_lo_evals),
				oblong(c_hi_evals),
			],
			r_x,
			r_rho,
		)
	}
}

impl<F: FieldOps> OperationClaim<F, BINMUL_ARITY> {
	/// Builds the BinMul claim by collapsing its per-bit operand claims to oblong claims.
	///
	/// The Lagrange weights fold the per-bit claims at the univariate challenge.
	/// This gives the oblong form the BitAnd claims already have.
	/// The BinMul row point splits into an instance part (low) and a constraint part (high).
	fn from_binmul(binmul_output: BinMulOutput<F>, lagrange: &[F], log_instances: usize) -> Self {
		let BinMulOutput {
			eval_point: r_out_binmul,
			a_lo_evals,
			a_hi_evals,
			b_lo_evals,
			b_hi_evals,
			c_lo_evals,
			c_hi_evals,
		} = binmul_output;
		let oblong = |evals: Vec<F>| inner_product_scalars(evals, lagrange.iter().cloned());
		let (r_rho, r_x) = r_out_binmul.split_at(log_instances);
		Self::new(
			[
				oblong(a_lo_evals),
				oblong(a_hi_evals),
				oblong(b_lo_evals),
				oblong(b_hi_evals),
				oblong(c_lo_evals),
				oblong(c_hi_evals),
			],
			r_x,
			r_rho,
		)
	}
}

/// The operations' claims entering the batched instance re-randomization.
///
/// BitAnd is always present. IntMul and BinMul enter only when the circuit carries their
/// constraints; an absent operation reduces to a zero claim contributing nothing to the shift.
struct RerandomizedOperations<F> {
	/// The BitAnd operand claims at the AND-check instance point.
	bitand: OperationClaim<F, BITAND_ARITY>,
	/// The IntMul operand claims at the IntMul instance point, when the circuit has IMUL
	/// constraints.
	intmul: Option<OperationClaim<F, INTMUL_ARITY>>,
	/// The BinMul operand claims at the BinMul instance point, when the circuit has BMUL
	/// constraints.
	binmul: Option<OperationClaim<F, BINMUL_ARITY>>,
}

impl<F: FieldOps> RerandomizedOperations<F> {
	/// Verifies the batched sumcheck that unifies every present operation's instance point.
	///
	/// - Check the sumcheck transporting every operand claim onto one shared instance point.
	/// - Read the reduced operand evaluations at that point.
	/// - Bind them to the sumcheck.
	///
	/// The operands are ordered [BitAnd | IntMul (if present) | BinMul (if present)], matching the
	/// prover's push order, so the sums, the reduced evaluations, and the binding all agree. An
	/// absent operation reduces to a zero claim at an empty point.
	///
	/// # Returns
	///
	/// The shared instance point, the BitAnd operand data, the IntMul operand data, and the BinMul
	/// operand data.
	fn verify<Channel>(self, channel: &mut Channel) -> Result<RerandOutput<F>, Error>
	where
		Channel: IPVerifierChannel<B128, Elem = F>,
	{
		// Every operation reduces over the same instance axis; recover its width from the BitAnd
		// point.
		let log_instances = self.bitand.r_rho.len();

		// Verify the batched sumcheck: one multilinear-eval claim per operand, in push order
		// [BitAnd a, b, c | IntMul a, b, lo, hi | BinMul a_lo, a_hi, b_lo, b_hi, c_lo, c_hi].
		let mut sums: Vec<F> = self.bitand.operand_claims.to_vec();
		if let Some(intmul) = &self.intmul {
			sums.extend(intmul.operand_claims.iter().cloned());
		}
		if let Some(binmul) = &self.binmul {
			sums.extend(binmul.operand_claims.iter().cloned());
		}
		let BatchSumcheckOutput {
			batch_coeff,
			eval,
			mut challenges,
		} = batch_verify(log_instances, RERAND_DEGREE, &sums, channel)?;
		challenges.reverse();
		let r_rho = challenges;

		// The prover wrote the reduced operand evaluations at `r_rho`, grouped by operation in push
		// order. These are the operand claims the shift consumes.
		let bitand_evals = channel.recv_array::<BITAND_ARITY>()?;
		let intmul_evals = self
			.intmul
			.as_ref()
			.map(|_| channel.recv_array::<INTMUL_ARITY>())
			.transpose()?;
		let binmul_evals = self
			.binmul
			.as_ref()
			.map(|_| channel.recv_array::<BINMUL_ARITY>())
			.transpose()?;

		// Bind the reduced evals to the sumcheck: each claim's contribution is its
		// eq(instance_point, r_rho) weight times its reduced eval, batched by `batch_coeff`. The
		// contributions follow the same push order as `sums`.
		let eq_and = eq_ind(&self.bitand.r_rho, &r_rho);
		let mut expected: Vec<F> = bitand_evals
			.iter()
			.map(|eval| eval.clone() * &eq_and)
			.collect();
		if let (Some(intmul), Some(evals)) = (&self.intmul, &intmul_evals) {
			let eq_mul = eq_ind(&intmul.r_rho, &r_rho);
			expected.extend(evals.iter().map(|eval| eval.clone() * &eq_mul));
		}
		if let (Some(binmul), Some(evals)) = (&self.binmul, &binmul_evals) {
			let eq_binmul = eq_ind(&binmul.r_rho, &r_rho);
			expected.extend(evals.iter().map(|eval| eval.clone() * &eq_binmul));
		}
		channel.assert_zero(evaluate_univariate(&expected, &batch_coeff) - eval)?;

		let bitand_data = OperatorData::new(self.bitand.r_x, bitand_evals);
		let intmul_data = match (self.intmul, intmul_evals) {
			(Some(intmul), Some(evals)) => OperatorData::new(intmul.r_x, evals),
			_ => OperatorData::new(Vec::new(), array::from_fn(|_| F::zero())),
		};
		let binmul_data = match (self.binmul, binmul_evals) {
			(Some(binmul), Some(evals)) => OperatorData::new(binmul.r_x, evals),
			_ => OperatorData::new(Vec::new(), array::from_fn(|_| F::zero())),
		};
		Ok((r_rho, bitand_data, intmul_data, binmul_data))
	}
}
