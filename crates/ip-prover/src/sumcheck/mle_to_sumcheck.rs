// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::multilinear::eq::eq_one_var;

use crate::sumcheck::{
	common::{MleCheckProver, SumcheckProver},
	round_evals::round_coeffs_by_eq,
	round_evaluator::{RoundContext, RoundEvaluator},
};

/// Adaptor that exposes a `SumcheckProver` interface for an internal `MleCheckProver`.
///
/// This struct implements the technique from [Gruen24] to convert an MLE-check protocol
/// into a standard sumcheck protocol. The key insight is that the MLE-check claim
/// $\sum_{v \in \{0,1\}^n} F(v) \cdot \text{eq}(v, z) = s$ can be rewritten as a sumcheck
/// claim by multiplying in the equality polynomial term-by-term during the protocol execution.
///
/// In each round, the adaptor multiplies the round polynomials from the inner MLE-check
/// prover by a linear polynomial term $(X - \alpha)$ where $\alpha$ is the corresponding
/// coordinate of the evaluation point. This effectively transforms the MLE-check round
/// polynomial into a sumcheck round polynomial that includes the equality check.
///
/// The `eq_prefix_eval` field accumulates the product of all previously factored equality
/// terms, ensuring the round polynomials maintain the correct scaling throughout the protocol.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
#[derive(Debug, Clone)]
pub struct MleToSumCheckDecorator<F: Field, InnerProver> {
	mlecheck_prover: InnerProver,
	eq_prefix_eval: F,
}

impl<F: Field, InnerProver: MleCheckProver<F>> MleToSumCheckDecorator<F, InnerProver> {
	pub const fn new(mlecheck_prover: InnerProver) -> Self {
		Self {
			mlecheck_prover,
			eq_prefix_eval: F::ONE,
		}
	}
}

impl<F: Field, InnerProver: MleCheckProver<F>> SumcheckProver<F>
	for MleToSumCheckDecorator<F, InnerProver>
{
	fn n_vars(&self) -> usize {
		self.mlecheck_prover.n_vars()
	}

	fn n_claims(&self) -> usize {
		self.mlecheck_prover.n_claims()
	}

	fn round_claim(&self) -> Vec<F> {
		// The sumcheck round claim is the inner MLE-check claim scaled by the accumulated equality
		// prefix: R^sc(0) + R^sc(1) = eq_prefix_eval * [(1 - α) p(0) + α p(1)] = eq_prefix_eval *
		// m, where m is the inner MLE-check round claim and p its round polynomial.
		self.mlecheck_prover
			.round_claim()
			.into_iter()
			.map(|m| m * self.eq_prefix_eval)
			.collect()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let round_coeffs_multi = self.mlecheck_prover.execute();

		// Multiply the round polynomials from the inner MLE-check prover by (X - α).
		let alpha = self.mlecheck_prover.eval_point()[self.n_vars() - 1];
		round_coeffs_multi
			.into_iter()
			.map(|round_coeffs| round_coeffs_by_eq(&round_coeffs, alpha) * self.eq_prefix_eval)
			.collect()
	}

	fn fold(&mut self, challenge: F) {
		assert_ne!(self.n_vars(), 0, "fold called out of order; expected finish");

		let alpha = self.mlecheck_prover.eval_point()[self.n_vars() - 1];
		self.eq_prefix_eval *= eq_one_var(challenge, alpha);

		self.mlecheck_prover.fold(challenge)
	}

	fn finish(self) -> Vec<F> {
		self.mlecheck_prover.finish()
	}
}

/// Adaptor that turns an MLE-check [`RoundEvaluator`] into a regular sumcheck one.
///
/// This is the evaluator-level mirror of [`MleToSumCheckDecorator`], applying the same [Gruen24]
/// technique: each round, the inner evaluator's prime round polynomials are multiplied by the
/// linear equality term in the bound coordinate and by the accumulated equality prefix. It lets
/// eq-weighted and plain claims live in one evaluator group behind a single
/// [`SumcheckProver`](super::round_evaluator::SharedSumcheckProver).
///
/// The accumulation pass is delegated to the inner evaluator untouched; only the interpolated
/// polynomials and the claim bookkeeping change.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub struct MleToSumCheckEvaluator<F: Field, Inner> {
	inner: Inner,
	// The inner MLE-check evaluation point; the highest remaining coordinate is the round's
	// alpha.
	eval_point: Vec<F>,
	// Local bookkeeping mirror of the store's remaining variable count.
	n_vars_remaining: usize,
	// Product of the equality terms of all previously bound coordinates.
	eq_prefix_eval: F,
}

impl<F: Field, Inner> MleToSumCheckEvaluator<F, Inner> {
	/// Wraps an MLE-check evaluator whose claims share the given evaluation point.
	pub const fn new(inner: Inner, eval_point: Vec<F>) -> Self {
		let n_vars_remaining = eval_point.len();
		Self {
			inner,
			eval_point,
			n_vars_remaining,
			eq_prefix_eval: F::ONE,
		}
	}
}

impl<F, P, Inner> RoundEvaluator<F, P> for MleToSumCheckEvaluator<F, Inner>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Inner: RoundEvaluator<F, P>,
{
	fn degree(&self) -> usize {
		// The eq factor multiplies the emitted round polynomial, not the accumulator: the wide
		// slots still hold the inner evaluator's prime-polynomial evaluations.
		self.inner.degree()
	}

	fn round_claim(&self) -> F {
		// The sumcheck round claim is the inner MLE-check claim scaled by the accumulated
		// equality prefix; see [`MleToSumCheckDecorator::round_claim`].
		self.inner.round_claim() * self.eq_prefix_eval
	}

	fn accumulate(
		&self,
		ctx: &RoundContext<'_, '_, P>,
		chunk_index: usize,
		accum: &mut [<P as WideMul>::Output],
	) {
		self.inner.accumulate(ctx, chunk_index, accum)
	}

	fn interpolate(&mut self, accum: &[<P as WideMul>::Output]) -> RoundCoeffs<F> {
		let round_coeffs = self.inner.interpolate(accum);

		// Multiply the round polynomial from the inner MLE-check evaluator by (X - α) and the
		// equality prefix.
		let alpha = self.eval_point[self.n_vars_remaining - 1];
		round_coeffs_by_eq(&round_coeffs, alpha) * self.eq_prefix_eval
	}

	fn fold(&mut self, challenge: F) {
		// precondition
		assert!(self.n_vars_remaining > 0, "fold called out of order; no variables remain");

		let alpha = self.eval_point[self.n_vars_remaining - 1];
		self.eq_prefix_eval *= eq_one_var(challenge, alpha);
		self.n_vars_remaining -= 1;

		self.inner.fold(challenge)
	}
}
