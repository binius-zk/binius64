// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::multilinear::eq::eq_one_var;

use crate::sumcheck::{
	common::{MleCheckProver, SumcheckProver},
	mle_store::{EqId, EvaluationChunk, MleStore},
	round_evals::round_coeffs_by_eq,
	round_evaluator::{MleCheckRoundEvaluator, SumcheckRoundEvaluator},
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

/// Adaptor that turns an [`MleCheckRoundEvaluator`] into a [`SumcheckRoundEvaluator`].
///
/// This is the evaluator-level mirror of [`MleToSumCheckDecorator`], applying the same [Gruen24]
/// technique: each round, the inner evaluator's prime round polynomials are multiplied by the
/// linear equality term in the bound coordinate and by the accumulated equality prefix. It lets
/// eq-weighted and plain claims live in one evaluator group behind a single
/// [`SharedSumcheckProver`].
///
/// Unlike a bare [`MleCheckRoundEvaluator`], whose driving [`SharedMleCheckProver`] owns the eq
/// tracker, this wrapper runs under a [`SharedSumcheckProver`] that knows nothing of eq indicators.
/// So the wrapper itself holds the [`EqId`] of the shared evaluation point's eq tracker, and
/// supplies the round's eq chunk to [`MleCheckRoundEvaluator::accumulate`] and its alpha and
/// equality prefix to [`MleCheckRoundEvaluator::interpolate`] — all read from that tracker, which
/// the store folds in lockstep, so the wrapper keeps no copy of the point and no equality
/// bookkeeping of its own.
///
/// [`SharedSumcheckProver`]: super::round_evaluator::SharedSumcheckProver
/// [`SharedMleCheckProver`]: super::round_evaluator::SharedMleCheckProver
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub struct MleToSumCheckEvaluator<Inner> {
	inner: Inner,
	// The shared eq tracker for the inner MLE-check evaluation point; the store maintains its
	// current alpha and equality prefix.
	eq_tracker: EqId,
}

impl<Inner> MleToSumCheckEvaluator<Inner> {
	/// Wraps an MLE-check evaluator whose claims share the point of eq tracker `eq_tracker`.
	pub const fn new(inner: Inner, eq_tracker: EqId) -> Self {
		Self { inner, eq_tracker }
	}
}

impl<F, P, Inner> SumcheckRoundEvaluator<F, P> for MleToSumCheckEvaluator<Inner>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Inner: MleCheckRoundEvaluator<F, P>,
{
	fn degree(&self) -> usize {
		// The eq factor multiplies the emitted round polynomial, not the accumulator: the wide
		// slots still hold the inner evaluator's prime-polynomial evaluations.
		self.inner.degree()
	}

	fn accumulate(&self, chunk: &EvaluationChunk<'_, P>, accum: &mut [<P as WideMul>::Output]) {
		self.inner
			.accumulate(chunk, chunk.eq(self.eq_tracker).to_ref(), accum)
	}

	fn interpolate(
		&self,
		store: &MleStore<'_, P>,
		accum: &[<P as WideMul>::Output],
		claim: F,
	) -> RoundCoeffs<F> {
		// `claim` is the sumcheck round claim: the inner MLE-check claim `m` scaled by the
		// accumulated equality prefix. Recover `m` for the inner evaluator by dividing the prefix
		// back out. (`invert_or_zero` mirrors the interpolation routines; the prefix is a product
		// of non-degenerate equality factors for honest challenges.)
		//
		// alpha and the equality prefix are read from the shared eq tracker (the store has not yet
		// folded this round, so the tracker is at the current round's alpha and prefix).
		let eq_prefix = store.eq_prefix(self.eq_tracker);
		let alpha = store.eq_alpha(self.eq_tracker);
		let inner_claim = claim * eq_prefix.invert_or_zero();
		// The inner MLE-check evaluator now takes a reduced accumulator; the plain sumcheck prover
		// driving this wrapper leaves the wide accumulators unreduced, so reduce them here.
		let reduced = accum
			.iter()
			.map(|slot| P::reduce(slot.clone()))
			.collect::<Vec<_>>();
		let round_coeffs = self.inner.interpolate(store, &reduced, inner_claim, alpha);

		// Multiply the inner MLE-check round polynomial by (X - α) and the equality prefix.
		round_coeffs_by_eq(&round_coeffs, alpha) * eq_prefix
	}
}
