// Copyright 2023-2025 Irreducible Inc.

use binius_field::Field;
use binius_ip::sumcheck::RoundCoeffs;
use either::Either;

/// A sumcheck prover with a round-by-round execution interface.
///
/// Sumcheck prover logic is accessed via a trait because important optimizations are available
/// depending on the structure of the multivariate polynomial that the protocol targets. For
/// example, [Gruen24] observes a significant optimization available to the sumcheck prover when
/// the multivariate is the product of a multilinear composite and an equality indicator
/// polynomial, which arises in the zerocheck protocol.
///
/// The trait exposes a round-by-round interface so that protocol execution logic that drives the
/// prover can interleave the executions of the interactive protocol, for example in the case of
/// batching several sumcheck protocols.
///
/// The caller must make a specific sequence of calls to the provers. For a prover where
/// [`Self::n_vars`] is $n$, the caller must call [`Self::execute`] and then [`Self::fold`] $n$
/// times, and finally call [`Self::finish`]. If the calls aren't made in that order, the prover
/// will panic.
///
/// This trait is _not_ object-safe.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub trait SumcheckProver<F: Field> {
	/// The number of variables in the remaining multivariate polynomial.
	///
	/// The number of variables decrements after each [`Self::fold`] call, as that binds one free
	/// variable with a concrete challenge.
	fn n_vars(&self) -> usize;

	/// Returns the number of claims (composite polynomials) being proved.
	///
	/// This is the expected length of the Vec returned by [`Self::execute`].
	fn n_claims(&self) -> usize;

	/// Computes the prover messages for this round as a univariate polynomial.
	///
	/// If [`Self::fold`] has already been called on the prover with the values $r_0$, ...,
	/// $r_{k-1}$ and the sumcheck prover is proving the sums of the composite polynomials $C_0,
	/// ..., C_{m-1}$, then the output of this method for low-to-high evaluation order would be:
	///
	/// $$
	/// R_i = \sum_{v \in B_{n-k-1}} C_i(r_0, ..., r_{k-1}, X, \{v\}), i \in \[0, ..., m-1\]
	/// $$
	///
	/// For high-to-low evaluation order the variables are specified in reverse order (starting with
	/// the highest indexed one) and hypercube sums are performed over the lower indexed variables.
	///
	/// The returned Vec must have length equal to [`Self::n_claims`].
	fn execute(&mut self) -> Vec<RoundCoeffs<F>>;

	/// Returns the claimed sums for the current round, one per claim.
	///
	/// The returned Vec has length equal to [`Self::n_claims`]. For claim $i$, this is the value
	/// $s_i$ that the round polynomial $R_i$ returned by [`Self::execute`] satisfies via the
	/// sumcheck identity $s_i = R_i(0) + R_i(1)$.
	///
	/// This may be called either before [`Self::execute`] (when the prover holds the claim sums
	/// directly) or after it (when the prover holds the round coefficients, in which case the value
	/// is recovered from them). A regular sumcheck prover recovers it as $R_i(0) + R_i(1)$ (see
	/// [`RoundCoeffs::sum_over_endpoints`]), while an MLE-check prover (see [`MleCheckProver`])
	/// recovers it as $(1 - \alpha) R_i(0) + \alpha R_i(1)$ with $\alpha$ the round's coordinate
	/// (see [`RoundCoeffs::lerp_over_endpoints`]).
	fn round_claim(&self) -> Vec<F>;

	/// Folds the sumcheck multilinears with a new verifier challenge.
	fn fold(&mut self, challenge: F);

	/// Finishes the sumcheck proving protocol and returns the evaluations of all multilinears at
	/// the challenge point.
	fn finish(self) -> Vec<F>;
}

impl<F, L, R> SumcheckProver<F> for Either<L, R>
where
	F: Field,
	L: SumcheckProver<F>,
	R: SumcheckProver<F>,
{
	fn n_vars(&self) -> usize {
		either::for_both!(self, inner => inner.n_vars())
	}

	fn n_claims(&self) -> usize {
		either::for_both!(self, inner => inner.n_claims())
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		either::for_both!(self, inner => inner.execute())
	}

	fn round_claim(&self) -> Vec<F> {
		either::for_both!(self, inner => inner.round_claim())
	}

	fn fold(&mut self, challenge: F) {
		either::for_both!(self, inner => inner.fold(challenge))
	}

	fn finish(self) -> Vec<F> {
		either::for_both!(self, inner => inner.finish())
	}
}

/// A prover for the MLE-check variant of the sumcheck protocol.
///
/// See [`binius_ip::mlecheck::verify`] for context on the protocol.
///
/// This trait inherits from [`SumcheckProver`] since it shares the same type-level interface
/// and protocol execution pattern. However, `MleCheckProver` instances provide different
/// guarantees for the values returned by [`SumcheckProver::execute`] compared to regular
/// sumcheck provers.
///
/// Note that while this technically violates the Liskov substitution principle (LSP), the
/// violation is contained and deemed acceptable for the current design. A future refactor
/// could introduce a common `SumcheckLikeProver` parent trait if stricter LSP compliance
/// becomes necessary.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub trait MleCheckProver<F: Field>: SumcheckProver<F> {
	/// Returns the evaluation point for the remaining claim.
	///
	/// The length of the evaluation point is equal to `self.n_vars()`.
	fn eval_point(&self) -> &[F];
}
