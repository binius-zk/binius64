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
	/// One entry per claim the prover carries.
	fn execute(&mut self) -> Vec<RoundCoeffs<F>>;

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

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		either::for_both!(self, inner => inner.execute())
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
/// The prover argues that a claimed value $s$ is the equality-weighted sum
///
/// $$
/// s = \sum_{v \in B_n} F(v) \cdot eq(v, z)
/// $$
///
/// over the hypercube $B_n$, for a point $z$ agreed in advance.
/// The weight $eq(v, z)$ is $1$ where $v = z$ and $0$ at every other vertex.
/// For multilinear $F$ the sum is then just $F(z)$.
///
/// The protocol runs one round per variable:
///
/// - The prover sends a univariate polynomial for the round.
/// - The verifier answers with a random challenge, binding that variable.
/// - After the last round the prover reports the evaluations it reached.
///
/// Driving it out of that order panics.
///
/// This trait is _not_ object-safe.
///
/// See [`binius_ip::mlecheck::verify`] for the verifier side, and [Gruen24] for the optimization.
///
/// A plain sumcheck proves the same claim by folding the weight into the summand.
/// This protocol leaves the weight out of the polynomial it sends, which is where it saves work.
/// The two send different polynomials:
///
/// ```text
///     plain sumcheck   R (X)    claim = R(0) + R(1)
///     MLE-check        R'(X)    claim = (1 - alpha) * R'(0) + alpha * R'(1)
/// ```
///
/// with `alpha` this round's coordinate of the point, and `R = R' * eq(X, alpha)`.
///
/// Each verifier rebuilds a different missing coefficient.
/// One protocol's polynomial fails the other's verifier, so neither trait inherits from the other.
/// Multiplying the weight back in converts a prover here into a plain sumcheck one.
/// The adaptor in this module that does so is the only sound bridge.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub trait MleCheckProver<F: Field> {
	/// The number of variables still free.
	///
	/// Each round binds one variable to a challenge, so this drops by one per round.
	fn n_vars(&self) -> usize;

	/// Computes this round's message, one univariate polynomial per claim.
	///
	/// With $r_0$, ..., $r_{k-1}$ already bound, for claims over $C_0, ..., C_{m-1}$, the output in
	/// low-to-high evaluation order is
	///
	/// $$
	/// R'_i = \sum_{v \in B_{n-k-1}} C_i(r_0, ..., r_{k-1}, X, \{v\}) \cdot eq(\{v\}, z')
	/// $$
	///
	/// where $z'$ is the part of the point summed over.
	///
	/// The coordinate bound this round is left out of the weight.
	/// Multiplying it back gives the round polynomial $R_i(X) = R'_i(X) \cdot eq(X, z_k)$.
	///
	/// For high-to-low order the variables are taken in reverse and summed over the lower indices.
	fn execute(&mut self) -> Vec<RoundCoeffs<F>>;

	/// Binds this round's variable to the verifier challenge.
	fn fold(&mut self, challenge: F);

	/// Consumes the prover and returns every multilinear's evaluation at the challenge point.
	fn finish(self) -> Vec<F>;

	/// The still-unbound part of the evaluation point, one coordinate per free variable.
	fn eval_point(&self) -> &[F];
}
