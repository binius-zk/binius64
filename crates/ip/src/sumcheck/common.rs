// Copyright 2023-2025 Irreducible Inc.

use std::ops::{Add, AddAssign, Index, Mul, MulAssign};

use binius_field::field::FieldOps;
use binius_math::univariate::evaluate_univariate;

/// A univariate polynomial in monomial basis.
///
/// The coefficient at position `i` in the inner vector corresponds to the term $X^i$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundCoeffs<F>(pub Vec<F>);

// The empty coefficient vector is the zero polynomial, a valid default for any element type.
// Deriving the default would instead demand that every element type have its own default value.
impl<F> Default for RoundCoeffs<F> {
	fn default() -> Self {
		// The zero polynomial has no coefficients.
		Self(Vec::new())
	}
}

impl<F> RoundCoeffs<F> {
	/// Truncate the highest-degree coefficient to produce a more compact round proof.
	///
	/// # Pre-conditions
	///
	/// - The coefficient vector must be non-empty.
	/// - A round polynomial always has degree at least one, so an empty vector signals a bug.
	pub fn truncate(mut self) -> RoundProof<F> {
		// Drop the highest-degree coefficient; the verifier reconstructs it from the claimed sum.
		self.0.pop();
		RoundProof(self)
	}

	/// The coefficients ordered from the constant term to the highest-degree term.
	pub fn as_slice(&self) -> &[F] {
		// Position `i` holds the coefficient of X^i, so the constant term comes first.
		&self.0
	}
}

impl<F: FieldOps> RoundCoeffs<F> {
	/// Evaluate the polynomial at a point.
	pub fn evaluate(&self, x: F) -> F {
		// Horner's method over the monomial coefficients.
		evaluate_univariate(&self.0, x)
	}

	/// The endpoint values $(R(0), R(1))$ of the round polynomial.
	///
	/// $R(0)$ is the constant coefficient $a_0$.
	/// $R(1) = \sum_j a_j$ is the sum of all coefficients.
	/// An empty coefficient vector is the zero polynomial, whose endpoints are both zero.
	fn endpoints(&self) -> (F, F) {
		// R(0) is the constant term, or zero when there are no coefficients.
		let r_0 = self.0.first().cloned().unwrap_or_else(F::zero);
		// R(1) is the polynomial at one, which sums every coefficient.
		let r_1 = self.0.iter().cloned().sum();
		(r_0, r_1)
	}

	/// The claimed sum $R(0) + R(1)$ that this round polynomial encodes.
	///
	/// For a sumcheck round polynomial, this is the round's claimed sum.
	/// The verifier expects the identity $s = R(0) + R(1)$ (see [`RoundProof::recover`]).
	pub fn sum_over_endpoints(&self) -> F {
		// The sumcheck claim is s = R(0) + R(1).
		let (r_0, r_1) = self.endpoints();
		r_0 + r_1
	}

	/// The claimed value $(1 - \alpha) R(0) + \alpha R(1)$ that this round polynomial encodes in an
	/// MLE-check.
	///
	/// This is the MLE-check analogue of [`Self::sum_over_endpoints`].
	/// An MLE-check round polynomial satisfies $s = (1 - \alpha) R(0) + \alpha R(1)$.
	/// Here $\alpha$ is the round's evaluation-point coordinate (see
	/// [`crate::mlecheck::RoundProof::recover`]). Equivalently, this is the linear extrapolation
	/// of $R$ from the endpoints $0$ and $1$ to $\alpha$.
	pub fn lerp_over_endpoints(&self, alpha: F) -> F {
		let (r_0, r_1) = self.endpoints();
		// Line through the endpoints, sampled at alpha: R(0) + alpha * (R(1) - R(0)).
		r_0.clone() + alpha * (r_1 - r_0)
	}
}

impl<F: FieldOps> Add<&Self> for RoundCoeffs<F> {
	type Output = Self;

	fn add(mut self, rhs: &Self) -> Self::Output {
		// Reuse the in-place addition and return the grown accumulator.
		self += rhs;
		self
	}
}

impl<F: FieldOps> AddAssign<&Self> for RoundCoeffs<F> {
	fn add_assign(&mut self, rhs: &Self) {
		// The two polynomials may have different degrees, hence different coefficient counts.
		// Pad the shorter accumulator with zeros so every addend coefficient has a partner.
		if self.0.len() < rhs.0.len() {
			self.0.resize(rhs.0.len(), F::zero());
		}

		// Add coefficient by coefficient at matching degrees.
		for (lhs_i, rhs_i) in self.0.iter_mut().zip(rhs.0.iter()) {
			*lhs_i += rhs_i;
		}
	}
}

impl<F: FieldOps> Mul<F> for RoundCoeffs<F> {
	type Output = Self;

	fn mul(mut self, rhs: F) -> Self::Output {
		// Reuse the in-place scaling.
		self *= rhs;
		self
	}
}

impl<F: FieldOps> MulAssign<F> for RoundCoeffs<F> {
	fn mul_assign(&mut self, rhs: F) {
		// Scaling a polynomial by a constant scales every coefficient.
		for coeff in &mut self.0 {
			*coeff *= &rhs;
		}
	}
}

impl<F: FieldOps> std::iter::Sum for RoundCoeffs<F> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		// Start from the zero polynomial and accumulate by polynomial addition.
		iter.fold(Self::default(), |acc, x| acc + &x)
	}
}

impl<F> Index<usize> for RoundCoeffs<F> {
	type Output = F;

	fn index(&self, index: usize) -> &F {
		// Position `i` selects the coefficient of X^i.
		&self.0[index]
	}
}

/// A sumcheck round proof is a univariate polynomial in monomial basis with the coefficient of the
/// highest-degree term truncated off.
///
/// Since the verifier knows the claimed sum of the polynomial values at the points 0 and 1, the
/// high-degree term coefficient can be easily recovered. Truncating the coefficient off saves a
/// small amount of proof data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundProof<F>(pub RoundCoeffs<F>);

// The empty proof is a valid default for any element type.
// Deriving the default would instead demand that every element type have its own default value.
impl<F> Default for RoundProof<F> {
	fn default() -> Self {
		// Mirror the empty (zero) polynomial default.
		Self(RoundCoeffs::default())
	}
}

impl<F> RoundProof<F> {
	/// The truncated polynomial coefficients.
	pub fn coeffs(&self) -> &[F] {
		// Every coefficient except the truncated highest-degree term.
		self.0.as_slice()
	}
}

impl<F: FieldOps> RoundProof<F> {
	/// Recovers all univariate polynomial coefficients from the compressed round proof.
	///
	/// The prover has sent coefficients for the purported ith round polynomial
	/// $r_i(X) = \sum_{j=0}^d a_j * X^j$.
	/// However, the prover has not sent the highest degree coefficient $a_d$.
	/// The verifier will need to recover this missing coefficient.
	///
	/// Let $s$ denote the current round's claimed sum.
	/// The verifier expects the round polynomial $r_i$ to satisfy the identity
	/// $s = r_i(0) + r_i(1)$.
	/// Using
	///     $r_i(0) = a_0$
	///     $r_i(1) = \sum_{j=0}^d a_j$
	/// There is a unique $a_d$ that allows $r_i$ to satisfy the above identity.
	/// Specifically
	///     $a_d = s - a_0 - \sum_{j=0}^{d-1} a_j$
	///
	/// Not sending the whole round polynomial is an optimization.
	/// In the unoptimized version of the protocol, the verifier will halt and reject
	/// if given a round polynomial that does not satisfy the above identity.
	pub fn recover(self, sum: F) -> RoundCoeffs<F> {
		let Self(RoundCoeffs(mut coeffs)) = self;
		// The received coefficients are a_0 through a_{d-1}; the top term a_d was truncated.
		// The claimed sum expands to s = R(0) + R(1) = a_0 + (a_0 + a_1 + ... + a_d).
		// Solving for the missing term gives a_d = s - a_0 - (a_0 + a_1 + ... + a_{d-1}).
		//
		// The constant term a_0 is subtracted twice on purpose.
		// The two copies cancel in characteristic 2 yet keep the identity correct over any field.

		// a_0, or zero for the degenerate empty proof.
		let first_coeff = coeffs.first().cloned().unwrap_or_else(F::zero);
		// a_d = s - a_0 - sum_{j=0}^{d-1} a_j.
		let last_coeff = sum - first_coeff - coeffs.iter().cloned().sum::<F>();
		// Append the recovered top coefficient to rebuild the full polynomial.
		coeffs.push(last_coeff);
		RoundCoeffs(coeffs)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Random, arch::OptimalB128 as B128};
	use binius_math::test_utils::random_scalars;
	use rand::prelude::*;

	use super::*;

	// Sumcheck round polynomials have small degree.
	// Degrees 1 through 4 cover the range the round provers in this workspace produce.
	const DEGREES: [usize; 4] = [1, 2, 3, 4];

	// Deterministic RNG seeded to a fixed value so any failure reproduces exactly.
	fn rng() -> StdRng {
		StdRng::seed_from_u64(0)
	}

	#[test]
	fn recover_round_trips_with_the_claimed_sum() {
		let mut rng = rng();
		// Invariant: truncating the top coefficient then recovering it rebuilds the polynomial,
		// as long as recovery is given the true claimed sum s = R(0) + R(1).
		for degree in DEGREES {
			// A random round polynomial with degree + 1 coefficients.
			let coeffs = RoundCoeffs(random_scalars::<B128>(&mut rng, degree + 1));

			// The verifier only learns the claimed sum s = R(0) + R(1).
			let sum = coeffs.sum_over_endpoints();
			// The proof drops the highest-degree coefficient to save transcript space.
			let proof = coeffs.clone().truncate();

			// Stripping one coefficient leaves exactly `degree` of them.
			assert_eq!(proof.coeffs().len(), degree);
			// The claimed sum uniquely determines the missing coefficient, so recovery is exact.
			assert_eq!(proof.recover(sum), coeffs);
		}
	}

	#[test]
	fn recovered_polynomial_satisfies_the_sumcheck_identity() {
		let mut rng = rng();
		// Invariant: the recovered polynomial must satisfy the verifier's check s = R(0) + R(1).
		for degree in DEGREES {
			let coeffs = RoundCoeffs(random_scalars::<B128>(&mut rng, degree + 1));
			// Claimed sum taken from the original polynomial.
			let sum = coeffs.sum_over_endpoints();
			// Round-trip through truncation and recovery.
			let recovered = coeffs.truncate().recover(sum);

			// Evaluate the recovered polynomial at both endpoints and confirm they sum to s.
			assert_eq!(recovered.evaluate(B128::ZERO) + recovered.evaluate(B128::ONE), sum);
		}
	}

	#[test]
	fn sum_over_endpoints_equals_evaluation_at_zero_and_one() {
		let mut rng = rng();
		// Invariant: the endpoint-sum shortcut equals R(0) + R(1) computed by direct evaluation.
		for degree in DEGREES {
			let coeffs = RoundCoeffs(random_scalars::<B128>(&mut rng, degree + 1));
			// Direct evaluation at the two endpoints.
			let expected = coeffs.evaluate(B128::ZERO) + coeffs.evaluate(B128::ONE);
			// The shortcut must agree with direct evaluation.
			assert_eq!(coeffs.sum_over_endpoints(), expected);
		}
	}

	#[test]
	fn lerp_over_endpoints_is_the_line_through_the_endpoints() {
		let mut rng = rng();
		// Invariant: the extrapolation is the straight line joining R(0) and R(1),
		// sampled at the round coordinate alpha.
		for degree in DEGREES {
			let coeffs = RoundCoeffs(random_scalars::<B128>(&mut rng, degree + 1));
			let alpha = B128::random(&mut rng);

			// The two endpoint values that define the line.
			let r_0 = coeffs.evaluate(B128::ZERO);
			let r_1 = coeffs.evaluate(B128::ONE);
			// The point on that line at alpha: R(0) + alpha * (R(1) - R(0)).
			let expected = r_0 + alpha * (r_1 - r_0);

			// The helper must reproduce that point.
			assert_eq!(coeffs.lerp_over_endpoints(alpha), expected);
			// alpha = 0 lands exactly on R(0).
			assert_eq!(coeffs.lerp_over_endpoints(B128::ZERO), r_0);
			// alpha = 1 lands exactly on R(1).
			assert_eq!(coeffs.lerp_over_endpoints(B128::ONE), r_1);
		}
	}

	#[test]
	fn addition_is_pointwise_across_ragged_lengths() {
		let mut rng = rng();
		// Invariant: adding polynomials adds their evaluations at every point,
		// even when the two operands have different degrees.
		//
		// Fixture: a degree-4 polynomial (5 coefficients) and a degree-1 polynomial (2).
		//
		//     long : [a_0, a_1, a_2, a_3, a_4]
		//     short: [b_0, b_1]
		//     sum  : [a_0+b_0, a_1+b_1, a_2, a_3, a_4]
		let long = RoundCoeffs(random_scalars::<B128>(&mut rng, 5));
		let short = RoundCoeffs(random_scalars::<B128>(&mut rng, 2));
		let x = B128::random(&mut rng);

		// Longer accumulator, shorter addend: the accumulator keeps its high-degree tail.
		assert_eq!((long.clone() + &short).evaluate(x), long.evaluate(x) + short.evaluate(x));
		// Shorter accumulator, longer addend: the accumulator is padded up to the larger degree.
		assert_eq!((short.clone() + &long).evaluate(x), short.evaluate(x) + long.evaluate(x));
	}

	#[test]
	fn scaling_multiplies_every_evaluation() {
		let mut rng = rng();
		// Invariant: scaling a polynomial by a constant scales its value at every point.
		let coeffs = RoundCoeffs(random_scalars::<B128>(&mut rng, 4));
		let scalar = B128::random(&mut rng);
		let x = B128::random(&mut rng);

		// (c * R)(x) must equal c * R(x).
		assert_eq!((coeffs.clone() * scalar).evaluate(x), coeffs.evaluate(x) * scalar);
	}

	#[test]
	fn sum_of_round_coeffs_adds_the_polynomials() {
		let mut rng = rng();
		// Invariant: summing polynomials sums their evaluations at every point.
		// Fixture: three degree-2 polynomials (3 coefficients each).
		let parts: Vec<_> = (0..3)
			.map(|_| RoundCoeffs(random_scalars::<B128>(&mut rng, 3)))
			.collect();
		let x = B128::random(&mut rng);

		// Fold the polynomials together with the iterator sum.
		let total: RoundCoeffs<B128> = parts.iter().cloned().sum();
		// Reference: add the individual evaluations at x.
		let expected: B128 = parts.iter().map(|c| c.evaluate(x)).sum();

		assert_eq!(total.evaluate(x), expected);
	}

	#[test]
	fn empty_round_coeffs_is_the_zero_polynomial() {
		// A polynomial with no coefficients is the zero polynomial.
		let coeffs = RoundCoeffs::<B128>(vec![]);
		// It evaluates to zero everywhere.
		assert_eq!(coeffs.evaluate(B128::random(rng())), B128::ZERO);
		// Both endpoints are zero, so their sum is zero.
		assert_eq!(coeffs.sum_over_endpoints(), B128::ZERO);
		// The line through (0, 0) and (1, 0) is zero at every alpha.
		assert_eq!(coeffs.lerp_over_endpoints(B128::random(rng())), B128::ZERO);
	}
}
