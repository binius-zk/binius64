// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Sampled round-polynomial evaluations, and their interpolation to monomial coefficients.
//!
//! A sumcheck round produces one univariate polynomial per claim.
//! The prover samples it at a few nodes, then solves for its coefficients.
//!
//! ```text
//!     accumulate    sum the composition over the halved hypercube, at each sampled node
//!     reduce        collapse the wide accumulator, once per round
//!     sum_scalars   sum the packed lanes into field elements
//!     interpolate   recover the node at 0 from the round claim, then solve
//! ```
//!
//! The evaluation at 0 is never sampled.
//! Recovering it from the round claim saves the prover one node per round.

use std::{
	array, iter,
	ops::{Add, AddAssign, Mul},
};

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;

/// The `D` sampled evaluations of one claim's round polynomial.
///
/// A degree-`D` polynomial needs `D + 1` values.
/// The round claim supplies the one at 0, so only `D` are sampled:
///
/// ```text
///     D = 1    [ R(1) ]
///     D = 2    [ R(1), R(inf) ]
/// ```
///
/// Invariant: slot 0 is `R(1)`, which every recovery of `R(0)` reads.
///
/// `R(inf)` is the leading coefficient, at one addition per element.
/// Any finite node past 1 would cost a full extrapolation.
///
/// `T` narrows as the round advances:
///
/// ```text
///     accumulate    wide, unreduced
///     reduce        packed
///     sum_scalars   scalar
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RoundEvals<T, const D: usize>(pub [T; D]);

impl<T: Default, const D: usize> Default for RoundEvals<T, D> {
	fn default() -> Self {
		Self(array::from_fn(|_| T::default()))
	}
}

impl<T, const D: usize> RoundEvals<T, D> {
	/// Reads one evaluator's run of accumulator slots.
	///
	/// # Panics
	///
	/// Panics unless the run holds exactly `D` slots.
	/// A mismatch means the evaluator's reported degree disagrees with the `D` its body uses.
	pub fn from_slots(slots: &[T]) -> Self
	where
		T: Copy,
	{
		Self(
			<[T; D]>::try_from(slots)
				.expect("slot run length must equal the evaluator's reported degree"),
		)
	}

	/// Adds these evaluations into one evaluator's run of accumulator slots.
	///
	/// An evaluator sums into locals through its hot loop, then calls this once per chunk.
	/// The running sums therefore stay in registers, not in the shared accumulator buffer.
	///
	/// # Panics
	///
	/// Panics unless the run holds exactly `D` slots, as in [`Self::from_slots`].
	pub fn add_to(self, slots: &mut [T])
	where
		T: AddAssign,
	{
		assert_eq!(slots.len(), D, "slot run length must equal the evaluator's reported degree");
		for (slot, eval) in iter::zip(slots, self.0) {
			*slot += eval;
		}
	}

	/// Reduces every wide slot, once the round's accumulation is complete.
	///
	/// Accumulation sums unreduced products.
	/// So the modular reduction is paid once per round, not once per hypercube element.
	pub fn reduce<P: PackedField + WideMul<Output = T>>(self) -> RoundEvals<P, D> {
		RoundEvals(self.0.map(P::reduce))
	}
}

impl<P: PackedField, const D: usize> RoundEvals<P, D> {
	/// Sums the packed lanes of every slot into a scalar.
	///
	/// A round narrower than one packed word leaves the trailing lanes dead.
	/// So only the first `2^n_vars` are summed.
	pub fn sum_scalars(self, n_vars: usize) -> RoundEvals<P::Scalar, D> {
		RoundEvals(self.0.map(|eval| eval.iter().take(1 << n_vars).sum()))
	}
}

impl<F: Field, const D: usize> RoundEvals<F, D> {
	/// Recovers `R(0)` from an MLE-check round claim, at any prime degree.
	///
	/// The verifier reduces with `claim = (1 - alpha) * R(0) + alpha * R(1)`.
	///
	/// Solving for `R(0)` divides by `1 - alpha`.
	/// That is non-zero for an honest challenge.
	fn eval_at_zero_eq(&self, claim: F, alpha: F) -> F {
		(claim - self.0[0] * alpha) * (F::ONE - alpha).invert_or_zero()
	}
}

impl<F: Field> RoundEvals<F, 1> {
	/// Interpolates the prime round polynomial of an MLE-check.
	///
	/// # Arguments
	///
	/// * `claim` - This round's claim on the prime polynomial.
	/// * `alpha` - The coordinate of the evaluation point that this round binds.
	pub fn interpolate_eq(self, claim: F, alpha: F) -> RoundCoeffs<F> {
		let y_0 = self.eval_at_zero_eq(claim, alpha);
		let [y_1] = self.0;

		// For P(X) = c_1 X + c_0:
		//
		//     P(0) =       c_0
		//     P(1) = c_1 + c_0
		RoundCoeffs(vec![y_0, y_1 - y_0])
	}
}

impl<F: Field> RoundEvals<F, 2> {
	/// Interpolates a plain sumcheck round polynomial.
	///
	/// # Arguments
	///
	/// * `claim` - This round's sum claim.
	pub fn interpolate(self, claim: F) -> RoundCoeffs<F> {
		// The verifier checks `claim = R(0) + R(1)`, so the prover never samples 0.
		let y_0 = claim - self.0[0];
		self.coeffs_from_zero(y_0)
	}

	/// Interpolates the prime round polynomial of an MLE-check.
	///
	/// # Arguments
	///
	/// * `claim` - This round's claim on the prime polynomial.
	/// * `alpha` - The coordinate of the evaluation point that this round binds.
	pub fn interpolate_eq(self, claim: F, alpha: F) -> RoundCoeffs<F> {
		let y_0 = self.eval_at_zero_eq(claim, alpha);
		self.coeffs_from_zero(y_0)
	}

	/// Solves for the monomial coefficients, given the recovered evaluation at 0.
	fn coeffs_from_zero(self, y_0: F) -> RoundCoeffs<F> {
		let [y_1, y_inf] = self.0;

		// For P(X) = c_2 X^2 + c_1 X + c_0:
		//
		//     P(0)   =             c_0
		//     P(1)   = c_2 + c_1 + c_0
		//     P(inf) = c_2
		RoundCoeffs(vec![y_0, y_1 - y_0 - y_inf, y_inf])
	}
}

impl<T: AddAssign + Copy, const D: usize> AddAssign<&Self> for RoundEvals<T, D> {
	fn add_assign(&mut self, rhs: &Self) {
		for (slot, &eval) in iter::zip(&mut self.0, &rhs.0) {
			*slot += eval;
		}
	}
}

impl<T: AddAssign + Copy, const D: usize> Add<&Self> for RoundEvals<T, D> {
	type Output = Self;

	fn add(mut self, rhs: &Self) -> Self::Output {
		self += rhs;
		self
	}
}

impl<P: PackedField, const D: usize> Mul<P::Scalar> for RoundEvals<P, D> {
	type Output = Self;

	fn mul(mut self, rhs: P::Scalar) -> Self::Output {
		for eval in &mut self.0 {
			*eval *= rhs;
		}
		self
	}
}

#[cfg(test)]
mod tests {
	use binius_field::FieldOps;
	use binius_math::test_utils::Packed128b;
	use proptest::prelude::*;

	use super::*;

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;

	#[test]
	fn from_slots_reads_the_run_in_order() {
		let slots = [F::from(1u128), F::from(2u128), F::from(3u128)];
		// Only the run handed over is read, not the slots on either side of it.
		let evals = RoundEvals::<F, 2>::from_slots(&slots[1..]);
		assert_eq!(evals.0, [F::from(2u128), F::from(3u128)]);
	}

	#[test]
	#[should_panic(expected = "slot run length must equal the evaluator's reported degree")]
	fn from_slots_rejects_a_run_of_the_wrong_length() {
		let slots = [F::from(1u128), F::from(2u128), F::from(3u128)];
		let _ = RoundEvals::<F, 2>::from_slots(&slots);
	}

	#[test]
	fn add_to_accumulates_into_the_run_alone() {
		let mut slots = [F::from(1u128), F::from(2u128), F::from(3u128)];
		// A two-slot run written at offset 1 must leave slot 0 untouched.
		RoundEvals::<F, 2>([F::from(4u128), F::from(5u128)]).add_to(&mut slots[1..]);
		assert_eq!(slots[0], F::from(1u128));
		assert_eq!(slots[1], F::from(2u128) + F::from(4u128));
		assert_eq!(slots[2], F::from(3u128) + F::from(5u128));
	}

	#[test]
	#[should_panic(expected = "slot run length must equal the evaluator's reported degree")]
	fn add_to_rejects_a_run_of_the_wrong_length() {
		let mut slots = [F::from(1u128)];
		RoundEvals::<F, 2>([F::from(2u128), F::from(3u128)]).add_to(&mut slots);
	}

	proptest! {
		// Invariant: a plain sumcheck round polynomial reproduces the claim over its endpoints.
		//
		// This is the identity the verifier checks each round.
		// The two sampled nodes must also come back out of the coefficients.
		#[test]
		fn interpolate_2_satisfies_the_sumcheck_identity(
			y_1 in any::<u128>(),
			y_inf in any::<u128>(),
			claim in any::<u128>(),
		) {
			let [y_1, y_inf, claim] = [y_1, y_inf, claim].map(F::from);

			let coeffs = RoundEvals::<F, 2>([y_1, y_inf]).interpolate(claim);

			prop_assert_eq!(coeffs.0.len(), 3);
			prop_assert_eq!(coeffs.sum_over_endpoints(), claim);
			prop_assert_eq!(coeffs.evaluate(&F::ONE), y_1);
			// The leading coefficient is by definition the evaluation at infinity.
			prop_assert_eq!(coeffs.0[2], y_inf);
		}

		// Invariant: a degree-2 prime round polynomial reproduces the MLE-check claim.
		//
		// The verifier reduces with `claim = (1 - alpha) R(0) + alpha R(1)`.
		// alpha = 1 makes that recovery singular, so an honest challenge excludes it.
		#[test]
		fn interpolate_eq_2_satisfies_the_mlecheck_identity(
			y_1 in any::<u128>(),
			y_inf in any::<u128>(),
			claim in any::<u128>(),
			alpha in any::<u128>(),
		) {
			let [y_1, y_inf, claim, alpha] = [y_1, y_inf, claim, alpha].map(F::from);
			prop_assume!(alpha != F::ONE);

			let coeffs = RoundEvals::<F, 2>([y_1, y_inf]).interpolate_eq(claim, alpha);

			prop_assert_eq!(coeffs.0.len(), 3);
			let reduced = (F::ONE - alpha) * coeffs.evaluate(&F::ZERO)
				+ alpha * coeffs.evaluate(&F::ONE);
			prop_assert_eq!(reduced, claim);
			prop_assert_eq!(coeffs.evaluate(&F::ONE), y_1);
			prop_assert_eq!(coeffs.0[2], y_inf);
		}

		// Invariant: the degree-1 prime polynomial satisfies the same MLE-check identity.
		//
		// One sampled node suffices, since the claim pins the second.
		#[test]
		fn interpolate_eq_1_satisfies_the_mlecheck_identity(
			y_1 in any::<u128>(),
			claim in any::<u128>(),
			alpha in any::<u128>(),
		) {
			let [y_1, claim, alpha] = [y_1, claim, alpha].map(F::from);
			prop_assume!(alpha != F::ONE);

			let coeffs = RoundEvals::<F, 1>([y_1]).interpolate_eq(claim, alpha);

			prop_assert_eq!(coeffs.0.len(), 2);
			let reduced = (F::ONE - alpha) * coeffs.evaluate(&F::ZERO)
				+ alpha * coeffs.evaluate(&F::ONE);
			prop_assert_eq!(reduced, claim);
			prop_assert_eq!(coeffs.evaluate(&F::ONE), y_1);
		}

		// Invariant: deferring the reduction changes nothing but the number of reductions.
		//
		// This is what lets an evaluator accumulate wide across a whole chunk.
		//
		//     reduce(w_a + w_b)  ==  reduce(w_a) + reduce(w_b)
		#[test]
		fn reduce_is_additive_over_the_wide_accumulator(
			a in any::<u128>(),
			b in any::<u128>(),
			c in any::<u128>(),
			d in any::<u128>(),
		) {
			let [a, b, c, d] = [a, b, c, d].map(|bits| P::broadcast(F::from(bits)));
			let first = RoundEvals::<_, 2>([P::wide_mul(a, b), P::wide_mul(b, c)]);
			let second = RoundEvals::<_, 2>([P::wide_mul(c, d), P::wide_mul(d, a)]);

			// Accumulate both contributions into one run, exactly as an evaluator does.
			let mut acc = RoundEvals::<<P as WideMul>::Output, 2>::default();
			first.add_to(&mut acc.0);
			second.add_to(&mut acc.0);
			let deferred = acc.reduce::<P>();

			let eager = first.reduce::<P>() + &second.reduce::<P>();

			prop_assert_eq!(deferred.0[0], eager.0[0]);
			prop_assert_eq!(deferred.0[1], eager.0[1]);
		}

		// Invariant: `sum_scalars` collapses only the live lanes of each slot.
		#[test]
		fn sum_scalars_sums_the_live_lanes(bits in any::<u128>(), n_vars in 0usize..=2) {
			let value = F::from(bits);
			let evals = RoundEvals::<P, 1>([P::broadcast(value)]);

			let expected: F = (0..1 << n_vars).map(|_| value).sum();
			prop_assert_eq!(evals.sum_scalars(n_vars).0[0], expected);
		}
	}
}
