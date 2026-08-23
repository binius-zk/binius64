// Copyright 2026 The Binius Developers

//! The numerator/denominator pair a fractional-addition instance is built from.

use binius_field::Field;

/// A numerator paired with its denominator.
///
/// Fractional addition never touches one half alone:
///
/// ```text
///     a_0/b_0 + a_1/b_1 = (a_0*b_1 + a_1*b_0) / (b_0*b_1)
/// ```
///
/// So the two travel together, through every layer and every claim.
/// Pairing them in one type makes a half-formed pair unrepresentable.
/// A bare tuple leaves each caller to remember which element is which.
///
/// `T` is whatever stands for one half.
/// A scalar gives a claimed fraction; a column buffer gives one layer of the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fraction<T> {
	/// The numerator.
	pub num: T,
	/// The denominator.
	pub den: T,
}

impl<T> Fraction<T> {
	/// Pairs a numerator with a denominator.
	pub const fn new(num: T, den: T) -> Self {
		Self { num, den }
	}

	/// Borrows both halves, so a fraction can be read without moving out of it.
	pub const fn as_ref(&self) -> Fraction<&T> {
		Fraction {
			num: &self.num,
			den: &self.den,
		}
	}

	/// Applies `f` to each half.
	///
	/// This is how a fraction changes representation, such as a pair of layer buffers reduced to
	/// the pair of scalars at their root.
	pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Fraction<U> {
		Fraction {
			num: f(self.num),
			den: f(self.den),
		}
	}
}

impl<F: Field> Fraction<F> {
	/// The zero fraction $0/1$.
	///
	/// This is the additive identity of fractional addition: adding it changes nothing.
	/// So it is what a padding leaf holds when a batch lifts a shallow tree to its depth, and what
	/// fills the selector slots past the last real instance.
	pub const ZERO: Self = Self {
		num: F::ZERO,
		den: F::ONE,
	};
}

impl<T> From<(T, T)> for Fraction<T> {
	fn from((num, den): (T, T)) -> Self {
		Self { num, den }
	}
}

impl<T> From<Fraction<T>> for (T, T) {
	fn from(Fraction { num, den }: Fraction<T>) -> Self {
		(num, den)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::FieldOps;
	use binius_math::test_utils::{Packed128b, random_scalars};
	use rand::prelude::*;

	use super::*;

	type F = <Packed128b as FieldOps>::Scalar;

	#[test]
	fn zero_fraction_is_the_additive_identity() {
		let mut rng = StdRng::seed_from_u64(0);
		let [num, den]: [F; 2] = random_scalars::<F>(&mut rng, 2)
			.try_into()
			.expect("two scalars");
		let f = Fraction::new(num, den);

		// Adding 0/1 to a/b by the fractional-addition rule must give back a/b unchanged.
		let pad = Fraction::<F>::ZERO;
		let sum = Fraction::new(f.num * pad.den + pad.num * f.den, f.den * pad.den);
		assert_eq!(sum, f);
	}

	#[test]
	fn tuple_conversion_round_trips() {
		let mut rng = StdRng::seed_from_u64(1);
		let [num, den]: [F; 2] = random_scalars::<F>(&mut rng, 2)
			.try_into()
			.expect("two scalars");

		let tuple: (F, F) = Fraction::new(num, den).into();
		assert_eq!(tuple, (num, den));
		assert_eq!(Fraction::from(tuple), Fraction::new(num, den));
	}

	#[test]
	fn map_and_as_ref_reach_both_halves() {
		let f = Fraction::new(vec![1u8, 2], vec![3u8]);
		assert_eq!(f.as_ref().map(Vec::len), Fraction::new(2, 1));
	}
}
