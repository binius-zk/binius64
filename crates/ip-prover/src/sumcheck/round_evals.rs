// Copyright 2023-2025 Irreducible Inc.

use std::ops::{Add, AddAssign, Mul};

use binius_field::{Field, PackedField, WideningMul};
use binius_ip::sumcheck::RoundCoeffs;

// Sumcheck round evaluations for degree-1 polynomials, on point 1 alone.
#[derive(Clone, Debug, Default)]
pub struct RoundEvals1<P: PackedField> {
	pub y_1: P,
}

impl<P: PackedField> RoundEvals1<P> {
	pub fn sum_scalars(self, n_vars: usize) -> RoundEvals1<P::Scalar> {
		RoundEvals1 {
			y_1: self.y_1.iter().take(1 << n_vars).sum(),
		}
	}
}

impl<F: Field> RoundEvals1<F> {
	// Interpolation routine for evaluations on P'(x) in Mlechecks.
	pub fn interpolate_eq(self, sum: F, alpha: F) -> RoundCoeffs<F> {
		let y_0 = (sum - self.y_1 * alpha) * (F::ONE - alpha).invert_or_zero();
		calculate_round_coeffs_from_evals_1(y_0, self.y_1)
	}
}

impl<P: PackedField> Add<&Self> for RoundEvals1<P> {
	type Output = Self;

	fn add(mut self, rhs: &Self) -> Self::Output {
		self += rhs;
		self
	}
}

impl<P: PackedField> AddAssign<&Self> for RoundEvals1<P> {
	fn add_assign(&mut self, rhs: &Self) {
		self.y_1 += rhs.y_1;
	}
}

impl<P: PackedField> Mul<P::Scalar> for RoundEvals1<P> {
	type Output = Self;

	fn mul(mut self, rhs: P::Scalar) -> Self::Output {
		self.y_1 *= rhs;
		self
	}
}

// Sumcheck round evaluations for degree-2 polynomials, on points 1 and ∞. The latter
// is defined as limit of P(X)/X^n as X approaches infinity, which equals the leading coefficient.
// This is the Karatsuba trick. Take note that it may require removing lower-degree terms from the
// composition polynomial.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoundEvals2<P: PackedField> {
	pub y_1: P,
	pub y_inf: P,
}

impl<P: PackedField> RoundEvals2<P> {
	pub fn sum_scalars(self, n_vars: usize) -> RoundEvals2<P::Scalar> {
		RoundEvals2 {
			y_1: self.y_1.iter().take(1 << n_vars).sum(),
			y_inf: self.y_inf.iter().take(1 << n_vars).sum(),
		}
	}
}

impl<F: Field> RoundEvals2<F> {
	// Regular degree-2 interpolation routine.
	pub fn interpolate(self, sum: F) -> RoundCoeffs<F> {
		// Computing evaluation at 0 from sum claim and evaluation on 1.
		let y_0 = sum - self.y_1;
		calculate_round_coeffs_from_evals_2(y_0, self.y_1, self.y_inf)
	}

	// Interpolation routine for evaluations on P'(x) in Mlechecks.
	pub fn interpolate_eq(self, sum: F, alpha: F) -> RoundCoeffs<F> {
		// We are given a sum claim on prime polynomial from the previous round, we also know that
		//  sum = (1 - alpha) * P'(0) + alpha * P'(1)
		let y_0 = (sum - self.y_1 * alpha) * (F::ONE - alpha).invert_or_zero();
		calculate_round_coeffs_from_evals_2(y_0, self.y_1, self.y_inf)
	}
}

impl<P: PackedField> Add<&Self> for RoundEvals2<P> {
	type Output = Self;

	fn add(mut self, rhs: &Self) -> Self::Output {
		self += rhs;
		self
	}
}

impl<P: PackedField> AddAssign<&Self> for RoundEvals2<P> {
	fn add_assign(&mut self, rhs: &Self) {
		self.y_1 += rhs.y_1;
		self.y_inf += rhs.y_inf;
	}
}

impl<P: PackedField> Mul<P::Scalar> for RoundEvals2<P> {
	type Output = Self;

	fn mul(mut self, rhs: P::Scalar) -> Self::Output {
		self.y_1 *= rhs;
		self.y_inf *= rhs;
		self
	}
}

/// Widening (unreduced) accumulator for degree-2 sumcheck round evaluations.
///
/// Stores `y_1` and `y_inf` as `P::Wide` values that can be summed via addition (XOR in GF(2))
/// without intermediate reduction. After accumulation, call [`reduce`](Self::reduce) to convert
/// back to a `RoundEvals2<P>`.
#[derive(Clone, Copy)]
pub struct WideRoundEvals2<W> {
	pub y_1: W,
	pub y_inf: W,
}

impl<W: Default> Default for WideRoundEvals2<W> {
	fn default() -> Self {
		Self {
			y_1: W::default(),
			y_inf: W::default(),
		}
	}
}

impl<W> WideRoundEvals2<W> {
	pub fn reduce<P: WideningMul<Wide = W>>(self) -> RoundEvals2<P> {
		RoundEvals2 {
			y_1: P::reduce_wide(self.y_1),
			y_inf: P::reduce_wide(self.y_inf),
		}
	}
}

impl<W: Add<Output = W>> Add for WideRoundEvals2<W> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self {
			y_1: self.y_1 + rhs.y_1,
			y_inf: self.y_inf + rhs.y_inf,
		}
	}
}

impl<W: AddAssign> AddAssign for WideRoundEvals2<W> {
	fn add_assign(&mut self, rhs: Self) {
		self.y_1 += rhs.y_1;
		self.y_inf += rhs.y_inf;
	}
}

// Computes the coefficients of a degree 1 polynomial interpolating two points (0, y_0) and (1,
// y_1).
fn calculate_round_coeffs_from_evals_1<F: Field>(y_0: F, y_1: F) -> RoundCoeffs<F> {
	// For a polynomial P(X) = c_1 x + c_0:
	//
	// P(0) =        c_0
	// P(1) = c_1  + c_0

	let c_0 = y_0;
	let c_1 = y_1 - c_0;
	RoundCoeffs(vec![c_0, c_1])
}

// Computes the coefficients of a degree 2 polynomial interpolating three points: (0, y_0),
// (1, y_1), and (infinity, y_inf).
fn calculate_round_coeffs_from_evals_2<F: Field>(y_0: F, y_1: F, y_inf: F) -> RoundCoeffs<F> {
	// For a polynomial P(X) = c_2 x² + c_1 x + c_0:
	//
	// P(0) =                  c_0
	// P(1) = c_2    + c_1   + c_0
	// P(∞) = c_2

	let c_0 = y_0;
	let c_2 = y_inf;
	let c_1 = y_1 - c_0 - c_2;
	RoundCoeffs(vec![c_0, c_1, c_2])
}

// Multiplication of a polynomial in monomial form by eq(x, alpha).
pub fn round_coeffs_by_eq<F: Field>(prime: &RoundCoeffs<F>, alpha: F) -> RoundCoeffs<F> {
	// eq(X, α) = (1 − α) + (2 α − 1) X
	// NB: In characteristic 2, this expression can be simplified to 1 + α + challenge.
	let (prime_by_constant_term, mut prime_by_linear_term) = if F::CHARACTERISTIC == 2 {
		(prime.clone() * (F::ONE + alpha), prime.clone())
	} else {
		(prime.clone() * (F::ONE - alpha), prime.clone() * (alpha.double() - F::ONE))
	};

	prime_by_linear_term.0.insert(0, F::ZERO); // Multiply prime polynomial by X
	prime_by_constant_term + &prime_by_linear_term
}
