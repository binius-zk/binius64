// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::{
	BigUint, BigUintDivideHint, ModDivideHint, ModInverseHint, PseudoMersenneModReduce,
	addsub::{add_with_carry_out, sub_with_borrow_out},
	biguint_lt, num_biguint_from_u64_limbs, sub, textbook_mul, textbook_square,
};

/// A struct that implements prime field arithmetic over pseudo-Mersenne modulus.
///
/// Field elements are `BigUint`s consisting of `PseudoMersennePrimeField::limbs_len` limbs.
/// It is assumed that all field elements are correctly represented (less than modulus).
pub struct PseudoMersennePrimeField {
	modulus: BigUint,
	modulus_po2: usize,
	modulus_subtrahend: BigUint,
}

impl PseudoMersennePrimeField {
	/// Create a new pseudo-Mersenne prime field.
	///
	/// See [`PseudoMersenneModReduce`] for description of the parameters.
	pub fn new(b: &CircuitBuilder, modulus_po2: usize, modulus_subtrahend: &[u64]) -> Self {
		let modulus_subtrahend = num_biguint_from_u64_limbs(modulus_subtrahend);
		let po2 = num_bigint::BigUint::from(2usize).pow(modulus_po2 as u32);
		assert!(po2 > modulus_subtrahend, "2^modulus_po2 - modulus_subtrahend > 0");

		let modulus = BigUint::new_constant(b, &(po2 - &modulus_subtrahend));
		let modulus_subtrahend = BigUint::new_constant(b, &modulus_subtrahend);

		Self {
			modulus,
			modulus_po2,
			modulus_subtrahend,
		}
	}

	/// Number of limbs in `BigUint`s representing field elements.
	pub const fn limbs_len(&self) -> usize {
		self.modulus.limbs.len()
	}

	/// Field modulus.
	pub const fn modulus(&self) -> &BigUint {
		&self.modulus
	}

	/// Field addition.
	///
	/// Equivalent formula: `(fe1 + fe2) % modulus`
	pub fn add(&self, b: &CircuitBuilder, fe1: &BigUint, fe2: &BigUint) -> BigUint {
		let l = self.limbs_len();
		assert!(fe1.limbs.len() == l && fe2.limbs.len() == l);

		// Both inputs are reduced, so the sum is below `2 * modulus` and the overflow past
		// the top limb is a single bit. Carrying it in a wire avoids widening every limbwise
		// operation below by an extra limb.
		let (sum, carry) = add_with_carry_out(b, fe1, fe2);

		// A carry means the sum reached `2^(64 * l)`, which is above the modulus, so the
		// comparison only matters when no carry occurred: with a carry the wrapped sum is
		// below the modulus. Carry and the comparison are therefore mutually exclusive, so
		// XOR matches the OR on the MSB that `zero_unless` reads.
		// TODO: consider nondeterminism
		let need_reduction = b.bxor(carry, b.bnot(biguint_lt(b, &sum, &self.modulus)));

		// Wrapping subtraction is exact here: without a carry the difference is non-negative,
		// and with one it borrows out by exactly the `2^(64 * l)` the carry stands for.
		let (reduced, _borrow) =
			sub_with_borrow_out(b, &sum, &self.modulus.zero_unless(b, need_reduction));

		reduced
	}

	/// Field subtraction.
	///
	/// Equivalent formula: `(fe1 - fe2) % modulus`
	pub fn sub(&self, b: &CircuitBuilder, fe1: &BigUint, fe2: &BigUint) -> BigUint {
		assert!(fe1.limbs.len() == self.limbs_len() && fe2.limbs.len() == self.limbs_len());
		// NB: fe1 - fe2 = fe1 + modulus - fe2 <= 2*modulus - 1 (one subtraction still normalizes)
		let fe2_add_inv = sub(b, &self.modulus, fe2);
		self.add(b, fe1, &fe2_add_inv)
	}

	/// Field squaring.
	///
	/// Equivalent formula: `(fe ** 2) % modulus`
	pub fn square(&self, b: &CircuitBuilder, fe: &BigUint) -> BigUint {
		assert!(fe.limbs.len() == self.limbs_len());
		self.reduce_product(b, textbook_square(b, fe))
	}

	/// Field multiplication.
	///
	/// Equivalent formula: `(fe1 * fe2) % modulus`
	/// Note: Both fe1 and fe2 may be greater or equal to modulus.
	pub fn mul(&self, b: &CircuitBuilder, fe1: &BigUint, fe2: &BigUint) -> BigUint {
		assert!(fe1.limbs.len() == self.limbs_len() && fe2.limbs.len() == self.limbs_len());
		self.reduce_product(b, textbook_mul(b, fe1, fe2))
	}

	fn reduce_product(&self, b: &CircuitBuilder, product: BigUint) -> BigUint {
		let (quotient, remainder) = BigUintDivideHint::call(b, &product.limbs, &self.modulus.limbs);

		let zero = b.add_constant(Word::ZERO);

		let quotient = BigUint { limbs: quotient };
		let remainder = BigUint { limbs: remainder }.pad_limbs_to(self.limbs_len(), zero);

		b.assert_true("remainder < modulus", biguint_lt(b, &remainder, &self.modulus));

		// constraint: product == remainder + quotient * modulus
		PseudoMersenneModReduce::new(
			b,
			&product,
			self.modulus_po2,
			&self.modulus_subtrahend,
			&quotient,
			&remainder,
		)
		.constrain(b);

		remainder
	}

	/// Field inverse.
	///
	/// Equivalent formula (for prime modulus): `(fe1 ** (modulus - 2)) % modulus`
	/// The wire parameter `exists` is a boolean-wire signifying the existence of the inverse;
	/// if `exists` is false, the modular reduction constraint is not applied. This is useful
	/// for avoiding overconstraining in skipped parts of larger circuits.
	pub fn inverse(&self, b: &CircuitBuilder, fe: &BigUint, exists: Wire) -> BigUint {
		assert!(fe.limbs.len() == self.limbs_len());
		let (quotient, inverse) = ModInverseHint::call(b, &fe.limbs, &self.modulus.limbs);

		let zero = b.add_constant(Word::ZERO);

		let quotient = BigUint { limbs: quotient };
		let inverse = BigUint { limbs: inverse }.pad_limbs_to(self.limbs_len(), zero);
		let one = BigUint::new_constant(b, &num_bigint::BigUint::from(1usize))
			.pad_limbs_to(self.limbs_len(), zero);

		let product = textbook_mul(b, &inverse, fe);

		b.assert_true("inverse < modulus", biguint_lt(b, &inverse, &self.modulus));

		// constraint: base * inverse = 1 + quotient * modulus
		PseudoMersenneModReduce::new(
			b,
			&product,
			self.modulus_po2,
			&self.modulus_subtrahend,
			&quotient,
			&one,
		)
		.constrain_cond(b, exists);

		inverse
	}

	/// Field division.
	///
	/// Equivalent formula (for prime modulus): `(dividend * divisor ** (modulus - 2)) % modulus`,
	/// i.e. `dividend / divisor (mod modulus)`.
	///
	/// This collapses a modular inverse followed by a multiplication into a single modular
	/// reduction, halving the multiplication cost relative to `mul(dividend, inverse(divisor))`.
	/// The returned `slope` is constrained by `slope * divisor = dividend + quotient * modulus`,
	/// which is `slope * divisor ≡ dividend (mod modulus)`.
	///
	/// The wire parameter `exists` is a boolean-wire signifying the existence of the quotient
	/// (i.e. that `divisor` is invertible modulo the modulus); if `exists` is false, the modular
	/// reduction constraint is not applied. This is useful for avoiding overconstraining in
	/// skipped parts of larger circuits. The `slope < modulus` range check is unconditional, so
	/// when `exists` is false the returned value is an unconstrained dummy `< modulus`.
	///
	/// # Precondition and incompleteness
	///
	/// `dividend` must be reduced (`dividend < modulus`): it plays the role of the remainder in
	/// the reduction `slope * divisor = dividend + quotient * modulus`. If `dividend >= modulus`
	/// there is no non-negative `quotient` satisfying that equation, so (when `exists` is true)
	/// the constraint system has no satisfying witness and proof generation fails — the gadget is
	/// *incomplete* for unreduced dividends. Callers must reduce the dividend beforehand; the
	/// field helpers [`add`](Self::add), [`sub`](Self::sub), [`mul`](Self::mul) and
	/// [`square`](Self::square) all return reduced values. (This matches the standard ECDSA
	/// convention of reducing the message hash to a scalar in `[0, n)` before computing
	/// `u1`/`u2`.)
	pub fn div(
		&self,
		b: &CircuitBuilder,
		dividend: &BigUint,
		divisor: &BigUint,
		exists: Wire,
	) -> BigUint {
		assert!(
			dividend.limbs.len() == self.limbs_len() && divisor.limbs.len() == self.limbs_len()
		);
		let (quotient, slope) =
			ModDivideHint::call(b, &dividend.limbs, &divisor.limbs, &self.modulus.limbs);

		let zero = b.add_constant(Word::ZERO);

		let quotient = BigUint { limbs: quotient };
		let slope = BigUint { limbs: slope }.pad_limbs_to(self.limbs_len(), zero);

		let product = textbook_mul(b, &slope, divisor);

		b.assert_true("slope < modulus", biguint_lt(b, &slope, &self.modulus));

		// constraint: slope * divisor = dividend + quotient * modulus
		PseudoMersenneModReduce::new(
			b,
			&product,
			self.modulus_po2,
			&self.modulus_subtrahend,
			&quotient,
			dividend,
		)
		.constrain_cond(b, exists);

		slope
	}
}
