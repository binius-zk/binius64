// Copyright 2026 The Binius Developers
//! Modular division hint implementation

use binius_core::Word;

use super::Hint;
use crate::util::num_biguint_from_u64_limbs;

/// ModDivide hint implementation.
///
/// Computes the modular quotient `slope = dividend * divisor^{-1} (mod modulus)` together with
/// the integer `quotient` witnessing the reduction `slope * divisor = dividend + quotient *
/// modulus`. Both outputs are set to zero when `divisor` is not invertible modulo `modulus`
/// (e.g. `divisor == 0` or `gcd(divisor, modulus) > 1`).
pub struct ModDivideHint;

impl ModDivideHint {
	pub const fn new() -> Self {
		Self
	}
}

impl Default for ModDivideHint {
	fn default() -> Self {
		Self::new()
	}
}

impl Hint for ModDivideHint {
	const NAME: &'static str = "binius.mod_divide";

	fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
		let [dividend_limbs, divisor_limbs, mod_limbs] = dimensions else {
			panic!("ModDivide requires 3 dimensions");
		};
		(*dividend_limbs + *divisor_limbs + *mod_limbs, 2 * *mod_limbs)
	}

	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let [n_dividend, n_divisor, n_mod] = dimensions else {
			panic!("ModDivide requires 3 dimensions");
		};

		let dividend_limbs = &inputs[0..*n_dividend];
		let divisor_limbs = &inputs[*n_dividend..*n_dividend + *n_divisor];
		let mod_limbs = &inputs[*n_dividend + *n_divisor..];

		let dividend = num_biguint_from_u64_limbs(dividend_limbs.iter().map(|w| w.as_u64()));
		let divisor = num_biguint_from_u64_limbs(divisor_limbs.iter().map(|w| w.as_u64()));
		let modulus = num_biguint_from_u64_limbs(mod_limbs.iter().map(|w| w.as_u64()));

		let zero = num_bigint::BigUint::ZERO;
		let (quotient, slope) = if let Some(inverse) = divisor.modinv(&modulus) {
			let slope = (&dividend * &inverse) % &modulus;
			let numerator = &slope * &divisor;
			// `slope * divisor ≡ dividend (mod modulus)`, so when `dividend < modulus` (the
			// gadget's precondition) `numerator >= dividend` and `quotient` is a non-negative
			// integer. If `dividend >= modulus` this subtraction would underflow; clamp to zero
			// so witness generation never panics and let the (then unsatisfiable) reduction
			// constraint surface the incompleteness. See `PseudoMersennePrimeField::div`.
			let quotient = if numerator >= dividend {
				(numerator - &dividend) / &modulus
			} else {
				zero
			};
			(quotient, slope)
		} else {
			(zero.clone(), zero)
		};

		assert_eq!(outputs.len(), 2 * *n_mod);
		let (quotient_words, slope_words) = outputs.split_at_mut(*n_mod);

		// Fill quotient limbs, ignoring any that exceed the output arity (only reachable when
		// the precondition is violated) and zeroing the remainder.
		for (i, limb) in quotient.iter_u64_digits().enumerate() {
			if i < *n_mod {
				quotient_words[i] = Word::from_u64(limb);
			}
		}
		for i in quotient.iter_u64_digits().len()..*n_mod {
			quotient_words[i] = Word::ZERO;
		}

		// Fill slope limbs and zero the remainder.
		for (i, limb) in slope.iter_u64_digits().enumerate() {
			if i < *n_mod {
				slope_words[i] = Word::from_u64(limb);
			}
		}
		for i in slope.iter_u64_digits().len()..*n_mod {
			slope_words[i] = Word::ZERO;
		}
	}
}
