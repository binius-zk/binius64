// Copyright 2025 Irreducible Inc.
//! BigUint division hint implementation

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, hints::Hint, util::num_biguint_from_u64_limbs};

pub struct BigUintDivideHint;

impl BigUintDivideHint {
	pub const fn new() -> Self {
		Self
	}

	/// BigUint division.
	///
	/// Returns `(quotient, remainder)` of the division of `dividend` by `divisor`.
	///
	/// This is a hint - a deterministic computation that happens only on the prover side.
	/// The result should be additionally constrained by using bignum circuits to check that
	/// `remainder + divisor * quotient == dividend`.
	pub fn call(
		builder: &CircuitBuilder,
		dividend: &[Wire],
		divisor: &[Wire],
	) -> (Vec<Wire>, Vec<Wire>) {
		let inputs: Vec<Wire> = dividend.iter().chain(divisor).copied().collect();
		let mut out = builder.call_hint(Self::new(), &[dividend.len(), divisor.len()], &inputs);
		let remainder = out.split_off(dividend.len());
		(out, remainder)
	}
}

impl Default for BigUintDivideHint {
	fn default() -> Self {
		Self::new()
	}
}

impl Hint for BigUintDivideHint {
	const NAME: &'static str = "binius.biguint_divide";

	fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
		let [dividend_limbs, divisor_limbs] = dimensions else {
			panic!("BigUintDivide requires 2 dimensions");
		};
		(*dividend_limbs + *divisor_limbs, *dividend_limbs + *divisor_limbs)
	}

	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let [n_dividend, n_divisor] = dimensions else {
			panic!("BigUintDivide requires 2 dimensions");
		};
		let n_quotient = *n_dividend;
		let n_remainder = *n_divisor;

		let dividend_limbs = &inputs[0..*n_dividend];
		let divisor_limbs = &inputs[*n_dividend..];

		let dividend = num_biguint_from_u64_limbs(dividend_limbs.iter().map(|w| w.as_u64()));
		let divisor = num_biguint_from_u64_limbs(divisor_limbs.iter().map(|w| w.as_u64()));

		let zero = num_bigint::BigUint::ZERO;
		let (quotient, remainder) = if divisor != zero {
			(dividend.clone() / divisor.clone(), dividend % divisor)
		} else {
			(zero.clone(), zero)
		};

		// Fill quotient limbs (first part of output)
		for (i, limb) in quotient.iter_u64_digits().enumerate() {
			if i < n_quotient {
				outputs[i] = Word::from_u64(limb);
			}
		}
		// Zero remaining quotient outputs
		for i in quotient.iter_u64_digits().len()..n_quotient {
			outputs[i] = Word::ZERO;
		}

		// Fill remainder limbs (second part of output)
		for (i, limb) in remainder.iter_u64_digits().enumerate() {
			if i < n_remainder {
				outputs[n_quotient + i] = Word::from_u64(limb);
			}
		}
		// Zero remaining remainder outputs
		for i in remainder.iter_u64_digits().len()..n_remainder {
			outputs[n_quotient + i] = Word::ZERO;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_biguint_divide_hint() {
		let builder = CircuitBuilder::new();

		// (2^128-1) % (2^64-5) = 24
		let d0 = builder.add_constant_64(u64::MAX);
		let d1 = builder.add_constant_64(u64::MAX);

		let m = builder.add_constant_64(u64::MAX - 4);

		let (q, r) = BigUintDivideHint::call(&builder, &[d0, d1], &[m]);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		assert_eq!(r.len(), 1);
		assert_eq!(w[r[0]], Word(24));

		assert_eq!(q.len(), 2);
		assert_eq!(w[q[0]], Word(5));
		assert_eq!(w[q[1]], Word(1));
	}

	#[test]
	fn test_biguint_divide_hint_div_by_zero() {
		let builder = CircuitBuilder::new();

		let d0 = builder.add_constant_64(u64::MAX);
		let d1 = builder.add_constant_64(u64::MAX);

		let m0 = builder.add_constant_64(0);
		let m1 = builder.add_constant_64(0);

		let (q, r) = BigUintDivideHint::call(&builder, &[d0, d1], &[m0, m1]);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		assert_eq!(r.len(), 2);
		assert_eq!(w[r[0]], Word(0));
		assert_eq!(w[r[1]], Word(0));

		assert_eq!(q.len(), 2);
		assert_eq!(w[q[0]], Word(0));
		assert_eq!(w[q[1]], Word(0));
	}
}
