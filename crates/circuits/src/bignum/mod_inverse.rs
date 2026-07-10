// Copyright 2025 Irreducible Inc.
//! Modular inverse hint implementation

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, hints::Hint, util::num_biguint_from_u64_limbs};

/// ModInverse hint implementation
pub struct ModInverseHint;

impl ModInverseHint {
	pub const fn new() -> Self {
		Self
	}

	/// Modular inverse.
	///
	/// Computes the modular inverse of `base` modulo `modulus`.
	/// Returns a pair `(quotient, inverse)` where both numbers are Bézout coefficients when
	/// `base` and `modulus` are coprime. Both numbers are set to zero if `gcd(base, modulus) > 1`.
	///
	/// This is a hint - a deterministic computation that happens only on the prover side.
	/// The result should be additionally constrained by using bignum circuits to check that
	/// `base * inverse = 1 + quotient * modulus`.
	pub fn call(
		builder: &CircuitBuilder,
		base: &[Wire],
		modulus: &[Wire],
	) -> (Vec<Wire>, Vec<Wire>) {
		let inputs: Vec<Wire> = base.iter().chain(modulus).copied().collect();
		let mut out = builder.call_hint(Self::new(), &[base.len(), modulus.len()], &inputs);
		let inverse = out.split_off(modulus.len());
		(out, inverse)
	}
}

impl Default for ModInverseHint {
	fn default() -> Self {
		Self::new()
	}
}

impl Hint for ModInverseHint {
	const NAME: &'static str = "binius.mod_inverse";

	fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
		let [base_limbs, mod_limbs] = dimensions else {
			panic!("ModInverse requires 2 dimensions");
		};
		(*base_limbs + *mod_limbs, 2 * *mod_limbs)
	}

	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let [n_base, n_mod] = dimensions else {
			panic!("ModInverse requires 2 dimensions");
		};

		let base_limbs = &inputs[0..*n_base];
		let mod_limbs = &inputs[*n_base..];

		let base = num_biguint_from_u64_limbs(base_limbs.iter().map(|w| w.as_u64()));
		let modulus = num_biguint_from_u64_limbs(mod_limbs.iter().map(|w| w.as_u64()));

		let zero = num_bigint::BigUint::ZERO;
		let (quotient, inverse) = if let Some(inverse) = base.modinv(&modulus) {
			let quotient = (base * &inverse - num_bigint::BigUint::from(1usize)) / &modulus;
			(quotient, inverse)
		} else {
			(zero.clone(), zero)
		};

		assert_eq!(outputs.len(), 2 * *n_mod);
		let (quotient_words, inverse_words) = outputs.split_at_mut(*n_mod);

		// Fill output quotient limbs
		for (i, limb) in quotient.iter_u64_digits().enumerate() {
			quotient_words[i] = Word::from_u64(limb);
		}

		// Zero remaining outputs if quotient has fewer limbs
		for i in quotient.iter_u64_digits().len()..*n_mod {
			quotient_words[i] = Word::ZERO;
		}

		// Fill output inverse limbs
		for (i, limb) in inverse.iter_u64_digits().enumerate() {
			inverse_words[i] = Word::from_u64(limb);
		}
		// Zero remaining outputs if inverse has fewer limbs
		for i in inverse.iter_u64_digits().len()..*n_mod {
			inverse_words[i] = Word::ZERO;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_mod_inverse_hint() {
		let builder = CircuitBuilder::new();

		let b = builder.add_constant_64(0x123456789abcdef0);

		// M12 = 2^127-1
		let m0 = builder.add_constant_64(u64::MAX);
		let m1 = builder.add_constant_64((1u64 << 63) - 1);

		let (quotient, inverse) = ModInverseHint::call(&builder, &[b], &[m0, m1]);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		assert_eq!(inverse.len(), 2);
		assert_eq!(w[inverse[0]], Word(0xe5a542e11f99750a));
		assert_eq!(w[inverse[1]], Word(0x1849faf75fbb9752));

		assert_eq!(quotient.len(), 2);
		assert_eq!(w[quotient[0]], Word(0x37455c1554b9aa1));
		assert_eq!(w[quotient[1]], Word::ZERO);
	}

	#[test]
	fn test_mod_inverse_hint_non_coprime() {
		let builder = CircuitBuilder::new();

		let b = builder.add_constant_64((1 << 19) - 1);

		// M7 * M11 = (2^19-1)*(2^107-1)
		let m0 = builder.add_constant_64(0xfffffffffff80001);
		let m1 = builder.add_constant_64(0x3ffff7ffffffffff);

		let (quotient, inverse) = ModInverseHint::call(&builder, &[b], &[m0, m1]);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		assert_eq!(inverse.len(), 2);
		assert_eq!(w[inverse[0]], Word::ZERO);
		assert_eq!(w[inverse[1]], Word::ZERO);

		assert_eq!(quotient.len(), 2);
		assert_eq!(w[quotient[0]], Word::ZERO);
		assert_eq!(w[quotient[1]], Word::ZERO);
	}
}
