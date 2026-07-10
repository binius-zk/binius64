// Copyright 2025 Irreducible Inc.
//! BigUint mod pow hint implementation

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, hints::Hint, util::num_biguint_from_u64_limbs};

pub struct BigUintModPowHint;

impl BigUintModPowHint {
	pub const fn new() -> Self {
		Self
	}

	/// Modular exponentiation.
	///
	/// Computes `(base^exp) % modulus`.
	/// This is a hint - a deterministic computation that happens only on the prover side.
	/// The result should be additionally constrained using bignum circuits.
	pub fn call(
		builder: &CircuitBuilder,
		base: &[Wire],
		exp: &[Wire],
		modulus: &[Wire],
	) -> Vec<Wire> {
		let inputs: Vec<Wire> = base.iter().chain(exp).chain(modulus).copied().collect();
		builder.call_hint(Self::new(), &[base.len(), exp.len(), modulus.len()], &inputs)
	}
}

impl Default for BigUintModPowHint {
	fn default() -> Self {
		Self::new()
	}
}

impl Hint for BigUintModPowHint {
	const NAME: &'static str = "binius.biguint_mod_pow";

	fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
		let [n_base_limbs, n_exp_limbs, n_modulus_limbs] = dimensions else {
			panic!("BigUintModPowHint requires 3 dimensions");
		};
		(*n_base_limbs + *n_exp_limbs + *n_modulus_limbs, *n_modulus_limbs)
	}

	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let [n_base_limbs, n_exp_limbs, n_modulus_limbs] = dimensions else {
			panic!("BigUintModPowHint requires 3 dimensions");
		};

		assert_eq!(inputs.len(), *n_base_limbs + *n_exp_limbs + *n_modulus_limbs);
		assert_eq!(outputs.len(), *n_modulus_limbs);

		let (base_limbs, inputs) = inputs.split_at(*n_base_limbs);
		let (exp_limbs, modulus_limbs) = inputs.split_at(*n_exp_limbs);

		let base = num_biguint_from_u64_limbs(base_limbs.iter().map(|w| w.as_u64()));
		let exp = num_biguint_from_u64_limbs(exp_limbs.iter().map(|w| w.as_u64()));
		let modulus = num_biguint_from_u64_limbs(modulus_limbs.iter().map(|w| w.as_u64()));

		let modpow = base.modpow(&exp, &modulus);

		// Fill modpow limbs (first part of the output)
		for (i, limb) in modpow.iter_u64_digits().enumerate() {
			outputs[i] = Word::from_u64(limb);
		}

		for i in modpow.iter_u64_digits().len()..*n_modulus_limbs {
			outputs[i] = Word::ZERO;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_mod_pow_hint() {
		let builder = CircuitBuilder::new();

		let c = builder.add_constant_64(0x123456789abcdef0);
		let modpow = BigUintModPowHint::call(&builder, &[c], &[c, c], &[c, c, c]);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		assert_eq!(modpow.len(), 3);
		assert_eq!(w[modpow[0]], Word(0x6f151e00d2c39f30));
		assert_eq!(w[modpow[1]], Word(0xfef75acc27ead52f));
		assert_eq!(w[modpow[2]], Word(0x00443adf222ea27));
	}
}
