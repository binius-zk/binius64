// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::Field;
use binius_ip::mlecheck::mask_buffer_dimensions;
pub use binius_spartan_frontend::constraint_system::BlindingInfo;
use binius_spartan_frontend::constraint_system::{
	ConstraintSystem, MulConstraint, Operand, Witness, WitnessIndex,
};
use binius_utils::checked_arithmetics::{checked_log_2, log2_ceil_usize};

/// A constraint system with blinding and power-of-two padding.
///
/// Wraps a [`ConstraintSystem`], adds dummy constraints for blinding, and pads the total
/// number of constraints to a power of two (required by the prover's multilinear extension
/// protocol).
#[derive(Debug, Clone)]
pub struct ConstraintSystemPadded<F: Field> {
	inner: ConstraintSystem<F>,
	log_private: u32,
	blinding_info: BlindingInfo,
	mul_constraints: Vec<MulConstraint<WitnessIndex>>,
	/// Mask buffer dimensions (m_n, m_d) for the ZK mulcheck mask polynomial.
	mask_dims: (usize, usize),
}

impl<F: Field> ConstraintSystemPadded<F> {
	/// Create a new padded constraint system with blinding.
	///
	/// This:
	/// 1. Adds dummy multiplication constraints for blinding (3 wires each: A * B = C)
	/// 2. Pads the total constraint count to a power of two with `one * one = one` constraints
	/// 3. Calculates the log_size based on witness requirements
	/// 4. Computes mask buffer dimensions for the ZK mulcheck mask polynomial
	pub fn new(cs: ConstraintSystem<F>, blinding_info: BlindingInfo) -> Self {
		let mut mul_constraints = cs.mul_constraints().to_vec();

		// Calculate padded private segment size
		let n_circuit_private = cs.n_private() as usize;
		let n_private =
			n_circuit_private + blinding_info.n_dummy_wires + 3 * blinding_info.n_dummy_constraints;
		let log_private = log2_ceil_usize(n_private) as u32;

		// Add dummy constraints for blinding
		// Each dummy constraint uses 3 consecutive private wires starting after n_dummy_wires
		let dummy_private_base = n_circuit_private + blinding_info.n_dummy_wires;
		for i in 0..blinding_info.n_dummy_constraints {
			let a = WitnessIndex::private((dummy_private_base + 3 * i) as u32);
			let b = WitnessIndex::private((dummy_private_base + 3 * i + 1) as u32);
			let c = WitnessIndex::private((dummy_private_base + 3 * i + 2) as u32);

			mul_constraints.push(MulConstraint {
				a: Operand::from(a),
				b: Operand::from(b),
				c: Operand::from(c),
			});
		}

		// Pad to next power of two with `one * one = one` constraints
		let one_operand = Operand::from(cs.one_wire());
		let current_len = mul_constraints.len();
		mul_constraints.resize(
			current_len.next_power_of_two(),
			MulConstraint {
				a: one_operand.clone(),
				b: one_operand.clone(),
				c: one_operand.clone(),
			},
		);

		// Calculate mask buffer dimensions
		let log_mul_constraints = checked_log_2(mul_constraints.len());
		let mask_degree = 2; // quadratic composition
		let mask_dims =
			mask_buffer_dimensions(log_mul_constraints, mask_degree, blinding_info.n_dummy_wires);

		Self {
			inner: cs,
			log_private,
			blinding_info,
			mul_constraints,
			mask_dims,
		}
	}

	pub fn constants(&self) -> &[F] {
		self.inner.constants()
	}

	pub fn n_inout(&self) -> u32 {
		self.inner.n_inout()
	}

	pub fn n_private(&self) -> u32 {
		self.inner.n_private()
	}

	pub fn log_public(&self) -> u32 {
		self.inner.log_public()
	}

	pub fn n_public(&self) -> u32 {
		self.inner.n_public()
	}

	pub fn one_wire(&self) -> WitnessIndex {
		self.inner.one_wire()
	}

	pub fn log_private(&self) -> u32 {
		self.log_private
	}

	pub fn private_size(&self) -> usize {
		1 << self.log_private as usize
	}

	pub fn blinding_info(&self) -> &BlindingInfo {
		&self.blinding_info
	}

	pub fn mul_constraints(&self) -> &[MulConstraint<WitnessIndex>] {
		&self.mul_constraints
	}

	/// Returns the mask buffer dimensions (m_n, m_d) for the ZK mulcheck mask polynomial.
	pub fn mask_dims(&self) -> (usize, usize) {
		self.mask_dims
	}

	pub fn validate(&self, witness: &Witness<F>) {
		assert_eq!(witness.public().len(), 1 << self.log_public() as usize);
		assert_eq!(witness.private().len(), self.private_size());

		let operand_val = |operand: &Operand<WitnessIndex>| {
			operand.wires().iter().map(|&idx| witness[idx]).sum::<F>()
		};

		for MulConstraint { a, b, c } in &self.mul_constraints {
			assert_eq!(operand_val(a) * operand_val(b), operand_val(c));
		}
	}
}
