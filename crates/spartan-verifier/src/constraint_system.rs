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
	log_precommit: u32,
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
		let blinding_size = blinding_info.n_dummy_wires + 3 * blinding_info.n_dummy_constraints;

		/// Adds dummy blinding constraints for a segment and returns its padded log-size.
		fn add_blinding_constraints<F: Field>(
			mul_constraints: &mut Vec<MulConstraint<WitnessIndex>>,
			make_index: fn(u32) -> WitnessIndex,
			n_circuit_wires: usize,
			blinding_info: &BlindingInfo,
			blinding_size: usize,
		) -> u32 {
			let dummy_base = n_circuit_wires + blinding_info.n_dummy_wires;
			for i in 0..blinding_info.n_dummy_constraints {
				let a = make_index((dummy_base + 3 * i) as u32);
				let b = make_index((dummy_base + 3 * i + 1) as u32);
				let c = make_index((dummy_base + 3 * i + 2) as u32);
				mul_constraints.push(MulConstraint {
					a: Operand::from(a),
					b: Operand::from(b),
					c: Operand::from(c),
				});
			}
			log2_ceil_usize(n_circuit_wires + blinding_size) as u32
		}

		let log_precommit = add_blinding_constraints::<F>(
			&mut mul_constraints,
			WitnessIndex::precommit,
			cs.n_precommit() as usize,
			&blinding_info,
			blinding_size,
		);
		let log_private = add_blinding_constraints::<F>(
			&mut mul_constraints,
			WitnessIndex::private,
			cs.n_private() as usize,
			&blinding_info,
			blinding_size,
		);

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
			log_precommit,
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

	pub fn n_precommit(&self) -> u32 {
		self.inner.n_precommit()
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

	pub fn log_precommit(&self) -> u32 {
		self.log_precommit
	}

	pub fn precommit_size(&self) -> usize {
		1 << self.log_precommit as usize
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
