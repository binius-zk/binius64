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

		/// Adds dummy blinding constraints for a segment and returns its padded log-size.
		fn add_blinding_constraints(
			mul_constraints: &mut Vec<MulConstraint<WitnessIndex>>,
			make_index: fn(u32) -> WitnessIndex,
			n_circuit_wires: usize,
			n_dummy_wires: usize,
			n_dummy_constraints: usize,
		) -> u32 {
			let dummy_base = n_circuit_wires + n_dummy_wires;
			for i in 0..n_dummy_constraints {
				let a = make_index((dummy_base + 3 * i) as u32);
				let b = make_index((dummy_base + 3 * i + 1) as u32);
				let c = make_index((dummy_base + 3 * i + 2) as u32);
				mul_constraints.push(MulConstraint {
					a: Operand::from(a),
					b: Operand::from(b),
					c: Operand::from(c),
				});
			}

			let blinding_size = n_dummy_wires + 3 * n_dummy_constraints;
			log2_ceil_usize(n_circuit_wires + blinding_size) as u32
		}

		// Both committed segments have evaluations revealed in the clear, so both need dummy
		// constraints to carry randomness into the wiring relation that masks them.
		let log_precommit = add_blinding_constraints(
			&mut mul_constraints,
			WitnessIndex::precommit,
			cs.n_precommit() as usize,
			blinding_info.n_dummy_wires,
			blinding_info.n_dummy_constraints,
		);
		let log_private = add_blinding_constraints(
			&mut mul_constraints,
			WitnessIndex::private,
			cs.n_private() as usize,
			blinding_info.n_dummy_wires,
			blinding_info.n_dummy_constraints,
		);

		// Pad to next power of two with `one * one = one` constraints
		let one_operand = Operand::from(cs.one_wire());
		let current_len = mul_constraints.len();
		mul_constraints.resize(
			current_len.next_power_of_two(),
			MulConstraint {
				a: one_operand.clone(),
				b: one_operand.clone(),
				c: one_operand,
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

	pub const fn n_inout(&self) -> u32 {
		self.inner.n_inout()
	}

	pub const fn n_precommit(&self) -> u32 {
		self.inner.n_precommit()
	}

	pub const fn n_private(&self) -> u32 {
		self.inner.n_private()
	}

	pub const fn log_public(&self) -> u32 {
		self.inner.log_public()
	}

	pub const fn n_public(&self) -> u32 {
		self.inner.n_public()
	}

	pub const fn one_wire(&self) -> WitnessIndex {
		self.inner.one_wire()
	}

	pub const fn log_precommit(&self) -> u32 {
		self.log_precommit
	}

	pub const fn precommit_size(&self) -> usize {
		1 << self.log_precommit as usize
	}

	pub const fn log_private(&self) -> u32 {
		self.log_private
	}

	pub const fn private_size(&self) -> usize {
		1 << self.log_private as usize
	}

	pub const fn blinding_info(&self) -> &BlindingInfo {
		&self.blinding_info
	}

	pub fn mul_constraints(&self) -> &[MulConstraint<WitnessIndex>] {
		&self.mul_constraints
	}

	/// Returns the mask buffer dimensions (m_n, m_d) for the ZK mulcheck mask polynomial.
	pub const fn mask_dims(&self) -> (usize, usize) {
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

#[cfg(test)]
mod tests {
	use std::collections::BTreeSet;

	use binius_field::Ghash128b as B128;
	use binius_spartan_frontend::{
		circuit_builder::{CircuitBuilder, ConstraintBuilder},
		compiler::compile,
		constraint_system::WitnessSegment,
	};

	use super::*;

	#[test]
	fn every_committed_segment_reserves_one_wire_beyond_the_fri_queries() {
		// Each FRI query opens one Merkle leaf, revealing one codeword symbol of the segment.
		const N_TEST_QUERIES: usize = 32;

		// Any circuit will do: blinding is padding appended after whatever real wires exist.
		let mut builder = ConstraintBuilder::<B128>::new();
		let x = builder.alloc_inout();
		let y = builder.alloc_inout();
		builder.assert_eq(x, y);
		let (cs, _layout) = compile(builder);

		let n_precommit = cs.n_precommit() as usize;
		let n_private = cs.n_private() as usize;

		let info = BlindingInfo::for_fri_queries(N_TEST_QUERIES);
		let padded = ConstraintSystemPadded::new(cs, info);

		// Invariant: the dummy wires must outnumber the queries.
		// Spending exactly one per query would leave the unopened leaves with no randomness of
		// their own, and the leaves carry no salt.
		assert!(info.n_dummy_wires > N_TEST_QUERIES);

		// Each segment is rounded up to a power of two, so its reserved size must still cover
		// every real wire plus the whole blinding allowance, which is the same for both:
		//
		//     n_dummy_wires + 3 * n_dummy_constraints
		let blinding = info.n_dummy_wires + 3 * info.n_dummy_constraints;
		assert!(padded.precommit_size() >= n_precommit + blinding);
		assert!(padded.private_size() >= n_private + blinding);
	}

	#[test]
	fn every_committed_segment_masks_its_revealed_evaluations() {
		// A revealed evaluation weights each wire by its coefficient in the wiring relation.
		//
		// That relation only ever sums over wires that appear in a multiplication constraint, so
		// a wire in no constraint is weighted by zero and can mask nothing.
		//
		// This pins where each segment's blinding lands:
		//
		//     dummy wires        in no constraint -> mask the codeword symbols FRI opens
		//     dummy constraints  in a constraint -> mask the evaluations sent in the clear
		const N_TEST_QUERIES: usize = 8;

		let mut builder = ConstraintBuilder::<B128>::new();
		let x = builder.alloc_inout();
		let y = builder.alloc_inout();
		builder.assert_eq(x, y);
		let (cs, _layout) = compile(builder);

		let n_circuit = [cs.n_precommit() as usize, cs.n_private() as usize];
		let info = BlindingInfo::for_fri_queries(N_TEST_QUERIES);
		let padded = ConstraintSystemPadded::new(cs, info);

		// Collect, per segment, the wire indices the constraints actually touch.
		let mut in_support = [BTreeSet::new(), BTreeSet::new()];
		for constraint in padded.mul_constraints() {
			for operand in [&constraint.a, &constraint.b, &constraint.c] {
				for wire in operand.wires() {
					let slot = match wire.segment {
						WitnessSegment::Precommit => Some(0),
						WitnessSegment::Private => Some(1),
						WitnessSegment::Public => None,
					};
					if let Some(slot) = slot {
						in_support[slot].insert(wire.index as usize);
					}
				}
			}
		}

		for (segment, n_circuit) in in_support.iter().zip(n_circuit) {
			// The dummy wires sit immediately after the circuit's own wires, and none of them may
			// appear in a constraint — that is what makes them useless against an evaluation.
			for offset in 0..info.n_dummy_wires {
				assert!(
					!segment.contains(&(n_circuit + offset)),
					"a dummy wire reached the wiring relation"
				);
			}

			// The dummy constraints follow, and all three of their wires must appear, or the
			// evaluations revealed for this segment have nothing masking them.
			let dummy_constraint_base = n_circuit + info.n_dummy_wires;
			for offset in 0..3 * info.n_dummy_constraints {
				assert!(
					segment.contains(&(dummy_constraint_base + offset)),
					"a dummy constraint wire never reached the wiring relation"
				);
			}
		}
	}
}
