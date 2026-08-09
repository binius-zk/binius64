// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Wire-level constraint records, holding [`WireOperand`]s until the wire mapping lowers them.

use binius_core::constraint_system::{
	AndConstraint, BmulConstraint, ImulConstraint, ShiftedValueIndex, ValueIndex, ZeroConstraint,
};
use cranelift_entity::{EntitySet, SecondaryMap};

use super::shift::WireOperand;
use crate::compiler::Wire;

/// AND constraint `A & B == C`, over wire operands.
pub struct WireAndConstraint {
	/// Left operand.
	pub a: WireOperand,
	/// Right operand.
	pub b: WireOperand,
	/// Result operand.
	pub c: WireOperand,
}

impl WireAndConstraint {
	pub(super) fn into_constraint(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> AndConstraint {
		AndConstraint([
			self.a.into_value_indices(wire_mapping),
			self.b.into_value_indices(wire_mapping),
			self.c.into_value_indices(wire_mapping),
		])
	}

	pub(super) fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
		self.a.mark_used(used_set);
		self.b.mark_used(used_set);
		self.c.mark_used(used_set);
	}
}

/// IMUL constraint `A * B == (HI << 64) | LO`, over wire operands.
pub struct WireImulConstraint {
	/// Left factor.
	pub a: WireOperand,
	/// Right factor.
	pub b: WireOperand,
	/// High 64 bits of the product.
	pub hi: WireOperand,
	/// Low 64 bits of the product.
	pub lo: WireOperand,
}

impl WireImulConstraint {
	pub(super) fn into_constraint(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> ImulConstraint {
		ImulConstraint([
			self.a.into_value_indices(wire_mapping),
			self.b.into_value_indices(wire_mapping),
			self.lo.into_value_indices(wire_mapping),
			self.hi.into_value_indices(wire_mapping),
		])
	}

	pub(super) fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
		self.a.mark_used(used_set);
		self.b.mark_used(used_set);
		self.hi.mark_used(used_set);
		self.lo.mark_used(used_set);
	}
}

/// BMUL constraint `(A_LO, A_HI) * (B_LO, B_HI) == (C_LO, C_HI)` in the GHASH field.
pub struct WireBmulConstraint {
	/// Low limb of the left factor.
	pub a_lo: WireOperand,
	/// High limb of the left factor.
	pub a_hi: WireOperand,
	/// Low limb of the right factor.
	pub b_lo: WireOperand,
	/// High limb of the right factor.
	pub b_hi: WireOperand,
	/// Low limb of the product.
	pub c_lo: WireOperand,
	/// High limb of the product.
	pub c_hi: WireOperand,
}

impl WireBmulConstraint {
	pub(super) fn into_constraint(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> BmulConstraint {
		BmulConstraint([
			self.a_lo.into_value_indices(wire_mapping),
			self.a_hi.into_value_indices(wire_mapping),
			self.b_lo.into_value_indices(wire_mapping),
			self.b_hi.into_value_indices(wire_mapping),
			self.c_lo.into_value_indices(wire_mapping),
			self.c_hi.into_value_indices(wire_mapping),
		])
	}

	pub(super) fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
		self.a_lo.mark_used(used_set);
		self.a_hi.mark_used(used_set);
		self.b_lo.mark_used(used_set);
		self.b_hi.mark_used(used_set);
		self.c_lo.mark_used(used_set);
		self.c_hi.mark_used(used_set);
	}
}

/// Linear constraint `RHS == DST`, over a wire operand and a destination wire.
pub struct WireLinearConstraint {
	/// XOR of shifted values that must equal `dst`.
	pub rhs: WireOperand,
	/// Destination wire.
	pub dst: Wire,
}

impl WireLinearConstraint {
	/// Lowers to the Zero constraint `RHS ^ DST == 0`.
	///
	/// This is the direct encoding of the equality: one constraint array instead of the three an
	/// AND constraint carries.
	pub(super) fn into_zero_constraint(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> ZeroConstraint {
		let dst = wire_mapping[self.dst];
		let mut operand = self.rhs.into_value_indices(wire_mapping);
		operand.push(ShiftedValueIndex::plain(dst));
		ZeroConstraint([operand])
	}

	pub(super) fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
		self.rhs.mark_used(used_set);
		used_set.insert(self.dst);
	}
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::{ShiftVariant, ValueIndex};
	use cranelift_entity::{EntityRef, SecondaryMap};

	use crate::compiler::{
		Wire,
		constraint_builder::{ConstraintBuilder, expr},
	};

	#[test]
	fn bmul_builder_lands_all_six_operands() {
		// Build (a_lo, a_hi) * (b_lo, b_hi) = (c_lo, c_hi) and check each of the six operands
		// reaches the produced BmulConstraint, including a shifted term in c_hi.
		let mut wire_mapping = SecondaryMap::new();
		let wires: Vec<Wire> = (0..7).map(Wire::new).collect();
		for (i, w) in wires.iter().enumerate() {
			wire_mapping[*w] = ValueIndex::private(i as u32);
		}
		let all_one_wire = Wire::new(7);
		wire_mapping[all_one_wire] = ValueIndex::private(7);

		let mut builder = ConstraintBuilder::new();
		builder.bmul(
			wires[0],
			wires[1],
			wires[2],
			wires[3],
			wires[4],
			expr::xor2(wires[5], expr::sll(wires[6], 5)),
		);

		let (_zero_constraints, and_constraints, imul_constraints, bmul_constraints) =
			builder.build(&wire_mapping);

		assert_eq!(and_constraints.len(), 0);
		assert_eq!(imul_constraints.len(), 0);
		assert_eq!(bmul_constraints.len(), 1);

		let bc = &bmul_constraints[0];
		assert_eq!(bc.a_lo()[0].value_index, ValueIndex::private(0));
		assert_eq!(bc.a_hi()[0].value_index, ValueIndex::private(1));
		assert_eq!(bc.b_lo()[0].value_index, ValueIndex::private(2));
		assert_eq!(bc.b_hi()[0].value_index, ValueIndex::private(3));
		assert_eq!(bc.c_lo()[0].value_index, ValueIndex::private(4));

		// c_hi is `wire5 ^ (wire6 << 5)`.
		assert_eq!(bc.c_hi().len(), 2);
		assert!(
			bc.c_hi()
				.iter()
				.any(|svi| svi.value_index == ValueIndex::private(5) && svi.amount == 0)
		);
		assert!(bc.c_hi().iter().any(|svi| {
			svi.value_index == ValueIndex::private(6)
				&& svi.amount == 5
				&& matches!(svi.shift_variant, ShiftVariant::Sll)
		}));
	}
}
