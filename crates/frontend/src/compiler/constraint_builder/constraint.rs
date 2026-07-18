// Copyright 2025 Irreducible Inc.

//! Wire-level constraint records, holding [`WireOperand`]s until the wire mapping lowers them.

use binius_core::constraint_system::{
	AndConstraint, BmulConstraint, ImulConstraint, ShiftedValueIndex, ValueIndex,
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
		AndConstraint {
			a: self.a.into_value_indices(wire_mapping),
			b: self.b.into_value_indices(wire_mapping),
			c: self.c.into_value_indices(wire_mapping),
		}
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
		ImulConstraint {
			a: self.a.into_value_indices(wire_mapping),
			b: self.b.into_value_indices(wire_mapping),
			hi: self.hi.into_value_indices(wire_mapping),
			lo: self.lo.into_value_indices(wire_mapping),
		}
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
		BmulConstraint {
			a_lo: self.a_lo.into_value_indices(wire_mapping),
			a_hi: self.a_hi.into_value_indices(wire_mapping),
			b_lo: self.b_lo.into_value_indices(wire_mapping),
			b_hi: self.b_hi.into_value_indices(wire_mapping),
			c_lo: self.c_lo.into_value_indices(wire_mapping),
			c_hi: self.c_hi.into_value_indices(wire_mapping),
		}
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
	/// Lowers to `RHS & all_ones == DST`.
	///
	/// AND against the all-ones wire is the identity on `&`, so this encodes
	/// the equality `RHS == DST` with the only opcode available for it.
	pub(super) fn into_and_constraint(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
		all_ones: ValueIndex,
	) -> AndConstraint {
		let dst = wire_mapping[self.dst];
		AndConstraint {
			a: self.rhs.into_value_indices(wire_mapping),
			b: vec![ShiftedValueIndex::plain(all_ones)],
			c: vec![ShiftedValueIndex::plain(dst)],
		}
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
			wire_mapping[*w] = ValueIndex(i as u32);
		}
		let all_one_wire = Wire::new(7);
		wire_mapping[all_one_wire] = ValueIndex(7);

		let mut builder = ConstraintBuilder::new();
		builder
			.bmul()
			.a_lo(wires[0])
			.a_hi(wires[1])
			.b_lo(wires[2])
			.b_hi(wires[3])
			.c_lo(wires[4])
			.c_hi(expr::xor2(wires[5], expr::sll(wires[6], 5)))
			.build();

		let (and_constraints, imul_constraints, bmul_constraints) =
			builder.build(&wire_mapping, all_one_wire);

		assert_eq!(and_constraints.len(), 0);
		assert_eq!(imul_constraints.len(), 0);
		assert_eq!(bmul_constraints.len(), 1);

		let bc = &bmul_constraints[0];
		assert_eq!(bc.a_lo[0].value_index, ValueIndex(0));
		assert_eq!(bc.a_hi[0].value_index, ValueIndex(1));
		assert_eq!(bc.b_lo[0].value_index, ValueIndex(2));
		assert_eq!(bc.b_hi[0].value_index, ValueIndex(3));
		assert_eq!(bc.c_lo[0].value_index, ValueIndex(4));

		// c_hi is `wire5 ^ (wire6 << 5)`.
		assert_eq!(bc.c_hi.len(), 2);
		assert!(
			bc.c_hi
				.iter()
				.any(|svi| svi.value_index == ValueIndex(5) && svi.amount == 0)
		);
		assert!(bc.c_hi.iter().any(|svi| {
			svi.value_index == ValueIndex(6)
				&& svi.amount == 5
				&& matches!(svi.shift_variant, ShiftVariant::Sll)
		}));
	}
}
