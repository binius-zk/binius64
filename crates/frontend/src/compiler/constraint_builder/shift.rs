// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The shift algebra shared by operands: a core [`Shift`] applied to a [`Wire`].

use std::ops::Index;

use binius_core::constraint_system::{Shift, ShiftedValueIndex, ValueIndex};
use cranelift_entity::{EntitySet, SecondaryMap};

use crate::compiler::Wire;

/// A single wire term of an operand, tagged with the shift to apply to it.
#[derive(Copy, Clone, Debug)]
pub struct ShiftedWire {
	/// The wire the shift applies to.
	pub wire: Wire,
	/// The shift folded into this term.
	pub shift: Shift,
}

impl ShiftedWire {
	/// Lowers this term to a core [`ShiftedValueIndex`] via the wire mapping.
	pub(super) fn to_shifted_value_index(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> ShiftedValueIndex {
		// The builder carries one shift per term, which the canonical form places inner.
		ShiftedValueIndex::single(wire_mapping[self.wire], self.shift)
	}
}

/// An operand: an XOR of shifted-wire terms, stored per constraint position.
#[derive(Clone, Debug, Default)]
pub struct WireOperand(Vec<ShiftedWire>);

impl WireOperand {
	/// Creates an empty operand.
	pub const fn new() -> Self {
		Self(Vec::new())
	}

	/// Appends a shifted-wire term.
	pub fn push(&mut self, term: ShiftedWire) {
		self.0.push(term);
	}

	/// The terms this operand XORs together.
	pub fn as_slice(&self) -> &[ShiftedWire] {
		&self.0
	}

	/// The number of terms.
	pub const fn len(&self) -> usize {
		self.0.len()
	}

	/// Whether the operand has no terms.
	///
	/// An empty XOR is the constant zero, so such an operand contributes nothing.
	pub const fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	/// Lowers the whole operand to core `ShiftedValueIndex` terms.
	pub(super) fn into_value_indices(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> Vec<ShiftedValueIndex> {
		self.0
			.into_iter()
			.map(|term| term.to_shifted_value_index(wire_mapping))
			.collect()
	}

	/// Inserts every wire this operand references into `used_set`.
	pub(super) fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
		for term in &self.0 {
			used_set.insert(term.wire);
		}
	}
}

impl Index<usize> for WireOperand {
	type Output = ShiftedWire;

	fn index(&self, term: usize) -> &Self::Output {
		&self.0[term]
	}
}

impl<'a> IntoIterator for &'a WireOperand {
	type Item = &'a ShiftedWire;
	type IntoIter = std::slice::Iter<'a, ShiftedWire>;

	fn into_iter(self) -> Self::IntoIter {
		self.0.iter()
	}
}

impl FromIterator<ShiftedWire> for WireOperand {
	fn from_iter<I: IntoIterator<Item = ShiftedWire>>(iter: I) -> Self {
		Self(iter.into_iter().collect())
	}
}

impl From<Vec<ShiftedWire>> for WireOperand {
	fn from(terms: Vec<ShiftedWire>) -> Self {
		Self(terms)
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
	fn rotr_zero_folds_to_plain_via_linear() {
		// A rotr-by-0 term must lower to a plain value index; a rotr-by-n>0 must stay native.
		let mut wire_mapping = SecondaryMap::with_default(ValueIndex::scratch(0));
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex::private(0);
		wire_mapping[wire_b] = ValueIndex::private(1);
		wire_mapping[wire_c] = ValueIndex::private(2);
		wire_mapping[all_one_wire] = ValueIndex::private(3);

		// c = rotr(a, 0) ^ b  ->  rotr(0) collapses to plain(a).
		{
			let mut builder = ConstraintBuilder::new();
			builder.linear(expr::xor2(expr::rotr(wire_a, 0), wire_b), wire_c);

			let (zero_constraints, and_constraints, imul_constraints, _bmul_constraints) =
				builder.build(&wire_mapping);

			// Linear lowers to the ZERO constraint `a ^ b ^ c = 0`.
			assert_eq!(zero_constraints.len(), 1);
			assert_eq!(and_constraints.len(), 0);
			assert_eq!(imul_constraints.len(), 0);

			let val = zero_constraints[0].val();
			assert_eq!(val.len(), 3);
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(0) && svi.is_unshifted())
			);
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(1) && svi.is_unshifted())
			);
			// The destination joins the operand rather than sitting in its own `c`.
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(2) && svi.is_unshifted())
			);
		}

		// c = rotr(a, 5) ^ b  ->  native rotr(a, 5).
		{
			let mut builder = ConstraintBuilder::new();
			builder.linear(expr::xor2(expr::rotr(wire_a, 5), wire_b), wire_c);

			let (zero_constraints, and_constraints, imul_constraints, _bmul_constraints) =
				builder.build(&wire_mapping);

			assert_eq!(zero_constraints.len(), 1);
			assert_eq!(and_constraints.len(), 0);
			assert_eq!(imul_constraints.len(), 0);

			let val = zero_constraints[0].val();
			assert_eq!(val.len(), 3);
			assert!(val.iter().any(|svi| {
				svi.value_index == ValueIndex::private(0)
					&& svi.inner().amount == 5
					&& matches!(svi.inner().variant, ShiftVariant::Rotr)
			}));
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(1) && svi.is_unshifted())
			);
		}
	}

	#[test]
	fn rotr_folds_inside_and_operand() {
		// The same rotr(0)->plain and rotr(n)->native folding must hold inside an AND operand.
		let mut wire_mapping = SecondaryMap::with_default(ValueIndex::scratch(0));
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex::private(0);
		wire_mapping[wire_b] = ValueIndex::private(1);
		wire_mapping[wire_c] = ValueIndex::private(2);
		wire_mapping[all_one_wire] = ValueIndex::private(3);

		// a & rotr(b, 0) = c  ->  b stays plain.
		{
			let mut builder = ConstraintBuilder::new();
			builder.and(wire_a, expr::rotr(wire_b, 0), wire_c);

			let (_, and_constraints, _, _) = builder.build(&wire_mapping);

			assert_eq!(and_constraints.len(), 1);
			let and_c = &and_constraints[0];

			assert_eq!(and_c.a().len(), 1);
			assert_eq!(and_c.a()[0].value_index, ValueIndex::private(0));
			assert_eq!(and_c.a()[0].inner().amount, 0);

			assert_eq!(and_c.b().len(), 1);
			assert_eq!(and_c.b()[0].value_index, ValueIndex::private(1));
			assert_eq!(and_c.b()[0].inner().amount, 0);

			assert_eq!(and_c.c().len(), 1);
			assert_eq!(and_c.c()[0].value_index, ValueIndex::private(2));
			assert_eq!(and_c.c()[0].inner().amount, 0);
		}

		// a & rotr(b, 8) = c  ->  b keeps native rotr(8).
		{
			let mut builder = ConstraintBuilder::new();
			builder.and(wire_a, expr::rotr(wire_b, 8), wire_c);

			let (_, and_constraints, _, _) = builder.build(&wire_mapping);

			assert_eq!(and_constraints.len(), 1);
			let and_c = &and_constraints[0];
			assert_eq!(and_c.b().len(), 1);
			assert!(and_c.b().iter().any(|svi| {
				svi.value_index == ValueIndex::private(1)
					&& svi.inner().amount == 8
					&& matches!(svi.inner().variant, ShiftVariant::Rotr)
			}));
		}
	}
}
