// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The operand expression DSL: build a [`WireExpr`] as an XOR of shifted-wire terms.

use binius_core::constraint_system::Shift;
use smallvec::{SmallVec, smallvec};

use super::{ConstraintBuilder, operand::ShiftedWire};
use crate::{ir::Wire, lower::WireOperand};

/// An operand under construction: the XOR of its terms.
#[derive(Clone)]
pub struct WireExpr(SmallVec<[WireExprTerm; 4]>);

impl WireExpr {
	/// Consumes the expression into the operand its terms describe.
	///
	/// The terms are appended to the shared arena.
	pub(super) fn into_operand(self, cb: &mut ConstraintBuilder) -> WireOperand {
		cb.push_operand(self.0.into_iter().map(WireExprTerm::to_shifted_wire))
	}
}

impl From<Wire> for WireExpr {
	fn from(w: Wire) -> Self {
		WireExpr(smallvec![WireExprTerm::Wire(w)])
	}
}

impl From<WireExprTerm> for WireExpr {
	fn from(expr: WireExprTerm) -> Self {
		WireExpr(smallvec![expr])
	}
}

/// One term of a [`WireExpr`]: a wire, optionally shifted.
#[derive(Copy, Clone)]
pub enum WireExprTerm {
	/// The wire, used as-is.
	Wire(Wire),
	/// The wire with a shift folded in.
	Shifted(Wire, Shift),
}

impl WireExprTerm {
	const fn to_shifted_wire(self) -> ShiftedWire {
		match self {
			WireExprTerm::Wire(wire) => ShiftedWire::single(wire, Shift::IDENTITY),
			WireExprTerm::Shifted(wire, shift) => ShiftedWire::single(wire, shift),
		}
	}
}

impl From<Wire> for WireExprTerm {
	fn from(w: Wire) -> Self {
		WireExprTerm::Wire(w)
	}
}

/// Left-shifts the whole word by `n`.
pub const fn sll(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::sll(n as usize))
}

/// Half-wise left-shifts each 32-bit lane by `n`.
pub const fn sll32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::sll32(n as usize))
}

/// Logically right-shifts the whole word by `n`.
pub const fn srl(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::srl(n as usize))
}

/// Half-wise logically right-shifts each 32-bit lane by `n`.
pub const fn srl32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::srl32(n as usize))
}

/// Arithmetically right-shifts the whole word by `n`.
pub const fn sar(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::sar(n as usize))
}

/// Half-wise arithmetically right-shifts each 32-bit lane by `n`.
pub const fn sra32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::sra32(n as usize))
}

/// Rotates the whole word right by `n`.
pub const fn rotr(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::rotr(n as usize))
}

/// Half-wise rotates each 32-bit lane right by `n`.
pub const fn rotr32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::rotr32(n as usize))
}

/// XOR of two terms.
pub fn xor2(a: impl Into<WireExprTerm>, b: impl Into<WireExprTerm>) -> WireExpr {
	WireExpr(smallvec![a.into(), b.into()])
}

/// XOR of three terms.
pub fn xor3(
	a: impl Into<WireExprTerm>,
	b: impl Into<WireExprTerm>,
	c: impl Into<WireExprTerm>,
) -> WireExpr {
	WireExpr(smallvec![a.into(), b.into(), c.into()])
}

/// XOR of four terms.
pub fn xor4(
	a: impl Into<WireExprTerm>,
	b: impl Into<WireExprTerm>,
	c: impl Into<WireExprTerm>,
	d: impl Into<WireExprTerm>,
) -> WireExpr {
	WireExpr(smallvec![a.into(), b.into(), c.into(), d.into()])
}

/// XOR of an arbitrary number of terms.
pub fn xor_multi(terms: impl IntoIterator<Item = WireExprTerm>) -> WireExpr {
	WireExpr(terms.into_iter().collect())
}

/// The empty operand, i.e. the constant zero.
pub fn empty() -> WireExpr {
	WireExpr(smallvec![])
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::{ShiftVariant, ValueIndex};
	use cranelift_entity::{EntityRef, SecondaryMap};

	use crate::{
		ir::Wire,
		lower::{ConstraintBuilder, expr},
	};

	#[test]
	fn multi_term_xor_expression_lowers_each_term() {
		// c = rotr(a, 0) ^ sll(b, 5) ^ rotr(a, 12) must lower to three operand terms:
		// plain(a), native sll(b, 5), native rotr(a, 12).
		let mut wire_mapping = SecondaryMap::with_default(ValueIndex::scratch(0));
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex::private(0);
		wire_mapping[wire_b] = ValueIndex::private(1);
		wire_mapping[wire_c] = ValueIndex::private(2);
		wire_mapping[all_one_wire] = ValueIndex::private(3);

		let mut builder = ConstraintBuilder::new();
		builder.linear(
			expr::xor3(expr::rotr(wire_a, 0), expr::sll(wire_b, 5), expr::rotr(wire_a, 12)),
			wire_c,
		);

		let (zero_constraints, and_constraints, imul_constraints, _bmul_constraints) =
			builder.build(&wire_mapping);

		assert_eq!(zero_constraints.len(), 1);
		assert_eq!(and_constraints.len(), 0);
		assert_eq!(imul_constraints.len(), 0);

		// The operand is the three RHS terms plus the destination.
		let val = zero_constraints[0].val();
		assert_eq!(val.len(), 4);

		assert!(
			val.iter()
				.any(|svi| svi.value_index == ValueIndex::private(0) && svi.is_unshifted()),
			"plain(a) from rotr(a, 0)"
		);
		assert!(
			val.iter().any(|svi| {
				svi.value_index == ValueIndex::private(1)
					&& svi.inner().amount == 5
					&& matches!(svi.inner().variant, ShiftVariant::Sll)
			}),
			"native sll(b, 5)"
		);
		assert!(
			val.iter().any(|svi| {
				svi.value_index == ValueIndex::private(0)
					&& svi.inner().amount == 12
					&& matches!(svi.inner().variant, ShiftVariant::Rotr)
			}),
			"native rotr(a, 12)"
		);
	}
}
