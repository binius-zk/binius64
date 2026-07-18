// Copyright 2025 Irreducible Inc.

//! The operand expression DSL: build a [`WireExpr`] as an XOR of shifted-wire terms.

use smallvec::{SmallVec, smallvec};

use super::shift::{Shift, ShiftedWire, WireOperand};
use crate::compiler::Wire;

/// An operand under construction: the XOR of its terms.
#[derive(Clone)]
pub struct WireExpr(SmallVec<[WireExprTerm; 4]>);

impl WireExpr {
	/// Consumes the expression into the operand its terms describe.
	pub(super) fn into_operand(self) -> WireOperand {
		self.0
			.into_iter()
			.map(WireExprTerm::to_shifted_wire)
			.collect()
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
			WireExprTerm::Wire(wire) => ShiftedWire {
				wire,
				shift: Shift::None,
			},
			WireExprTerm::Shifted(wire, shift) => ShiftedWire { wire, shift },
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
	WireExprTerm::Shifted(w, Shift::Sll(n))
}

/// Half-wise left-shifts each 32-bit lane by `n`.
pub const fn sll32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::Sll32(n))
}

/// Logically right-shifts the whole word by `n`.
pub const fn srl(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::Srl(n))
}

/// Half-wise logically right-shifts each 32-bit lane by `n`.
pub const fn srl32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::Srl32(n))
}

/// Arithmetically right-shifts the whole word by `n`.
pub const fn sar(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::Sar(n))
}

/// Half-wise arithmetically right-shifts each 32-bit lane by `n`.
pub const fn sra32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::Sra32(n))
}

/// Rotates the whole word right by `n`.
pub const fn rotr(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::Rotr(n))
}

/// Half-wise rotates each 32-bit lane right by `n`.
pub const fn rotr32(w: Wire, n: u32) -> WireExprTerm {
	WireExprTerm::Shifted(w, Shift::Rotr32(n))
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

	use crate::compiler::{
		Wire,
		constraint_builder::{ConstraintBuilder, expr},
	};

	#[test]
	fn multi_term_xor_expression_lowers_each_term() {
		// c = rotr(a, 0) ^ sll(b, 5) ^ rotr(a, 12) must lower to three operand terms:
		// plain(a), native sll(b, 5), native rotr(a, 12).
		let mut wire_mapping = SecondaryMap::new();
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex(0);
		wire_mapping[wire_b] = ValueIndex(1);
		wire_mapping[wire_c] = ValueIndex(2);
		wire_mapping[all_one_wire] = ValueIndex(3);

		let mut builder = ConstraintBuilder::new();
		builder
			.linear()
			.rhs(expr::xor3(expr::rotr(wire_a, 0), expr::sll(wire_b, 5), expr::rotr(wire_a, 12)))
			.dst(wire_c)
			.build();

		let (and_constraints, imul_constraints, _bmul_constraints) =
			builder.build(&wire_mapping, all_one_wire);

		assert_eq!(and_constraints.len(), 1);
		assert_eq!(imul_constraints.len(), 0);

		let and_c = &and_constraints[0];
		assert_eq!(and_c.a.len(), 3);

		assert!(
			and_c
				.a
				.iter()
				.any(|svi| svi.value_index == ValueIndex(0) && svi.amount == 0),
			"plain(a) from rotr(a, 0)"
		);
		assert!(
			and_c.a.iter().any(|svi| {
				svi.value_index == ValueIndex(1)
					&& svi.amount == 5
					&& matches!(svi.shift_variant, ShiftVariant::Sll)
			}),
			"native sll(b, 5)"
		);
		assert!(
			and_c.a.iter().any(|svi| {
				svi.value_index == ValueIndex(0)
					&& svi.amount == 12
					&& matches!(svi.shift_variant, ShiftVariant::Rotr)
			}),
			"native rotr(a, 12)"
		);
	}
}
