// Copyright 2025 Irreducible Inc.

//! Fluent per-constraint builders returned by `ConstraintBuilder::and`/`imul`/`bmul`/`linear`.

use super::{
	ConstraintBuilder, WireAndConstraint, WireBmulConstraint, WireImulConstraint,
	WireLinearConstraint, expr::WireExpr, shift::WireOperand,
};
use crate::compiler::Wire;

/// Builds an AND constraint `A & B == C`.
pub struct AndConstraintBuilder<'a> {
	builder: &'a mut ConstraintBuilder,
	a: WireOperand,
	b: WireOperand,
	c: WireOperand,
}

impl<'a> AndConstraintBuilder<'a> {
	pub(super) const fn new(builder: &'a mut ConstraintBuilder) -> Self {
		Self {
			builder,
			a: WireOperand::new(),
			b: WireOperand::new(),
			c: WireOperand::new(),
		}
	}

	/// Sets the `a` operand.
	pub fn a(mut self, expr: impl Into<WireExpr>) -> Self {
		self.a = expr.into().into_operand();
		self
	}

	/// Sets the `b` operand.
	pub fn b(mut self, expr: impl Into<WireExpr>) -> Self {
		self.b = expr.into().into_operand();
		self
	}

	/// Sets the `c` operand.
	pub fn c(mut self, expr: impl Into<WireExpr>) -> Self {
		self.c = expr.into().into_operand();
		self
	}

	/// Finalizes the constraint and appends it to the builder.
	pub fn build(self) {
		self.builder.and_constraints.push(WireAndConstraint {
			a: self.a,
			b: self.b,
			c: self.c,
		});
	}
}

/// Builds an IMUL constraint `A * B == (HI << 64) | LO`.
pub struct ImulConstraintBuilder<'a> {
	builder: &'a mut ConstraintBuilder,
	a: WireOperand,
	b: WireOperand,
	hi: WireOperand,
	lo: WireOperand,
}

impl<'a> ImulConstraintBuilder<'a> {
	pub(super) const fn new(builder: &'a mut ConstraintBuilder) -> Self {
		Self {
			builder,
			a: WireOperand::new(),
			b: WireOperand::new(),
			hi: WireOperand::new(),
			lo: WireOperand::new(),
		}
	}

	/// Sets the `a` operand.
	pub fn a(mut self, expr: impl Into<WireExpr>) -> Self {
		self.a = expr.into().into_operand();
		self
	}

	/// Sets the `b` operand.
	pub fn b(mut self, expr: impl Into<WireExpr>) -> Self {
		self.b = expr.into().into_operand();
		self
	}

	/// Sets the `hi` operand (high 64 bits of the product).
	pub fn hi(mut self, expr: impl Into<WireExpr>) -> Self {
		self.hi = expr.into().into_operand();
		self
	}

	/// Sets the `lo` operand (low 64 bits of the product).
	pub fn lo(mut self, expr: impl Into<WireExpr>) -> Self {
		self.lo = expr.into().into_operand();
		self
	}

	/// Finalizes the constraint and appends it to the builder.
	pub fn build(self) {
		self.builder.imul_constraints.push(WireImulConstraint {
			a: self.a,
			b: self.b,
			hi: self.hi,
			lo: self.lo,
		});
	}
}

/// Builds a BMUL constraint `(A_LO, A_HI) * (B_LO, B_HI) == (C_LO, C_HI)`.
pub struct BmulConstraintBuilder<'a> {
	builder: &'a mut ConstraintBuilder,
	a_lo: WireOperand,
	a_hi: WireOperand,
	b_lo: WireOperand,
	b_hi: WireOperand,
	c_lo: WireOperand,
	c_hi: WireOperand,
}

impl<'a> BmulConstraintBuilder<'a> {
	pub(super) const fn new(builder: &'a mut ConstraintBuilder) -> Self {
		Self {
			builder,
			a_lo: WireOperand::new(),
			a_hi: WireOperand::new(),
			b_lo: WireOperand::new(),
			b_hi: WireOperand::new(),
			c_lo: WireOperand::new(),
			c_hi: WireOperand::new(),
		}
	}

	/// Sets the `a_lo` operand.
	pub fn a_lo(mut self, expr: impl Into<WireExpr>) -> Self {
		self.a_lo = expr.into().into_operand();
		self
	}

	/// Sets the `a_hi` operand.
	pub fn a_hi(mut self, expr: impl Into<WireExpr>) -> Self {
		self.a_hi = expr.into().into_operand();
		self
	}

	/// Sets the `b_lo` operand.
	pub fn b_lo(mut self, expr: impl Into<WireExpr>) -> Self {
		self.b_lo = expr.into().into_operand();
		self
	}

	/// Sets the `b_hi` operand.
	pub fn b_hi(mut self, expr: impl Into<WireExpr>) -> Self {
		self.b_hi = expr.into().into_operand();
		self
	}

	/// Sets the `c_lo` operand.
	pub fn c_lo(mut self, expr: impl Into<WireExpr>) -> Self {
		self.c_lo = expr.into().into_operand();
		self
	}

	/// Sets the `c_hi` operand.
	pub fn c_hi(mut self, expr: impl Into<WireExpr>) -> Self {
		self.c_hi = expr.into().into_operand();
		self
	}

	/// Finalizes the constraint and appends it to the builder.
	pub fn build(self) {
		self.builder.bmul_constraints.push(WireBmulConstraint {
			a_lo: self.a_lo,
			a_hi: self.a_hi,
			b_lo: self.b_lo,
			b_hi: self.b_hi,
			c_lo: self.c_lo,
			c_hi: self.c_hi,
		});
	}
}

/// Builds a linear constraint `RHS == DST`.
///
/// Unlike the other builders, `dst` is a single wire rather than an operand.
pub struct LinearConstraintBuilder<'a> {
	builder: &'a mut ConstraintBuilder,
	rhs: WireOperand,
	dst: Option<Wire>,
}

impl<'a> LinearConstraintBuilder<'a> {
	pub(super) const fn new(builder: &'a mut ConstraintBuilder) -> Self {
		Self {
			builder,
			rhs: WireOperand::new(),
			dst: None,
		}
	}

	/// Sets the `rhs` operand (an XOR combination of shifted values).
	pub fn rhs(mut self, expr: impl Into<WireExpr>) -> Self {
		self.rhs = expr.into().into_operand();
		self
	}

	/// Sets the `dst` wire.
	pub const fn dst(mut self, wire: Wire) -> Self {
		self.dst = Some(wire);
		self
	}

	/// Finalizes the constraint and appends it to the builder.
	///
	/// # Panics
	/// Panics if `dst` was never set.
	pub fn build(self) {
		self.builder.linear_constraints.push(WireLinearConstraint {
			rhs: self.rhs,
			dst: self.dst.expect("dst wire must be assigned"),
		});
	}
}
