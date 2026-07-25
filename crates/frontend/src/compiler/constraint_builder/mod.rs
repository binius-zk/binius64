// Copyright 2025 Irreducible Inc.

//! Wire-level constraint DSL and its lowering to core `ValueIndex` constraints.

mod constraint;
pub mod expr;
mod shift;

use binius_core::constraint_system::{AndConstraint, BmulConstraint, ImulConstraint, ValueIndex};
pub use constraint::{
	WireAndConstraint, WireBmulConstraint, WireImulConstraint, WireLinearConstraint,
};
use cranelift_entity::{EntitySet, SecondaryMap};
use expr::WireExpr;
pub use expr::WireExprTerm;
pub use shift::{Shift, ShiftedWire, WireOperand};

use crate::compiler::Wire;

/// Accumulates the constraints a circuit emits, expressed over [`Wire`]s.
///
/// Gates push into the four typed buckets, one method per constraint shape.
///
/// Every operand is a parameter of that method:
///
/// - leaving one out is a compile error rather than a silent zero;
/// - an operand that is meant to be zero says so with [`expr::empty`].
///
/// [`build`](Self::build) then converts every wire to its [`ValueIndex`] and
/// produces the core constraint lists the prover and verifier consume.
pub struct ConstraintBuilder {
	/// AND constraints: `A & B == C`.
	pub and_constraints: Vec<WireAndConstraint>,
	/// Integer-multiply constraints: `A * B == (HI << 64) | LO`.
	pub imul_constraints: Vec<WireImulConstraint>,
	/// GHASH-field multiply constraints over `(lo, hi)` limb pairs.
	pub bmul_constraints: Vec<WireBmulConstraint>,
	/// Linear constraints `RHS == DST`, lowered to AND against the all-ones wire.
	pub linear_constraints: Vec<WireLinearConstraint>,
}

impl ConstraintBuilder {
	/// Creates an empty builder.
	pub const fn new() -> Self {
		Self {
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
			linear_constraints: Vec::new(),
		}
	}

	/// Appends an AND constraint `a & b == c`.
	pub fn and(&mut self, a: impl Into<WireExpr>, b: impl Into<WireExpr>, c: impl Into<WireExpr>) {
		self.and_constraints.push(WireAndConstraint {
			a: a.into().into_operand(),
			b: b.into().into_operand(),
			c: c.into().into_operand(),
		});
	}

	/// Appends an IMUL constraint `a * b == (hi << 64) | lo`.
	pub fn imul(
		&mut self,
		a: impl Into<WireExpr>,
		b: impl Into<WireExpr>,
		hi: impl Into<WireExpr>,
		lo: impl Into<WireExpr>,
	) {
		self.imul_constraints.push(WireImulConstraint {
			a: a.into().into_operand(),
			b: b.into().into_operand(),
			hi: hi.into().into_operand(),
			lo: lo.into().into_operand(),
		});
	}

	/// Appends a BMUL constraint `(a_lo, a_hi) * (b_lo, b_hi) == (c_lo, c_hi)` in the GHASH field.
	pub fn bmul(
		&mut self,
		a_lo: impl Into<WireExpr>,
		a_hi: impl Into<WireExpr>,
		b_lo: impl Into<WireExpr>,
		b_hi: impl Into<WireExpr>,
		c_lo: impl Into<WireExpr>,
		c_hi: impl Into<WireExpr>,
	) {
		self.bmul_constraints.push(WireBmulConstraint {
			a_lo: a_lo.into().into_operand(),
			a_hi: a_hi.into().into_operand(),
			b_lo: b_lo.into().into_operand(),
			b_hi: b_hi.into().into_operand(),
			c_lo: c_lo.into().into_operand(),
			c_hi: c_hi.into().into_operand(),
		});
	}

	/// Appends a linear constraint `rhs == dst`.
	///
	/// `rhs` is an XOR of shifted values; `dst` is a single wire.
	pub fn linear(&mut self, rhs: impl Into<WireExpr>, dst: Wire) {
		self.linear_constraints.push(WireLinearConstraint {
			rhs: rhs.into().into_operand(),
			dst,
		});
	}

	/// Lowers every wire-level constraint to its core `ValueIndex` form.
	///
	/// Linear constraints have no native core opcode, so each becomes
	/// `RHS & all_one == DST` — an AND against the all-ones wire that acts as
	/// the identity for `&`.
	pub fn build(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
		all_one: Wire,
	) -> (Vec<AndConstraint>, Vec<ImulConstraint>, Vec<BmulConstraint>) {
		let mut and_constraints = self
			.and_constraints
			.into_iter()
			.map(|c| c.into_constraint(wire_mapping))
			.collect::<Vec<_>>();

		let imul_constraints = self
			.imul_constraints
			.into_iter()
			.map(|c| c.into_constraint(wire_mapping))
			.collect();

		let bmul_constraints = self
			.bmul_constraints
			.into_iter()
			.map(|c| c.into_constraint(wire_mapping))
			.collect();

		if !self.linear_constraints.is_empty() {
			let all_one = wire_mapping[all_one];
			for linear_constraint in self.linear_constraints {
				let and_constraint = linear_constraint.into_and_constraint(wire_mapping, all_one);
				and_constraints.push(and_constraint);
			}
		}

		(and_constraints, imul_constraints, bmul_constraints)
	}

	/// Collects every wire referenced by any pending constraint.
	///
	/// Dead-code elimination uses this to keep wires that feed a constraint.
	pub fn mark_used_wires(&self) -> EntitySet<Wire> {
		let mut used_set = EntitySet::new();
		for ac in &self.and_constraints {
			ac.mark_used(&mut used_set);
		}
		for mc in &self.imul_constraints {
			mc.mark_used(&mut used_set);
		}
		for bc in &self.bmul_constraints {
			bc.mark_used(&mut used_set);
		}
		for lc in &self.linear_constraints {
			lc.mark_used(&mut used_set);
		}
		used_set
	}
}

impl Default for ConstraintBuilder {
	fn default() -> Self {
		Self::new()
	}
}
