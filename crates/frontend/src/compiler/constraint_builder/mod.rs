// Copyright 2025 Irreducible Inc.

//! Wire-level constraint DSL and its lowering to core `ValueIndex` constraints.

mod builder;
mod constraint;
pub mod expr;
mod shift;

use binius_core::constraint_system::{AndConstraint, BmulConstraint, ImulConstraint, ValueIndex};
pub use builder::{
	AndConstraintBuilder, BmulConstraintBuilder, ImulConstraintBuilder, LinearConstraintBuilder,
};
pub use constraint::{
	WireAndConstraint, WireBmulConstraint, WireImulConstraint, WireLinearConstraint,
};
use cranelift_entity::{EntitySet, SecondaryMap};
pub use expr::WireExprTerm;
pub use shift::{Shift, ShiftedWire, WireOperand};

use crate::compiler::Wire;

/// Accumulates the constraints a circuit emits, expressed over [`Wire`]s.
///
/// Gates push into the four typed buckets through the fluent builders.
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

	/// Starts an AND constraint `A & B == C`.
	pub const fn and(&mut self) -> AndConstraintBuilder<'_> {
		AndConstraintBuilder::new(self)
	}

	/// Starts an IMUL constraint `A * B == (HI << 64) | LO`.
	pub const fn imul(&mut self) -> ImulConstraintBuilder<'_> {
		ImulConstraintBuilder::new(self)
	}

	/// Starts a BMUL constraint `(A_LO, A_HI) * (B_LO, B_HI) == (C_LO, C_HI)` in the GHASH field.
	pub const fn bmul(&mut self) -> BmulConstraintBuilder<'_> {
		BmulConstraintBuilder::new(self)
	}

	/// Starts a linear constraint `RHS == DST`.
	///
	/// `RHS` is an XOR of shifted values; `DST` is a single wire.
	pub const fn linear(&mut self) -> LinearConstraintBuilder<'_> {
		LinearConstraintBuilder::new(self)
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
