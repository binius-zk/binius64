// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Wire-level constraint DSL and its lowering to core `ValueIndex` constraints.

mod constraint;
pub mod expr;
mod operand;

use binius_core::constraint_system::{
	AndConstraint, BmulConstraint, ImulConstraint, ValueIndex, ZeroConstraint,
};
pub use constraint::{
	WireAndConstraint, WireBmulConstraint, WireImulConstraint, WireLinearConstraint,
	WireZeroConstraint,
};
use cranelift_entity::{EntitySet, SecondaryMap};
use expr::WireExpr;
pub use expr::WireExprTerm;
pub use operand::{PushInner, ShiftedWire, WireOperand, push_inner};

use crate::ir::Wire;

/// Accumulates the constraints a circuit emits, expressed over [`Wire`]s.
///
/// Gates push into the five typed buckets, one method per constraint shape.
///
/// Every operand is a parameter of that method:
///
/// - leaving one out is a compile error rather than a silent zero;
/// - an operand that is meant to be zero says so with [`expr::empty`].
///
/// Every operand's terms live in one shared arena rather than in a `Vec` of their own.
/// An operand handle is therefore `Copy`.
/// Building an operand never allocates on its own account.
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
	/// Linear constraints `RHS == DST`, lowered by [`build`](Self::build) to Zero constraints.
	pub linear_constraints: Vec<WireLinearConstraint>,
	/// Zero constraints `VAL == 0`, which assert rather than define.
	pub zero_constraints: Vec<WireZeroConstraint>,
	/// Every operand's terms, in the order they were appended.
	///
	/// A [`WireOperand`] names its own slice of this by range rather than owning a `Vec`.
	terms: Vec<ShiftedWire>,
}

impl ConstraintBuilder {
	/// Creates an empty builder.
	pub const fn new() -> Self {
		Self {
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
			linear_constraints: Vec::new(),
			zero_constraints: Vec::new(),
			terms: Vec::new(),
		}
	}

	/// Builds an operand from a ready sequence of terms, appending them to the shared arena.
	pub fn push_operand(&mut self, terms: impl IntoIterator<Item = ShiftedWire>) -> WireOperand {
		let start = self.terms.len();
		self.terms.extend(terms);
		WireOperand::from_range(start, self.terms.len() - start)
	}

	/// The terms one operand XORs together, resolved through the shared arena.
	pub fn operand_terms(&self, operand: WireOperand) -> &[ShiftedWire] {
		operand.as_slice(&self.terms)
	}

	/// One term of an already-built operand, resolved through the shared arena.
	///
	/// Gate fusion reads terms this way, one at a time, while it is also appending new ones.
	/// Copying a term out this way never holds a live borrow of the arena across a later push.
	pub(crate) fn term(&self, operand: WireOperand, index: usize) -> ShiftedWire {
		operand.as_slice(&self.terms)[index]
	}

	/// Where the next pushed term would land.
	///
	/// Pairs with [`operand_since`](Self::operand_since) to grow an operand term by term.
	/// This suits a caller that discovers its terms one at a time, not as a ready sequence.
	pub(crate) const fn next_term_start(&self) -> usize {
		self.terms.len()
	}

	/// Appends one term to the shared arena.
	pub(crate) fn push_term(&mut self, term: ShiftedWire) {
		self.terms.push(term);
	}

	/// The handle spanning every term pushed since `start`.
	pub(crate) fn operand_since(&self, start: usize) -> WireOperand {
		WireOperand::from_range(start, self.terms.len() - start)
	}

	/// Appends an AND constraint `a & b == c`.
	pub fn and(&mut self, a: impl Into<WireExpr>, b: impl Into<WireExpr>, c: impl Into<WireExpr>) {
		let constraint = WireAndConstraint {
			a: a.into().into_operand(self),
			b: b.into().into_operand(self),
			c: c.into().into_operand(self),
		};
		self.and_constraints.push(constraint);
	}

	/// Appends an IMUL constraint `a * b == (hi << 64) | lo`.
	pub fn imul(
		&mut self,
		a: impl Into<WireExpr>,
		b: impl Into<WireExpr>,
		hi: impl Into<WireExpr>,
		lo: impl Into<WireExpr>,
	) {
		let constraint = WireImulConstraint {
			a: a.into().into_operand(self),
			b: b.into().into_operand(self),
			hi: hi.into().into_operand(self),
			lo: lo.into().into_operand(self),
		};
		self.imul_constraints.push(constraint);
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
		let constraint = WireBmulConstraint {
			a_lo: a_lo.into().into_operand(self),
			a_hi: a_hi.into().into_operand(self),
			b_lo: b_lo.into().into_operand(self),
			b_hi: b_hi.into().into_operand(self),
			c_lo: c_lo.into().into_operand(self),
			c_hi: c_hi.into().into_operand(self),
		};
		self.bmul_constraints.push(constraint);
	}

	/// Appends a linear constraint `rhs == dst`.
	///
	/// `rhs` is an XOR of shifted values; `dst` is a single wire.
	pub fn linear(&mut self, rhs: impl Into<WireExpr>, dst: Wire) {
		let constraint = WireLinearConstraint {
			rhs: rhs.into().into_operand(self),
			dst,
		};
		self.linear_constraints.push(constraint);
	}

	/// Appends the assertion `val == 0`.
	///
	/// Unlike [`linear`](Self::linear) this defines no wire, so it is the right shape for an
	/// assertion gate: the equation it states is enforced without naming a value it produces.
	pub fn zero(&mut self, val: impl Into<WireExpr>) {
		let constraint = WireZeroConstraint {
			val: val.into().into_operand(self),
		};
		self.zero_constraints.push(constraint);
	}

	/// Lowers every wire-level constraint to its core `ValueIndex` form.
	///
	/// A linear constraint lowers to the Zero constraint `RHS ^ DST == 0`, which carries one
	/// constraint array where an AND against the all-ones constant would carry three.
	pub fn build(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> (Vec<ZeroConstraint>, Vec<AndConstraint>, Vec<ImulConstraint>, Vec<BmulConstraint>) {
		let terms = &self.terms;

		let and_constraints = self
			.and_constraints
			.into_iter()
			.map(|c| c.into_constraint(terms, wire_mapping))
			.collect::<Vec<_>>();

		let imul_constraints = self
			.imul_constraints
			.into_iter()
			.map(|c| c.into_constraint(terms, wire_mapping))
			.collect();

		let bmul_constraints = self
			.bmul_constraints
			.into_iter()
			.map(|c| c.into_constraint(terms, wire_mapping))
			.collect();

		let zero_constraints = self
			.linear_constraints
			.into_iter()
			.map(|c| c.into_zero_constraint(terms, wire_mapping))
			.chain(
				self.zero_constraints
					.into_iter()
					.map(|c| c.into_constraint(terms, wire_mapping)),
			)
			.collect();

		(zero_constraints, and_constraints, imul_constraints, bmul_constraints)
	}

	/// Collects every wire referenced by any pending constraint.
	///
	/// Dead-code elimination uses this to keep wires that feed a constraint.
	pub fn mark_used_wires(&self) -> EntitySet<Wire> {
		let mut used_set = EntitySet::new();
		for ac in &self.and_constraints {
			ac.mark_used(&self.terms, &mut used_set);
		}
		for mc in &self.imul_constraints {
			mc.mark_used(&self.terms, &mut used_set);
		}
		for bc in &self.bmul_constraints {
			bc.mark_used(&self.terms, &mut used_set);
		}
		for lc in &self.linear_constraints {
			lc.mark_used(&self.terms, &mut used_set);
		}
		for zc in &self.zero_constraints {
			zc.mark_used(&self.terms, &mut used_set);
		}
		used_set
	}
}

impl Default for ConstraintBuilder {
	fn default() -> Self {
		Self::new()
	}
}
