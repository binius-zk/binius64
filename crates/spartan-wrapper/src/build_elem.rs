// Copyright 2026 The Binius Developers

//! Symbolic field element types for building constraint systems.

use std::{
	cell::RefCell,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_field::{BinaryField128bGhash as B128, Field};
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder},
	constraint_system::ConstraintWire,
};

/// An opaque wire in the constraint builder, carrying a reference to the builder.
#[derive(Clone, Copy)]
pub struct BuildWire<'a> {
	builder: &'a RefCell<ConstraintBuilder>,
	wire: ConstraintWire,
}

impl<'a> BuildWire<'a> {
	pub(crate) fn new(builder: &'a RefCell<ConstraintBuilder>, wire: ConstraintWire) -> Self {
		Self { builder, wire }
	}

	pub(crate) fn builder(&self) -> &'a RefCell<ConstraintBuilder> {
		self.builder
	}

	pub(crate) fn wire(&self) -> ConstraintWire {
		self.wire
	}
}

/// A symbolic field element that is either a known constant or a wire in a constraint system.
#[derive(Clone, Copy)]
pub enum BuildElem<'a> {
	Constant(B128),
	Wire(BuildWire<'a>),
}

impl<'a> BuildElem<'a> {
	/// Returns the builder reference if this is a Wire variant.
	pub(crate) fn builder(&self) -> Option<&'a RefCell<ConstraintBuilder>> {
		match self {
			BuildElem::Constant(_) => None,
			BuildElem::Wire(w) => Some(w.builder),
		}
	}

	/// Given two BuildElems, return the builder that at least one of them references.
	///
	/// Panics if both are Wire variants referencing different builders.
	pub(crate) fn resolve_builder(
		a: &BuildElem<'a>,
		b: &BuildElem<'a>,
	) -> &'a RefCell<ConstraintBuilder> {
		match (a.builder(), b.builder()) {
			(Some(ba), Some(bb)) => {
				assert!(
					std::ptr::eq(ba, bb),
					"BuildElem wires reference different ConstraintBuilders"
				);
				ba
			}
			(Some(b), None) | (None, Some(b)) => b,
			(None, None) => panic!("cannot resolve builder: both operands are constants"),
		}
	}

	/// Convert this element to a ConstraintWire, allocating a constant wire if necessary.
	pub(crate) fn to_wire(&self, builder: &mut ConstraintBuilder) -> ConstraintWire {
		match self {
			BuildElem::Constant(val) => builder.constant(*val),
			BuildElem::Wire(w) => w.wire,
		}
	}

	fn make_wire(builder: &'a RefCell<ConstraintBuilder>, wire: ConstraintWire) -> Self {
		BuildElem::Wire(BuildWire { builder, wire })
	}
}

// In characteristic 2, negation is identity.
impl Neg for BuildElem<'_> {
	type Output = Self;

	fn neg(self) -> Self {
		self
	}
}

impl<'a> Add for BuildElem<'a> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(BuildElem::Constant(a), BuildElem::Constant(b)) => BuildElem::Constant(*a + *b),
			_ => {
				if matches!(&self, BuildElem::Constant(c) if *c == B128::ZERO) {
					return rhs;
				}
				if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ZERO) {
					return self;
				}
				let builder_ref = BuildElem::resolve_builder(&self, &rhs);
				let mut builder = builder_ref.borrow_mut();
				let a_wire = self.to_wire(&mut builder);
				let b_wire = rhs.to_wire(&mut builder);
				let out = builder.add(a_wire, b_wire);
				BuildElem::make_wire(builder_ref, out)
			}
		}
	}
}

impl<'a> Sub for BuildElem<'a> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(BuildElem::Constant(a), BuildElem::Constant(b)) => BuildElem::Constant(*a + *b),
			_ => {
				if matches!(&self, BuildElem::Constant(c) if *c == B128::ZERO) {
					return rhs;
				}
				if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ZERO) {
					return self;
				}
				let builder_ref = BuildElem::resolve_builder(&self, &rhs);
				let mut builder = builder_ref.borrow_mut();
				let a_wire = self.to_wire(&mut builder);
				let b_wire = rhs.to_wire(&mut builder);
				let out = builder.sub(a_wire, b_wire);
				BuildElem::make_wire(builder_ref, out)
			}
		}
	}
}

impl<'a> Mul for BuildElem<'a> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(BuildElem::Constant(a), BuildElem::Constant(b)) => BuildElem::Constant(*a * *b),
			_ => {
				if matches!(&self, BuildElem::Constant(c) if *c == B128::ZERO) {
					return BuildElem::Constant(B128::ZERO);
				}
				if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ZERO) {
					return BuildElem::Constant(B128::ZERO);
				}
				if matches!(&self, BuildElem::Constant(c) if *c == B128::ONE) {
					return rhs;
				}
				if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ONE) {
					return self;
				}
				let builder_ref = BuildElem::resolve_builder(&self, &rhs);
				let mut builder = builder_ref.borrow_mut();
				let a_wire = self.to_wire(&mut builder);
				let b_wire = rhs.to_wire(&mut builder);
				let out = builder.mul(a_wire, b_wire);
				BuildElem::make_wire(builder_ref, out)
			}
		}
	}
}

// By-reference variants: clone and delegate.

impl<'a> Add<&BuildElem<'a>> for BuildElem<'a> {
	type Output = Self;

	fn add(self, rhs: &BuildElem<'a>) -> Self {
		self + rhs.clone()
	}
}

impl<'a> Sub<&BuildElem<'a>> for BuildElem<'a> {
	type Output = Self;

	fn sub(self, rhs: &BuildElem<'a>) -> Self {
		self + rhs.clone()
	}
}

impl<'a> Mul<&BuildElem<'a>> for BuildElem<'a> {
	type Output = Self;

	fn mul(self, rhs: &BuildElem<'a>) -> Self {
		self * rhs.clone()
	}
}

// Assign variants

impl<'a> AddAssign for BuildElem<'a> {
	fn add_assign(&mut self, rhs: Self) {
		*self = self.clone() + rhs;
	}
}

impl<'a> SubAssign for BuildElem<'a> {
	fn sub_assign(&mut self, rhs: Self) {
		*self = self.clone() + rhs;
	}
}

impl<'a> MulAssign for BuildElem<'a> {
	fn mul_assign(&mut self, rhs: Self) {
		*self = self.clone() * rhs;
	}
}

impl<'a> AddAssign<&BuildElem<'a>> for BuildElem<'a> {
	fn add_assign(&mut self, rhs: &BuildElem<'a>) {
		*self = self.clone() + rhs.clone();
	}
}

impl<'a> SubAssign<&BuildElem<'a>> for BuildElem<'a> {
	fn sub_assign(&mut self, rhs: &BuildElem<'a>) {
		*self = self.clone() + rhs.clone();
	}
}

impl<'a> MulAssign<&BuildElem<'a>> for BuildElem<'a> {
	fn mul_assign(&mut self, rhs: &BuildElem<'a>) {
		*self = self.clone() * rhs.clone();
	}
}

// Sum and Product

impl<'a> Sum for BuildElem<'a> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(BuildElem::Constant(B128::ZERO), |acc, x| acc + x)
	}
}

impl<'a, 'b> Sum<&'b BuildElem<'a>> for BuildElem<'a> {
	fn sum<I: Iterator<Item = &'b BuildElem<'a>>>(iter: I) -> Self {
		iter.cloned().sum()
	}
}

impl<'a> Product for BuildElem<'a> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(BuildElem::Constant(B128::ONE), |acc, x| acc * x)
	}
}

impl<'a, 'b> Product<&'b BuildElem<'a>> for BuildElem<'a> {
	fn product<I: Iterator<Item = &'b BuildElem<'a>>>(iter: I) -> Self {
		iter.cloned().product()
	}
}
