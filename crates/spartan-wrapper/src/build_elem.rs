// Copyright 2026 The Binius Developers

//! Symbolic field element types for building constraint systems.

use std::cell::RefCell;

use binius_field::BinaryField128bGhash as B128;
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
}
