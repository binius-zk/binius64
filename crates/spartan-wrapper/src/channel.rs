// Copyright 2026 The Binius Developers

//! Builder channel that symbolically executes a verifier to build constraint systems.

use std::cell::RefCell;

use binius_spartan_frontend::circuit_builder::ConstraintBuilder;

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations as constraints
/// in a [`ConstraintBuilder`]. The typical usage pattern is:
///
/// 1. Create a `RefCell<ConstraintBuilder>`
/// 2. Create an `IronSpartanBuilderChannel` referencing it
/// 3. Run the verifier on the channel
/// 4. Drop the channel and extract the built constraint system from the builder
pub struct IronSpartanBuilderChannel<'a> {
	builder: &'a RefCell<ConstraintBuilder>,
}

impl<'a> IronSpartanBuilderChannel<'a> {
	/// Creates a new builder channel referencing the given constraint builder.
	pub fn new(builder: &'a RefCell<ConstraintBuilder>) -> Self {
		Self { builder }
	}
}
