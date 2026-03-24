// Copyright 2026 The Binius Developers

//! Builder channel that symbolically executes a verifier to build constraint systems.

use std::cell::RefCell;

use binius_field::{BinaryField128bGhash as B128, Field};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::circuit_builder::ConstraintBuilder;

use crate::build_elem::{BuildElem, BuildWire};

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

	fn alloc_inout_elem(&self) -> BuildElem<'a> {
		let wire = self.builder.borrow_mut().alloc_inout();
		BuildElem::Wire(BuildWire::new(self.builder, wire))
	}
}

impl<'a> IPVerifierChannel<B128> for IronSpartanBuilderChannel<'a> {
	type Elem = BuildElem<'a>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		Ok(self.alloc_inout_elem())
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, binius_ip::channel::Error> {
		Ok((0..n).map(|_| self.alloc_inout_elem()).collect())
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], binius_ip::channel::Error> {
		Ok(std::array::from_fn(|_| self.alloc_inout_elem()))
	}

	fn sample(&mut self) -> Self::Elem {
		self.alloc_inout_elem()
	}

	fn observe_one(&mut self, _val: B128) -> Self::Elem {
		self.alloc_inout_elem()
	}

	fn observe_many(&mut self, vals: &[B128]) -> Vec<Self::Elem> {
		(0..vals.len()).map(|_| self.alloc_inout_elem()).collect()
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		use binius_spartan_frontend::circuit_builder::CircuitBuilder;

		match val {
			BuildElem::Constant(c) if c == B128::ZERO => Ok(()),
			BuildElem::Constant(_) => Err(binius_ip::channel::Error::InvalidAssert),
			BuildElem::Wire(w) => {
				self.builder.borrow_mut().assert_zero(w.wire());
				Ok(())
			}
		}
	}
}
