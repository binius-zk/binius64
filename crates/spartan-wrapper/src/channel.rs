// Copyright 2026 The Binius Developers

//! Builder channel that symbolically executes a verifier to build constraint systems.

use std::{cell::RefCell, rc::Rc};

use binius_field::{BinaryField128bGhash as B128, Field};
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::circuit_builder::ConstraintBuilder;

use crate::build_elem::{BuildElem, BuildWire};

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations as constraints
/// in a [`ConstraintBuilder`]. The typical usage pattern is:
///
/// 1. Create an `IronSpartanBuilderChannel` from a `ConstraintBuilder`
/// 2. Run the verifier on the channel (e.g., `verify_iop`)
/// 3. The channel's `finish()` method returns the `ConstraintBuilder` with all recorded constraints
pub struct IronSpartanBuilderChannel {
	builder: Rc<RefCell<ConstraintBuilder>>,
}

impl IronSpartanBuilderChannel {
	/// Creates a new builder channel that takes ownership of the given constraint builder.
	pub fn new(builder: ConstraintBuilder) -> Self {
		Self {
			builder: Rc::new(RefCell::new(builder)),
		}
	}

	fn alloc_inout_elem(&self) -> BuildElem {
		let wire = self.builder.borrow_mut().alloc_inout();
		BuildElem::Wire(BuildWire::new(&self.builder, wire))
	}
}

impl IPVerifierChannel<B128> for IronSpartanBuilderChannel {
	type Elem = BuildElem;

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

impl IOPVerifierChannel<B128> for IronSpartanBuilderChannel {
	type Oracle = ();
	type Finish = ConstraintBuilder;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&[]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
		Ok(())
	}

	fn finish(
		self,
		_oracle_relations: &[OracleLinearRelation<'_, Self::Oracle, Self::Elem>],
	) -> Result<Self::Finish, binius_iop::channel::Error> {
		Ok(Rc::try_unwrap(self.builder)
			.expect("BuildElem values should only hold Weak references")
			.into_inner())
	}
}
