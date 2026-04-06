// Copyright 2026 The Binius Developers

//! Builder channel that symbolically executes a verifier to build constraint systems.

use std::{cell::RefCell, rc::Rc};

use binius_field::{BinaryField128bGhash as B128, Field};
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
	constraint_system::{ConstraintWire, WitnessLayout},
};

use crate::circuit_elem::{CircuitElem, CircuitWire};

type BuildElem = CircuitElem<ConstraintBuilder<B128>>;
type BuildWire = CircuitWire<ConstraintBuilder<B128>>;

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations as constraints
/// in a [`ConstraintBuilder<B128>`]. The typical usage pattern is:
///
/// 1. Create an `IronSpartanBuilderChannel` from a `ConstraintBuilder<B128>`
/// 2. Run the verifier on the channel (e.g., `verify_iop`)
/// 3. The channel's `finish()` method returns the `ConstraintBuilder<B128>` with all recorded
///    constraints
pub struct IronSpartanBuilderChannel {
	builder: Rc<RefCell<ConstraintBuilder<B128>>>,
}

impl IronSpartanBuilderChannel {
	/// Creates a new builder channel that takes ownership of the given constraint builder.
	pub fn new(builder: ConstraintBuilder<B128>) -> Self {
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

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&[]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
		Ok(())
	}

	fn verify_oracle_relations<'a>(
		&mut self,
		_oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'a, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		Ok(())
	}
}

impl IronSpartanBuilderChannel {
	/// Consumes the channel and returns the underlying [`ConstraintBuilder<B128>`].
	///
	/// This must be called after all `BuildElem` values derived from this channel have been
	/// dropped, as it requires sole ownership of the builder via `Rc::try_unwrap`.
	pub fn finish(self) -> ConstraintBuilder<B128> {
		Rc::try_unwrap(self.builder)
			.expect("BuildElem values should only hold Weak references")
			.into_inner()
	}
}

type WitnessElem<'a> = CircuitElem<WitnessGenerator<'a, B128>>;
type WitnessWire<'a> = CircuitWire<WitnessGenerator<'a, B128>>;

/// A channel that replays recorded interaction values through a [`WitnessGenerator`], filling
/// both inout and private wires in the outer witness.
///
/// This mirrors [`IronSpartanBuilderChannel`] but uses concrete evaluation instead of symbolic
/// constraint building. Each operation consumes the next value and writes it to the corresponding
/// inout wire in the [`WitnessGenerator`]. When the verifier's arithmetic runs on the returned
/// [`CircuitElem`] values, the [`WitnessGenerator`] fills private wires.
pub struct ReplayChannel<'a> {
	witness_gen: Rc<RefCell<WitnessGenerator<'a, B128>>>,
	events: std::vec::IntoIter<B128>,
	next_inout_id: u32,
}

impl<'a> ReplayChannel<'a> {
	/// Creates a new replay channel.
	pub fn new(layout: &'a WitnessLayout<B128>, events: Vec<B128>) -> Self {
		Self {
			witness_gen: Rc::new(RefCell::new(WitnessGenerator::new(layout))),
			events: events.into_iter(),
			next_inout_id: 0,
		}
	}

	fn next_inout_elem(&mut self, value: B128) -> WitnessElem<'a> {
		let wire = ConstraintWire::inout(self.next_inout_id);
		self.next_inout_id += 1;
		let witness_wire = self.witness_gen.borrow_mut().write_inout(wire, value);
		WitnessElem::Wire(WitnessWire::new(&self.witness_gen, witness_wire))
	}

	fn next_event(&mut self) -> B128 {
		self.events
			.next()
			.unwrap_or_else(|| panic!("replay exhausted: no more events"))
	}

	/// Consumes the channel and builds the outer witness.
	pub fn finish(
		self,
	) -> Result<Vec<B128>, binius_spartan_frontend::circuit_builder::WitnessError> {
		Rc::try_unwrap(self.witness_gen)
			.expect("CircuitElem values should only hold Weak references")
			.into_inner()
			.build()
	}
}

impl<'a> IPVerifierChannel<B128> for ReplayChannel<'a> {
	type Elem = WitnessElem<'a>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		let val = self.next_event();
		Ok(self.next_inout_elem(val))
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, binius_ip::channel::Error> {
		(0..n).map(|_| self.recv_one()).collect()
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], binius_ip::channel::Error> {
		let mut result = [(); N].map(|_| WitnessElem::Constant(B128::ZERO));
		for elem in &mut result {
			*elem = self.recv_one()?;
		}
		Ok(result)
	}

	fn sample(&mut self) -> Self::Elem {
		let val = self.next_event();
		self.next_inout_elem(val)
	}

	fn observe_one(&mut self, _val: B128) -> Self::Elem {
		let val = self.next_event();
		self.next_inout_elem(val)
	}

	fn observe_many(&mut self, vals: &[B128]) -> Vec<Self::Elem> {
		vals.iter().map(|&val| self.observe_one(val)).collect()
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			WitnessElem::Constant(c) if c == B128::ZERO => Ok(()),
			WitnessElem::Constant(_) => Err(binius_ip::channel::Error::InvalidAssert),
			WitnessElem::Wire(w) => {
				self.witness_gen.borrow_mut().assert_zero(w.wire());
				Ok(())
			}
		}
	}
}

impl<'a> IOPVerifierChannel<B128> for ReplayChannel<'a> {
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&[]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
		Ok(())
	}

	fn verify_oracle_relations<'b>(
		&mut self,
		_oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'b, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		Ok(())
	}
}
