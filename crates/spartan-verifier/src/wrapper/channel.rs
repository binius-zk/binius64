// Copyright 2026 The Binius Developers

//! Channels that symbolically execute a verifier to build constraint systems or fill witnesses.

use std::{cell::RefCell, rc::Rc};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
	constraint_system::{ConstraintWire, WitnessLayout},
};

use super::circuit_elem::{CircuitElem, CircuitWire};

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations as constraints
/// in a [`ConstraintBuilder`]. The typical usage pattern is:
///
/// 1. Create an `IronSpartanBuilderChannel` from a [`ConstraintBuilder`]
/// 2. Run the verifier on the channel (e.g., `verify_iop`)
/// 3. The channel's `finish()` method returns the [`ConstraintBuilder`] with all recorded
///    constraints
pub struct IronSpartanBuilderChannel<F: Field> {
	builder: Rc<RefCell<ConstraintBuilder<F>>>,
}

impl<F: Field> IronSpartanBuilderChannel<F> {
	/// Creates a new builder channel that takes ownership of the given constraint builder.
	pub fn new(builder: ConstraintBuilder<F>) -> Self {
		Self {
			builder: Rc::new(RefCell::new(builder)),
		}
	}

	fn alloc_inout_elem(&self) -> CircuitElem<ConstraintBuilder<F>> {
		let wire = self.builder.borrow_mut().alloc_inout();
		CircuitElem::Wire(CircuitWire::new(&self.builder, wire))
	}

	/// Consumes the channel and returns the underlying [`ConstraintBuilder`].
	///
	/// This must be called after all `CircuitElem` values derived from this channel have been
	/// dropped, as it requires sole ownership of the builder via `Rc::try_unwrap`.
	pub fn finish(self) -> ConstraintBuilder<F> {
		Rc::try_unwrap(self.builder)
			.expect("CircuitElem values should only hold Weak references")
			.into_inner()
	}
}

impl<F: Field> IPVerifierChannel<F> for IronSpartanBuilderChannel<F> {
	type Elem = CircuitElem<ConstraintBuilder<F>>;

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

	fn observe_one(&mut self, _val: F) -> Self::Elem {
		self.alloc_inout_elem()
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem> {
		(0..vals.len()).map(|_| self.alloc_inout_elem()).collect()
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			CircuitElem::Constant(c) if c == F::ZERO => Ok(()),
			CircuitElem::Constant(_) => Err(binius_ip::channel::Error::InvalidAssert),
			CircuitElem::Wire(w) => {
				self.builder.borrow_mut().assert_zero(w.wire());
				Ok(())
			}
		}
	}
}

impl<F: Field> IOPVerifierChannel<F> for IronSpartanBuilderChannel<F> {
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

/// A channel that replays recorded interaction values through a [`WitnessGenerator`], filling
/// both inout and private wires in the outer witness.
///
/// This mirrors [`IronSpartanBuilderChannel`] but uses concrete evaluation instead of symbolic
/// constraint building. Each operation consumes the next value and writes it to the corresponding
/// inout wire in the [`WitnessGenerator`]. When the verifier's arithmetic runs on the returned
/// [`CircuitElem`] values, the [`WitnessGenerator`] fills private wires.
pub struct ReplayChannel<'a, F: Field> {
	witness_gen: Rc<RefCell<WitnessGenerator<'a, F>>>,
	events: std::vec::IntoIter<F>,
	next_inout_id: u32,
}

impl<'a, F: Field> ReplayChannel<'a, F> {
	/// Creates a new replay channel.
	pub fn new(layout: &'a WitnessLayout<F>, events: Vec<F>) -> Self {
		Self {
			witness_gen: Rc::new(RefCell::new(WitnessGenerator::new(layout))),
			events: events.into_iter(),
			next_inout_id: 0,
		}
	}

	fn next_inout_elem(&mut self, value: F) -> CircuitElem<WitnessGenerator<'a, F>> {
		let wire = ConstraintWire::inout(self.next_inout_id);
		self.next_inout_id += 1;
		let witness_wire = self.witness_gen.borrow_mut().write_inout(wire, value);
		CircuitElem::Wire(CircuitWire::new(&self.witness_gen, witness_wire))
	}

	fn next_event(&mut self) -> F {
		self.events
			.next()
			.unwrap_or_else(|| panic!("replay exhausted: no more events"))
	}

	/// Consumes the channel and builds the outer witness.
	pub fn finish(self) -> Result<Vec<F>, binius_spartan_frontend::circuit_builder::WitnessError> {
		Rc::try_unwrap(self.witness_gen)
			.expect("CircuitElem values should only hold Weak references")
			.into_inner()
			.build()
	}
}

impl<'a, F: Field> IPVerifierChannel<F> for ReplayChannel<'a, F> {
	type Elem = CircuitElem<WitnessGenerator<'a, F>>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		let val = self.next_event();
		Ok(self.next_inout_elem(val))
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, binius_ip::channel::Error> {
		(0..n).map(|_| self.recv_one()).collect()
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], binius_ip::channel::Error> {
		let mut result = [(); N].map(|_| CircuitElem::Constant(F::ZERO));
		for elem in &mut result {
			*elem = self.recv_one()?;
		}
		Ok(result)
	}

	fn sample(&mut self) -> Self::Elem {
		let val = self.next_event();
		self.next_inout_elem(val)
	}

	fn observe_one(&mut self, _val: F) -> Self::Elem {
		let val = self.next_event();
		self.next_inout_elem(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem> {
		vals.iter().map(|&val| self.observe_one(val)).collect()
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			CircuitElem::Constant(c) if c == F::ZERO => Ok(()),
			CircuitElem::Constant(_) => Err(binius_ip::channel::Error::InvalidAssert),
			CircuitElem::Wire(w) => {
				self.witness_gen.borrow_mut().assert_zero(w.wire());
				Ok(())
			}
		}
	}
}

impl<'a, F: Field> IOPVerifierChannel<F> for ReplayChannel<'a, F> {
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
