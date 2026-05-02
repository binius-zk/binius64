// Copyright 2026 The Binius Developers

//! Channels that symbolically execute a verifier to build constraint systems or fill witnesses.

use std::{
	cell::RefCell,
	rc::{Rc, Weak},
	vec::IntoIter as VecIntoIter,
};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessError, WitnessGenerator},
	constraint_system::{ConstraintWire, Witness, WitnessLayout},
};

use super::circuit_elem::{CircuitElem, CircuitWire};
use crate::wrapper::circuit_elem::{BuilderWire, WitnessGenWire};

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

	fn alloc_precommit_elem(&self) -> CircuitElem<ConstraintBuilder<F>> {
		let wire = self.builder.borrow_mut().alloc_precommit();
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
	type Elem = CircuitElem<F, BuilderWire<F>>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		// For each element that the inner prover sends, the wrapped prover allocates a one-time-pad
		// encryption key in the precommit segment and encrypts the underlying value before sending.
		// Here the verifier gets the encryption key from the precommit segment and decrypts.
		let inout = self.alloc_inout_elem();
		let key = self.alloc_precommit_elem();
		Ok(inout - key)
	}

	fn sample(&mut self) -> Self::Elem {
		self.alloc_inout_elem()
	}

	fn observe_one(&mut self, _val: F) -> Self::Elem {
		self.alloc_inout_elem()
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			CircuitElem::Constant(c)
			| CircuitElem::Wire {
				wire: BuilderWire::Constant(c),
				..
			} => (c == F::ZERO).ok_or(binius_ip::channel::Error::InvalidAssert),
			CircuitElem::Wire {
				builder,
				wire: BuilderWire::Wire(wire),
			} => {
				assert!(Weak::ptr_eq(&Rc::downgrade(&self.builder), &builder));
				self.builder.borrow_mut().assert_zero(wire);
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
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'a, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		// For each oracle opening, the prover sends the decrypted evaluation. The outer verifier
		// checks in the circuit equality of this value with the expected expression over encrypted
		// values.
		for relation in oracle_relations {
			let decrypted_claim = self.alloc_inout_elem();
			self.assert_zero(relation.claim - decrypted_claim)?;
		}
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
	keys: VecIntoIter<F>,
	events: VecIntoIter<F>,
	next_inout_id: u32,
	next_precommit_id: u32,
}

impl<'a, F: Field> ReplayChannel<'a, F> {
	/// Creates a new replay channel.
	///
	/// TODO: Document args. Keys are the symmetric OTP keys for the received values.
	pub fn new(layout: &'a WitnessLayout<F>, keys: Vec<F>, events: Vec<F>) -> Self {
		Self {
			witness_gen: Rc::new(RefCell::new(WitnessGenerator::new(layout))),
			keys: keys.into_iter(),
			events: events.into_iter(),
			next_inout_id: 0,
			next_precommit_id: 0,
		}
	}

	fn next_inout_elem(&mut self) -> CircuitElem<F, WitnessGenWire<'a, F>> {
		let value = self
			.events
			.next()
			.unwrap_or_else(|| panic!("replay exhausted: no more events"));

		let wire = ConstraintWire::inout(self.next_inout_id);
		self.next_inout_id += 1;
		let witness_wire = self.witness_gen.borrow_mut().write_inout(wire, value);
		CircuitElem::Wire { builder: Rc::downgrade(&self.witness_gen), wire: WitnessGenWire::wire(witness_wire) }
	}

	fn next_precommit_elem(&mut self) -> CircuitElem<WitnessGenerator<'a, F>> {
		let value = self
			.keys
			.next()
			.expect("precommit segment is sized incorrectly");

		let wire = ConstraintWire::precommit(self.next_precommit_id);
		self.next_precommit_id += 1;
		let witness_wire = self.witness_gen.borrow_mut().write_precommit(wire, value);
		CircuitElem::Wire { builder: Rc::downgrade(&self.witness_gen), wire: WitnessGenWire::wire(witness_wire) }
	}

	/// Consumes the channel and builds the outer witness.
	pub fn finish(self) -> Result<Witness<F>, WitnessError> {
		Rc::try_unwrap(self.witness_gen)
			.expect("CircuitElem values should only hold Weak references")
			.into_inner()
			.build()
	}
}

impl<'a, F: Field> IPVerifierChannel<F> for ReplayChannel<'a, F> {
	type Elem = CircuitElem<F, WitnessGenWire<'a, F>>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		let encrypted_elem = self.next_inout_elem();
		let key = self.next_precommit_elem();
		Ok(encrypted_elem + key)
	}

	fn sample(&mut self) -> Self::Elem {
		self.next_inout_elem()
	}

	fn observe_one(&mut self, _val: F) -> Self::Elem {
		self.next_inout_elem()
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			CircuitElem::Constant(c)
			| CircuitElem::Wire {
				wire: WitnessGenWire::Constant(c),
				..
			} => (c == F::ZERO).ok_or(binius_ip::channel::Error::InvalidAssert),
			CircuitElem::Wire {
				builder,
				wire: WitnessGenWire::Wire(wire),
			} => {
				assert!(Weak::ptr_eq(&Rc::downgrade(&self.witness_gen), &builder));
				self.witness_gen.borrow_mut().assert_zero(wire);
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
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'b, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		// For each oracle opening, the prover sends the decrypted evaluation. The outer verifier
		// checks in the circuit equality of this value with the expected expression over encrypted
		// values.
		for relation in oracle_relations {
			let decrypted_claim = self.next_inout_elem();
			self.assert_zero(relation.claim - decrypted_claim)?;
		}
		Ok(())
	}
}
