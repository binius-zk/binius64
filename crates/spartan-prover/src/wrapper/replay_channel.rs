// Copyright 2026 The Binius Developers

//! [`ReplayChannel`]: an [`IPVerifierChannel`] that replays recorded interaction values through a
//! [`WitnessGenerator`], filling both inout and private wires in the outer witness.

use std::{
	cell::RefCell,
	rc::{Rc, Weak},
	vec::IntoIter as VecIntoIter,
};

use binius_field::{Field, util::FieldFn};
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, WireAllocator, WitnessError, WitnessGenerator},
	constraint_system::{WireKind, Witness, WitnessLayout},
};
use binius_spartan_verifier::wrapper::circuit_elem::CircuitElem;

/// A channel that replays recorded interaction values through a [`WitnessGenerator`], filling
/// both inout and private wires in the outer witness.
///
/// This mirrors
/// [`IronSpartanBuilderChannel`](binius_spartan_verifier::wrapper::IronSpartanBuilderChannel)
/// but uses concrete evaluation instead of symbolic constraint building. Each operation consumes
/// the next value and writes it to the corresponding inout wire in the [`WitnessGenerator`]. When
/// the verifier's arithmetic runs on the returned [`CircuitElem`] values, the [`WitnessGenerator`]
/// fills private wires.
pub struct ReplayChannel<'a, F: Field> {
	witness_gen: Rc<RefCell<WitnessGenerator<'a, F>>>,
	/// Allocators for the InOut and Precommit segments. They live here, not on the
	/// [`WitnessGenerator`], because allocating wires in interaction order is the channel's job;
	/// the generator just writes a value to a given wire. Allocation order must match the symbolic
	/// [`IronSpartanBuilderChannel`](binius_spartan_verifier::wrapper::IronSpartanBuilderChannel)
	/// so the wire ids align with `layout`.
	inout_alloc: WireAllocator,
	precommit_alloc: WireAllocator,
	keys: VecIntoIter<F>,
	events: VecIntoIter<F>,
}

impl<'a, F: Field> ReplayChannel<'a, F> {
	/// Creates a new replay channel.
	///
	/// TODO: Document args. Keys are the symmetric OTP keys for the received values.
	pub fn new(layout: &'a WitnessLayout<F>, keys: Vec<F>, events: Vec<F>) -> Self {
		Self {
			witness_gen: Rc::new(RefCell::new(WitnessGenerator::new(layout))),
			inout_alloc: WireAllocator::new(WireKind::InOut),
			precommit_alloc: WireAllocator::new(WireKind::Precommit),
			keys: keys.into_iter(),
			events: events.into_iter(),
		}
	}

	fn next_inout_elem(&mut self) -> CircuitElem<F, WitnessGenerator<'a, F>> {
		let value = self
			.events
			.next()
			.unwrap_or_else(|| panic!("replay exhausted: no more events"));

		let wire = self.inout_alloc.alloc();
		let witness_wire = self.witness_gen.borrow_mut().write_inout(wire, value);
		CircuitElem::wire(&self.witness_gen, witness_wire)
	}

	fn next_precommit_elem(&mut self) -> CircuitElem<F, WitnessGenerator<'a, F>> {
		let value = self
			.keys
			.next()
			.expect("precommit segment is sized incorrectly");

		let wire = self.precommit_alloc.alloc();
		let witness_wire = self.witness_gen.borrow_mut().write_precommit(wire, value);
		CircuitElem::wire(&self.witness_gen, witness_wire)
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
	type Elem = CircuitElem<F, WitnessGenerator<'a, F>>;

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
			// A compile-time constant is checked here; any other wire's value is checked by the
			// witness generator, which records a constraint violation as a build error.
			CircuitElem::Constant(c) => {
				if c == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			CircuitElem::Wire { builder, wire } => {
				assert!(Weak::ptr_eq(&Rc::downgrade(&self.witness_gen), &builder));
				self.witness_gen.borrow_mut().assert_zero(wire);
				Ok(())
			}
		}
	}

	fn compute_public_value(&mut self, inputs: &[Self::Elem], f: impl FieldFn<F>) -> Self::Elem {
		// The function's result enters as a single derived public wire (matching the symbolic
		// builder's `hint_varsize`), whose value the prover computes natively from the
		// public-derived inputs. See `IronSpartanBuilderChannel::compute_public_value`.
		let out_wire = {
			let mut witness_gen = self.witness_gen.borrow_mut();
			let input_wires: Vec<_> = inputs
				.iter()
				.map(|elem| elem.to_wire(&mut witness_gen))
				.collect();
			witness_gen.hint_varsize(&input_wires, 1, move |vals| vec![f.call::<F>(vals)])[0]
		};
		CircuitElem::wire(&self.witness_gen, out_wire)
	}
}

impl<'r, 'a, F: Field> IOPVerifierChannel<'r, F> for ReplayChannel<'a, F> {
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&[]
	}

	fn recv_oracle(
		&mut self,
		_log_msg_len: usize,
		_is_witness_dependent: bool,
	) -> Result<Self::Oracle, binius_iop::channel::Error> {
		Ok(())
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'r, Self::Oracle, Self::Elem>>,
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

#[cfg(test)]
mod tests {
	use std::{cell::RefCell, rc::Rc};

	use binius_field::{
		BinaryField1b as B1, BinaryField128bGhash as B128, ExtensionField, field::FieldOps,
	};
	use binius_spartan_frontend::circuit_builder::{ConstraintBuilder, WitnessGenerator};
	use binius_spartan_verifier::wrapper::circuit_elem::CircuitElem;

	type BuildElem = CircuitElem<B128, ConstraintBuilder<B128>>;
	type WitnessElem<'a> = CircuitElem<B128, WitnessGenerator<'a, B128>>;

	#[test]
	fn test_square_transpose_wires() {
		// Test that square_transpose on wire elements builds a valid constraint system,
		// and that a WitnessGenerator with correct values satisfies all constraints.
		type FSub = B1;
		let degree = <B128 as ExtensionField<FSub>>::DEGREE;

		// Phase 1: Build the constraint system symbolically.
		let mut constraint_builder = ConstraintBuilder::<B128>::new();
		let inout_wires: Vec<_> = (0..degree)
			.map(|_| constraint_builder.alloc_inout())
			.collect();

		// Build CircuitElem wires via a shared Rc.
		let rc = Rc::new(RefCell::new(constraint_builder));
		let mut elems: Vec<BuildElem> = inout_wires
			.iter()
			.map(|&w| BuildElem::wire(&rc, w))
			.collect();

		<BuildElem as FieldOps>::square_transpose::<FSub>(&mut elems);

		// The transposed outputs are wires; drop them so we can extract the builder.
		drop(elems);
		let constraint_builder = Rc::try_unwrap(rc).unwrap().into_inner();
		let (cs, layout) = constraint_builder.build().finalize();

		// The constraint system should have multiplication constraints from
		// Frobenius checks, reconstruction, and transposed output.
		assert!(!cs.mul_constraints().is_empty());

		// Phase 2: Generate a witness with concrete values and verify all constraints.
		let test_values: Vec<B128> = (0..degree)
			.map(<B128 as ExtensionField<FSub>>::basis)
			.collect();

		let mut witness_gen = WitnessGenerator::new(&layout);
		let witness_wires: Vec<_> = inout_wires
			.iter()
			.zip(&test_values)
			.map(|(&wire, &val)| witness_gen.write_inout(wire, val))
			.collect();

		let witness_rc = Rc::new(RefCell::new(witness_gen));
		let mut witness_elems: Vec<WitnessElem> = witness_wires
			.iter()
			.map(|&w| WitnessElem::wire(&witness_rc, w))
			.collect();

		<WitnessElem as FieldOps>::square_transpose::<FSub>(&mut witness_elems);

		drop(witness_elems);
		let witness_gen = Rc::try_unwrap(witness_rc).unwrap().into_inner();
		let witness = witness_gen
			.build()
			.expect("witness generation should succeed (all constraints satisfied)");

		cs.validate(&witness);
	}
}
