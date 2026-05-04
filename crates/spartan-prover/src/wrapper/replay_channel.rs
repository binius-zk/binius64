// Copyright 2026 The Binius Developers

//! [`ReplayChannel`]: an [`IPVerifierChannel`] that replays recorded interaction values through a
//! [`WitnessGenerator`], filling both inout and private wires in the outer witness.

use std::{
	cell::RefCell,
	marker::PhantomData,
	rc::{Rc, Weak},
	vec::IntoIter as VecIntoIter,
};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, WitnessError, WitnessGenerator, WitnessWire},
	constraint_system::{ConstraintWire, Witness, WitnessLayout},
};
use binius_spartan_verifier::wrapper::circuit_elem::{CircuitElem, CircuitWire};

/// [`CircuitWire`] backend over [`WitnessGenerator`] — used by [`ReplayChannel`] to evaluate
/// arithmetic concretely while filling private witness wires.
#[derive(Debug, Clone, Copy)]
pub enum WitnessGenWire<'a, F: Field> {
	Constant(F),
	Wire(WitnessWire<F>, PhantomData<&'a ()>),
}

impl<'a, F: Field> WitnessGenWire<'a, F> {
	pub fn wire(wire: WitnessWire<F>) -> Self {
		Self::Wire(wire, PhantomData)
	}
}

impl<'a, F: Field> CircuitWire<F> for WitnessGenWire<'a, F> {
	type Builder = WitnessGenerator<'a, F>;

	fn combine<const IN: usize, const OUT: usize>(
		builder: &mut Self::Builder,
		wires: [&Self; IN],
		f_op: impl Fn([F; IN]) -> [F; OUT],
		builder_op: impl Fn(&mut Self::Builder, [WitnessWire<F>; IN]) -> [WitnessWire<F>; OUT],
	) -> [Self; OUT] {
		let inner_constants = array_util::try_map(wires, |wire| {
			if let Self::Constant(val) = wire {
				Some(*val)
			} else {
				None
			}
		});

		if let Some(inner_constants) = inner_constants {
			f_op(inner_constants).map(Self::Constant)
		} else {
			let inner_wires = wires.map(|wire| match wire {
				Self::Constant(val) => builder.constant(*val),
				Self::Wire(wire, _) => *wire,
			});
			builder_op(builder, inner_wires).map(|val| Self::Wire(val, PhantomData))
		}
	}

	fn combine_varlen(
		builder: &mut Self::Builder,
		wires: &[&Self],
		n_out: usize,
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		builder_op: impl FnOnce(&mut Self::Builder, &[WitnessWire<F>]) -> Vec<WitnessWire<F>>,
	) -> Vec<Self> {
		let inner_constants = wires
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) => Some(*val),
				Self::Wire(_, _) => None,
			})
			.collect::<Option<Vec<_>>>();

		if let Some(inner_constants) = inner_constants {
			let result = f_op(&inner_constants);
			debug_assert_eq!(result.len(), n_out);
			result.into_iter().map(Self::Constant).collect()
		} else {
			let inner_wires = wires
				.iter()
				.map(|wire| match wire {
					Self::Constant(val) => builder.constant(*val),
					Self::Wire(wire, _) => *wire,
				})
				.collect::<Vec<_>>();
			let result = builder_op(builder, &inner_wires);
			debug_assert_eq!(result.len(), n_out);
			result
				.into_iter()
				.map(|val| Self::Wire(val, PhantomData))
				.collect()
		}
	}
}

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
		CircuitElem::wire(&self.witness_gen, WitnessGenWire::wire(witness_wire))
	}

	fn next_precommit_elem(&mut self) -> CircuitElem<F, WitnessGenWire<'a, F>> {
		let value = self
			.keys
			.next()
			.expect("precommit segment is sized incorrectly");

		let wire = ConstraintWire::precommit(self.next_precommit_id);
		self.next_precommit_id += 1;
		let witness_wire = self.witness_gen.borrow_mut().write_precommit(wire, value);
		CircuitElem::wire(&self.witness_gen, WitnessGenWire::wire(witness_wire))
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
			} => {
				if c == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			CircuitElem::Wire {
				builder,
				wire: WitnessGenWire::Wire(wire, _),
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

#[cfg(test)]
mod tests {
	use std::{cell::RefCell, rc::Rc};

	use binius_field::{
		BinaryField1b as B1, BinaryField128bGhash as B128, ExtensionField, field::FieldOps,
	};
	use binius_spartan_frontend::circuit_builder::{ConstraintBuilder, WitnessGenerator};
	use binius_spartan_verifier::wrapper::{
		builder_channel::BuilderWire, circuit_elem::CircuitElem,
	};

	use super::WitnessGenWire;

	type BuildElem = CircuitElem<B128, BuilderWire<B128>>;
	type WitnessElem<'a> = CircuitElem<B128, WitnessGenWire<'a, B128>>;

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
			.map(|&w| BuildElem::wire(&rc, BuilderWire::Private(w)))
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
			.map(|(&w, &val)| witness_gen.write_inout(w, val))
			.collect();

		let witness_rc = Rc::new(RefCell::new(witness_gen));
		let mut witness_elems: Vec<WitnessElem> = witness_wires
			.iter()
			.map(|&w| WitnessElem::wire(&witness_rc, WitnessGenWire::wire(w)))
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
