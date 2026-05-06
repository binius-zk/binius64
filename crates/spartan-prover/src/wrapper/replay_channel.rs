// Copyright 2026 The Binius Developers

//! [`ReplayChannel`]: an [`IPVerifierChannel`] that replays recorded interaction values through a
//! [`WitnessGenerator`], filling both inout and private wires in the outer witness.

use std::{
	cell::{Cell, RefCell},
	marker::PhantomData,
	rc::{Rc, Weak},
	vec::IntoIter as VecIntoIter,
};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, WireAllocator, WitnessError, WitnessGenerator, WitnessWire},
	constraint_system::{WireKind, Witness, WitnessLayout},
};
use binius_spartan_verifier::wrapper::circuit_elem::{CircuitElem, CircuitWire};

/// [`CircuitWire`] backend over [`WitnessGenerator`] — used by [`ReplayChannel`] to evaluate
/// arithmetic concretely while filling private witness wires.
#[derive(Debug, Clone)]
pub enum WitnessGenWire<'a, F: Field> {
	Constant(F),
	InOut(Rc<Cell<LazyInOut<F>>>),
	Private(WitnessWire<F>, PhantomData<&'a ()>),
}

impl<'a, F: Field> WitnessGenWire<'a, F> {
	fn lazy_inout(value: F) -> Self {
		Self::InOut(Rc::new(Cell::new(LazyInOut::Unmaterialized(value))))
	}

	pub fn private(wire: WitnessWire<F>) -> Self {
		Self::Private(wire, PhantomData)
	}

	fn materialize(builder: &mut WitnessGeneratorWithAlloc<'a, F>, wire: &Self) -> WitnessWire<F> {
		match wire {
			Self::Constant(val) => builder.constant(*val),
			Self::InOut(lazy_inout) => match lazy_inout.get() {
				LazyInOut::Unmaterialized(value) => {
					let witness_wire = builder.next_inout(value);
					lazy_inout.set(LazyInOut::Materialized(witness_wire));
					witness_wire
				}
				LazyInOut::Materialized(wire) => wire,
			},
			Self::Private(wire, _) => *wire,
		}
	}
}

impl<'a, F: Field> CircuitWire<F> for WitnessGenWire<'a, F> {
	type Builder = WitnessGeneratorWithAlloc<'a, F>;

	fn combine<const IN: usize, const OUT: usize>(
		builder: &mut Self::Builder,
		wires: [&Self; IN],
		f_op: impl Fn([F; IN]) -> [F; OUT],
		builder_op: impl Fn(&mut Self::Builder, [WitnessWire<F>; IN]) -> [WitnessWire<F>; OUT],
	) -> [Self; OUT] {
		let inner_values = array_util::try_map(wires, |wire| match wire {
			Self::Constant(val) => Some(*val),
			Self::InOut(lazy_inout) => Some(lazy_inout.get().val()),
			Self::Private(..) => None,
		});

		if let Some(inner_values) = inner_values {
			let ret_values = f_op(inner_values);
			let all_constant = wires.iter().all(|wire| matches!(wire, Self::Constant(..)));
			if all_constant {
				ret_values.map(Self::Constant)
			} else {
				ret_values.map(Self::lazy_inout)
			}
		} else {
			let inner_wires = wires.map(|wire| Self::materialize(builder, wire));
			builder_op(builder, inner_wires).map(Self::private)
		}
	}

	fn combine_varlen(
		builder: &mut Self::Builder,
		wires: &[&Self],
		n_out: usize,
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		builder_op: impl FnOnce(&mut Self::Builder, &[WitnessWire<F>]) -> Vec<WitnessWire<F>>,
	) -> Vec<Self> {
		let inner_values = wires
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) => Some(*val),
				Self::InOut(lazy_inout) => Some(lazy_inout.get().val()),
				Self::Private(..) => None,
			})
			.collect::<Option<Vec<_>>>();

		if let Some(inner_values) = inner_values {
			let result = f_op(&inner_values);
			debug_assert_eq!(result.len(), n_out);
			let all_constant = wires.iter().all(|wire| matches!(wire, Self::Constant(..)));
			if all_constant {
				result.into_iter().map(Self::Constant).collect()
			} else {
				result.into_iter().map(Self::lazy_inout).collect()
			}
		} else {
			let inner_wires = wires
				.iter()
				.map(|wire| Self::materialize(builder, wire))
				.collect::<Vec<_>>();
			let result = builder_op(builder, &inner_wires);
			debug_assert_eq!(result.len(), n_out);
			result.into_iter().map(Self::private).collect()
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
	witness_gen: Rc<RefCell<WitnessGeneratorWithAlloc<'a, F>>>,
	keys: VecIntoIter<F>,
	events: VecIntoIter<F>,
}

impl<'a, F: Field> ReplayChannel<'a, F> {
	/// Creates a new replay channel.
	///
	/// TODO: Document args. Keys are the symmetric OTP keys for the received values.
	pub fn new(layout: &'a WitnessLayout<F>, keys: Vec<F>, events: Vec<F>) -> Self {
		Self {
			witness_gen: Rc::new(RefCell::new(WitnessGeneratorWithAlloc::new(layout))),
			keys: keys.into_iter(),
			events: events.into_iter(),
		}
	}

	fn next_inout_elem(&mut self) -> CircuitElem<F, WitnessGenWire<'a, F>> {
		let value = self
			.events
			.next()
			.unwrap_or_else(|| panic!("replay exhausted: no more events"));

		CircuitElem::wire(&self.witness_gen, WitnessGenWire::lazy_inout(value))
	}

	fn next_precommit_elem(&mut self) -> CircuitElem<F, WitnessGenWire<'a, F>> {
		let value = self
			.keys
			.next()
			.expect("precommit segment is sized incorrectly");

		let witness_wire = self.witness_gen.borrow_mut().next_precommit(value);
		CircuitElem::wire(&self.witness_gen, WitnessGenWire::private(witness_wire))
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
				wire: WitnessGenWire::InOut(lazy_inout),
				..
			} => {
				if lazy_inout.get().val() == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}

			CircuitElem::Wire {
				builder,
				wire: WitnessGenWire::Private(wire, _),
			} => {
				assert!(Weak::ptr_eq(&Rc::downgrade(&self.witness_gen), &builder));
				self.witness_gen.borrow_mut().assert_zero(wire);
				Ok(())
			}
		}
	}

	fn compute_public_value(
		&mut self,
		inputs: &[Self::Elem],
		f: impl FnOnce(&[F]) -> F,
	) -> Self::Elem {
		let input_refs = inputs.iter().collect::<Vec<_>>();
		let outs = CircuitElem::combine_varlen(
			&input_refs,
			1,
			move |inputs| vec![f(inputs)],
			|_, _| {
				// Self::Elem::combine_varlen will only call the builder_op closure if any inputs
				// are non-public.
				panic!("compute_public_value: input is not public")
			},
		);
		outs.into_iter()
			.next()
			.expect("combine_varlen returns Vec with len = n_out; n_out = 1")
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

#[derive(Debug, Clone, Copy)]

pub enum LazyInOut<F: Field> {
	Unmaterialized(F),
	Materialized(WitnessWire<F>),
}

impl<F: Field> LazyInOut<F> {
	pub fn val(self) -> F {
		match self {
			Self::Unmaterialized(value) => value,
			Self::Materialized(wire) => wire.val(),
		}
	}
}

/// A [`WitnessGenerator`] decorator that internalizes [`WireAllocator`]s for the InOut and
/// Precommit segments. Exposes `next_inout(value)` and `next_precommit(value)` for callers that
/// allocate-and-write in one step (e.g. [`ReplayChannel`] and the lazy InOut materialization in
/// [`WitnessGenWire`]).
///
/// All [`CircuitBuilder`] methods delegate to the inner [`WitnessGenerator`].
#[derive(Debug)]
pub struct WitnessGeneratorWithAlloc<'a, F: Field> {
	inner: WitnessGenerator<'a, F>,
	inout_alloc: WireAllocator,
	precommit_alloc: WireAllocator,
}

impl<'a, F: Field> WitnessGeneratorWithAlloc<'a, F> {
	pub fn new(layout: &'a WitnessLayout<F>) -> Self {
		Self {
			inner: WitnessGenerator::new(layout),
			inout_alloc: WireAllocator::new(WireKind::InOut),
			precommit_alloc: WireAllocator::new(WireKind::Precommit),
		}
	}

	/// Allocates the next InOut wire and writes `value` to it.
	pub fn next_inout(&mut self, value: F) -> WitnessWire<F> {
		let wire = self.inout_alloc.alloc();
		self.inner.write_inout(wire, value)
	}

	/// Allocates the next Precommit wire and writes `value` to it.
	pub fn next_precommit(&mut self, value: F) -> WitnessWire<F> {
		let wire = self.precommit_alloc.alloc();
		self.inner.write_precommit(wire, value)
	}

	pub fn build(self) -> Result<Witness<F>, WitnessError> {
		self.inner.build()
	}
}

impl<'a, F: Field> CircuitBuilder for WitnessGeneratorWithAlloc<'a, F> {
	type Wire = WitnessWire<F>;
	type Field = F;

	fn assert_zero(&mut self, wire: Self::Wire) {
		self.inner.assert_zero(wire)
	}

	fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
		self.inner.assert_eq(lhs, rhs)
	}

	fn constant(&mut self, val: F) -> Self::Wire {
		self.inner.constant(val)
	}

	fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		self.inner.add(lhs, rhs)
	}

	fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		self.inner.mul(lhs, rhs)
	}

	fn hint<H: Fn([F; IN]) -> [F; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		inputs: [Self::Wire; IN],
		f: H,
	) -> [Self::Wire; OUT] {
		self.inner.hint(inputs, f)
	}
}

#[cfg(test)]
mod tests {
	use std::{cell::RefCell, rc::Rc};

	use binius_field::{
		BinaryField1b as B1, BinaryField128bGhash as B128, ExtensionField, field::FieldOps,
	};
	use binius_spartan_frontend::circuit_builder::ConstraintBuilder;
	use binius_spartan_verifier::wrapper::{
		builder_channel::BuilderWire, circuit_elem::CircuitElem,
	};

	use super::{WitnessGenWire, WitnessGeneratorWithAlloc};

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

		let mut witness_gen = WitnessGeneratorWithAlloc::new(&layout);
		let witness_wires: Vec<_> = test_values
			.iter()
			.map(|&val| witness_gen.next_inout(val))
			.collect();

		let witness_rc = Rc::new(RefCell::new(witness_gen));
		let mut witness_elems: Vec<WitnessElem> = witness_wires
			.iter()
			.map(|&w| WitnessElem::wire(&witness_rc, WitnessGenWire::private(w)))
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
