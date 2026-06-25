// Copyright 2026 The Binius Developers

//! [`IronSpartanBuilderChannel`]: an [`IPVerifierChannel`] that symbolically executes a verifier
//! and records the computation on a [`CircuitBuilderWithAlloc`] backend.

use std::{
	cell::RefCell,
	rc::{Rc, Weak},
};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::circuit_builder::{
	CircuitBuilder, CircuitBuilderWithAlloc, ConstraintBuilder,
};

use super::circuit_elem::CircuitElem;

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations on a
/// [`CircuitBuilderWithAlloc`] backend `B`:
///
/// - `B = ConstraintBuilder` (the default) records only the constraint system.
/// - `B = GateRecordingConstraintBuilder` additionally records a replayable
///   [`GateSequence`](binius_spartan_frontend::gate::GateSequence) — the prover replays it to fill
///   the outer witness.
///
/// The typical usage pattern is:
///
/// 1. Construct an [`IronSpartanBuilderChannel`] via [`Self::new`] from a backing builder
/// 2. Run the verifier on the channel (e.g., `verify`)
/// 3. The channel's [`Self::finish`] method returns the backing builder `B` (compile it, or call
///    `into_parts()` on a `GateRecordingConstraintBuilder` to also take the gate sequence)
pub struct IronSpartanBuilderChannel<
	F: Field,
	B: CircuitBuilderWithAlloc + CircuitBuilder<Field = F> = ConstraintBuilder<F>,
> {
	builder: Rc<RefCell<B>>,
}

impl<F: Field, B: CircuitBuilderWithAlloc + CircuitBuilder<Field = F>>
	IronSpartanBuilderChannel<F, B>
{
	/// Creates a new builder channel backed by `builder` (e.g. `ConstraintBuilder::new()` or
	/// `GateRecordingConstraintBuilder::new()`).
	pub fn new(builder: B) -> Self {
		Self {
			builder: Rc::new(RefCell::new(builder)),
		}
	}

	fn alloc_inout_elem(&self) -> CircuitElem<F, B> {
		let wire = self.builder.borrow_mut().alloc_inout();
		CircuitElem::wire(&self.builder, wire)
	}

	fn alloc_precommit_elem(&self) -> CircuitElem<F, B> {
		let wire = self.builder.borrow_mut().alloc_precommit();
		CircuitElem::wire(&self.builder, wire)
	}

	/// Consumes the channel and returns the underlying builder `B`.
	///
	/// This must be called after all `CircuitElem` values derived from this channel have been
	/// dropped, as it requires sole ownership of the builder via `Rc::try_unwrap`.
	pub fn finish(self) -> B {
		Rc::try_unwrap(self.builder)
			.unwrap_or_else(|_| panic!("CircuitElem values should only hold Weak references"))
			.into_inner()
	}
}

impl<F: Field, B: CircuitBuilderWithAlloc + CircuitBuilder<Field = F>> IPVerifierChannel<F>
	for IronSpartanBuilderChannel<F, B>
{
	type Elem = CircuitElem<F, B>;

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
			// A compile-time constant is checked here; a non-zero one is an unsatisfiable
			// assertion.
			CircuitElem::Constant(c) => {
				if c == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			// Record the assertion as a constraint over the wire (whether public-derivable or
			// private). The outer verifier enforces it; with derived wires there is no need to
			// special-case public values out of the constraint system.
			CircuitElem::Wire { builder, wire } => {
				assert!(Weak::ptr_eq(&Rc::downgrade(&self.builder), &builder));
				self.builder.borrow_mut().assert_zero(wire);
				Ok(())
			}
		}
	}

	fn compute_public_value(
		&mut self,
		inputs: &[Self::Elem],
		f: impl Fn(&[F]) -> F + 'static,
	) -> Self::Elem {
		// The closure is an arbitrary native computation the constraint system cannot replay, so
		// its result enters as a single derived public wire (a one-output `hint_varsize`) rather
		// than a sub-circuit's worth of constraints. A recording backend captures `f` (keyed on
		// the input wires) so replay can recompute the value; a symbolic backend ignores it.
		let wire = {
			let mut builder = self.builder.borrow_mut();
			let input_wires = inputs
				.iter()
				.map(|elem| match elem {
					CircuitElem::Constant(val) => builder.constant(*val),
					CircuitElem::Wire { wire, .. } => *wire,
				})
				.collect::<Vec<_>>();
			let outputs = builder.hint_varsize(&input_wires, 1, move |vals| vec![f(vals)]);
			outputs[0]
		};
		CircuitElem::wire(&self.builder, wire)
	}
}

impl<'r, F: Field, B: CircuitBuilderWithAlloc + CircuitBuilder<Field = F>> IOPVerifierChannel<'r, F>
	for IronSpartanBuilderChannel<F, B>
{
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&[]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
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
			let decrypted_claim = self.alloc_inout_elem();
			self.assert_zero(relation.claim - decrypted_claim)?;
		}
		Ok(())
	}
}
