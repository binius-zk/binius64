// Copyright 2026 The Binius Developers

//! [`IronSpartanBuilderChannel`]: an [`IPVerifierChannel`] that symbolically executes a verifier
//! and records the computation as constraints on a [`ConstraintBuilder`].

use std::{
	cell::RefCell,
	rc::{Rc, Weak},
};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::circuit_builder::{CircuitBuilder, ConstraintBuilder};

use super::circuit_elem::CircuitElem;

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations as constraints
/// in a [`ConstraintBuilder`]. The typical usage pattern is:
///
/// 1. Construct a fresh [`IronSpartanBuilderChannel`] via [`Self::new`]
/// 2. Run the verifier on the channel (e.g., `verify_iop`)
/// 3. The channel's `finish()` method returns the [`ConstraintBuilder`] with all recorded
///    constraints
pub struct IronSpartanBuilderChannel<F: Field> {
	builder: Rc<RefCell<ConstraintBuilder<F>>>,
}

impl<F: Field> Default for IronSpartanBuilderChannel<F> {
	fn default() -> Self {
		Self::new()
	}
}

impl<F: Field> IronSpartanBuilderChannel<F> {
	/// Creates a new builder channel backed by a fresh [`ConstraintBuilder`].
	pub fn new() -> Self {
		Self {
			builder: Rc::new(RefCell::new(ConstraintBuilder::new())),
		}
	}

	fn alloc_inout_elem(&self) -> CircuitElem<F, ConstraintBuilder<F>> {
		let wire = self.builder.borrow_mut().alloc_inout();
		CircuitElem::wire(&self.builder, wire)
	}

	fn alloc_precommit_elem(&self) -> CircuitElem<F, ConstraintBuilder<F>> {
		let wire = self.builder.borrow_mut().alloc_precommit();
		CircuitElem::wire(&self.builder, wire)
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
	type Elem = CircuitElem<F, ConstraintBuilder<F>>;

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
		f: impl FnOnce(&[F]) -> F,
	) -> Self::Elem {
		// The closure is an arbitrary native computation the constraint system cannot replay, so
		// its result enters as a single derived public wire (a one-output `hint_varsize`,
		// computed from the public inputs, emitting no constraints) rather than a sub-circuit's
		// worth of constraints. Symbolically we only allocate the wire; the value is filled
		// concretely by the instance/witness channels, which recompute it via their own
		// `hint_varsize`.
		let out_wire = {
			let mut builder = self.builder.borrow_mut();
			let input_wires: Vec<_> = inputs
				.iter()
				.map(|elem| elem.to_wire(&mut builder))
				.collect();
			builder.hint_varsize(&input_wires, 1, move |vals| vec![f(vals)])[0]
		};
		CircuitElem::wire(&self.builder, out_wire)
	}
}

impl<'r, F: Field> IOPVerifierChannel<'r, F> for IronSpartanBuilderChannel<F> {
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
