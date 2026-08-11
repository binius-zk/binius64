// Copyright 2026 The Binius Developers

//! [`IronSpartanBuilderChannel`]: an [`IPVerifierChannel`] that symbolically executes a verifier
//! and records the computation as constraints on a [`ConstraintBuilder`].

use std::{
	cell::RefCell,
	rc::{Rc, Weak},
};

use binius_core::word::Word;
use binius_field::{BinaryField1b as B1, ExtensionField, Field, util::FieldFn};
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_spartan_frontend::circuit_builder::{CircuitBuilder, ConstraintBuilder};

use super::{circuit_elem::CircuitElem, circuit_word::CircuitWord};

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

	/// Allocates the inner statement as this circuit's public input.
	///
	/// Pass the result to the verifier this channel drives. The words are wires, so what is
	/// recorded is a circuit about the inner system rather than about one statement — which it
	/// could not be anyway, since the build precedes every instance.
	pub fn statement(&mut self, n_inout: usize) -> Vec<CircuitWord<F, ConstraintBuilder<F>>>
	where
		F: ExtensionField<B1>,
	{
		(0..n_inout)
			.map(|_| {
				let wire = self.builder.borrow_mut().alloc_inout();
				CircuitWord::from_wire(&self.builder, wire)
			})
			.collect()
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

	fn compute_public_value(&mut self, inputs: &[Self::Elem], f: impl FieldFn<F>) -> Self::Elem {
		// The function is an arbitrary native computation the constraint system cannot replay, so
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
			builder.hint_varsize(&input_wires, 1, move |vals| vec![f.call_native(vals)])[0]
		};
		CircuitElem::wire(&self.builder, out_wire)
	}
}

impl<F: Field> WordIPVerifierChannel<F> for IronSpartanBuilderChannel<F> {
	type Word = CircuitWord<F, ConstraintBuilder<F>>;

	// The outer verifier rebinds the public inputs, so the wrapper records no Fiat-Shamir state.
	fn observe_words(&mut self, _words: &[Self::Word]) {}

	fn subset_sum(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		word.subset_sum(elems)
	}

	fn select(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		word.select(elems)
	}

	// A code proximity test's query indices are the only words that reach this, and that test runs
	// against the inner channel rather than symbolically here, so nothing reads what it returns.
	fn sample_bits(&mut self, _bits: usize) -> Self::Word {
		CircuitWord::Constant(Word::ZERO)
	}
}

impl<F: Field> IOPVerifierChannel<F> for IronSpartanBuilderChannel<F> {
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
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<Self::Oracle, Self::Elem>>,
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
