// Copyright 2026 The Binius Developers

//! The builder and the record of what the witness has to supply.

use std::cell::RefCell;

use binius_frontend::{CircuitBuilder, Wire};

/// A wire the witness must supply, tagged with the operation that allocated it.
///
/// The tag is what makes a divergence between the build and the replay loud: the replay names the
/// operation it is filling for, and a mismatch is caught at that wire rather than showing up as a
/// wrong value several thousand wires later.
#[derive(Clone, Copy, Debug)]
pub struct Input {
	pub wire: Wire,
	pub kind: &'static str,
}

/// The circuit under construction, plus the wires whose values the witness must supply.
///
/// Shared between the channel and every [`SymbolicElem`](crate::SymbolicElem) and
/// [`SymbolicWord`](crate::SymbolicWord) derived from it, so an operation can allocate a wire
/// wherever it happens rather than only where the channel is in scope.
pub struct Shared {
	builder: CircuitBuilder,
	/// Wires the circuit cannot derive, in allocation order.
	///
	/// Two kinds end up here: words read off the proof, and values a missing gadget would
	/// otherwise have computed. Both are filled by replaying the verifier over the real
	/// transcript, which visits the same operations in the same order.
	inputs: RefCell<Vec<Input>>,
}

impl Shared {
	pub fn new() -> Self {
		Self::with_builder(CircuitBuilder::new())
	}

	/// Anchors on a builder the caller already holds.
	///
	/// A `CircuitBuilder` is a handle, so a clone emits into the same circuit.
	pub const fn with_builder(builder: CircuitBuilder) -> Self {
		Self {
			builder,
			inputs: RefCell::new(Vec::new()),
		}
	}

	pub const fn builder(&self) -> &CircuitBuilder {
		&self.builder
	}

	/// Allocates a wire the witness must supply, recording which operation asked for it.
	pub fn input_wire(&self, kind: &'static str) -> Wire {
		let wire = self.builder.add_witness();
		self.inputs.borrow_mut().push(Input { wire, kind });
		wire
	}

	/// The wires the witness must supply, in allocation order.
	pub fn inputs(&self) -> Vec<Input> {
		self.inputs.borrow().clone()
	}
}

impl Default for Shared {
	fn default() -> Self {
		Self::new()
	}
}
