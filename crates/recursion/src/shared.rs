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

/// The circuit under construction, plus the wires someone other than the gates has to fill.
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
	/// Wires the inner statement enters on, in observation order.
	statement: RefCell<Vec<Wire>>,
}

impl Shared {
	pub fn new() -> Self {
		Self {
			builder: CircuitBuilder::new(),
			inputs: RefCell::new(Vec::new()),
			statement: RefCell::new(Vec::new()),
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

	/// Allocates a wire the inner statement enters on.
	///
	/// The wire is inout rather than witness, so the statement is public.
	/// It is kept out of [`Self::inputs`] on purpose.
	/// A statement is given, not derived, so a replay has nothing to supply for it.
	pub fn statement_wire(&self) -> Wire {
		let wire = self.builder.add_inout();
		self.statement.borrow_mut().push(wire);
		wire
	}

	/// The wires the witness must supply, in allocation order.
	pub fn inputs(&self) -> Vec<Input> {
		self.inputs.borrow().clone()
	}

	/// The wires the inner statement enters on, in observation order.
	pub fn statement(&self) -> Vec<Wire> {
		self.statement.borrow().clone()
	}
}

impl Default for Shared {
	fn default() -> Self {
		Self::new()
	}
}
