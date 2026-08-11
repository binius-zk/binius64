// Copyright 2026 The Binius Developers

//! The builder and the record of what the witness has to supply.

use std::cell::RefCell;

use binius_frontend::{CircuitBuilder, Wire};

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
	inputs: RefCell<Vec<Wire>>,
}

impl Shared {
	pub fn new() -> Self {
		Self {
			builder: CircuitBuilder::new(),
			inputs: RefCell::new(Vec::new()),
		}
	}

	pub const fn builder(&self) -> &CircuitBuilder {
		&self.builder
	}

	/// Allocates a wire the witness must supply, and records it.
	pub fn input_wire(&self) -> Wire {
		let wire = self.builder.add_witness();
		self.inputs.borrow_mut().push(wire);
		wire
	}

	/// The wires the witness must supply, in allocation order.
	pub fn inputs(&self) -> Vec<Wire> {
		self.inputs.borrow().clone()
	}
}

impl Default for Shared {
	fn default() -> Self {
		Self::new()
	}
}
