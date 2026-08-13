// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Opcode {
	// Bitwise operations
	Band,
	Bxor,
	BxorMulti,
	Bor,
	Fax,

	// Selection
	Select,

	// Arithmetic
	IaddCinCout,
	Iadd32,
	Iadd32CinCout,
	IsubBinBout,
	Imul,
	Bmul,

	// Shifts
	Shift,

	// Comparisons
	IcmpUlt,
	IcmpEq,

	// Assertions
	AssertEq,
	AssertZero,
	AssertNonZero,
	AssertFalse,
	AssertTrue,
	AssertEqCond,

	/// Generic hint gate. The hint's [`HintId`](crate::ir::hints::HintId) is stored in
	/// `immediates[0]` and the user dimensions (passed to
	/// [`Hint::shape`](crate::ir::hints::Hint::shape) /
	/// [`Hint::execute`](crate::ir::hints::Hint::execute)) are `&dimensions`.
	Hint,
}

/// The shape of an opcode is a description of it's inputs and outputs. It allows treating a gate as
/// a black box, correctly identifying its inputs or outputs.
#[derive(Clone, Copy)]
pub struct OpcodeShape {
	/// The constants the gate with this opcode expects.
	pub const_in: &'static [Word],
	/// The number of inputs this opcode expects.
	///
	/// In case this opcode has a dynamic shape, it specifies the fixed number of inputs.
	pub n_in: usize,
	/// The number of outputs this opcode provides.
	///
	/// In case this opcode has a dynamic shape, it specifies the fixed number of outputs.
	pub n_out: usize,
	/// The number of wires of aux wires.
	///
	/// Aux wires are neither inputs nor outputs, but are still being used within constraint
	/// system.
	///
	/// In case this opcode has a dynamic shape, it specifies the fixed number of aux wires.
	pub n_aux: usize,
	/// The number of scratch wires.
	///
	/// Scratch wires are the wires that are neither inputs nor outputs. They also do not
	/// get referenced in the constraint system. Those are only needed for the witness evaluation.
	///
	/// In case this opcode has a dynamic shape, it specifies the fixed number of scratch wires.
	pub n_scratch: usize,
	/// The number of immediate operands.
	///
	/// Those are the fixed constant parameters for the opcode. Those include the constant shift
	/// amounts and things like that.
	///
	/// In case this opcode has a dynamic shape, it specifies the fixed number of immediates.
	pub n_imm: usize,
}

impl OpcodeShape {
	/// A shape reading `n_in` inputs and producing `n_out` outputs, carrying nothing else.
	pub const fn new(n_in: usize, n_out: usize) -> Self {
		Self {
			const_in: &[],
			n_in,
			n_out,
			n_aux: 0,
			n_scratch: 0,
			n_imm: 0,
		}
	}

	/// The same shape, reading the given constants ahead of its inputs.
	pub const fn with_consts(mut self, const_in: &'static [Word]) -> Self {
		self.const_in = const_in;
		self
	}

	/// The same shape, with wires the constraint system references but no caller passes.
	pub const fn with_aux(mut self, n_aux: usize) -> Self {
		self.n_aux = n_aux;
		self
	}

	/// The same shape, with wires only witness evaluation reads.
	pub const fn with_scratch(mut self, n_scratch: usize) -> Self {
		self.n_scratch = n_scratch;
		self
	}

	/// The same shape, with compile-time parameters carried alongside the wires.
	pub const fn with_imm(mut self, n_imm: usize) -> Self {
		self.n_imm = n_imm;
		self
	}
}
