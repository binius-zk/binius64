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

impl Opcode {
	pub fn shape(self, dimensions: &[usize]) -> OpcodeShape {
		assert_eq!(self.is_const_shape(), dimensions.is_empty());

		match self {
			// Bitwise operations
			Opcode::Band => super::band::shape(),
			Opcode::Bxor => super::bxor::shape(),
			// TODO: Can we get rid of this gate? This is the only non-hint one with dimensions
			Opcode::BxorMulti => super::bxor_multi::shape(dimensions),
			Opcode::Bor => super::bor::shape(),
			Opcode::Fax => super::fax::shape(),

			// Selection
			Opcode::Select => super::select::shape(),

			// Arithmetic
			Opcode::IaddCinCout => super::iadd_cin_cout::shape(),
			Opcode::Iadd32 => super::iadd32::shape(),
			Opcode::Iadd32CinCout => super::iadd32_cin_cout::shape(),
			Opcode::IsubBinBout => super::isub_bin_bout::shape(),
			Opcode::Imul => super::imul::shape(),
			Opcode::Bmul => super::bmul::shape(),

			// Shifts
			Opcode::Shift => super::shift::shape(),

			// Comparisons
			Opcode::IcmpUlt => super::icmp_ult::shape(),
			Opcode::IcmpEq => super::icmp_eq::shape(),

			// Assertions (no outputs)
			Opcode::AssertEq => super::assert_eq::shape(),
			Opcode::AssertZero => super::assert_zero::shape(),
			Opcode::AssertNonZero => super::assert_non_zero::shape(),
			Opcode::AssertFalse => super::assert_false::shape(),
			Opcode::AssertTrue => super::assert_true::shape(),
			Opcode::AssertEqCond => super::assert_eq_cond::shape(),

			// Hints (no constraints)
			Opcode::Hint => {
				panic!("Opcode::Hint shape requires the HintRegistry; use GateData::shape instead")
			}
		}
	}

	pub const fn is_const_shape(self) -> bool {
		#[allow(clippy::match_like_matches_macro)]
		match self {
			Opcode::BxorMulti => false,
			Opcode::Hint => false,
			_ => true,
		}
	}
}
