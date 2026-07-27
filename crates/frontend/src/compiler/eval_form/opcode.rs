// Copyright 2026 The Binius Developers
//! The opcode byte at the head of every evaluation instruction.

/// An evaluation instruction's opcode.
///
/// The emitter writes a variant and the executor decodes back to that same variant.
/// So dispatch matches on this enum rather than on a byte.
/// An opcode the emitter can write but the executor cannot run is then a compile error.
///
/// A byte is never reassigned, so the gaps between families are permanent.
/// That way an old bytecode dump keeps meaning one thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EvalOpcode {
	// Bitwise.
	Band = 0x01,
	Bor = 0x02,
	Bxor = 0x03,
	Select = 0x05,
	BxorMulti = 0x06,
	Fax = 0x07,

	// Shifts.
	Shift = 0x10,

	// Arithmetic.
	IaddCout = 0x20,
	IaddCinCout = 0x21,
	IsubBinBout = 0x23,
	Imul = 0x30,
	Bmul = 0x31,

	// 32-bit lanewise arithmetic.
	Iadd32CinCout = 0x40,
	Iadd32Cout = 0x46,

	// Masks.
	MaskLow = 0x50,
	MaskHigh = 0x51,

	// Assertions.
	AssertEq = 0x60,
	AssertEqCond = 0x61,
	AssertZero = 0x62,
	AssertNonZero = 0x63,
	AssertFalse = 0x64,
	AssertTrue = 0x65,

	// Hint calls.
	Hint = 0x80,
}

impl EvalOpcode {
	/// Decodes an opcode byte, or `None` when no opcode claims it.
	///
	/// Each arm repeats the byte its variant is declared with.
	/// `every_opcode_decodes_back_to_itself` below is what holds the two in agreement.
	pub const fn from_byte(byte: u8) -> Option<Self> {
		Some(match byte {
			0x01 => Self::Band,
			0x02 => Self::Bor,
			0x03 => Self::Bxor,
			0x05 => Self::Select,
			0x06 => Self::BxorMulti,
			0x07 => Self::Fax,
			0x10 => Self::Shift,
			0x20 => Self::IaddCout,
			0x21 => Self::IaddCinCout,
			0x23 => Self::IsubBinBout,
			0x30 => Self::Imul,
			0x31 => Self::Bmul,
			0x40 => Self::Iadd32CinCout,
			0x46 => Self::Iadd32Cout,
			0x50 => Self::MaskLow,
			0x51 => Self::MaskHigh,
			0x60 => Self::AssertEq,
			0x61 => Self::AssertEqCond,
			0x62 => Self::AssertZero,
			0x63 => Self::AssertNonZero,
			0x64 => Self::AssertFalse,
			0x65 => Self::AssertTrue,
			0x80 => Self::Hint,
			_ => return None,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::EvalOpcode;

	/// Every opcode, so the round-trip test can walk them.
	///
	/// A variant missing here is a variant the round trip never checks.
	const ALL: &[EvalOpcode] = &[
		EvalOpcode::Band,
		EvalOpcode::Bor,
		EvalOpcode::Bxor,
		EvalOpcode::Select,
		EvalOpcode::BxorMulti,
		EvalOpcode::Fax,
		EvalOpcode::Shift,
		EvalOpcode::IaddCout,
		EvalOpcode::IaddCinCout,
		EvalOpcode::IsubBinBout,
		EvalOpcode::Imul,
		EvalOpcode::Bmul,
		EvalOpcode::Iadd32CinCout,
		EvalOpcode::Iadd32Cout,
		EvalOpcode::MaskLow,
		EvalOpcode::MaskHigh,
		EvalOpcode::AssertEq,
		EvalOpcode::AssertEqCond,
		EvalOpcode::AssertZero,
		EvalOpcode::AssertNonZero,
		EvalOpcode::AssertFalse,
		EvalOpcode::AssertTrue,
		EvalOpcode::Hint,
	];

	#[test]
	fn every_opcode_decodes_back_to_itself() {
		// Invariant: the byte a variant is declared with is the byte that decodes to it.
		// This is what keeps the enum and `from_byte` from drifting apart.
		for &opcode in ALL {
			assert_eq!(
				EvalOpcode::from_byte(opcode as u8),
				Some(opcode),
				"{opcode:?} does not decode from its own discriminant"
			);
		}
	}

	#[test]
	fn no_byte_decodes_to_an_opcode_it_does_not_belong_to() {
		// Invariant: decoding is injective, so no two opcodes claim one byte.
		for byte in 0..=u8::MAX {
			let Some(opcode) = EvalOpcode::from_byte(byte) else {
				continue;
			};
			assert_eq!(opcode as u8, byte, "{opcode:?} decoded from the wrong byte");
		}
	}

	#[test]
	fn an_unassigned_byte_decodes_to_nothing() {
		// Fixture state: 0x04 sits in a gap the bitwise family left behind.
		assert_eq!(EvalOpcode::from_byte(0x04), None);
		// Mutation: 0xff was never assigned either.
		assert_eq!(EvalOpcode::from_byte(0xff), None);
	}
}
