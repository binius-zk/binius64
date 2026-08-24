// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

/// The constraint kind a key names.
/// Every constraint in a Binius64 system reduces to one of these four checks over 64-bit words.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operation {
	/// A single word equals zero.
	///
	/// ```text
	/// a = 0
	/// ```
	Zero,
	/// The bitwise AND of two words, XORed with a third, equals zero.
	///
	/// ```text
	/// a & b ^ c = 0
	/// ```
	BitwiseAnd,
	/// Two 64-bit words multiplied as integers equal a 128-bit product split across two words.
	///
	/// ```text
	/// a * b = (hi << 64) | lo
	/// ```
	IntegerMul,
	/// Two words multiplied in the GHASH field equal a third.
	///
	/// ```text
	/// a * b = c   (GHASH field multiplication)
	/// ```
	BinMul,
}

impl Operation {
	/// The operation's code in the builder's packed references: two bits, in the walk order the
	/// key collection is built in (`Zero, BitwiseAnd, IntegerMul, BinMul`).
	///
	/// Not the serialized wire value below, which is pinned to a different numbering.
	pub(super) const fn packed_code(self) -> u8 {
		match self {
			Self::Zero => 0,
			Self::BitwiseAnd => 1,
			Self::IntegerMul => 2,
			Self::BinMul => 3,
		}
	}

	/// Decodes [`Self::packed_code`].
	///
	/// Total because the caller masks its input to the two bits the packing wrote, so every
	/// reachable value names the operation that produced it.
	pub(super) const fn from_packed_code(code: u8) -> Self {
		match code {
			0 => Self::Zero,
			1 => Self::BitwiseAnd,
			2 => Self::IntegerMul,
			_ => Self::BinMul,
		}
	}
}

impl SerializeBytes for Operation {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		// Wire values do not follow declaration order.
		// This pins the format independently of any future reordering of the variants.
		let val = match self {
			Operation::BitwiseAnd => 0u8,
			Operation::IntegerMul => 1u8,
			Operation::BinMul => 2u8,
			Operation::Zero => 3u8,
		};
		val.serialize(write_buf)
	}
}

impl DeserializeBytes for Operation {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let val = u8::deserialize(&mut read_buf)?;
		match val {
			0 => Ok(Operation::BitwiseAnd),
			1 => Ok(Operation::IntegerMul),
			2 => Ok(Operation::BinMul),
			3 => Ok(Operation::Zero),
			_ => Err(SerializationError::UnknownEnumVariant {
				name: "Operation",
				index: val,
			}),
		}
	}
}
