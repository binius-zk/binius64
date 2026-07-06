// Copyright 2025 Irreducible Inc.
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::{ValueIndex, ValueVec};
use crate::word::Word;

/// A different variants of shifting a value.
///
/// Note that there is no shift left arithmetic because it is redundant.
///
/// The discriminant is stored in a single byte.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ShiftVariant {
	/// Shift logical left.
	Sll = 0,
	/// Shift logical right.
	Slr = 1,
	/// Shift arithmetic right.
	///
	/// This is similar to the logical shift right but instead of shifting in 0 bits it will
	/// replicate the sign bit.
	Sar = 2,
	/// Rotate right.
	///
	/// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
	Rotr = 3,
	/// Shift logical left on 32-bit halves.
	///
	/// Performs independent logical left shifts on the upper and lower 32-bit halves of the word.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	Sll32 = 4,
	/// Shift logical right on 32-bit halves.
	///
	/// Performs independent logical right shifts on the upper and lower 32-bit halves of the word.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	Srl32 = 5,
	/// Shift arithmetic right on 32-bit halves.
	///
	/// Performs independent arithmetic right shifts on the upper and lower 32-bit halves of the
	/// word. Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift
	/// amount (0-31).
	Sra32 = 6,
	/// Rotate right on 32-bit halves.
	///
	/// Performs independent rotate right operations on the upper and lower 32-bit halves of the
	/// word. Bits shifted off the right end wrap around to the left within each 32-bit half.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	Rotr32 = 7,
}

impl ShiftVariant {
	/// Whether this variant operates on the two 32-bit halves independently.
	///
	/// - The `*32` family shifts each half on its own.
	/// - It reads only the lower 5 bits of the amount.
	/// - Every other variant acts on the whole 64-bit word.
	#[inline]
	pub const fn is_half_word(self) -> bool {
		matches!(
			self,
			ShiftVariant::Sll32 | ShiftVariant::Srl32 | ShiftVariant::Sra32 | ShiftVariant::Rotr32
		)
	}

	/// The exclusive upper bound on a valid shift amount for this variant.
	///
	/// - Half-word (`*32`) variants read only the lower 5 bits, so amounts run `0..32`.
	/// - Full-width variants take amounts `0..64`.
	///
	/// Construction, validation, and deserialization all enforce this same bound.
	/// A value that passes any of them therefore denotes the same shift everywhere.
	#[inline]
	pub const fn max_amount(self) -> usize {
		if self.is_half_word() { 32 } else { 64 }
	}

	/// Applies this shift to a 64-bit word and returns the result.
	///
	/// The variant selects which word-level operation runs.
	/// Full-width variants act on the whole 64-bit word.
	/// The 32-bit variants act on the upper and lower halves independently.
	///
	/// # Arguments
	/// - The word to shift.
	/// - The shift amount in bits.
	#[inline]
	pub fn apply(self, word: Word, amount: usize) -> Word {
		// The word-level operators take the amount as a 32-bit count.
		let amount = amount as u32;
		// Dispatch to the matching operation:
		// - logical left / logical right shift in zeros.
		// - arithmetic right replicates the sign bit.
		// - rotate wraps bits around the word.
		// - the `*32` family applies the same op to each 32-bit half on its own.
		match self {
			ShiftVariant::Sll => word << amount,
			ShiftVariant::Slr => word >> amount,
			ShiftVariant::Sar => word.sar(amount),
			ShiftVariant::Rotr => word.rotr(amount),
			ShiftVariant::Sll32 => word.sll32(amount),
			ShiftVariant::Srl32 => word.srl32(amount),
			ShiftVariant::Sra32 => word.sra32(amount),
			ShiftVariant::Rotr32 => word.rotr32(amount),
		}
	}
}

impl SerializeBytes for ShiftVariant {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		(*self as u8).serialize(write_buf)
	}
}

impl DeserializeBytes for ShiftVariant {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let index = u8::deserialize(read_buf)?;
		match index {
			0 => Ok(ShiftVariant::Sll),
			1 => Ok(ShiftVariant::Slr),
			2 => Ok(ShiftVariant::Sar),
			3 => Ok(ShiftVariant::Rotr),
			4 => Ok(ShiftVariant::Sll32),
			5 => Ok(ShiftVariant::Srl32),
			6 => Ok(ShiftVariant::Sra32),
			7 => Ok(ShiftVariant::Rotr32),
			_ => Err(SerializationError::UnknownEnumVariant {
				name: "ShiftVariant",
				index,
			}),
		}
	}
}

/// Similar to [`ValueIndex`], but represents a value that has been shifted by a certain amount.
///
/// This is used in the operands to constraints like [`AndConstraint`](super::AndConstraint).
///
/// The canonical formto represent a value without any shifting is [`ShiftVariant::Sll`] with
/// amount equals 0.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShiftedValueIndex {
	/// The index of this value in the input values vector.
	pub value_index: ValueIndex,
	/// The flavour of the shift that the value must be shifted by.
	pub shift_variant: ShiftVariant,
	/// The number of bits to shift by.
	///
	/// Stored as a byte to keep the struct small: constraint systems hold millions of these.
	/// Must be less than 64.
	pub amount: u8,
}

impl ShiftedValueIndex {
	/// Create a value index that just uses the specified value. Equivalent to [`Self::sll`] with
	/// amount equals 0.
	pub const fn plain(value_index: ValueIndex) -> Self {
		Self {
			value_index,
			shift_variant: ShiftVariant::Sll,
			amount: 0,
		}
	}

	/// Shift Left Logical by the given number of bits.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn sll(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 64, "shift amount n={amount} out of range");
		Self {
			value_index,
			shift_variant: ShiftVariant::Sll,
			amount: amount as u8,
		}
	}

	/// Shift Right Logical by the given number of bits.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn srl(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 64, "shift amount n={amount} out of range");
		Self {
			value_index,
			shift_variant: ShiftVariant::Slr,
			amount: amount as u8,
		}
	}

	/// Shift Right Arithmetic by the given number of bits.
	///
	/// This is similar to the Shift Right Logical but instead of shifting in 0 bits it will
	/// replicate the sign bit.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn sar(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 64, "shift amount n={amount} out of range");
		Self {
			value_index,
			shift_variant: ShiftVariant::Sar,
			amount: amount as u8,
		}
	}

	/// Rotate Right by the given number of bits.
	///
	/// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn rotr(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 64, "shift amount n={amount} out of range");
		Self {
			value_index,
			shift_variant: ShiftVariant::Rotr,
			amount: amount as u8,
		}
	}

	/// Shift Left Logical on 32-bit halves by the given number of bits.
	///
	/// Performs independent logical left shifts on the upper and lower 32-bit halves.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn sll32(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 32, "shift amount n={amount} out of range for 32-bit shift");
		Self {
			value_index,
			shift_variant: ShiftVariant::Sll32,
			amount: amount as u8,
		}
	}

	/// Shift Right Logical on 32-bit halves by the given number of bits.
	///
	/// Performs independent logical right shifts on the upper and lower 32-bit halves.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn srl32(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 32, "shift amount n={amount} out of range for 32-bit shift");
		Self {
			value_index,
			shift_variant: ShiftVariant::Srl32,
			amount: amount as u8,
		}
	}

	/// Shift Right Arithmetic on 32-bit halves by the given number of bits.
	///
	/// Performs independent arithmetic right shifts on the upper and lower 32-bit halves.
	/// Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift amount
	/// (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn sra32(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 32, "shift amount n={amount} out of range for 32-bit shift");
		Self {
			value_index,
			shift_variant: ShiftVariant::Sra32,
			amount: amount as u8,
		}
	}

	/// Rotate Right on 32-bit halves by the given number of bits.
	///
	/// Performs independent rotate right operations on the upper and lower 32-bit halves.
	/// Bits shifted off the right end wrap around to the left within each 32-bit half.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn rotr32(value_index: ValueIndex, amount: usize) -> Self {
		assert!(amount < 32, "shift amount n={amount} out of range for 32-bit rotate");
		Self {
			value_index,
			shift_variant: ShiftVariant::Rotr32,
			amount: amount as u8,
		}
	}

	/// Evaluates this term against a witness.
	///
	/// A term names one value and a shift to apply to it.
	/// It contributes one shifted word to the XOR that forms an operand.
	#[inline]
	pub fn eval(&self, witness: &ValueVec) -> Word {
		// Look up the referenced word, then apply this term's shift.
		self.shift_variant
			.apply(witness[self.value_index], self.amount as usize)
	}
}

impl SerializeBytes for ShiftedValueIndex {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.value_index.serialize(&mut write_buf)?;
		self.shift_variant.serialize(&mut write_buf)?;
		// Keep the wire format a u32 so serialized systems stay byte-compatible.
		(self.amount as usize).serialize(write_buf)
	}
}

impl DeserializeBytes for ShiftedValueIndex {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let value_index = ValueIndex::deserialize(&mut read_buf)?;
		let shift_variant = ShiftVariant::deserialize(&mut read_buf)?;
		let amount = usize::deserialize(read_buf)?;

		// Reject any amount the variant cannot represent.
		// Half-word variants cap at 32, full-width at 64.
		// This mirrors the bound the constructors enforce.
		// A value below 64 always fits in the byte-sized field.
		if amount >= shift_variant.max_amount() {
			return Err(SerializationError::InvalidConstruction {
				name: "ShiftedValueIndex::amount",
			});
		}

		Ok(ShiftedValueIndex {
			value_index,
			shift_variant,
			amount: amount as u8,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_shift_variant_serialization_round_trip() {
		let variants = [
			ShiftVariant::Sll,
			ShiftVariant::Slr,
			ShiftVariant::Sar,
			ShiftVariant::Rotr,
		];

		for variant in variants {
			let mut buf = Vec::new();
			variant.serialize(&mut buf).unwrap();

			let deserialized = ShiftVariant::deserialize(&mut buf.as_slice()).unwrap();
			match (variant, deserialized) {
				(ShiftVariant::Sll, ShiftVariant::Sll)
				| (ShiftVariant::Slr, ShiftVariant::Slr)
				| (ShiftVariant::Sar, ShiftVariant::Sar)
				| (ShiftVariant::Rotr, ShiftVariant::Rotr) => {}
				_ => panic!("ShiftVariant round trip failed: {:?} != {:?}", variant, deserialized),
			}
		}
	}

	#[test]
	fn test_shift_variant_unknown_variant() {
		// Create invalid variant index
		let mut buf = Vec::new();
		255u8.serialize(&mut buf).unwrap();

		let result = ShiftVariant::deserialize(&mut buf.as_slice());
		assert!(result.is_err());
		match result.unwrap_err() {
			SerializationError::UnknownEnumVariant { name, index } => {
				assert_eq!(name, "ShiftVariant");
				assert_eq!(index, 255);
			}
			_ => panic!("Expected UnknownEnumVariant error"),
		}
	}

	#[test]
	fn test_shifted_value_index_serialization_round_trip() {
		let shifted_value_index = ShiftedValueIndex::srl(ValueIndex(42), 23);

		let mut buf = Vec::new();
		shifted_value_index.serialize(&mut buf).unwrap();

		let deserialized = ShiftedValueIndex::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(shifted_value_index.value_index, deserialized.value_index);
		assert_eq!(shifted_value_index.amount, deserialized.amount);
		match (shifted_value_index.shift_variant, deserialized.shift_variant) {
			(ShiftVariant::Slr, ShiftVariant::Slr) => {}
			_ => panic!("ShiftVariant mismatch"),
		}
	}

	#[test]
	fn test_shifted_value_index_invalid_amount() {
		// Create a buffer with invalid shift amount (>= 64)
		let mut buf = Vec::new();
		ValueIndex(0).serialize(&mut buf).unwrap();
		ShiftVariant::Sll.serialize(&mut buf).unwrap();
		64usize.serialize(&mut buf).unwrap(); // Invalid amount

		let result = ShiftedValueIndex::deserialize(&mut buf.as_slice());
		assert!(result.is_err());
		match result.unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "ShiftedValueIndex::amount");
			}
			_ => panic!("Expected InvalidConstruction error"),
		}
	}

	#[test]
	fn test_max_amount_and_is_half_word() {
		// Full-width variants take amounts up to 63.
		for variant in [
			ShiftVariant::Sll,
			ShiftVariant::Slr,
			ShiftVariant::Sar,
			ShiftVariant::Rotr,
		] {
			assert!(!variant.is_half_word());
			assert_eq!(variant.max_amount(), 64);
		}
		// Half-word variants take amounts up to 31.
		for variant in [
			ShiftVariant::Sll32,
			ShiftVariant::Srl32,
			ShiftVariant::Sra32,
			ShiftVariant::Rotr32,
		] {
			assert!(variant.is_half_word());
			assert_eq!(variant.max_amount(), 32);
		}
	}

	// Deserializes a raw (variant, amount) buffer, bypassing the constructors.
	// This lets out-of-range half-word amounts reach the deserialization path.
	fn deserialize_amount(
		shift_variant: ShiftVariant,
		amount: usize,
	) -> Result<ShiftedValueIndex, SerializationError> {
		let mut buf = Vec::new();
		ValueIndex(0).serialize(&mut buf).unwrap();
		shift_variant.serialize(&mut buf).unwrap();
		amount.serialize(&mut buf).unwrap();
		ShiftedValueIndex::deserialize(&mut buf.as_slice())
	}

	#[test]
	fn test_deserialize_rejects_half_word_amount_at_or_above_32() {
		// 31 is the largest amount a half-word variant can carry.
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll32, 31).unwrap(),
			ShiftedValueIndex {
				value_index: ValueIndex(0),
				shift_variant: ShiftVariant::Sll32,
				amount: 31,
			}
		);
		// 32 exceeds the 5-bit range and must be rejected.
		match deserialize_amount(ShiftVariant::Sll32, 32).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "ShiftedValueIndex::amount");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
		// A full-width variant still accepts 32 and up to 63.
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll, 32).unwrap(),
			ShiftedValueIndex {
				value_index: ValueIndex(0),
				shift_variant: ShiftVariant::Sll,
				amount: 32,
			}
		);
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll, 63).unwrap(),
			ShiftedValueIndex {
				value_index: ValueIndex(0),
				shift_variant: ShiftVariant::Sll,
				amount: 63,
			}
		);
	}

	#[test]
	fn shifted_value_index_fits_in_a_word() {
		// Layout: value_index (u32, 4 bytes) + shift_variant (1 byte) + amount (u8, 1 byte).
		// Padded to the u32 alignment, that is 8 bytes.
		// Holding this at one word matters: systems carry millions of these on the prover hot path.
		assert_eq!(size_of::<ShiftedValueIndex>(), 8);
	}
}
