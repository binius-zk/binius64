// Copyright 2025 Irreducible Inc.
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::{ShiftedValueIndex, ValueIndex};

/// Operand type.
///
/// An operand in Binius64 is a vector of shifted values. Each item in the vector represents a
/// term in a XOR combination of shifted values.
///
/// To give a couple examples:
///
/// ```ignore
/// vec![] == 0
/// vec![1] == 1
/// vec![1, 1] == 1 ^ 1
/// vec![x >> 5, y << 5] = (x >> 5) ^ (y << 5)
/// ```
pub type Operand = Vec<ShiftedValueIndex>;

/// AND constraint: `A & B = C`.
///
/// This constraint verifies that the bitwise AND of operands A and B equals operand C.
/// Each operand is computed as the XOR of multiple shifted values from the value vector.
#[derive(Debug, Clone, Default)]
pub struct AndConstraint {
	/// Operand A.
	pub a: Operand,
	/// Operand B.
	pub b: Operand,
	/// Operand C.
	pub c: Operand,
}

impl AndConstraint {
	/// Creates a new AND constraint from XOR combinations of the given unshifted values.
	pub fn plain_abc(
		a: impl IntoIterator<Item = ValueIndex>,
		b: impl IntoIterator<Item = ValueIndex>,
		c: impl IntoIterator<Item = ValueIndex>,
	) -> AndConstraint {
		AndConstraint {
			a: a.into_iter().map(ShiftedValueIndex::plain).collect(),
			b: b.into_iter().map(ShiftedValueIndex::plain).collect(),
			c: c.into_iter().map(ShiftedValueIndex::plain).collect(),
		}
	}

	/// Creates a new AND constraint from XOR combinations of the given shifted values.
	pub fn abc(
		a: impl IntoIterator<Item = ShiftedValueIndex>,
		b: impl IntoIterator<Item = ShiftedValueIndex>,
		c: impl IntoIterator<Item = ShiftedValueIndex>,
	) -> AndConstraint {
		AndConstraint {
			a: a.into_iter().collect(),
			b: b.into_iter().collect(),
			c: c.into_iter().collect(),
		}
	}
}

impl SerializeBytes for AndConstraint {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.a.serialize(&mut write_buf)?;
		self.b.serialize(&mut write_buf)?;
		self.c.serialize(write_buf)
	}
}

impl DeserializeBytes for AndConstraint {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let a = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let b = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let c = Vec::<ShiftedValueIndex>::deserialize(read_buf)?;

		Ok(AndConstraint { a, b, c })
	}
}

/// IMUL constraint: `A * B = (HI << 64) | LO`.
///
/// 64-bit unsigned integer multiplication producing 128-bit result split into high and low 64-bit
/// words.
#[derive(Debug, Clone, Default)]
pub struct ImulConstraint {
	/// A operand.
	pub a: Operand,
	/// B operand.
	pub b: Operand,
	/// HI operand.
	///
	/// The high 64 bits of the result of the multiplication.
	pub hi: Operand,
	/// LO operand.
	///
	/// The low 64 bits of the result of the multiplication.
	pub lo: Operand,
}

impl SerializeBytes for ImulConstraint {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.a.serialize(&mut write_buf)?;
		self.b.serialize(&mut write_buf)?;
		self.hi.serialize(&mut write_buf)?;
		self.lo.serialize(write_buf)
	}
}

impl DeserializeBytes for ImulConstraint {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let a = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let b = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let hi = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let lo = Vec::<ShiftedValueIndex>::deserialize(read_buf)?;

		Ok(ImulConstraint { a, b, hi, lo })
	}
}

/// BMUL constraint: `A * B = C` in the GHASH field `GF(2^128)`.
///
/// Multiplication of two GHASH binary-field elements. Because a field element spans 128 bits while
/// a word holds only 64, each operand is carried by a pair of words: the `lo` word supplies the low
/// 64 coefficients (of `1, X, ..., X^63`) and the `hi` word the high 64 (of `X^64, ..., X^127`).
#[derive(Debug, Clone, Default)]
pub struct BmulConstraint {
	/// Low word of the A operand.
	pub a_lo: Operand,
	/// High word of the A operand.
	pub a_hi: Operand,
	/// Low word of the B operand.
	pub b_lo: Operand,
	/// High word of the B operand.
	pub b_hi: Operand,
	/// Low word of the C (product) operand.
	pub c_lo: Operand,
	/// High word of the C (product) operand.
	pub c_hi: Operand,
}

impl SerializeBytes for BmulConstraint {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.a_lo.serialize(&mut write_buf)?;
		self.a_hi.serialize(&mut write_buf)?;
		self.b_lo.serialize(&mut write_buf)?;
		self.b_hi.serialize(&mut write_buf)?;
		self.c_lo.serialize(&mut write_buf)?;
		self.c_hi.serialize(write_buf)
	}
}

impl DeserializeBytes for BmulConstraint {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let a_lo = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let a_hi = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let b_lo = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let b_hi = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let c_lo = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
		let c_hi = Vec::<ShiftedValueIndex>::deserialize(read_buf)?;

		Ok(BmulConstraint {
			a_lo,
			a_hi,
			b_lo,
			b_hi,
			c_lo,
			c_hi,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_and_constraint_serialization_round_trip() {
		let constraint = AndConstraint::abc(
			vec![ShiftedValueIndex::sll(ValueIndex(1), 5)],
			vec![ShiftedValueIndex::srl(ValueIndex(2), 10)],
			vec![
				ShiftedValueIndex::sar(ValueIndex(3), 15),
				ShiftedValueIndex::plain(ValueIndex(4)),
			],
		);

		let mut buf = Vec::new();
		constraint.serialize(&mut buf).unwrap();

		let deserialized = AndConstraint::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(constraint.a.len(), deserialized.a.len());
		assert_eq!(constraint.b.len(), deserialized.b.len());
		assert_eq!(constraint.c.len(), deserialized.c.len());

		for (orig, deser) in constraint.a.iter().zip(deserialized.a.iter()) {
			assert_eq!(orig.value_index, deser.value_index);
			assert_eq!(orig.amount, deser.amount);
		}
	}

	#[test]
	fn test_imul_constraint_serialization_round_trip() {
		let constraint = ImulConstraint {
			a: vec![ShiftedValueIndex::plain(ValueIndex(0))],
			b: vec![ShiftedValueIndex::srl(ValueIndex(1), 32)],
			hi: vec![ShiftedValueIndex::plain(ValueIndex(2))],
			lo: vec![ShiftedValueIndex::plain(ValueIndex(3))],
		};

		let mut buf = Vec::new();
		constraint.serialize(&mut buf).unwrap();

		let deserialized = ImulConstraint::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(constraint.a.len(), deserialized.a.len());
		assert_eq!(constraint.b.len(), deserialized.b.len());
		assert_eq!(constraint.hi.len(), deserialized.hi.len());
		assert_eq!(constraint.lo.len(), deserialized.lo.len());
	}

	#[test]
	fn test_bmul_constraint_serialization_round_trip() {
		let constraint = BmulConstraint {
			a_lo: vec![ShiftedValueIndex::plain(ValueIndex(0))],
			a_hi: vec![ShiftedValueIndex::srl(ValueIndex(1), 32)],
			b_lo: vec![ShiftedValueIndex::plain(ValueIndex(2))],
			b_hi: vec![ShiftedValueIndex::sll(ValueIndex(3), 5)],
			c_lo: vec![ShiftedValueIndex::plain(ValueIndex(4))],
			c_hi: vec![
				ShiftedValueIndex::sar(ValueIndex(5), 15),
				ShiftedValueIndex::plain(ValueIndex(6)),
			],
		};

		let mut buf = Vec::new();
		constraint.serialize(&mut buf).unwrap();

		let deserialized = BmulConstraint::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(constraint.a_lo.len(), deserialized.a_lo.len());
		assert_eq!(constraint.a_hi.len(), deserialized.a_hi.len());
		assert_eq!(constraint.b_lo.len(), deserialized.b_lo.len());
		assert_eq!(constraint.b_hi.len(), deserialized.b_hi.len());
		assert_eq!(constraint.c_lo.len(), deserialized.c_lo.len());
		assert_eq!(constraint.c_hi.len(), deserialized.c_hi.len());
	}
}
