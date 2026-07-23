// Copyright 2025 Irreducible Inc.
use std::array;

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

/// Number of operands of an [`AndConstraint`].
const AND_ARITY: usize = 3;
/// Number of operands of an [`ImulConstraint`].
const IMUL_ARITY: usize = 4;
/// Number of operands of a [`BmulConstraint`].
const BMUL_ARITY: usize = 6;

/// Serializes the operands of a constraint in storage order.
fn serialize_operands(
	operands: &[Operand],
	mut write_buf: impl BufMut,
) -> Result<(), SerializationError> {
	for operand in operands {
		operand.serialize(&mut write_buf)?;
	}
	Ok(())
}

/// Deserializes the operands of a constraint of the given arity in storage order.
fn deserialize_operands<const ARITY: usize>(
	mut read_buf: impl Buf,
) -> Result<[Operand; ARITY], SerializationError> {
	let mut operands = array::from_fn::<_, ARITY, _>(|_| Operand::new());
	for operand in &mut operands {
		*operand = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
	}
	Ok(operands)
}

/// AND constraint: `A & B = C`.
///
/// This constraint verifies that the bitwise AND of operands A and B equals operand C.
/// Each operand is computed as the XOR of multiple shifted values from the value vector.
///
/// The operands are stored in the order given by [`AndConstraint::OPERAND_NAMES`].
#[derive(Debug, Clone, Default)]
pub struct AndConstraint(pub [Operand; AND_ARITY]);

impl AndConstraint {
	/// Number of operands.
	pub const ARITY: usize = AND_ARITY;
	/// Names of the operands, in storage order.
	pub const OPERAND_NAMES: [&'static str; AND_ARITY] = ["a", "b", "c"];

	/// Creates a new AND constraint from XOR combinations of the given unshifted values.
	pub fn plain_abc(
		a: impl IntoIterator<Item = ValueIndex>,
		b: impl IntoIterator<Item = ValueIndex>,
		c: impl IntoIterator<Item = ValueIndex>,
	) -> AndConstraint {
		AndConstraint::abc(
			a.into_iter().map(ShiftedValueIndex::plain),
			b.into_iter().map(ShiftedValueIndex::plain),
			c.into_iter().map(ShiftedValueIndex::plain),
		)
	}

	/// Creates a new AND constraint from XOR combinations of the given shifted values.
	pub fn abc(
		a: impl IntoIterator<Item = ShiftedValueIndex>,
		b: impl IntoIterator<Item = ShiftedValueIndex>,
		c: impl IntoIterator<Item = ShiftedValueIndex>,
	) -> AndConstraint {
		AndConstraint([
			a.into_iter().collect(),
			b.into_iter().collect(),
			c.into_iter().collect(),
		])
	}

	/// Operand A.
	pub const fn a(&self) -> &Operand {
		&self.0[0]
	}

	/// Operand B.
	pub const fn b(&self) -> &Operand {
		&self.0[1]
	}

	/// Operand C.
	pub const fn c(&self) -> &Operand {
		&self.0[2]
	}
}

impl SerializeBytes for AndConstraint {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_operands(&self.0, write_buf)
	}
}

impl DeserializeBytes for AndConstraint {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(AndConstraint(deserialize_operands(read_buf)?))
	}
}

/// IMUL constraint: `A * B = (HI << 64) | LO`.
///
/// 64-bit unsigned integer multiplication producing 128-bit result split into high and low 64-bit
/// words.
///
/// The operands are stored in the order given by [`ImulConstraint::OPERAND_NAMES`].
#[derive(Debug, Clone, Default)]
pub struct ImulConstraint(pub [Operand; IMUL_ARITY]);

impl ImulConstraint {
	/// Number of operands.
	pub const ARITY: usize = IMUL_ARITY;
	/// Names of the operands, in storage order.
	pub const OPERAND_NAMES: [&'static str; IMUL_ARITY] = ["a", "b", "hi", "lo"];

	/// A operand.
	pub const fn a(&self) -> &Operand {
		&self.0[0]
	}

	/// B operand.
	pub const fn b(&self) -> &Operand {
		&self.0[1]
	}

	/// HI operand.
	///
	/// The high 64 bits of the result of the multiplication.
	pub const fn hi(&self) -> &Operand {
		&self.0[2]
	}

	/// LO operand.
	///
	/// The low 64 bits of the result of the multiplication.
	pub const fn lo(&self) -> &Operand {
		&self.0[3]
	}
}

impl SerializeBytes for ImulConstraint {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_operands(&self.0, write_buf)
	}
}

impl DeserializeBytes for ImulConstraint {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(ImulConstraint(deserialize_operands(read_buf)?))
	}
}

/// BMUL constraint: `A * B = C` in the GHASH field `GF(2^128)`.
///
/// Multiplication of two GHASH binary-field elements. Because a field element spans 128 bits while
/// a word holds only 64, each operand is carried by a pair of words: the `lo` word supplies the low
/// 64 coefficients (of `1, X, ..., X^63`) and the `hi` word the high 64 (of `X^64, ..., X^127`).
///
/// The operands are stored in the order given by [`BmulConstraint::OPERAND_NAMES`].
#[derive(Debug, Clone, Default)]
pub struct BmulConstraint(pub [Operand; BMUL_ARITY]);

impl BmulConstraint {
	/// Number of operands.
	pub const ARITY: usize = BMUL_ARITY;
	/// Names of the operands, in storage order.
	pub const OPERAND_NAMES: [&'static str; BMUL_ARITY] =
		["a_lo", "a_hi", "b_lo", "b_hi", "c_lo", "c_hi"];

	/// Low word of the A operand.
	pub const fn a_lo(&self) -> &Operand {
		&self.0[0]
	}

	/// High word of the A operand.
	pub const fn a_hi(&self) -> &Operand {
		&self.0[1]
	}

	/// Low word of the B operand.
	pub const fn b_lo(&self) -> &Operand {
		&self.0[2]
	}

	/// High word of the B operand.
	pub const fn b_hi(&self) -> &Operand {
		&self.0[3]
	}

	/// Low word of the C (product) operand.
	pub const fn c_lo(&self) -> &Operand {
		&self.0[4]
	}

	/// High word of the C (product) operand.
	pub const fn c_hi(&self) -> &Operand {
		&self.0[5]
	}
}

impl SerializeBytes for BmulConstraint {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_operands(&self.0, write_buf)
	}
}

impl DeserializeBytes for BmulConstraint {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(BmulConstraint(deserialize_operands(read_buf)?))
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
		assert_eq!(constraint.a().len(), deserialized.a().len());
		assert_eq!(constraint.b().len(), deserialized.b().len());
		assert_eq!(constraint.c().len(), deserialized.c().len());

		for (orig, deser) in constraint.a().iter().zip(deserialized.a().iter()) {
			assert_eq!(orig.value_index, deser.value_index);
			assert_eq!(orig.amount, deser.amount);
		}
	}

	#[test]
	fn test_imul_constraint_serialization_round_trip() {
		let constraint = ImulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex(0))],
			vec![ShiftedValueIndex::srl(ValueIndex(1), 32)],
			vec![ShiftedValueIndex::plain(ValueIndex(2))],
			vec![ShiftedValueIndex::plain(ValueIndex(3))],
		]);

		let mut buf = Vec::new();
		constraint.serialize(&mut buf).unwrap();

		let deserialized = ImulConstraint::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(constraint.a().len(), deserialized.a().len());
		assert_eq!(constraint.b().len(), deserialized.b().len());
		assert_eq!(constraint.hi().len(), deserialized.hi().len());
		assert_eq!(constraint.lo().len(), deserialized.lo().len());
	}

	#[test]
	fn test_bmul_constraint_serialization_round_trip() {
		let constraint = BmulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex(0))],
			vec![ShiftedValueIndex::srl(ValueIndex(1), 32)],
			vec![ShiftedValueIndex::plain(ValueIndex(2))],
			vec![ShiftedValueIndex::sll(ValueIndex(3), 5)],
			vec![ShiftedValueIndex::plain(ValueIndex(4))],
			vec![
				ShiftedValueIndex::sar(ValueIndex(5), 15),
				ShiftedValueIndex::plain(ValueIndex(6)),
			],
		]);

		let mut buf = Vec::new();
		constraint.serialize(&mut buf).unwrap();

		let deserialized = BmulConstraint::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(constraint.a_lo().len(), deserialized.a_lo().len());
		assert_eq!(constraint.a_hi().len(), deserialized.a_hi().len());
		assert_eq!(constraint.b_lo().len(), deserialized.b_lo().len());
		assert_eq!(constraint.b_hi().len(), deserialized.b_hi().len());
		assert_eq!(constraint.c_lo().len(), deserialized.c_lo().len());
		assert_eq!(constraint.c_hi().len(), deserialized.c_hi().len());
	}
}
