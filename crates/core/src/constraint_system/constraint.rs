// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::fmt;

use binius_field::Ghash128b as B128;
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::{ShiftedValueIndex, ValueIndex, ValueVec};
use crate::{error::ConstraintViolation, word::Word};

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

/// The kind of a constraint of a [`ConstraintSystem`](super::ConstraintSystem).
///
/// Each variant identifies one of the constraint types the system holds, and its [`fmt::Display`]
/// impl gives the lowercase name used in diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
	/// A [`ZeroConstraint`].
	Zero,
	/// An [`AndConstraint`].
	And,
	/// An [`ImulConstraint`].
	Imul,
	/// A [`BmulConstraint`].
	Bmul,
}

impl fmt::Display for ConstraintKind {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let name = match self {
			Self::Zero => "zero",
			Self::And => "and",
			Self::Imul => "imul",
			Self::Bmul => "bmul",
		};
		f.write_str(name)
	}
}

/// Zero constraint: `VAL = 0`.
///
/// This constraint verifies that a single operand vanishes, where the operand is the XOR of
/// multiple shifted values from the value vector. It expresses an arbitrary `F_2`-linear relation
/// among shifted words: a two-term operand constrains two shifted values to be equal, and a
/// three-term operand `[x, y, z]` forces `z = x ^ y`.
///
/// The operands are stored in the order given by [`ZeroConstraint::OPERAND_NAMES`].
#[derive(Debug, Clone, Default)]
pub struct ZeroConstraint(pub [Operand; ZeroConstraint::ARITY]);

impl ZeroConstraint {
	/// Number of operands.
	pub const ARITY: usize = 1;
	/// Kind of this constraint.
	pub const KIND: ConstraintKind = ConstraintKind::Zero;
	/// Names of the operands, in storage order.
	pub const OPERAND_NAMES: [&'static str; Self::ARITY] = ["val"];

	/// Creates a new Zero constraint from an XOR combination of the given unshifted values.
	pub fn plain(val: impl IntoIterator<Item = ValueIndex>) -> ZeroConstraint {
		ZeroConstraint::new(val.into_iter().map(ShiftedValueIndex::plain))
	}

	/// Creates a new Zero constraint from an XOR combination of the given shifted values.
	pub fn new(val: impl IntoIterator<Item = ShiftedValueIndex>) -> ZeroConstraint {
		ZeroConstraint([val.into_iter().collect()])
	}

	/// Operand VAL.
	pub const fn val(&self) -> &Operand {
		&self.0[0]
	}

	/// Checks that the given value vector makes the single operand vanish.
	///
	/// # Errors
	///
	/// Returns the word the operand evaluates to, when that word is nonzero.
	pub fn verify(&self, values: &ValueVec) -> Result<(), ConstraintViolation> {
		let Word(val) = values.eval_operand(self.val());

		if val != 0 {
			return Err(ConstraintViolation::Zero { val });
		}
		Ok(())
	}
}

impl AsRef<[Operand; ZeroConstraint::ARITY]> for ZeroConstraint {
	fn as_ref(&self) -> &[Operand; Self::ARITY] {
		&self.0
	}
}

impl SerializeBytes for ZeroConstraint {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for ZeroConstraint {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		<[Operand; Self::ARITY]>::deserialize(read_buf).map(Self)
	}
}

/// AND constraint: `A & B = C`.
///
/// This constraint verifies that the bitwise AND of operands A and B equals operand C.
/// Each operand is computed as the XOR of multiple shifted values from the value vector.
///
/// The operands are stored in the order given by [`AndConstraint::OPERAND_NAMES`].
#[derive(Debug, Clone, Default)]
pub struct AndConstraint(pub [Operand; AndConstraint::ARITY]);

impl AndConstraint {
	/// Number of operands.
	pub const ARITY: usize = 3;
	/// Kind of this constraint.
	pub const KIND: ConstraintKind = ConstraintKind::And;
	/// Names of the operands, in storage order.
	pub const OPERAND_NAMES: [&'static str; Self::ARITY] = ["a", "b", "c"];

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

	/// Checks that the conjunction of operands A and B equals operand C.
	///
	/// # Errors
	///
	/// Returns the three evaluated operands, together with the bits on which they disagree.
	pub fn verify(&self, values: &ValueVec) -> Result<(), ConstraintViolation> {
		let Word(a) = values.eval_operand(self.a());
		let Word(b) = values.eval_operand(self.b());
		let Word(c) = values.eval_operand(self.c());

		let residue = (a & b) ^ c;
		if residue != 0 {
			return Err(ConstraintViolation::And { a, b, c, residue });
		}
		Ok(())
	}
}

impl AsRef<[Operand; AndConstraint::ARITY]> for AndConstraint {
	fn as_ref(&self) -> &[Operand; Self::ARITY] {
		&self.0
	}
}

impl SerializeBytes for AndConstraint {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for AndConstraint {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		<[Operand; Self::ARITY]>::deserialize(read_buf).map(Self)
	}
}

/// IMUL constraint: `A * B = (HI << 64) | LO`.
///
/// 64-bit unsigned integer multiplication producing 128-bit result split into high and low 64-bit
/// words.
///
/// The operands are stored in the order given by [`ImulConstraint::OPERAND_NAMES`].
#[derive(Debug, Clone, Default)]
pub struct ImulConstraint(pub [Operand; ImulConstraint::ARITY]);

impl ImulConstraint {
	/// Number of operands.
	pub const ARITY: usize = 4;
	/// Kind of this constraint.
	pub const KIND: ConstraintKind = ConstraintKind::Imul;
	/// Names of the operands, in storage order.
	pub const OPERAND_NAMES: [&'static str; Self::ARITY] = ["a", "b", "lo", "hi"];

	/// A operand.
	pub const fn a(&self) -> &Operand {
		&self.0[0]
	}

	/// B operand.
	pub const fn b(&self) -> &Operand {
		&self.0[1]
	}

	/// LO operand.
	///
	/// The low 64 bits of the result of the multiplication.
	pub const fn lo(&self) -> &Operand {
		&self.0[2]
	}

	/// HI operand.
	///
	/// The high 64 bits of the result of the multiplication.
	pub const fn hi(&self) -> &Operand {
		&self.0[3]
	}

	/// Checks that the product of operands A and B equals the HI and LO word pair.
	///
	/// The product is formed over the 128-bit integers.
	/// Nothing is truncated before it is split into the two 64-bit halves.
	///
	/// # Errors
	///
	/// Returns the evaluated operands, alongside the halves the product actually has.
	pub fn verify(&self, values: &ValueVec) -> Result<(), ConstraintViolation> {
		let Word(a) = values.eval_operand(self.a());
		let Word(b) = values.eval_operand(self.b());
		let Word(lo) = values.eval_operand(self.lo());
		let Word(hi) = values.eval_operand(self.hi());

		let product = a as u128 * b as u128;
		let expected_lo = product as u64;
		let expected_hi = (product >> 64) as u64;

		if lo != expected_lo || hi != expected_hi {
			return Err(ConstraintViolation::Imul {
				a,
				b,
				lo,
				hi,
				expected_lo,
				expected_hi,
			});
		}
		Ok(())
	}
}

impl AsRef<[Operand; ImulConstraint::ARITY]> for ImulConstraint {
	fn as_ref(&self) -> &[Operand; Self::ARITY] {
		&self.0
	}
}

impl SerializeBytes for ImulConstraint {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for ImulConstraint {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		<[Operand; Self::ARITY]>::deserialize(read_buf).map(Self)
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
pub struct BmulConstraint(pub [Operand; BmulConstraint::ARITY]);

impl BmulConstraint {
	/// Number of operands.
	pub const ARITY: usize = 6;
	/// Kind of this constraint.
	pub const KIND: ConstraintKind = ConstraintKind::Bmul;
	/// Names of the operands, in storage order.
	pub const OPERAND_NAMES: [&'static str; Self::ARITY] =
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

	/// Checks that the product of operands A and B equals operand C in the GHASH field.
	///
	/// The multiply runs through the same field implementation the proving system uses.
	/// This check therefore cannot drift from the arithmetic it mirrors.
	///
	/// # Errors
	///
	/// Returns the three reassembled elements, alongside the product the factors actually have.
	pub fn verify(&self, values: &ValueVec) -> Result<(), ConstraintViolation> {
		let a = eval_element(values, self.a_lo(), self.a_hi());
		let b = eval_element(values, self.b_lo(), self.b_hi());
		let c = eval_element(values, self.c_lo(), self.c_hi());

		let expected = u128::from(B128::new(a) * B128::new(b));
		if c != expected {
			return Err(ConstraintViolation::Bmul { a, b, c, expected });
		}
		Ok(())
	}
}

/// Reassembles one GHASH field element from the pair of words carrying it.
///
/// A field element spans 128 coefficients while a word holds only 64, so it takes two words:
///
/// - bit `i` of the low word is the coefficient of `X^i`.
/// - bit `i` of the high word is the coefficient of `X^(64 + i)`.
fn eval_element(values: &ValueVec, lo: &Operand, hi: &Operand) -> u128 {
	let Word(lo) = values.eval_operand(lo);
	let Word(hi) = values.eval_operand(hi);
	lo as u128 | ((hi as u128) << 64)
}

impl AsRef<[Operand; BmulConstraint::ARITY]> for BmulConstraint {
	fn as_ref(&self) -> &[Operand; Self::ARITY] {
		&self.0
	}
}

impl SerializeBytes for BmulConstraint {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for BmulConstraint {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		<[Operand; Self::ARITY]>::deserialize(read_buf).map(Self)
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use super::*;
	use crate::constraint_system::ConstraintKind;

	#[test]
	fn test_zero_constraint_serialization_round_trip() {
		// One term per segment a constraint may name, so the round trip covers the segment tag as
		// well as the index.
		let constraint = ZeroConstraint::new([
			ShiftedValueIndex::sll(ValueIndex::constant(1), 5),
			ShiftedValueIndex::srl(ValueIndex::inout(2), 10),
			ShiftedValueIndex::plain(ValueIndex::private(3)),
		]);

		let mut buf = Vec::new();
		constraint.serialize(&mut buf).unwrap();

		let deserialized = ZeroConstraint::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(constraint.val().len(), deserialized.val().len());

		for (orig, deser) in constraint.val().iter().zip(deserialized.val().iter()) {
			assert_eq!(orig.value_index, deser.value_index);
			assert_eq!(orig.shift_seq, deser.shift_seq);
		}
	}

	#[test]
	fn test_and_constraint_serialization_round_trip() {
		let constraint = AndConstraint::abc(
			vec![ShiftedValueIndex::sll(ValueIndex::constant(1), 5)],
			vec![ShiftedValueIndex::srl(ValueIndex::constant(2), 10)],
			vec![
				ShiftedValueIndex::sar(ValueIndex::constant(3), 15),
				ShiftedValueIndex::plain(ValueIndex::constant(4)),
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
			assert_eq!(orig.shift_seq, deser.shift_seq);
		}
	}

	#[test]
	fn test_imul_constraint_serialization_round_trip() {
		let constraint = ImulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex::constant(0))],
			vec![ShiftedValueIndex::srl(ValueIndex::constant(1), 32)],
			vec![ShiftedValueIndex::plain(ValueIndex::constant(2))],
			vec![ShiftedValueIndex::plain(ValueIndex::constant(3))],
		]);

		let mut buf = Vec::new();
		constraint.serialize(&mut buf).unwrap();

		let deserialized = ImulConstraint::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(constraint.a().len(), deserialized.a().len());
		assert_eq!(constraint.b().len(), deserialized.b().len());
		assert_eq!(constraint.lo().len(), deserialized.lo().len());
		assert_eq!(constraint.hi().len(), deserialized.hi().len());
	}

	#[test]
	fn test_bmul_constraint_serialization_round_trip() {
		let constraint = BmulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex::constant(0))],
			vec![ShiftedValueIndex::srl(ValueIndex::constant(1), 32)],
			vec![ShiftedValueIndex::plain(ValueIndex::constant(2))],
			vec![ShiftedValueIndex::sll(ValueIndex::constant(3), 5)],
			vec![ShiftedValueIndex::plain(ValueIndex::constant(4))],
			vec![
				ShiftedValueIndex::sar(ValueIndex::constant(5), 15),
				ShiftedValueIndex::plain(ValueIndex::constant(6)),
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

	/// A GHASH-field triple satisfying `A * B = C`.
	///
	/// The product is hard-coded rather than computed.
	/// The BMUL cases below therefore do not lean on the same multiply they are checking.
	const A: u128 = 0x0123456789abcdef_fedcba9876543210;
	const B: u128 = 0x0f1e2d3c4b5a6978_8796a5b4c3d2e1f0;
	const A_TIMES_B: u128 = 0x7f2984f784967f5a_7b881bf2b700d768;

	/// Builds a value vector whose public segment opens the given words as constants.
	///
	/// Both segments are eight words long, which is the shortest shape the shape check accepts.
	/// The whole public segment is constants, so the inout section is empty and starts at its end:
	///
	///     [ w_0 w_1 ... 0 0 ][ 0 0 0 0 0 0 0 0 ]
	///       0 1          7    8 ...         15
	fn values(words: &[u64]) -> ValueVec {
		let mut public = [Word::ZERO; 8];
		for (slot, &word) in iter::zip(&mut public, words) {
			*slot = Word::from_u64(word);
		}
		ValueVec::new_from_data(public.len(), &public, &[Word::ZERO; 8])
	}

	/// An operand reading constant `index` unshifted.
	fn at(index: u32) -> Operand {
		vec![ShiftedValueIndex::plain(ValueIndex::constant(index))]
	}

	/// The two words carrying a GHASH element, low half first.
	fn split(x: u128) -> [u64; 2] {
		[x as u64, (x >> 64) as u64]
	}

	#[test]
	fn zero_constraint_accepts_operand_whose_terms_cancel() {
		// Two terms reading equal words XOR to zero.
		// The operand vanishes even though neither word is zero.
		let values = values(&[0xfeed_face, 0xfeed_face]);
		let constraint = ZeroConstraint::plain([ValueIndex::constant(0), ValueIndex::constant(1)]);

		assert!(constraint.verify(&values).is_ok());
	}

	#[test]
	fn zero_constraint_rejects_operand_that_survives() {
		let values = values(&[0xfeed_face, 0x0bad_cafe]);
		let constraint = ZeroConstraint::plain([ValueIndex::constant(0), ValueIndex::constant(1)]);

		match constraint.verify(&values).unwrap_err() {
			ConstraintViolation::Zero { val } => assert_eq!(val, 0xfeed_face ^ 0x0bad_cafe),
			other => panic!("wrong violation: {other:?}"),
		}
	}

	#[test]
	fn and_constraint_accepts_matching_conjunction() {
		let values = values(&[0b1100, 0b1010, 0b1000]);
		let constraint = AndConstraint::plain_abc(
			[ValueIndex::constant(0)],
			[ValueIndex::constant(1)],
			[ValueIndex::constant(2)],
		);

		assert!(constraint.verify(&values).is_ok());
	}

	#[test]
	fn and_constraint_rejects_mismatched_conjunction() {
		// C claims one bit too many: 0b1100 & 0b1010 is 0b1000, not 0b1001.
		let values = values(&[0b1100, 0b1010, 0b1001]);
		let constraint = AndConstraint::plain_abc(
			[ValueIndex::constant(0)],
			[ValueIndex::constant(1)],
			[ValueIndex::constant(2)],
		);

		match constraint.verify(&values).unwrap_err() {
			ConstraintViolation::And { a, b, c, residue } => {
				assert_eq!(a, 0b1100);
				assert_eq!(b, 0b1010);
				assert_eq!(c, 0b1001);
				assert_eq!(residue, 0b0001);
			}
			other => panic!("wrong violation: {other:?}"),
		}
	}

	#[test]
	fn imul_constraint_accepts_both_halves_of_the_product() {
		// Both factors exceed 2^32, so the product fills the high word as well as the low one.
		let a = 0x1234_5678_9abc_def0u64;
		let b = 0x0fed_cba9_8765_4321u64;
		let product = a as u128 * b as u128;
		let values = values(&[a, b, product as u64, (product >> 64) as u64]);
		let constraint = ImulConstraint([at(0), at(1), at(2), at(3)]);

		assert!(constraint.verify(&values).is_ok());
	}

	#[test]
	fn imul_constraint_rejects_a_dropped_high_word() {
		// A prover keeping only the low half is the failure a truncating check would miss.
		let a = 0x1234_5678_9abc_def0u64;
		let b = 0x0fed_cba9_8765_4321u64;
		let product = a as u128 * b as u128;
		let values = values(&[a, b, product as u64, 0]);
		let constraint = ImulConstraint([at(0), at(1), at(2), at(3)]);

		match constraint.verify(&values).unwrap_err() {
			ConstraintViolation::Imul {
				a: got_a,
				b: got_b,
				lo,
				hi,
				expected_lo,
				expected_hi,
			} => {
				assert_eq!(got_a, a);
				assert_eq!(got_b, b);
				assert_eq!(lo, product as u64);
				assert_eq!(hi, 0);
				assert_eq!(expected_lo, product as u64);
				assert_eq!(expected_hi, (product >> 64) as u64);
			}
			other => panic!("wrong violation: {other:?}"),
		}
	}

	#[test]
	fn bmul_constraint_accepts_the_field_product() {
		let [a_lo, a_hi] = split(A);
		let [b_lo, b_hi] = split(B);
		let [c_lo, c_hi] = split(A_TIMES_B);
		let values = values(&[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi]);
		let constraint = BmulConstraint([at(0), at(1), at(2), at(3), at(4), at(5)]);

		assert!(constraint.verify(&values).is_ok());
	}

	#[test]
	fn bmul_constraint_rejects_a_product_off_by_one_coefficient() {
		let [a_lo, a_hi] = split(A);
		let [b_lo, b_hi] = split(B);
		let [c_lo, c_hi] = split(A_TIMES_B ^ 1);
		let values = values(&[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi]);
		let constraint = BmulConstraint([at(0), at(1), at(2), at(3), at(4), at(5)]);

		match constraint.verify(&values).unwrap_err() {
			ConstraintViolation::Bmul { a, b, c, expected } => {
				assert_eq!(a, A);
				assert_eq!(b, B);
				assert_eq!(c, A_TIMES_B ^ 1);
				assert_eq!(expected, A_TIMES_B);
			}
			other => panic!("wrong violation: {other:?}"),
		}
	}

	#[test]
	fn violation_reports_the_kind_of_constraint_that_failed() {
		// The kind is derived from the violation rather than stored beside it.
		// The two therefore cannot disagree.
		assert_eq!(ConstraintViolation::Zero { val: 1 }.kind(), ConstraintKind::Zero);
		assert_eq!(
			ConstraintViolation::And {
				a: 1,
				b: 1,
				c: 0,
				residue: 1
			}
			.kind(),
			ConstraintKind::And
		);
		assert_eq!(
			ConstraintViolation::Imul {
				a: 1,
				b: 1,
				lo: 0,
				hi: 0,
				expected_lo: 1,
				expected_hi: 0
			}
			.kind(),
			ConstraintKind::Imul
		);
		assert_eq!(
			ConstraintViolation::Bmul {
				a: 1,
				b: 1,
				c: 0,
				expected: 1
			}
			.kind(),
			ConstraintKind::Bmul
		);
	}
}
