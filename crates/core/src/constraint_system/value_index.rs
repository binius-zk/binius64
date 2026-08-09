// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

/// The section of the [`ValueVec`](super::ValueVec) a [`ValueIndex`] names.
///
/// The sections partition every word a circuit allocates. The first three hold the values a
/// [`ConstraintSystem`](super::ConstraintSystem) may reference; [`Self::Scratch`] holds the
/// uncommitted temporaries that only exist while a circuit is evaluated.
///
/// The discriminants are the two-bit tag [`ValueIndex`] packs, and their order is the order the
/// sections occupy in the value vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ValueSegment {
	/// The constants the circuit declares, known to both prover and verifier.
	Constant = 0,
	/// The input/output values, which are public but chosen per instance.
	InOut = 1,
	/// The values only the prover knows: the declared witness and the values the gates create.
	Private = 2,
	/// The uncommitted temporaries, live only while a circuit is evaluated.
	///
	/// These words are not committed and no constraint may reference them, so a
	/// [`ValueIndex`] in this segment is meaningful only in the circuit's wire mapping and its
	/// evaluation form. [`ConstraintSystem::validate`](super::ConstraintSystem::validate) rejects
	/// any operand term that names it.
	Scratch = 3,
}

impl ValueSegment {
	/// The four segments, in value-vector order.
	pub const ALL: [ValueSegment; 4] = [
		ValueSegment::Constant,
		ValueSegment::InOut,
		ValueSegment::Private,
		ValueSegment::Scratch,
	];

	/// Whether a [`ConstraintSystem`](super::ConstraintSystem) operand may reference this segment.
	pub const fn is_referenceable(self) -> bool {
		!matches!(self, ValueSegment::Scratch)
	}

	/// The segment a two-bit tag encodes.
	const fn from_tag(tag: u32) -> Self {
		match tag {
			0 => ValueSegment::Constant,
			1 => ValueSegment::InOut,
			2 => ValueSegment::Private,
			3 => ValueSegment::Scratch,
			_ => panic!("tag is masked to two bits"),
		}
	}
}

/// A type safe reference to one word of the [`ValueVec`](super::ValueVec), as a segment and an
/// index within it.
///
/// # Representation
///
/// The pair is packed into a single `u32`: the [`ValueSegment`] in the top two bits and the
/// index in the bottom [`Self::INDEX_BITS`]. Constraint systems hold millions of these, so the
/// packing keeps a [`ShiftedValueIndex`](super::ShiftedValueIndex) at eight bytes rather than
/// twelve.
///
/// The packing also makes the derived [`Ord`] order the words by segment and then by index, which
/// is the order they occupy in the value vector.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueIndex(u32);

impl ValueIndex {
	/// The number of bits the index occupies, the segment tag taking the remaining two.
	pub const INDEX_BITS: u32 = 30;

	/// The number of values one segment can hold.
	///
	/// This is one short of the index space, because [`Self::INVALID`] reserves the last index.
	pub const SEGMENT_CAPACITY: u32 = Self::INDEX_MASK;

	/// The bits of the packed word holding the index.
	const INDEX_MASK: u32 = (1 << Self::INDEX_BITS) - 1;

	/// The value index that is not considered to be valid.
	///
	/// This is the last index of the scratch segment, whose words no constraint may reference
	/// anyway. Reserving it keeps the all-ones word invalid, so a zeroed-then-defaulted map reads
	/// back as unassigned.
	pub const INVALID: ValueIndex = ValueIndex(u32::MAX);

	/// Creates an index naming the given word of the given segment.
	///
	/// # Panics
	///
	/// Panics if the index is not below [`Self::SEGMENT_CAPACITY`].
	pub const fn new(segment: ValueSegment, index: u32) -> Self {
		assert!(index < Self::SEGMENT_CAPACITY, "value index out of range");
		Self(((segment as u32) << Self::INDEX_BITS) | index)
	}

	/// Creates an index naming a constant.
	pub const fn constant(index: u32) -> Self {
		Self::new(ValueSegment::Constant, index)
	}

	/// Creates an index naming an inout value.
	pub const fn inout(index: u32) -> Self {
		Self::new(ValueSegment::InOut, index)
	}

	/// Creates an index naming a private value.
	pub const fn private(index: u32) -> Self {
		Self::new(ValueSegment::Private, index)
	}

	/// Creates an index naming a scratch word.
	pub const fn scratch(index: u32) -> Self {
		Self::new(ValueSegment::Scratch, index)
	}

	/// The segment this index names.
	pub const fn segment(self) -> ValueSegment {
		ValueSegment::from_tag(self.0 >> Self::INDEX_BITS)
	}

	/// The index within [`Self::segment`].
	pub const fn index(self) -> u32 {
		self.0 & Self::INDEX_MASK
	}

	/// The flat position of this word, given the word each segment starts at.
	///
	/// This is the one place the segment-relative form is resolved back to an absolute position.
	/// Its callers differ only in where they read the starts from: a
	/// [`ValueVecLayout`](super::ValueVecLayout) knows them directly, while a
	/// [`ConstraintSystem`](super::ConstraintSystem) derives them from its own section sizes.
	#[inline]
	pub const fn offset_within(self, segment_starts: [usize; 4]) -> usize {
		segment_starts[self.segment() as usize] + self.index() as usize
	}
}

/// The most sensible default for a value index is invalid.
impl Default for ValueIndex {
	fn default() -> Self {
		Self::INVALID
	}
}

/// Prints the segment and index rather than the packed word, which reads as a nonsense integer.
impl std::fmt::Debug for ValueIndex {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		if *self == Self::INVALID {
			return f.write_str("ValueIndex::INVALID");
		}
		write!(f, "ValueIndex({:?}, {})", self.segment(), self.index())
	}
}

impl SerializeBytes for ValueIndex {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for ValueIndex {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(ValueIndex(u32::deserialize(read_buf)?))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn round_trips_every_segment_through_the_packing() {
		for segment in ValueSegment::ALL {
			for index in [0, 1, 12345, ValueIndex::SEGMENT_CAPACITY - 1] {
				let value_index = ValueIndex::new(segment, index);
				assert_eq!(value_index.segment(), segment);
				assert_eq!(value_index.index(), index);
			}
		}
	}

	#[test]
	fn orders_words_by_segment_then_index() {
		// The value vector holds the segments in this order, so the packed order must match.
		let ascending = [
			ValueIndex::constant(0),
			ValueIndex::constant(1),
			ValueIndex::inout(0),
			ValueIndex::private(0),
			ValueIndex::private(7),
			ValueIndex::scratch(0),
		];
		assert!(ascending.is_sorted());
	}

	#[test]
	fn invalid_is_the_reserved_scratch_index() {
		assert_eq!(ValueIndex::default(), ValueIndex::INVALID);
		assert_eq!(ValueIndex::INVALID.segment(), ValueSegment::Scratch);
		assert_eq!(ValueIndex::INVALID.index(), ValueIndex::SEGMENT_CAPACITY);
	}

	#[test]
	#[should_panic(expected = "value index out of range")]
	fn rejects_the_index_invalid_reserves() {
		ValueIndex::scratch(ValueIndex::SEGMENT_CAPACITY);
	}

	#[test]
	fn only_scratch_is_unreferenceable() {
		for segment in ValueSegment::ALL {
			assert_eq!(segment.is_referenceable(), segment != ValueSegment::Scratch);
		}
	}

	#[test]
	fn test_value_index_serialization_round_trip() {
		let value_index = ValueIndex::private(12345);

		let mut buf = Vec::new();
		value_index.serialize(&mut buf).unwrap();

		let deserialized = ValueIndex::deserialize(&mut buf.as_slice()).unwrap();

		assert_eq!(value_index, deserialized);
	}
}
