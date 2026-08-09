// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

/// The section of the [`ValueVec`](super::ValueVec) a [`WitnessIndex`] names.
///
/// The sections partition every word a circuit allocates. The first three hold the values a
/// [`ConstraintSystem`](super::ConstraintSystem) may reference; [`Self::Scratch`] holds the
/// uncommitted temporaries that only exist while a circuit is evaluated.
///
/// The discriminants are the two-bit tag [`WitnessIndex`] packs, and their order is the order the
/// sections occupy in the value vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum WitnessSegment {
	/// The constants the circuit declares, known to both prover and verifier.
	Constant = 0,
	/// The input/output values, which are public but chosen per instance.
	InOut = 1,
	/// The values only the prover knows: the declared witness and the values the gates create.
	Private = 2,
	/// The uncommitted temporaries, live only while a circuit is evaluated.
	///
	/// These words are not committed and no constraint may reference them, so a
	/// [`WitnessIndex`] in this segment is meaningful only in the circuit's wire mapping and its
	/// evaluation form. [`ConstraintSystem::validate`](super::ConstraintSystem::validate) rejects
	/// any operand term that names it.
	Scratch = 3,
}

impl WitnessSegment {
	/// The four segments, in value-vector order.
	pub const ALL: [WitnessSegment; 4] = [
		WitnessSegment::Constant,
		WitnessSegment::InOut,
		WitnessSegment::Private,
		WitnessSegment::Scratch,
	];

	/// Whether a [`ConstraintSystem`](super::ConstraintSystem) operand may reference this segment.
	pub const fn is_referenceable(self) -> bool {
		!matches!(self, WitnessSegment::Scratch)
	}

	/// The segment a two-bit tag encodes.
	const fn from_tag(tag: u32) -> Self {
		match tag {
			0 => WitnessSegment::Constant,
			1 => WitnessSegment::InOut,
			2 => WitnessSegment::Private,
			3 => WitnessSegment::Scratch,
			_ => panic!("tag is masked to two bits"),
		}
	}
}

/// A type safe reference to one word of the [`ValueVec`](super::ValueVec), as a segment and an
/// index within it.
///
/// # Representation
///
/// The pair is packed into a single `u32`: the [`WitnessSegment`] in the top two bits and the
/// index in the bottom [`Self::INDEX_BITS`]. Constraint systems hold millions of these, so the
/// packing keeps a [`ShiftedValueIndex`](super::ShiftedValueIndex) at eight bytes rather than
/// twelve.
///
/// The packing also makes the derived [`Ord`] order the words by segment and then by index, which
/// is the order they occupy in the value vector.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WitnessIndex(u32);

impl WitnessIndex {
	/// The number of bits the index occupies, the segment tag taking the remaining two.
	pub const INDEX_BITS: u32 = 30;

	/// The number of values one segment can hold.
	///
	/// This is one short of the index space, because [`Self::INVALID`] reserves the last index.
	pub const SEGMENT_CAPACITY: u32 = Self::INDEX_MASK;

	/// The bits of the packed word holding the index.
	const INDEX_MASK: u32 = (1 << Self::INDEX_BITS) - 1;

	/// The witness index that is not considered to be valid.
	///
	/// This is the last index of the scratch segment, whose words no constraint may reference
	/// anyway. Reserving it keeps the all-ones word invalid, so a zeroed-then-defaulted map reads
	/// back as unassigned.
	pub const INVALID: WitnessIndex = WitnessIndex(u32::MAX);

	/// Creates an index naming the given word of the given segment.
	///
	/// # Panics
	///
	/// Panics if the index is not below [`Self::SEGMENT_CAPACITY`].
	pub const fn new(segment: WitnessSegment, index: u32) -> Self {
		assert!(index < Self::SEGMENT_CAPACITY, "witness index out of range");
		Self(((segment as u32) << Self::INDEX_BITS) | index)
	}

	/// Creates an index naming a constant.
	pub const fn constant(index: u32) -> Self {
		Self::new(WitnessSegment::Constant, index)
	}

	/// Creates an index naming an inout value.
	pub const fn inout(index: u32) -> Self {
		Self::new(WitnessSegment::InOut, index)
	}

	/// Creates an index naming a private value.
	pub const fn private(index: u32) -> Self {
		Self::new(WitnessSegment::Private, index)
	}

	/// Creates an index naming a scratch word.
	pub const fn scratch(index: u32) -> Self {
		Self::new(WitnessSegment::Scratch, index)
	}

	/// The segment this index names.
	pub const fn segment(self) -> WitnessSegment {
		WitnessSegment::from_tag(self.0 >> Self::INDEX_BITS)
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

/// The most sensible default for a witness index is invalid.
impl Default for WitnessIndex {
	fn default() -> Self {
		Self::INVALID
	}
}

/// Prints the segment and index rather than the packed word, which reads as a nonsense integer.
impl std::fmt::Debug for WitnessIndex {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		if *self == Self::INVALID {
			return f.write_str("WitnessIndex::INVALID");
		}
		write!(f, "WitnessIndex({:?}, {})", self.segment(), self.index())
	}
}

impl SerializeBytes for WitnessIndex {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for WitnessIndex {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(WitnessIndex(u32::deserialize(read_buf)?))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn round_trips_every_segment_through_the_packing() {
		for segment in WitnessSegment::ALL {
			for index in [0, 1, 12345, WitnessIndex::SEGMENT_CAPACITY - 1] {
				let witness_index = WitnessIndex::new(segment, index);
				assert_eq!(witness_index.segment(), segment);
				assert_eq!(witness_index.index(), index);
			}
		}
	}

	#[test]
	fn orders_words_by_segment_then_index() {
		// The value vector holds the segments in this order, so the packed order must match.
		let ascending = [
			WitnessIndex::constant(0),
			WitnessIndex::constant(1),
			WitnessIndex::inout(0),
			WitnessIndex::private(0),
			WitnessIndex::private(7),
			WitnessIndex::scratch(0),
		];
		assert!(ascending.is_sorted());
	}

	#[test]
	fn invalid_is_the_reserved_scratch_index() {
		assert_eq!(WitnessIndex::default(), WitnessIndex::INVALID);
		assert_eq!(WitnessIndex::INVALID.segment(), WitnessSegment::Scratch);
		assert_eq!(WitnessIndex::INVALID.index(), WitnessIndex::SEGMENT_CAPACITY);
	}

	#[test]
	#[should_panic(expected = "witness index out of range")]
	fn rejects_the_index_invalid_reserves() {
		WitnessIndex::scratch(WitnessIndex::SEGMENT_CAPACITY);
	}

	#[test]
	fn only_scratch_is_unreferenceable() {
		for segment in WitnessSegment::ALL {
			assert_eq!(segment.is_referenceable(), segment != WitnessSegment::Scratch);
		}
	}

	#[test]
	fn test_witness_index_serialization_round_trip() {
		let witness_index = WitnessIndex::private(12345);

		let mut buf = Vec::new();
		witness_index.serialize(&mut buf).unwrap();

		let deserialized = WitnessIndex::deserialize(&mut buf.as_slice()).unwrap();

		assert_eq!(witness_index, deserialized);
	}
}
