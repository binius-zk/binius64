// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::collections::BTreeSet;

use binius_core::{ShiftVariant, constraint_system::Shift};
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use binius_verifier::protocols::shift::LOG_SHIFT_COUNT;
use bytes::{Buf, BufMut};

/// A dense re-encoding of the shift sequences occurring in a key segment.
/// A key names a sequence of two shifts, each slot drawn from a fixed alphabet of 512 spellings.
///
/// - The sequence space is therefore 512^2 = 262,144 entries.
/// - A constraint system uses only a few dozen of them.
/// - The segment's own sequences are re-encoded as a contiguous range.
/// - A per-sequence table is sized by the sequences present, not the whole space.
/// - The space is too large for a lookup table, so a sequence is located by binary search.
#[derive(Debug, Clone, Default)]
pub struct DenseShiftEncoding {
	/// The shift sequence each dense index encodes, in ascending sequence order.
	///
	/// Invariant: sorted and distinct.
	/// This is what makes finding a sequence's index a binary search.
	/// A deserialized encoding is checked for both properties.
	shifts: Vec<[Shift; 2]>,
}

impl DenseShiftEncoding {
	/// Builds the encoding of the shift sequences in an iterator, neither sorted nor distinct.
	///
	/// # Panics
	///
	/// Panics if the sequences do not fit the `u16` a key addresses them with.
	/// That needs more than 65,536 distinct shift sequences in one segment.
	/// No real system reaches that many.
	pub(super) fn new(shifts: impl IntoIterator<Item = [Shift; 2]>) -> Self {
		// A sorted set dedupes and orders in one pass, over the sequences present.
		let shifts = shifts.into_iter().collect::<BTreeSet<_>>();
		assert!(
			shifts.len() <= u16::MAX as usize + 1,
			"a key segment uses {} distinct shift sequences, more than the u16 dense index addresses",
			shifts.len()
		);
		Self {
			shifts: shifts.into_iter().collect(),
		}
	}

	/// The number of distinct shift sequences the segment uses.
	pub const fn len(&self) -> usize {
		self.shifts.len()
	}

	/// Whether the segment uses no shifted values at all.
	pub const fn is_empty(&self) -> bool {
		self.shifts.is_empty()
	}

	/// The shift sequences the segment uses, in dense index order.
	pub fn iter(&self) -> impl Iterator<Item = [Shift; 2]> + '_ {
		self.shifts.iter().copied()
	}

	/// Where every sequence the segment uses sits in the space two shift slots span.
	///
	/// - The space is addressed outer-major: the outer slot's index sits above the inner slot's.
	/// - This matches the reduction's round order, which peels the outer shift first.
	/// - Distinct sequences land on distinct indices, so two segments' encodings can merge.
	/// - The indices do not come out ascending, since sequences are sorted by inner slot first.
	/// - Nothing needs them ascending.
	///
	/// ```text
	/// index = (outer_index << LOG_SHIFT_COUNT) | inner_index
	/// ```
	pub fn shift_indices(&self) -> impl Iterator<Item = usize> + '_ {
		self.shifts
			.iter()
			.map(|&[inner, outer]| outer.index() << LOG_SHIFT_COUNT | inner.index())
	}

	/// The dense index of one shift sequence, for lookup while the keys are built.
	/// The sequences are sorted, so this is a binary search over the ones the segment uses.
	/// A lookup table over the whole sequence space would instead be 262,144 entries wide.
	///
	/// # Panics
	///
	/// Panics if the sequence is not one this encoding covers.
	pub(super) fn dense_idx(&self, shift_seq: [Shift; 2]) -> u16 {
		let index = self
			.shifts
			.binary_search(&shift_seq)
			.expect("the encoding covers every shift sequence its segment's keys name");
		// `new` bounds the length, so every index it yields fits.
		index as u16
	}
}

impl SerializeBytes for DenseShiftEncoding {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		(self.shifts.len() as u32).serialize(&mut write_buf)?;
		for shift_seq in &self.shifts {
			for shift in shift_seq {
				shift.variant.serialize(&mut write_buf)?;
				shift.amount.serialize(&mut write_buf)?;
			}
		}
		Ok(())
	}
}

impl DeserializeBytes for DenseShiftEncoding {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let len = u32::deserialize(&mut read_buf)? as usize;
		let mut shifts = Vec::with_capacity(len);
		for _ in 0..len {
			let mut shift_seq = [Shift::IDENTITY; 2];
			for shift in &mut shift_seq {
				let variant = ShiftVariant::deserialize(&mut read_buf)?;
				let amount = u8::deserialize(&mut read_buf)?;
				*shift = Shift { variant, amount };
			}
			shifts.push(shift_seq);
		}

		// Half-word (*32) variants cap at 32, full-width ones at 64.
		// An amount past its variant's bound denotes no shift at all.
		let amounts_in_range = shifts
			.iter()
			.flatten()
			.all(|shift| (shift.amount as usize) < shift.variant.max_amount());
		// A dense index is only meaningful against a list of distinct sequences, which the strictly
		// ascending order this writes them in also gives.
		let strictly_ascending = shifts.windows(2).all(|window| window[0] < window[1]);
		if !amounts_in_range || !strictly_ascending {
			return Err(SerializationError::InvalidConstruction {
				name: "DenseShiftEncoding::shifts",
			});
		}
		// A key addresses a sequence with a `u16`, so a longer list could not be indexed.
		if len > u16::MAX as usize + 1 {
			return Err(SerializationError::InvalidConstruction {
				name: "DenseShiftEncoding::shifts",
			});
		}

		Ok(DenseShiftEncoding { shifts })
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;

	use super::*;

	/// A shift sequence carrying one shift, which the canonical form places in the inner slot.
	fn single(shift: Shift) -> [Shift; 2] {
		[shift, Shift::IDENTITY]
	}

	// Serializes an encoding built raw, bypassing `new`'s sorting and deduplication, so that a
	// malformed list reaches the deserializer.
	fn deserialize_raw(shifts: Vec<[Shift; 2]>) -> Result<DenseShiftEncoding, SerializationError> {
		let mut buf = Vec::new();
		DenseShiftEncoding { shifts }.serialize(&mut buf).unwrap();
		DenseShiftEncoding::deserialize(buf.as_slice())
	}

	#[test]
	fn dense_shift_encoding_rejects_an_unordered_serialization() {
		// A dense index means nothing against an unsorted list: `dense_idx` binary-searches it.
		match deserialize_raw(vec![single(Shift::srl(3)), single(Shift::IDENTITY)]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_rejects_a_repeated_sequence() {
		// Ascending order is checked strictly, so a repeat is rejected along with a swap: two equal
		// sequences would give one shift sequence two dense indices.
		match deserialize_raw(vec![single(Shift::srl(3)), single(Shift::srl(3))]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_rejects_an_out_of_range_shift_amount() {
		// The bound is the variant's own: a half-word (*32) variant caps at 32, not at 64.
		let out_of_range = Shift {
			variant: ShiftVariant::Sll32,
			amount: 32,
		};
		match deserialize_raw(vec![single(out_of_range)]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_rejects_an_out_of_range_outer_shift_amount() {
		// Both slots are checked, so an outer amount its variant cannot represent is rejected too.
		let out_of_range = Shift {
			variant: ShiftVariant::Sll,
			amount: Word::BITS as u8,
		};
		match deserialize_raw(vec![[Shift::srl(3), out_of_range]]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_indexes_every_sequence_it_covers() {
		// Indexing inverts the dense index directly: no production caller needs a decode-by-index
		// method, so the inverse is checked here against the private field instead of one.
		// The input is unsorted and repeats one sequence, which also pins the sort and the dedupe.
		let sequences = [
			[Shift::srl(3), Shift::sll(3)],
			single(Shift::rotr(1)),
			single(Shift::IDENTITY),
			single(Shift::rotr(1)),
		];
		let encoding = DenseShiftEncoding::new(sequences);

		assert_eq!(encoding.len(), 3);
		for dense_idx in 0..encoding.len() {
			let sequence = encoding.shifts[dense_idx];
			assert_eq!(encoding.dense_idx(sequence) as usize, dense_idx);
		}
	}
}
