// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Range;

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::{
	builder::BuilderKey,
	dense_shift_encoding::DenseShiftEncoding,
	key::{ConstraintIndex, Key},
};

/// One value-vector segment's keys, public or hidden.
/// Indexed so each word's constraints can be found without a scan.
#[derive(Debug, Clone)]
pub struct KeySegment {
	/// Every key of the segment, flattened into one vector.
	pub keys: Vec<Key>,
	/// One range per word, at that word's segment-relative index.
	/// The range names that word's keys inside the flattened keys vector.
	pub key_ranges: Vec<Range<u32>>,
	/// The constraint indices the keys reference, flattened into one vector.
	pub constraint_indices: Vec<ConstraintIndex>,
	/// The shift sequences the segment's keys name.
	pub dense_shift_enc: DenseShiftEncoding,
}

impl KeySegment {
	/// The number of words the segment covers.
	pub const fn n_words(&self) -> usize {
		self.key_ranges.len()
	}

	/// The keys for the word at the given segment-relative index.
	pub fn word_keys(&self, index: usize) -> &[Key] {
		let Range { start, end } = self.key_ranges[index];
		&self.keys[start as usize..end as usize]
	}

	/// Builds the segment's keys from the builder keys lists of its words.
	pub(super) fn build(builder_key_lists: Vec<Vec<BuilderKey>>) -> Self {
		// Every distinct shift sequence across every word, before any per-key index is assigned.
		let dense_shift_enc = DenseShiftEncoding::new(
			builder_key_lists
				.iter()
				.flatten()
				.map(|builder_key| builder_key.shift_seq),
		);

		// Word w's keys occupy a contiguous run in the flattened keys vector.
		// A running offset gives each word's run its start and end.
		let key_ranges = builder_key_lists
			.iter()
			.scan(0u32, |offset, builder_keys| {
				let start = *offset;
				*offset += builder_keys.len() as u32;
				Some(start..*offset)
			})
			.collect();

		let mut keys = Vec::new();
		let mut constraint_indices = Vec::new();

		for builder_key in builder_key_lists.into_iter().flatten() {
			let BuilderKey {
				shift_seq,
				operation,
				constraint_indices: mut builder_constraint_indices,
			} = builder_key;

			// Sort constraint indices by operand index, so a later linear scan can detect each
			// operand's boundary with no extra bookkeeping.
			builder_constraint_indices
				.sort_by_key(|constraint_index| constraint_index.operand_index);

			let start = constraint_indices.len() as u32;
			constraint_indices.extend(builder_constraint_indices);
			let end = constraint_indices.len() as u32;
			keys.push(Key {
				dense_shift_idx: dense_shift_enc.dense_idx(shift_seq),
				operation,
				range: start..end,
			});
		}

		Self {
			keys,
			key_ranges,
			constraint_indices,
			dense_shift_enc,
		}
	}
}

impl SerializeBytes for KeySegment {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.keys.serialize(&mut write_buf)?;

		// Serialize key_ranges as pairs of start/end
		(self.key_ranges.len() as u32).serialize(&mut write_buf)?;
		for range in &self.key_ranges {
			range.start.serialize(&mut write_buf)?;
			range.end.serialize(&mut write_buf)?;
		}

		self.constraint_indices.serialize(&mut write_buf)?;
		self.dense_shift_enc.serialize(write_buf)
	}
}

impl DeserializeBytes for KeySegment {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let keys = Vec::<Key>::deserialize(&mut read_buf)?;

		// Deserialize key_ranges
		let len = u32::deserialize(&mut read_buf)? as usize;
		let mut key_ranges = Vec::with_capacity(len);
		for _ in 0..len {
			let start = u32::deserialize(&mut read_buf)?;
			let end = u32::deserialize(&mut read_buf)?;
			key_ranges.push(start..end);
		}

		let constraint_indices = Vec::<ConstraintIndex>::deserialize(&mut read_buf)?;
		let dense_shift_enc = DenseShiftEncoding::deserialize(&mut read_buf)?;

		// A key indexes its own segment's shift encoding and constraint list.
		// Rejecting a bad index here beats panicking mid-proof in `build_g` or `word_scalar`.
		let keys_index_siblings = keys.iter().all(|key| {
			(key.dense_shift_idx as usize) < dense_shift_enc.len()
				&& key.range.end as usize <= constraint_indices.len()
		});
		if !keys_index_siblings {
			return Err(SerializationError::InvalidConstruction {
				name: "KeySegment::keys",
			});
		}
		// A word's range names its keys inside the flattened keys vector, which `word_keys` slices.
		let key_ranges_index_keys = key_ranges
			.iter()
			.all(|range| range.start <= range.end && range.end as usize <= keys.len());
		if !key_ranges_index_keys {
			return Err(SerializationError::InvalidConstruction {
				name: "KeySegment::key_ranges",
			});
		}

		Ok(KeySegment {
			keys,
			key_ranges,
			constraint_indices,
			dense_shift_enc,
		})
	}
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::Shift;

	use super::{super::operation::Operation, *};

	// Serializes a segment built raw, bypassing `build`, so malformed indices reach the
	// deserializer.
	fn deserialize_raw(segment: &KeySegment) -> Result<KeySegment, SerializationError> {
		let mut buf = Vec::new();
		segment.serialize(&mut buf).unwrap();
		KeySegment::deserialize(buf.as_slice())
	}

	#[test]
	fn key_segment_round_trips_a_well_formed_segment() {
		// Pins the checks against the shape `build` produces: word ranges tile the keys vector, and
		// every index addresses a sibling long enough to hold it.
		let segment = KeySegment {
			keys: vec![
				Key {
					operation: Operation::Zero,
					dense_shift_idx: 0,
					range: Range { start: 0, end: 2 },
				},
				Key {
					operation: Operation::Zero,
					dense_shift_idx: 1,
					range: Range { start: 2, end: 3 },
				},
			],
			key_ranges: vec![Range { start: 0, end: 1 }, Range { start: 1, end: 2 }],
			constraint_indices: (0..3)
				.map(|constraint_index| ConstraintIndex {
					operand_index: 0,
					constraint_index,
				})
				.collect(),
			dense_shift_enc: DenseShiftEncoding::new([
				[Shift::srl(1), Shift::IDENTITY],
				[Shift::srl(2), Shift::IDENTITY],
			]),
		};

		let segment = deserialize_raw(&segment).expect("a well-formed segment deserializes");

		assert_eq!(segment.n_words(), 2);
		assert_eq!(segment.keys.len(), 2);
		assert_eq!(segment.constraint_indices.len(), 3);
		assert_eq!(segment.dense_shift_enc.len(), 2);
		assert_eq!(segment.word_keys(0).len(), 1);
		assert_eq!(segment.word_keys(1).len(), 1);
	}

	#[test]
	fn key_segment_rejects_a_key_indexing_past_the_shift_encoding() {
		// `build_g` scales this index by the row length to reach into the multilinears buffer.
		// The encoding holds two sequences, so index 2 is one past its end.
		let segment = KeySegment {
			keys: vec![Key {
				operation: Operation::Zero,
				dense_shift_idx: 2,
				range: Range { start: 0, end: 1 },
			}],
			key_ranges: vec![Range { start: 0, end: 1 }],
			constraint_indices: vec![ConstraintIndex {
				operand_index: 0,
				constraint_index: 0,
			}],
			dense_shift_enc: DenseShiftEncoding::new([
				[Shift::srl(1), Shift::IDENTITY],
				[Shift::srl(2), Shift::IDENTITY],
			]),
		};

		match deserialize_raw(&segment).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "KeySegment::keys");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn key_segment_rejects_a_key_range_past_the_constraint_indices() {
		// `accumulate_wide` slices the segment's flattened constraint list with this range, which
		// holds one entry against the key's claim of four.
		let segment = KeySegment {
			keys: vec![Key {
				operation: Operation::Zero,
				dense_shift_idx: 0,
				range: Range { start: 0, end: 4 },
			}],
			key_ranges: vec![Range { start: 0, end: 1 }],
			constraint_indices: vec![ConstraintIndex {
				operand_index: 0,
				constraint_index: 0,
			}],
			dense_shift_enc: DenseShiftEncoding::new([[Shift::srl(1), Shift::IDENTITY]]),
		};

		match deserialize_raw(&segment).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "KeySegment::keys");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn key_segment_rejects_a_word_range_past_the_keys() {
		// `word_keys` slices the flattened keys vector, which holds one key against a range of two.
		let segment = KeySegment {
			keys: vec![Key {
				operation: Operation::Zero,
				dense_shift_idx: 0,
				range: Range { start: 0, end: 1 },
			}],
			key_ranges: vec![Range { start: 0, end: 2 }],
			constraint_indices: vec![ConstraintIndex {
				operand_index: 0,
				constraint_index: 0,
			}],
			dense_shift_enc: DenseShiftEncoding::new([[Shift::srl(1), Shift::IDENTITY]]),
		};

		match deserialize_raw(&segment).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "KeySegment::key_ranges");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn key_segment_rejects_a_reversed_word_range() {
		// Slicing `start..end` panics when start runs past end, in bounds or not.
		let segment = KeySegment {
			keys: vec![Key {
				operation: Operation::Zero,
				dense_shift_idx: 0,
				range: Range { start: 0, end: 1 },
			}],
			key_ranges: vec![Range { start: 1, end: 0 }],
			constraint_indices: vec![ConstraintIndex {
				operand_index: 0,
				constraint_index: 0,
			}],
			dense_shift_enc: DenseShiftEncoding::new([[Shift::srl(1), Shift::IDENTITY]]),
		};

		match deserialize_raw(&segment).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "KeySegment::key_ranges");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}
}
