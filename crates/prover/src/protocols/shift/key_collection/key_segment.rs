// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Range;

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

#[cfg(test)]
use super::builder::BuilderKey;
use super::{
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
	/// The reference layout the counting-sort builder is checked against.
	#[cfg(test)]
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

		Ok(KeySegment {
			keys,
			key_ranges,
			constraint_indices,
			dense_shift_enc,
		})
	}
}
