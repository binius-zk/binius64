// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	serialization::{DeserializeBytes, SerializationError, SerializeBytes},
};
use bytes::{Buf, BufMut};

use super::key_segment::KeySegment;

/// The prover's complete view of a constraint system's shift keys, split by value-vector segment.
///
/// - The public segment covers value-vector indices from 0 up to the public word count.
/// - The hidden segment covers the rest, up to the combined length.
/// - Word indices inside each segment are relative to that segment's own start.
/// - Both phases of the shift reduction iterate the two segments in absolute value-vector order.
#[derive(Debug, Clone)]
pub struct KeyCollection {
	/// The keys of the public segment: constants and inout words.
	pub public: KeySegment,
	/// The keys of the hidden segment: private words.
	pub hidden: KeySegment,
}

impl KeyCollection {
	/// The base-2 logarithm of the hidden segment length in words, rounded up to a power of two.
	///
	/// Matches the corresponding quantity for the constraint system the collection was built from.
	/// That system guarantees this is at least the public segment's logarithm.
	///
	/// ```text
	/// log_witness_words = ceil_log2( hidden segment length in words )
	/// ```
	pub const fn log_witness_words(&self) -> usize {
		log2_ceil_usize(self.hidden.n_words())
	}
}

impl SerializeBytes for KeyCollection {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		// Version for forward compatibility; version 3 introduced the dense shift encoding.
		const VERSION: u32 = 3;
		VERSION.serialize(&mut write_buf)?;

		self.public.serialize(&mut write_buf)?;
		self.hidden.serialize(write_buf)
	}
}

impl DeserializeBytes for KeyCollection {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		const VERSION: u32 = 3;
		let version = u32::deserialize(&mut read_buf)?;
		if version != VERSION {
			return Err(SerializationError::InvalidConstruction {
				name: "KeyCollection::version",
			});
		}

		let public = KeySegment::deserialize(&mut read_buf)?;
		let hidden = KeySegment::deserialize(read_buf)?;

		Ok(KeyCollection { public, hidden })
	}
}
