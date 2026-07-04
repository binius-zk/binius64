// Copyright 2025 Irreducible Inc.
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

/// A type safe wrapper over an index into the [`ValueVec`](super::ValueVec).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ValueIndex(pub u32);

impl ValueIndex {
	/// The value index that is not considered to be valid.
	pub const INVALID: ValueIndex = ValueIndex(u32::MAX);
}

/// The most sensible default for a value index is invalid.
impl Default for ValueIndex {
	fn default() -> Self {
		Self::INVALID
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
	fn test_value_index_serialization_round_trip() {
		let value_index = ValueIndex(12345);

		let mut buf = Vec::new();
		value_index.serialize(&mut buf).unwrap();

		let deserialized = ValueIndex::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(value_index, deserialized);
	}
}
