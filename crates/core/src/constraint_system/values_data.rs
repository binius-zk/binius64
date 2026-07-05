// Copyright 2025 Irreducible Inc.
use std::borrow::Cow;

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use crate::word::Word;

/// Values data for zero-knowledge proofs (either public witness or non-public part - private inputs
/// and internal values).
///
/// It uses `Cow<[Word]>` to avoid unnecessary clones while supporting
/// both borrowed and owned data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValuesData<'a> {
	data: Cow<'a, [Word]>,
}

impl<'a> ValuesData<'a> {
	/// Serialization format version for compatibility checking
	pub const SERIALIZATION_VERSION: u32 = 1;

	/// Create a new ValuesData from borrowed data
	pub const fn borrowed(data: &'a [Word]) -> Self {
		Self {
			data: Cow::Borrowed(data),
		}
	}

	/// Create a new ValuesData from owned data
	pub const fn owned(data: Vec<Word>) -> Self {
		Self {
			data: Cow::Owned(data),
		}
	}

	/// Get the values data as a slice
	pub fn as_slice(&self) -> &[Word] {
		&self.data
	}

	/// Get the number of words in the values data
	pub fn len(&self) -> usize {
		self.data.len()
	}

	/// Check if the witness is empty
	pub fn is_empty(&self) -> bool {
		self.data.is_empty()
	}

	/// Convert to owned data, consuming self
	pub fn into_owned(self) -> Vec<Word> {
		self.data.into_owned()
	}

	/// Convert to owned version of ValuesData
	pub fn to_owned(&self) -> ValuesData<'static> {
		ValuesData {
			data: Cow::Owned(self.data.to_vec()),
		}
	}
}

impl<'a> SerializeBytes for ValuesData<'a> {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		Self::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

		self.data.as_ref().serialize(write_buf)
	}
}

impl DeserializeBytes for ValuesData<'static> {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let version = u32::deserialize(&mut read_buf)?;
		if version != Self::SERIALIZATION_VERSION {
			return Err(SerializationError::InvalidConstruction {
				name: "Witness::version",
			});
		}

		let data = Vec::<Word>::deserialize(read_buf)?;

		Ok(ValuesData::owned(data))
	}
}

impl<'a> From<&'a [Word]> for ValuesData<'a> {
	fn from(data: &'a [Word]) -> Self {
		ValuesData::borrowed(data)
	}
}

impl From<Vec<Word>> for ValuesData<'static> {
	fn from(data: Vec<Word>) -> Self {
		ValuesData::owned(data)
	}
}

impl<'a> AsRef<[Word]> for ValuesData<'a> {
	fn as_ref(&self) -> &[Word] {
		self.as_slice()
	}
}

impl<'a> std::ops::Deref for ValuesData<'a> {
	type Target = [Word];

	fn deref(&self) -> &Self::Target {
		self.as_slice()
	}
}

impl<'a> From<ValuesData<'a>> for Vec<Word> {
	fn from(value: ValuesData<'a>) -> Self {
		value.into_owned()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_witness_serialization_round_trip_owned() {
		let data = vec![
			Word::from_u64(1),
			Word::from_u64(42),
			Word::from_u64(0xDEADBEEF),
			Word::from_u64(0x1234567890ABCDEF),
		];
		let witness = ValuesData::owned(data.clone());

		let mut buf = Vec::new();
		witness.serialize(&mut buf).unwrap();

		let deserialized = ValuesData::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(witness, deserialized);
		assert_eq!(deserialized.as_slice(), data.as_slice());
	}

	#[test]
	fn test_witness_serialization_round_trip_borrowed() {
		let data = vec![Word::from_u64(123), Word::from_u64(456)];
		let witness = ValuesData::borrowed(&data);

		let mut buf = Vec::new();
		witness.serialize(&mut buf).unwrap();

		let deserialized = ValuesData::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(witness, deserialized);
		assert_eq!(deserialized.as_slice(), data.as_slice());
	}

	#[test]
	fn test_witness_version_mismatch() {
		let mut buf = Vec::new();
		999u32.serialize(&mut buf).unwrap(); // Wrong version
		vec![Word::from_u64(1)].serialize(&mut buf).unwrap(); // Some data

		let result = ValuesData::deserialize(&mut buf.as_slice());
		assert!(result.is_err());
		match result.unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "Witness::version");
			}
			_ => panic!("Expected version mismatch error"),
		}
	}

	/// Helper function to create or update the reference binary file for Witness version
	/// compatibility testing.
	#[test]
	#[ignore] // Use `cargo test -- --ignored create_witness_reference_binary` to run this
	fn create_witness_reference_binary_file() {
		let data = vec![
			Word::from_u64(1),
			Word::from_u64(42),
			Word::from_u64(0xDEADBEEF),
			Word::from_u64(0x1234567890ABCDEF),
		];
		let witness = ValuesData::owned(data);

		let mut buf = Vec::new();
		witness.serialize(&mut buf).unwrap();

		let test_data_path = std::path::Path::new("verifier/core/test_data/witness_v1.bin");

		if let Some(parent) = test_data_path.parent() {
			std::fs::create_dir_all(parent).unwrap();
		}

		std::fs::write(test_data_path, &buf).unwrap();

		println!("Created Witness reference binary file at: {:?}", test_data_path);
		println!("Binary data length: {} bytes", buf.len());
	}

	/// Test deserialization from a reference binary file to ensure Witness version
	/// compatibility. This test will fail if breaking changes are made without incrementing the
	/// version.
	#[test]
	fn test_witness_deserialize_from_reference_binary_file() {
		let binary_data = include_bytes!("../../test_data/witness_v1.bin");

		let deserialized = ValuesData::deserialize(&mut binary_data.as_slice()).unwrap();

		assert_eq!(deserialized.len(), 4);
		assert_eq!(deserialized.as_slice()[0].as_u64(), 1);
		assert_eq!(deserialized.as_slice()[1].as_u64(), 42);
		assert_eq!(deserialized.as_slice()[2].as_u64(), 0xDEADBEEF);
		assert_eq!(deserialized.as_slice()[3].as_u64(), 0x1234567890ABCDEF);

		// Verify that the version is what we expect
		// This is implicitly checked during deserialization, but we can also verify
		// the file starts with the correct version bytes
		let version_bytes = &binary_data[0..4]; // First 4 bytes should be version
		let expected_version_bytes = 1u32.to_le_bytes(); // Version 1 in little-endian
		assert_eq!(
			version_bytes, expected_version_bytes,
			"WitnessData binary file version mismatch. If you made breaking changes, increment WitnessData::SERIALIZATION_VERSION"
		);
	}
}
