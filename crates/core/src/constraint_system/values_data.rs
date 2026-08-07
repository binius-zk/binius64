// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::ops::Deref;

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use crate::word::Word;

/// A run of value-vector words, decoded from its versioned on-disk form.
///
/// The words of one proving job travel as two files, holding what the circuit itself does not fix:
///
/// ```text
///     file        | holds                        | who reads it
///     ------------+------------------------------+------------------
///     inout       | inputs and outputs           | prover, verifier
///     non-public  | witness and internal values  | prover only
/// ```
///
/// Neither file holds the circuit's constants, nor the padding between sections. Both of those
/// are the same for every instance, so they stay in the constraint system and are put back by
/// [`ConstraintSystem::public_segment`](super::ConstraintSystem::public_segment).
///
/// Those two files plus the circuit's constraint system are all another host needs.
/// From the three it rebuilds the witness and proves against it.
///
/// This is the owned end of the format, the one decoding produces.
/// It owns its words because the byte buffer they came from need not outlive the call.
/// Writing starts from the borrowed counterpart below, which copies nothing.
#[derive(Clone, Debug)]
pub struct ValuesData(Vec<Word>);

impl ValuesData {
	/// Version of the byte layout, written ahead of the words in both directions.
	///
	/// # Why this exists
	///
	/// An older layout would decode into plausible but wrong words.
	/// A wrong witness proves nothing, so the mismatch has to surface here.
	/// Bumping this on any layout change turns silent corruption into a hard error.
	pub const SERIALIZATION_VERSION: u32 = 1;
}

impl Deref for ValuesData {
	type Target = [Word];

	fn deref(&self) -> &[Word] {
		// A segment is read-only once decoded, so it is handed out as a plain word slice.
		&self.0
	}
}

impl DeserializeBytes for ValuesData {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		// A mismatched tag means the words that follow were written to another layout.
		let version = u32::deserialize(&mut read_buf)?;
		if version != Self::SERIALIZATION_VERSION {
			return Err(SerializationError::InvalidConstruction {
				name: "ValuesData::version",
			});
		}

		// The word count leads the words, so a short buffer fails instead of truncating.
		Ok(Self(Vec::deserialize(read_buf)?))
	}
}

/// A segment of a value vector borrowed straight from a witness, ready to write.
///
/// This is the borrowed end of the format.
/// It relates to the owned counterpart above as a string slice relates to an owned string.
///
/// Borrowing is what keeps writing cheap.
/// A witness segment runs to tens of megabytes, and none of it is copied to reach the buffer.
pub struct ValuesRef<'a>(&'a [Word]);

impl<'a> ValuesRef<'a> {
	/// Wraps one segment of a value vector for writing.
	pub const fn new(words: &'a [Word]) -> Self {
		Self(words)
	}
}

impl SerializeBytes for ValuesRef<'_> {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		// The tag leads the bytes, so a reader can reject the layout before decoding any word.
		ValuesData::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

		// Then a word count, then that many little-endian words.
		self.0.serialize(write_buf)
	}
}

#[cfg(test)]
mod tests {
	use std::{fs, path::Path};

	use proptest::{collection, prelude::any, prop_assert_eq, proptest};

	use super::*;

	// The four words held by the committed reference file.
	fn reference_words() -> [Word; 4] {
		[
			Word::from_u64(1),
			Word::from_u64(42),
			Word::from_u64(0xDEAD_BEEF),
			Word::from_u64(0x1234_5678_90AB_CDEF),
		]
	}

	proptest! {
		#[test]
		fn round_trip_preserves_words(words in collection::vec(any::<u64>(), 0..64usize)) {
			// Invariant: the words read back are the words written, in order.
			//
			// Fixture state: 0 to 63 arbitrary words, so the empty segment is covered too.
			let words: Vec<Word> = words.into_iter().map(Word).collect();

			// Writing borrows the slice, reading returns an owned segment:
			//
			//     words in --write--> [ 1 | n | word_0 .. word_n-1 ] --read--> words out
			let mut buf = Vec::new();
			ValuesRef::new(&words).serialize(&mut buf).unwrap();
			let read = ValuesData::deserialize(buf.as_slice()).unwrap();

			prop_assert_eq!(&*read, &words[..]);
		}
	}

	#[test]
	fn deserialize_rejects_version_mismatch() {
		// Invariant: a segment written to an unknown layout is rejected, never decoded.
		//
		// Fixture state: one word, tagged one version past the current one.
		//
		//     on disk:  [ 2 | 1 | word_0 ]
		//     expected:   1
		//     -> reject without reading word_0
		let mut buf = Vec::new();
		(ValuesData::SERIALIZATION_VERSION + 1)
			.serialize(&mut buf)
			.unwrap();
		vec![Word::ONE].serialize(&mut buf).unwrap();

		match ValuesData::deserialize(buf.as_slice()).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "ValuesData::version");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn deserialize_rejects_truncated_segment() {
		// Invariant: a file cut short fails, rather than yielding a shorter segment.
		//
		// Fixture state: two words written, then one byte dropped.
		//
		//     written:  [ 1 | 2 | word_0 | word_1 ]   4 + 4 + 8 + 8 = 24 bytes
		//     on disk:  same, minus one byte          23 bytes
		//     -> the count promises 16 bytes of words, 15 remain
		let mut buf = Vec::new();
		ValuesRef::new(&[Word::ONE, Word::ALL_ONE])
			.serialize(&mut buf)
			.unwrap();
		buf.truncate(buf.len() - 1);

		match ValuesData::deserialize(buf.as_slice()).unwrap_err() {
			SerializationError::NotEnoughBytes => {}
			other => panic!("Expected NotEnoughBytes, got: {other:?}"),
		}
	}

	#[test]
	fn reference_binary_deserializes_at_current_version() {
		// Invariant: the committed file still decodes to the words it was written from.
		//
		// This is what forces a layout change to bump the version tag.
		// Change the bytes without touching the tag, and the words stop matching.
		let bytes = include_bytes!("../../test_data/values_data_v1.bin");

		// The tag occupies the leading four bytes, little-endian.
		assert_eq!(
			&bytes[..4],
			&ValuesData::SERIALIZATION_VERSION.to_le_bytes(),
			"reference file version mismatch: regenerate it with the ignored test below"
		);

		let read = ValuesData::deserialize(bytes.as_slice()).unwrap();
		assert_eq!(&*read, &reference_words()[..]);
	}

	// Regenerates the reference file after an intentional layout change.
	// Run: `cargo test -p binius-core -- --ignored create_values_data_reference_binary`.
	#[test]
	#[ignore]
	fn create_values_data_reference_binary_file() {
		let mut buf = Vec::new();
		ValuesRef::new(&reference_words())
			.serialize(&mut buf)
			.unwrap();

		// Relative to the crate root, which is the working directory of a test run.
		let path = Path::new("test_data/values_data_v1.bin");
		fs::write(path, &buf).unwrap();

		println!("Wrote {} bytes to {}", buf.len(), path.display());
	}
}
