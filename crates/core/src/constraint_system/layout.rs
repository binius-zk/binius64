// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	serialization::{DeserializeBytes, SerializationError, SerializeBytes},
};
use bytes::{Buf, BufMut};

use super::ValueIndex;
use crate::{consts, error::ConstraintSystemError};

/// Description of a layout of the value vector for a particular circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueVecLayout {
	/// The number of the constants declared by the circuit.
	pub n_const: usize,
	/// The number of the input output parameters declared by the circuit.
	pub n_inout: usize,
	/// The number of the witness parameters declared by the circuit.
	pub n_witness: usize,
	/// The number of the internal values declared by the circuit.
	///
	/// Those are outputs and intermediaries created by the gates.
	pub n_internal: usize,

	/// The offset at which `inout` parameters start.
	pub offset_inout: usize,
	/// The offset at which `witness` parameters start.
	///
	/// The public section of the value vec has the power-of-two size and is greater than the
	/// minimum number of words. By public section we mean the constants and the inout values.
	pub offset_witness: usize,
	/// The number of words in the hidden segment: the witness and internal values, including
	/// padding up to the segment length. This does not include the public segment or any
	/// scratch values.
	pub n_hidden_words: usize,
	/// The number of scratch values at the end of the value vec.
	pub n_scratch: usize,
}

impl ValueVecLayout {
	/// Validates that the value vec layout has a correct shape.
	///
	/// Specifically checks that:
	///
	/// - the public segment (constants and inout values) is padded to the power of two.
	/// - the public segment is not less than the minimum size.
	/// - the hidden segment is at least as long as the public segment, so
	///   [`Self::log_witness_words`] is at least [`Self::log_public_words`].
	pub const fn validate(&self) -> Result<(), ConstraintSystemError> {
		if !self.offset_witness.is_power_of_two() {
			return Err(ConstraintSystemError::PublicInputPowerOfTwo);
		}

		let pub_input_size = self.offset_witness;
		if pub_input_size < consts::MIN_WORDS_PER_SEGMENT {
			return Err(ConstraintSystemError::PublicInputTooShort { pub_input_size });
		}

		if self.n_hidden_words < self.n_public_words() {
			return Err(ConstraintSystemError::HiddenSegmentTooShort {
				public_len: self.n_public_words(),
				hidden_len: self.n_hidden_words,
			});
		}

		Ok(())
	}

	/// Returns the number of words in the public segment: the constants and inout values,
	/// including padding up to the power-of-two segment length.
	pub const fn n_public_words(&self) -> usize {
		self.offset_witness
	}

	/// Returns the base-2 logarithm of the public segment length in words.
	///
	/// [`Self::validate`] guarantees that the public segment length is a power of two.
	pub const fn log_public_words(&self) -> usize {
		self.offset_witness.trailing_zeros() as usize
	}

	/// Returns the combined number of public and hidden words, excluding scratch.
	///
	/// This is the length of the value vector prefix that constraint operands can reference.
	pub const fn combined_len(&self) -> usize {
		self.offset_witness + self.n_hidden_words
	}

	/// Returns the base-2 logarithm of the hidden segment length in words, rounded up to a
	/// power of two.
	///
	/// [`Self::validate`] guarantees this is at least [`Self::log_public_words`].
	pub const fn log_witness_words(&self) -> usize {
		log2_ceil_usize(self.n_hidden_words)
	}

	/// Returns true if the given index points to an area that is considered to be padding.
	pub(super) const fn is_padding(&self, index: ValueIndex) -> bool {
		let idx = index.0 as usize;

		// padding 1: between constants and inout section
		if idx >= self.n_const && idx < self.offset_inout {
			return true;
		}

		// padding 2: between the end of inout section and the start of witness section
		let end_of_inout = self.offset_inout + self.n_inout;
		if idx >= end_of_inout && idx < self.offset_witness {
			return true;
		}

		// padding 3: between the last internal value and the total len
		let end_of_internal = self.offset_witness + self.n_witness + self.n_internal;
		if idx >= end_of_internal && idx < self.combined_len() {
			return true;
		}

		false
	}

	/// Returns true if the given index is out-of-bounds for the committed part of this layout.
	pub(super) const fn is_committed_oob(&self, index: ValueIndex) -> bool {
		index.0 as usize >= self.combined_len()
	}
}

impl SerializeBytes for ValueVecLayout {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.n_const.serialize(&mut write_buf)?;
		self.n_inout.serialize(&mut write_buf)?;
		self.n_witness.serialize(&mut write_buf)?;
		self.n_internal.serialize(&mut write_buf)?;
		self.offset_inout.serialize(&mut write_buf)?;
		self.offset_witness.serialize(&mut write_buf)?;
		self.n_hidden_words.serialize(&mut write_buf)?;
		self.n_scratch.serialize(write_buf)
	}
}

impl DeserializeBytes for ValueVecLayout {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let n_const = usize::deserialize(&mut read_buf)?;
		let n_inout = usize::deserialize(&mut read_buf)?;
		let n_witness = usize::deserialize(&mut read_buf)?;
		let n_internal = usize::deserialize(&mut read_buf)?;
		let offset_inout = usize::deserialize(&mut read_buf)?;
		let offset_witness = usize::deserialize(&mut read_buf)?;
		let n_hidden_words = usize::deserialize(&mut read_buf)?;
		let n_scratch = usize::deserialize(read_buf)?;

		Ok(ValueVecLayout {
			n_const,
			n_inout,
			n_witness,
			n_internal,
			offset_inout,
			offset_witness,
			n_hidden_words,
			n_scratch,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_value_vec_layout_serialization_round_trip() {
		let layout = ValueVecLayout {
			n_const: 5,
			n_inout: 3,
			n_witness: 12,
			n_internal: 7,
			offset_inout: 8,
			offset_witness: 16,
			n_hidden_words: 16,
			n_scratch: 0,
		};

		let mut buf = Vec::new();
		layout.serialize(&mut buf).unwrap();

		let deserialized = ValueVecLayout::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(layout, deserialized);
	}

	#[test]
	fn test_log_witness_words() {
		let layout = |offset_witness: usize, n_hidden_words: usize| ValueVecLayout {
			n_const: 0,
			n_inout: 0,
			n_witness: 0,
			n_internal: 0,
			offset_inout: 0,
			offset_witness,
			n_hidden_words,
			n_scratch: 0,
		};
		// Typical: more witness words than public words.
		assert_eq!(layout(4, 64).log_witness_words(), 6);
		// Exact power-of-two witness count.
		assert_eq!(layout(4, 32).log_witness_words(), 5);
	}

	#[test]
	fn test_validate_rejects_short_hidden_segment() {
		// The hidden segment (4 words) is shorter than the public segment (16 words).
		let layout = ValueVecLayout {
			n_const: 1,
			n_inout: 8,
			n_witness: 4,
			n_internal: 0,
			offset_inout: 1,
			offset_witness: 16,
			n_hidden_words: 4,
			n_scratch: 0,
		};
		assert!(matches!(
			layout.validate(),
			Err(ConstraintSystemError::HiddenSegmentTooShort { .. })
		));
	}

	#[test]
	fn test_is_padding_comprehensive() {
		// Test layout with all types of padding
		let layout = ValueVecLayout {
			n_const: 2,         // constants at indices 0-1
			n_inout: 3,         // inout at indices 4-6
			n_witness: 5,       // witness at indices 16-20
			n_internal: 10,     // internal at indices 21-30
			offset_inout: 4,    // gap between constants and inout (indices 2-3 are padding)
			offset_witness: 16, // public section is 16 (power of 2), gap 7-15 is padding
			n_hidden_words: 48, // total must be power of 2, gap 31-63 is padding
			n_scratch: 0,
		};

		// Test constants (indices 0-1): NOT padding
		assert!(!layout.is_padding(ValueIndex(0)), "index 0 should be constant");
		assert!(!layout.is_padding(ValueIndex(1)), "index 1 should be constant");

		// Test padding between constants and inout (indices 2-3): PADDING
		assert!(
			layout.is_padding(ValueIndex(2)),
			"index 2 should be padding between const and inout"
		);
		assert!(
			layout.is_padding(ValueIndex(3)),
			"index 3 should be padding between const and inout"
		);

		// Test inout values (indices 4-6): NOT padding
		assert!(!layout.is_padding(ValueIndex(4)), "index 4 should be inout");
		assert!(!layout.is_padding(ValueIndex(5)), "index 5 should be inout");
		assert!(!layout.is_padding(ValueIndex(6)), "index 6 should be inout");

		// Test padding between inout and witness (indices 7-15): PADDING
		for i in 7..16 {
			assert!(
				layout.is_padding(ValueIndex(i)),
				"index {} should be padding between inout and witness",
				i
			);
		}

		// Test witness values (indices 16-20): NOT padding
		for i in 16..21 {
			assert!(!layout.is_padding(ValueIndex(i)), "index {} should be witness", i);
		}

		// Test internal values (indices 21-30): NOT padding
		for i in 21..31 {
			assert!(!layout.is_padding(ValueIndex(i)), "index {} should be internal", i);
		}

		// Test padding after internal values (indices 31-63): PADDING
		for i in 31..64 {
			assert!(
				layout.is_padding(ValueIndex(i)),
				"index {} should be padding after internal",
				i
			);
		}
	}

	#[test]
	fn test_is_padding_minimal_layout() {
		// Test a minimal layout with no gaps except required end padding
		let layout = ValueVecLayout {
			n_const: 4,        // constants at indices 0-3
			n_inout: 4,        // inout at indices 4-7
			n_witness: 4,      // witness at indices 8-11
			n_internal: 4,     // internal at indices 12-15
			offset_inout: 4,   // no gap between constants and inout
			offset_witness: 8, // no gap between inout and witness
			n_hidden_words: 8, // exactly fits all values
			n_scratch: 0,
		};

		// No padding anywhere in this layout
		for i in 0..16 {
			assert!(
				!layout.is_padding(ValueIndex(i)),
				"index {} should not be padding in minimal layout",
				i
			);
		}
	}

	#[test]
	fn test_is_padding_public_section_min_size() {
		// Test layout where public section must be padded to meet MIN_WORDS_PER_SEGMENT
		let layout = ValueVecLayout {
			n_const: 1,        // only 1 constant
			n_inout: 1,        // only 1 inout
			n_witness: 2,      // 2 witness values
			n_internal: 2,     // 2 internal values
			offset_inout: 4,   // padding between const and inout to reach min size
			offset_witness: 8, // public section padded to 8 (MIN_WORDS_PER_SEGMENT)
			n_hidden_words: 8, // power of 2
			n_scratch: 0,
		};

		// Test the single constant
		assert!(!layout.is_padding(ValueIndex(0)), "index 0 should be constant");

		// Test padding between constant and inout (indices 1-3)
		assert!(layout.is_padding(ValueIndex(1)), "index 1 should be padding");
		assert!(layout.is_padding(ValueIndex(2)), "index 2 should be padding");
		assert!(layout.is_padding(ValueIndex(3)), "index 3 should be padding");

		// Test the single inout value
		assert!(!layout.is_padding(ValueIndex(4)), "index 4 should be inout");

		// Test padding between inout and witness (indices 5-7)
		assert!(layout.is_padding(ValueIndex(5)), "index 5 should be padding");
		assert!(layout.is_padding(ValueIndex(6)), "index 6 should be padding");
		assert!(layout.is_padding(ValueIndex(7)), "index 7 should be padding");

		// Test witness values (indices 8-9)
		assert!(!layout.is_padding(ValueIndex(8)), "index 8 should be witness");
		assert!(!layout.is_padding(ValueIndex(9)), "index 9 should be witness");

		// Test internal values (indices 10-11)
		assert!(!layout.is_padding(ValueIndex(10)), "index 10 should be internal");
		assert!(!layout.is_padding(ValueIndex(11)), "index 11 should be internal");

		// Test padding at the end (indices 12-15)
		for i in 12..16 {
			assert!(layout.is_padding(ValueIndex(i)), "index {} should be end padding", i);
		}
	}

	#[test]
	fn test_is_padding_boundary_conditions() {
		let layout = ValueVecLayout {
			n_const: 2,
			n_inout: 2,
			n_witness: 4,
			n_internal: 4,
			offset_inout: 4,
			offset_witness: 8,
			n_hidden_words: 8,
			n_scratch: 0,
		};

		// Test exact boundaries
		assert!(!layout.is_padding(ValueIndex(1)), "last constant should not be padding");
		assert!(layout.is_padding(ValueIndex(2)), "first padding after const should be padding");

		assert!(layout.is_padding(ValueIndex(3)), "last padding before inout should be padding");
		assert!(!layout.is_padding(ValueIndex(4)), "first inout should not be padding");

		assert!(!layout.is_padding(ValueIndex(5)), "last inout should not be padding");
		assert!(layout.is_padding(ValueIndex(6)), "first padding after inout should be padding");

		assert!(layout.is_padding(ValueIndex(7)), "last padding before witness should be padding");
		assert!(!layout.is_padding(ValueIndex(8)), "first witness should not be padding");

		assert!(!layout.is_padding(ValueIndex(11)), "last witness should not be padding");
		assert!(!layout.is_padding(ValueIndex(12)), "first internal should not be padding");

		assert!(!layout.is_padding(ValueIndex(15)), "last internal should not be padding");
		// Note: index 16 would be out of bounds, not tested here
	}

	#[test]
	fn test_is_padding_matches_compiler_requirements() {
		// Test that is_padding correctly handles the MIN_WORDS_PER_SEGMENT requirement
		// as seen in the compiler mod.rs:
		// cur_index = cur_index.max(MIN_WORDS_PER_SEGMENT as u32);
		// cur_index = cur_index.next_power_of_two();

		// Case 1: Very small public section (1 const + 1 inout = 2 total)
		// Should be padded to MIN_WORDS_PER_SEGMENT (8)
		let layout1 = ValueVecLayout {
			n_const: 1,
			n_inout: 1,
			n_witness: 4,
			n_internal: 4,
			offset_inout: 1,   // right after constants
			offset_witness: 8, // padded to MIN_WORDS_PER_SEGMENT
			n_hidden_words: 8,
			n_scratch: 0,
		};

		// Verify padding between end of inout (index 2) and offset_witness (8)
		assert!(!layout1.is_padding(ValueIndex(0)), "const should not be padding");
		assert!(!layout1.is_padding(ValueIndex(1)), "inout should not be padding");
		for i in 2..8 {
			assert!(
				layout1.is_padding(ValueIndex(i)),
				"index {} should be padding to meet MIN_WORDS_PER_SEGMENT",
				i
			);
		}

		// Case 2: Public section exactly MIN_WORDS_PER_SEGMENT (no extra padding needed)
		let layout2 = ValueVecLayout {
			n_const: 4,
			n_inout: 4,
			n_witness: 8,
			n_internal: 0,
			offset_inout: 4,
			offset_witness: 8, // exactly MIN_WORDS_PER_SEGMENT, already power of 2
			n_hidden_words: 8,
			n_scratch: 0,
		};

		// No padding in public section
		for i in 0..8 {
			assert!(!layout2.is_padding(ValueIndex(i)), "index {} should not be padding", i);
		}

		// Case 3: Public section between MIN_WORDS_PER_SEGMENT and next power of 2
		// e.g., 10 total needs to round up to 16
		let layout3 = ValueVecLayout {
			n_const: 5,
			n_inout: 5,
			n_witness: 16,
			n_internal: 0,
			offset_inout: 5,
			offset_witness: 16, // rounded up from 10 to 16 (next power of 2)
			n_hidden_words: 16,
			n_scratch: 0,
		};

		// Check padding from end of inout (10) to offset_witness (16)
		for i in 0..5 {
			assert!(!layout3.is_padding(ValueIndex(i)), "const {} should not be padding", i);
		}
		for i in 5..10 {
			assert!(!layout3.is_padding(ValueIndex(i)), "inout {} should not be padding", i);
		}
		for i in 10..16 {
			assert!(
				layout3.is_padding(ValueIndex(i)),
				"index {} should be padding for power-of-2 alignment",
				i
			);
		}

		// Case 4: Test with offsets that show all three padding types
		let layout4 = ValueVecLayout {
			n_const: 2,         // indices 0-1
			n_inout: 2,         // indices 8-9
			n_witness: 4,       // indices 16-19
			n_internal: 4,      // indices 20-23
			offset_inout: 8,    // padding after constants to align
			offset_witness: 16, // padding after inout to reach power of 2
			n_hidden_words: 16, // padding after internal to reach total
			n_scratch: 0,
		};

		// Constants
		assert!(!layout4.is_padding(ValueIndex(0)));
		assert!(!layout4.is_padding(ValueIndex(1)));

		// Padding between constants and inout (indices 2-7)
		for i in 2..8 {
			assert!(layout4.is_padding(ValueIndex(i)), "padding between const and inout at {}", i);
		}

		// Inout values
		assert!(!layout4.is_padding(ValueIndex(8)));
		assert!(!layout4.is_padding(ValueIndex(9)));

		// Padding between inout and witness (indices 10-15)
		for i in 10..16 {
			assert!(
				layout4.is_padding(ValueIndex(i)),
				"padding between inout and witness at {}",
				i
			);
		}

		// Witness values
		for i in 16..20 {
			assert!(!layout4.is_padding(ValueIndex(i)), "witness at {}", i);
		}

		// Internal values
		for i in 20..24 {
			assert!(!layout4.is_padding(ValueIndex(i)), "internal at {}", i);
		}

		// Padding after internal to total_len (indices 24-31)
		for i in 24..32 {
			assert!(layout4.is_padding(ValueIndex(i)), "padding after internal at {}", i);
		}
	}
}
