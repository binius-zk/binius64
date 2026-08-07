// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::iter;

use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	serialization::{DeserializeBytes, SerializationError, SerializeBytes},
};
use bytes::{Buf, BufMut};

#[cfg(doc)]
use super::ValueVec;
use super::{
	AndConstraint, BmulConstraint, ConstraintKind, ImulConstraint, Operand, ShiftVariant,
	ValueIndex, ZeroConstraint,
};
use crate::{error::ConstraintSystemError, word::Word};

/// The ConstraintSystem is the core data structure in Binius64 that defines the computational
/// constraints to be proven in zero-knowledge. It represents a system of equations over 64-bit
/// words that must be satisfied by a valid values vector [`ValueVec`].
///
/// # Value vector shape
///
/// The constraints reference words of a value vector partitioned into two segments. The public
/// segment holds the constants and the inout values; the hidden segment holds the private values.
/// Each group of values is followed by padding, so the value vector runs
///
/// ```text
/// [ constants | const pad | inout | inout pad ][ private | private pad ]
///  \------------- public segment -----------/  \----- hidden segment -/
/// ```
///
/// A constraint may reference any value word, but not a padding word.
///
/// # Constraint counts
///
/// The ZERO, AND, IMUL and BMUL constraint counts are the true counts; none of them is rounded up
/// to a power of two. The reductions run over power-of-two-sized operand columns, so the prover
/// rounds each count up to a power of two and zero-fills the tail when it materializes the column;
/// [`Self::log_and_constraints`] and its ZERO, IMUL and BMUL siblings report the resulting variable
/// count. Zero is a valid padding row for every constraint type: an empty operand evaluates to
/// [`Word::ZERO`], and `0 = 0`, `0 & 0 ^ 0 = 0` and `0 * 0 = 0 || 0` all hold.
///
/// # Clone
///
/// While this type is cloneable it may be expensive to do so since the constraint systems often
/// can have millions of constraints.
#[derive(Debug, Clone)]
pub struct ConstraintSystem {
	/// The constants that this constraint system defines.
	///
	/// Those constants will be going to be available for constraints in the value vector. Those
	/// are known to both prover and verifier.
	pub constants: Vec<Word>,
	/// The number of padding words between the constants and the inout values.
	pub n_const_pad: usize,
	/// The number of input/output values, which are public but chosen per instance.
	pub n_inout: usize,
	/// The number of padding words between the inout values and the end of the public segment.
	pub n_inout_pad: usize,
	/// The number of private values, which only the prover knows.
	pub n_private: usize,
	/// The number of padding words between the private values and the end of the hidden segment.
	pub n_private_pad: usize,
	/// List of ZERO constraints that must be satisfied by the values vector.
	pub zero_constraints: Vec<ZeroConstraint>,
	/// List of AND constraints that must be satisfied by the values vector.
	pub and_constraints: Vec<AndConstraint>,
	/// List of IMUL constraints that must be satisfied by the values vector.
	pub imul_constraints: Vec<ImulConstraint>,
	/// List of BMUL constraints that must be satisfied by the values vector.
	pub bmul_constraints: Vec<BmulConstraint>,
}

impl ConstraintSystem {
	/// Serialization format version for compatibility checking
	pub const SERIALIZATION_VERSION: u32 = 7;

	/// The minimum number of words in the public segment.
	///
	/// [`Self::validate_shape`] rejects any system whose public segment is shorter than this.
	pub const MIN_WORDS_PER_SEGMENT: usize = 2;

	/// Returns the number of constants.
	pub const fn n_const(&self) -> usize {
		self.constants.len()
	}

	/// Returns the index of the first inout value.
	pub const fn offset_inout(&self) -> usize {
		self.n_const() + self.n_const_pad
	}

	/// Returns the number of words in the public segment: the constants and inout values,
	/// including padding up to the power-of-two segment length.
	pub const fn n_public_words(&self) -> usize {
		self.offset_inout() + self.n_inout + self.n_inout_pad
	}

	/// Returns the base-2 logarithm of the public segment length in words.
	///
	/// [`Self::validate_shape`] guarantees that the public segment length is a power of two.
	pub const fn log_public_words(&self) -> usize {
		self.n_public_words().trailing_zeros() as usize
	}

	/// Returns the number of words in the hidden segment: the private values and their padding.
	pub const fn n_hidden_words(&self) -> usize {
		self.n_private + self.n_private_pad
	}

	/// Returns the base-2 logarithm of the hidden segment length in words, rounded up to a
	/// power of two.
	///
	/// [`Self::validate_shape`] guarantees this is at least [`Self::log_public_words`].
	pub const fn log_witness_words(&self) -> usize {
		log2_ceil_usize(self.n_hidden_words())
	}

	/// Ensures that the value vector shape of this constraint system is well-formed.
	///
	/// Specifically checks that:
	///
	/// - the public segment (constants and inout values) is padded to the power of two.
	/// - the public segment is not less than the minimum size.
	/// - the hidden segment is at least as long as the public segment, so
	///   [`Self::log_witness_words`] is at least [`Self::log_public_words`].
	pub const fn validate_shape(&self) -> Result<(), ConstraintSystemError> {
		let pub_input_size = self.n_public_words();
		if !pub_input_size.is_power_of_two() {
			return Err(ConstraintSystemError::PublicInputPowerOfTwo);
		}
		if pub_input_size < Self::MIN_WORDS_PER_SEGMENT {
			return Err(ConstraintSystemError::PublicInputTooShort { pub_input_size });
		}

		if self.n_hidden_words() < pub_input_size {
			return Err(ConstraintSystemError::HiddenSegmentTooShort {
				public_len: pub_input_size,
				hidden_len: self.n_hidden_words(),
			});
		}

		Ok(())
	}

	/// Returns true if the given index points to an area that is considered to be padding.
	const fn is_padding(&self, index: ValueIndex) -> bool {
		let idx = index.0 as usize;

		// padding 1: between constants and inout section
		if idx >= self.n_const() && idx < self.offset_inout() {
			return true;
		}

		// padding 2: between the end of inout section and the end of the public segment
		let end_of_inout = self.offset_inout() + self.n_inout;
		if idx >= end_of_inout && idx < self.n_public_words() {
			return true;
		}

		// padding 3: between the last private value and the end of the hidden segment
		let end_of_private = self.n_public_words() + self.n_private;
		if idx >= end_of_private && idx < self.value_vec_len() {
			return true;
		}

		false
	}

	/// Ensures that this constraint system is well-formed and ready for proving.
	///
	/// Specifically checks that:
	///
	/// - the value vector shape is [valid][`Self::validate_shape`].
	/// - every [shifted value index][super::ShiftedValueIndex] is canonical.
	/// - referenced values indices are in the range.
	/// - constraints do not reference values in the padding area.
	/// - shifts amounts are valid.
	pub fn validate(&self) -> Result<(), ConstraintSystemError> {
		tracing::debug_span!("Validating constraint system");

		self.validate_shape()?;

		self.validate_constraints(
			&self.zero_constraints,
			ZeroConstraint::KIND,
			ZeroConstraint::OPERAND_NAMES,
		)?;
		self.validate_constraints(
			&self.and_constraints,
			AndConstraint::KIND,
			AndConstraint::OPERAND_NAMES,
		)?;
		self.validate_constraints(
			&self.imul_constraints,
			ImulConstraint::KIND,
			ImulConstraint::OPERAND_NAMES,
		)?;
		self.validate_constraints(
			&self.bmul_constraints,
			BmulConstraint::KIND,
			BmulConstraint::OPERAND_NAMES,
		)?;

		Ok(())
	}

	/// Checks every operand of every constraint of one kind, in storage order.
	fn validate_constraints<C: AsRef<[Operand; ARITY]>, const ARITY: usize>(
		&self,
		constraints: &[C],
		constraint_kind: ConstraintKind,
		operand_names: [&'static str; ARITY],
	) -> Result<(), ConstraintSystemError> {
		for (i, constraint) in constraints.iter().enumerate() {
			for (operand, name) in iter::zip(constraint.as_ref(), operand_names) {
				self.validate_operand(operand, constraint_kind, i, name)?;
			}
		}
		Ok(())
	}

	/// Checks that every term of an operand is canonical and references a value word.
	fn validate_operand(
		&self,
		operand: &Operand,
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
	) -> Result<(), ConstraintSystemError> {
		for term in operand {
			// check canonicity. SLL is the canonical form of the operand.
			if term.amount == 0 && term.shift_variant != ShiftVariant::Sll {
				return Err(ConstraintSystemError::NonCanonicalShift {
					constraint_kind,
					constraint_index,
					operand_name,
				});
			}
			// Half-word (*32) variants cap at 32, full-width at 64.
			let max_amount = term.shift_variant.max_amount();
			if usize::from(term.amount) >= max_amount {
				return Err(ConstraintSystemError::ShiftAmountTooLarge {
					constraint_kind,
					constraint_index,
					operand_name,
					shift_amount: term.amount as usize,
					max_amount,
				});
			}
			// Check if the value index is out of bounds.
			if term.value_index.0 as usize >= self.value_vec_len() {
				return Err(ConstraintSystemError::OutOfRangeValueIndex {
					constraint_kind,
					constraint_index,
					operand_name,
					value_index: term.value_index.0,
					total_len: self.value_vec_len(),
				});
			}
			// No value should refer to padding.
			if self.is_padding(term.value_index) {
				return Err(ConstraintSystemError::PaddingValueIndex {
					constraint_kind,
					constraint_index,
					operand_name,
				});
			}
		}
		Ok(())
	}

	/// Returns the number of ZERO constraints in the system.
	pub const fn n_zero_constraints(&self) -> usize {
		self.zero_constraints.len()
	}

	/// Returns the number of AND constraints in the system.
	pub const fn n_and_constraints(&self) -> usize {
		self.and_constraints.len()
	}

	/// Returns the number of IMUL  constraints in the system.
	pub const fn n_imul_constraints(&self) -> usize {
		self.imul_constraints.len()
	}

	/// Returns the number of BMUL constraints in the system.
	pub const fn n_bmul_constraints(&self) -> usize {
		self.bmul_constraints.len()
	}

	/// Returns the number of variables the Zero reduction runs over, or `None` when the system has
	/// no ZERO constraints.
	///
	/// This is `ceil(log2(n_zero_constraints))`, matching the zero-padded operand column the
	/// reduction consumes. As with [`Self::log_and_constraints`], the Zero reduction always runs: a
	/// system with no ZERO constraints still gets a single all-zero row, which the constraint
	/// vacuously satisfies. Such a system reduces over zero variables, so its callers read `None`
	/// as zero.
	pub const fn log_zero_constraints(&self) -> Option<usize> {
		match self.n_zero_constraints() {
			0 => None,
			n => Some(log2_ceil_usize(n)),
		}
	}

	/// Returns the number of variables the BitAnd reduction runs over, or `None` when the system
	/// has no AND constraints.
	///
	/// The reduction operates on operand columns with one row per AND constraint, zero-padded up
	/// to a power of two, so it has `ceil(log2(n_and_constraints))` variables. Unlike the two
	/// multiplication reductions, the BitAnd reduction always runs: a system with no AND
	/// constraints still gets a single all-zero row, which every constraint type satisfies. Such a
	/// system reduces over zero variables, so its callers read `None` as zero.
	pub const fn log_and_constraints(&self) -> Option<usize> {
		match self.n_and_constraints() {
			0 => None,
			n => Some(log2_ceil_usize(n)),
		}
	}

	/// Returns the number of variables the IntMul reduction runs over, or `None` when the system
	/// has no IMUL constraints.
	///
	/// This is `ceil(log2(n_imul_constraints))`, matching the zero-padded operand columns the
	/// reduction consumes. `None` is the skip signal: an empty IMUL set makes the prover and
	/// verifier skip the IntMul reduction entirely, rather than run it over a single dummy
	/// constraint.
	pub const fn log_imul_constraints(&self) -> Option<usize> {
		match self.n_imul_constraints() {
			0 => None,
			n => Some(log2_ceil_usize(n)),
		}
	}

	/// Returns the number of variables the BinMul reduction runs over, or `None` when the system
	/// has no BMUL constraints.
	///
	/// This is `ceil(log2(n_bmul_constraints))`, matching the zero-padded operand columns the
	/// reduction consumes. As with [`Self::log_imul_constraints`], `None` is the skip signal; both
	/// sides skip the BinMul reduction for an empty BMUL set.
	pub const fn log_bmul_constraints(&self) -> Option<usize> {
		match self.n_bmul_constraints() {
			0 => None,
			n => Some(log2_ceil_usize(n)),
		}
	}

	/// The total length of the [`ValueVec`] expected by this constraint system.
	pub const fn value_vec_len(&self) -> usize {
		self.n_public_words() + self.n_hidden_words()
	}

	/// Picks the inout values out of the public segment of a [`ValueVec`].
	///
	/// The inout values are the only part of that segment an instance chooses; the constants and
	/// the padding around them are fixed by this system. [`Self::public_segment`] is the inverse.
	///
	/// # Panics
	///
	/// If `public` is shorter than the public segment of this system.
	pub fn inout_values<'a>(&self, public: &'a [Word]) -> &'a [Word] {
		&public[self.offset_inout()..self.offset_inout() + self.n_inout]
	}

	/// Rebuilds the public segment of a [`ValueVec`] from the inout values of one instance.
	///
	/// Everything else in the segment already lives here, so the instance need not carry it:
	///
	/// ```text
	///     [ constants | const pad | inout | inout pad ]
	///       from self   zeros       given   zeros
	/// ```
	pub fn public_segment(&self, inout: &[Word]) -> Result<Vec<Word>, ConstraintSystemError> {
		if inout.len() != self.n_inout {
			return Err(ConstraintSystemError::IncorrectInoutLength {
				expected: self.n_inout,
				actual: inout.len(),
			});
		}

		// Each `resize` grows to the start of the next section, zero-filling the padding between.
		let mut public = Vec::with_capacity(self.n_public_words());
		public.extend_from_slice(&self.constants);
		public.resize(self.offset_inout(), Word::ZERO);
		public.extend_from_slice(inout);
		public.resize(self.n_public_words(), Word::ZERO);
		Ok(public)
	}
}

impl SerializeBytes for ConstraintSystem {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		Self::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

		self.constants.serialize(&mut write_buf)?;
		self.n_const_pad.serialize(&mut write_buf)?;
		self.n_inout.serialize(&mut write_buf)?;
		self.n_inout_pad.serialize(&mut write_buf)?;
		self.n_private.serialize(&mut write_buf)?;
		self.n_private_pad.serialize(&mut write_buf)?;
		self.zero_constraints.serialize(&mut write_buf)?;
		self.and_constraints.serialize(&mut write_buf)?;
		self.imul_constraints.serialize(&mut write_buf)?;
		self.bmul_constraints.serialize(write_buf)
	}
}

impl DeserializeBytes for ConstraintSystem {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let version = u32::deserialize(&mut read_buf)?;
		if version != Self::SERIALIZATION_VERSION {
			return Err(SerializationError::InvalidConstruction {
				name: "ConstraintSystem::version",
			});
		}

		let constants = Vec::<Word>::deserialize(&mut read_buf)?;
		let n_const_pad = usize::deserialize(&mut read_buf)?;
		let n_inout = usize::deserialize(&mut read_buf)?;
		let n_inout_pad = usize::deserialize(&mut read_buf)?;
		let n_private = usize::deserialize(&mut read_buf)?;
		let n_private_pad = usize::deserialize(&mut read_buf)?;
		let zero_constraints = Vec::<ZeroConstraint>::deserialize(&mut read_buf)?;
		let and_constraints = Vec::<AndConstraint>::deserialize(&mut read_buf)?;
		let imul_constraints = Vec::<ImulConstraint>::deserialize(&mut read_buf)?;
		let bmul_constraints = Vec::<BmulConstraint>::deserialize(read_buf)?;

		Ok(ConstraintSystem {
			constants,
			n_const_pad,
			n_inout,
			n_inout_pad,
			n_private,
			n_private_pad,
			zero_constraints,
			and_constraints,
			imul_constraints,
			bmul_constraints,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constraint_system::{ShiftedValueIndex, ValueVec, ValuesData, ValuesRef};

	/// A shape with one padding word after the constants and two after the inout values, so the
	/// public segment is 8 words, followed by a hidden segment of 6 values and 2 padding words.
	///
	///     [ c c c _ | i i _ _ ][ p p p p p p _ _ ]
	///       0 1 2 3   4 5 6 7   8 ...      13
	fn test_shape() -> ConstraintSystem {
		ConstraintSystem {
			constants: vec![
				Word::from_u64(1),
				Word::from_u64(42),
				Word::from_u64(0xDEADBEEF),
			],
			n_const_pad: 1,
			n_inout: 2,
			n_inout_pad: 2,
			n_private: 6,
			n_private_pad: 2,
			zero_constraints: vec![],
			and_constraints: vec![],
			imul_constraints: vec![],
			bmul_constraints: vec![],
		}
	}

	pub(crate) fn create_test_constraint_system() -> ConstraintSystem {
		ConstraintSystem {
			zero_constraints: vec![ZeroConstraint::plain([
				ValueIndex(0),
				ValueIndex(4),
				ValueIndex(8),
			])],
			and_constraints: vec![
				AndConstraint::plain_abc(
					vec![ValueIndex(0), ValueIndex(1)],
					vec![ValueIndex(2)],
					vec![ValueIndex(4), ValueIndex(5)],
				),
				AndConstraint::abc(
					vec![ShiftedValueIndex::sll(ValueIndex(0), 5)],
					vec![ShiftedValueIndex::srl(ValueIndex(1), 10)],
					vec![ShiftedValueIndex::sar(ValueIndex(2), 15)],
				),
			],
			imul_constraints: vec![ImulConstraint([
				vec![ShiftedValueIndex::plain(ValueIndex(0))],
				vec![ShiftedValueIndex::plain(ValueIndex(1))],
				vec![ShiftedValueIndex::plain(ValueIndex(2))],
				vec![ShiftedValueIndex::plain(ValueIndex(8))],
			])],
			bmul_constraints: vec![BmulConstraint([
				vec![ShiftedValueIndex::plain(ValueIndex(0))],
				vec![ShiftedValueIndex::plain(ValueIndex(1))],
				vec![ShiftedValueIndex::plain(ValueIndex(2))],
				vec![ShiftedValueIndex::plain(ValueIndex(4))],
				vec![ShiftedValueIndex::plain(ValueIndex(5))],
				vec![ShiftedValueIndex::sll(ValueIndex(0), 5)],
			])],
			..test_shape()
		}
	}

	#[test]
	fn test_constraint_system_serialization_round_trip() {
		let original = create_test_constraint_system();

		let mut buf = Vec::new();
		original.serialize(&mut buf).unwrap();

		let deserialized = ConstraintSystem::deserialize(&mut buf.as_slice()).unwrap();

		// Check version
		assert_eq!(ConstraintSystem::SERIALIZATION_VERSION, 7);

		// Check the value vector shape
		assert_eq!(original.constants, deserialized.constants);
		assert_eq!(original.n_const_pad, deserialized.n_const_pad);
		assert_eq!(original.n_inout, deserialized.n_inout);
		assert_eq!(original.n_inout_pad, deserialized.n_inout_pad);
		assert_eq!(original.n_private, deserialized.n_private);
		assert_eq!(original.n_private_pad, deserialized.n_private_pad);

		// Check zero_constraints
		assert_eq!(original.zero_constraints.len(), deserialized.zero_constraints.len());

		// Check and_constraints
		assert_eq!(original.and_constraints.len(), deserialized.and_constraints.len());

		// Check imul_constraints
		assert_eq!(original.imul_constraints.len(), deserialized.imul_constraints.len());

		// Check bmul_constraints
		assert_eq!(original.bmul_constraints.len(), deserialized.bmul_constraints.len());
	}

	#[test]
	fn test_constraint_system_version_mismatch() {
		// Create a buffer with wrong version
		let mut buf = Vec::new();
		999u32.serialize(&mut buf).unwrap(); // Wrong version

		let result = ConstraintSystem::deserialize(&mut buf.as_slice());
		assert!(result.is_err());
		match result.unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "ConstraintSystem::version");
			}
			_ => panic!("Expected InvalidConstruction error"),
		}
	}

	#[test]
	fn test_serialization_with_different_sources() {
		let original = create_test_constraint_system();

		// Test with Vec<u8> (memory buffer)
		let mut vec_buf = Vec::new();
		original.serialize(&mut vec_buf).unwrap();
		let deserialized1 = ConstraintSystem::deserialize(&mut vec_buf.as_slice()).unwrap();
		assert_eq!(original.constants.len(), deserialized1.constants.len());

		// Test with bytes::BytesMut (another common buffer type)
		let mut bytes_buf = bytes::BytesMut::new();
		original.serialize(&mut bytes_buf).unwrap();
		let deserialized2 = ConstraintSystem::deserialize(bytes_buf.freeze()).unwrap();
		assert_eq!(original.constants.len(), deserialized2.constants.len());
	}

	/// Helper function to create or update the reference binary file for version compatibility
	/// testing. This is not run automatically but can be used to regenerate the reference file
	/// when needed.
	#[test]
	#[ignore] // Use `cargo test -- --ignored create_reference_binary` to run this
	fn create_reference_binary_file() {
		let constraint_system = create_test_constraint_system();

		// Serialize to binary data
		let mut buf = Vec::new();
		constraint_system.serialize(&mut buf).unwrap();

		// Write to reference file.
		let test_data_path = std::path::Path::new("test_data/constraint_system_v7.bin");

		// Create directory if it doesn't exist
		if let Some(parent) = test_data_path.parent() {
			std::fs::create_dir_all(parent).unwrap();
		}

		std::fs::write(test_data_path, &buf).unwrap();

		println!("Created reference binary file at: {:?}", test_data_path);
		println!("Binary data length: {} bytes", buf.len());
	}

	/// Test deserialization from a reference binary file to ensure version compatibility.
	/// This test will fail if breaking changes are made without incrementing the version.
	#[test]
	fn test_deserialize_from_reference_binary_file() {
		// The v7 format adds the ZERO constraints ahead of the AND constraints. Older files are no
		// longer compatible.
		let binary_data = include_bytes!("../../test_data/constraint_system_v7.bin");

		let deserialized = ConstraintSystem::deserialize(&mut binary_data.as_slice()).unwrap();

		assert_eq!(deserialized.n_const(), 3);
		assert_eq!(deserialized.n_const_pad, 1);
		assert_eq!(deserialized.n_inout, 2);
		assert_eq!(deserialized.n_inout_pad, 2);
		assert_eq!(deserialized.n_private, 6);
		assert_eq!(deserialized.n_private_pad, 2);

		assert_eq!(deserialized.constants[0].as_u64(), 1);
		assert_eq!(deserialized.constants[1].as_u64(), 42);
		assert_eq!(deserialized.constants[2].as_u64(), 0xDEADBEEF);

		assert_eq!(deserialized.zero_constraints.len(), 1);
		assert_eq!(deserialized.and_constraints.len(), 2);
		assert_eq!(deserialized.imul_constraints.len(), 1);
		assert_eq!(deserialized.bmul_constraints.len(), 1);

		// Verify that the version is what we expect
		// This is implicitly checked during deserialization, but we can also verify
		// the file starts with the correct version bytes
		let version_bytes = &binary_data[0..4]; // First 4 bytes should be version
		let expected_version_bytes = 7u32.to_le_bytes(); // Version 7 in little-endian
		assert_eq!(
			version_bytes, expected_version_bytes,
			"Binary file version mismatch. If you made breaking changes, increment ConstraintSystem::SERIALIZATION_VERSION"
		);
	}

	#[test]
	fn test_log_witness_words() {
		let cs = |n_private: usize, n_private_pad: usize| ConstraintSystem {
			n_private,
			n_private_pad,
			..test_shape()
		};
		// Typical: more hidden words than public words.
		assert_eq!(cs(60, 4).log_witness_words(), 6);
		// Exact power-of-two hidden count.
		assert_eq!(cs(30, 2).log_witness_words(), 5);
	}

	#[test]
	fn test_validate_shape_rejects_short_hidden_segment() {
		// The hidden segment (4 words) is shorter than the public segment (8 words).
		let cs = ConstraintSystem {
			n_private: 4,
			n_private_pad: 0,
			..test_shape()
		};
		assert!(matches!(
			cs.validate_shape(),
			Err(ConstraintSystemError::HiddenSegmentTooShort {
				public_len: 8,
				hidden_len: 4,
			})
		));
	}

	#[test]
	fn test_validate_shape_rejects_non_power_of_two_public_segment() {
		// Nine public words: the segment must be padded to a power of two.
		let cs = ConstraintSystem {
			n_inout_pad: 3,
			..test_shape()
		};
		assert!(matches!(cs.validate_shape(), Err(ConstraintSystemError::PublicInputPowerOfTwo)));
	}

	#[test]
	fn test_is_padding_covers_every_gap() {
		//     [ c c | _ _ _ _ _ _ | i i | _ _ _ _ _ _ ][ p p p p p p p p | _ _ _ _ _ _ _ _ ]
		//       0 1   2 ...     7   8 9   10 ...   15    16 ...      23   24 ...       31
		let cs = ConstraintSystem {
			constants: vec![Word::ONE, Word::ALL_ONE],
			n_const_pad: 6,
			n_inout: 2,
			n_inout_pad: 6,
			n_private: 8,
			n_private_pad: 8,
			..test_shape()
		};
		cs.validate_shape().unwrap();

		let padding = (0..cs.value_vec_len())
			.filter(|&i| cs.is_padding(ValueIndex(i as u32)))
			.collect::<Vec<_>>();
		let expected = (2..8).chain(10..16).chain(24..32).collect::<Vec<_>>();
		assert_eq!(padding, expected);
	}

	#[test]
	fn test_is_padding_gapless_shape_has_no_padding() {
		//     [ c c c c | i i i i ][ p p p p p p p p ]
		let cs = ConstraintSystem {
			constants: vec![Word::ONE; 4],
			n_const_pad: 0,
			n_inout: 4,
			n_inout_pad: 0,
			n_private: 8,
			n_private_pad: 0,
			..test_shape()
		};
		cs.validate_shape().unwrap();

		for i in 0..cs.value_vec_len() {
			assert!(!cs.is_padding(ValueIndex(i as u32)), "index {i} should not be padding");
		}
	}

	#[test]
	fn test_validate_rejects_padding_references() {
		let mut cs = test_shape();

		// Index 3 is the padding word between the constants and the inout values.
		cs.and_constraints.push(AndConstraint::plain_abc(
			vec![ValueIndex(0)], // valid constant
			vec![ValueIndex(3)], // PADDING!
			vec![ValueIndex(8)], // valid private value
		));

		let result = cs.validate();
		assert!(result.is_err(), "Should reject constraint referencing padding");

		match result.unwrap_err() {
			ConstraintSystemError::PaddingValueIndex {
				constraint_kind, ..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::And);
			}
			other => panic!("Expected PaddingValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_accepts_non_padding_references() {
		let mut cs = test_shape();

		// Add constraints that only reference valid non-padding indices
		cs.and_constraints.push(AndConstraint::plain_abc(
			vec![ValueIndex(0), ValueIndex(1)], // constants
			vec![ValueIndex(4), ValueIndex(5)], // inout
			vec![ValueIndex(8), ValueIndex(9)], // private
		));

		cs.imul_constraints.push(ImulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex(10))], // a
			vec![ShiftedValueIndex::plain(ValueIndex(11))], // b
			vec![ShiftedValueIndex::plain(ValueIndex(12))], // lo
			vec![ShiftedValueIndex::plain(ValueIndex(13))], // hi
		]));

		let result = cs.validate();
		assert!(
			result.is_ok(),
			"Should accept constraints with only valid references: {:?}",
			result
		);
	}

	#[test]
	fn test_validate_keeps_true_constraint_counts() {
		let cs = create_test_constraint_system();
		let (n_zero, n_and, n_imul, n_bmul) = (
			cs.n_zero_constraints(),
			cs.n_and_constraints(),
			cs.n_imul_constraints(),
			cs.n_bmul_constraints(),
		);
		cs.validate().unwrap();
		assert_eq!(cs.n_zero_constraints(), n_zero);
		assert_eq!(cs.n_and_constraints(), n_and);
		assert_eq!(cs.n_imul_constraints(), n_imul);
		assert_eq!(cs.n_bmul_constraints(), n_bmul);
	}

	#[test]
	fn test_validate_rejects_out_of_range_in_zero_constraint() {
		let mut cs = test_shape();

		cs.zero_constraints
			.push(ZeroConstraint::plain([ValueIndex(0), ValueIndex(100)]));

		match cs.validate().unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				total_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::Zero);
				assert_eq!(operand_name, "val");
				assert_eq!(value_index, 100);
				assert_eq!(total_len, 16);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_log_constraint_counts_round_up() {
		let mut cs = test_shape();
		// An empty set reports `None`; the BitAnd reduction reads that as its single all-zero
		// padding row, i.e. zero variables.
		assert_eq!(cs.log_and_constraints(), None);
		assert_eq!(cs.log_zero_constraints(), None);

		cs.zero_constraints = vec![ZeroConstraint::plain([ValueIndex(0), ValueIndex(8)]); 3];
		assert_eq!(cs.log_zero_constraints(), Some(2));

		let and =
			AndConstraint::plain_abc(vec![ValueIndex(0)], vec![ValueIndex(4)], vec![ValueIndex(8)]);
		cs.and_constraints = vec![and; 3];
		assert_eq!(cs.log_and_constraints(), Some(2));
		cs.and_constraints.push(cs.and_constraints[0].clone());
		assert_eq!(cs.log_and_constraints(), Some(2));
	}

	#[test]
	fn test_validate_rejects_out_of_range_indices() {
		let mut cs = test_shape();

		// Add AND constraint that references an out-of-range index
		cs.and_constraints.push(AndConstraint::plain_abc(
			vec![ValueIndex(0)],  // valid constant
			vec![ValueIndex(16)], // OUT OF RANGE! (total_len is 16, so max valid index is 15)
			vec![ValueIndex(8)],  // valid private value
		));

		let result = cs.validate();
		assert!(result.is_err(), "Should reject constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				total_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::And);
				assert_eq!(operand_name, "b");
				assert_eq!(value_index, 16);
				assert_eq!(total_len, 16);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_rejects_out_of_range_in_imul_constraint() {
		let mut cs = test_shape();

		// Add IMUL constraint with out-of-range index in 'hi' operand
		cs.imul_constraints.push(ImulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex(0))], // a: valid
			vec![ShiftedValueIndex::plain(ValueIndex(1))], // b: valid
			vec![ShiftedValueIndex::plain(ValueIndex(8))], // lo: valid
			vec![ShiftedValueIndex::plain(ValueIndex(100))], // hi: WAY out of range!
		]));

		let result = cs.validate();
		assert!(result.is_err(), "Should reject IMUL constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				total_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::Imul);
				assert_eq!(operand_name, "hi");
				assert_eq!(value_index, 100);
				assert_eq!(total_len, 16);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_rejects_out_of_range_in_bmul_constraint() {
		let mut cs = test_shape();

		// Add BMUL constraint with out-of-range index in 'c_hi' operand
		cs.bmul_constraints.push(BmulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex(0))], // a_lo: valid const
			vec![ShiftedValueIndex::plain(ValueIndex(4))], // a_hi: valid inout
			vec![ShiftedValueIndex::plain(ValueIndex(5))], // b_lo: valid inout
			vec![ShiftedValueIndex::plain(ValueIndex(8))], // b_hi: valid private
			vec![ShiftedValueIndex::plain(ValueIndex(9))], // c_lo: valid private
			vec![ShiftedValueIndex::plain(ValueIndex(100))], // c_hi: WAY out of range!
		]));

		let result = cs.validate();
		assert!(result.is_err(), "Should reject BMUL constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				total_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::Bmul);
				assert_eq!(operand_name, "c_hi");
				assert_eq!(value_index, 100);
				assert_eq!(total_len, 16);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_checks_out_of_range_before_padding() {
		// This test verifies that out-of-range checking happens before padding checking
		// by using an index that is both out-of-range AND would be in a padding area if it were
		// valid
		let mut cs = test_shape();

		// Index 19 is out of range (>= 16)
		// If it were in range, indices 3 and 6-7 would be padding
		cs.and_constraints.push(AndConstraint::plain_abc(
			vec![ValueIndex(0)],
			vec![ValueIndex(19)], // out of range
			vec![ValueIndex(8)],
		));

		let result = cs.validate();
		assert!(result.is_err());

		// Should get OutOfRangeValueIndex, not PaddingValueIndex
		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex { .. } => {
				// Good, out-of-range was detected first
			}
			other => panic!(
				"Expected OutOfRangeValueIndex to be detected before padding check, got: {:?}",
				other
			),
		}
	}

	#[test]
	fn test_validate_rejects_half_word_shift_amount_out_of_range() {
		let mut cs = test_shape();

		// A half-word (*32) shift may only use amounts < 32.
		// 32 is out of range even though it is below the full-width bound of 64.
		cs.and_constraints.push(AndConstraint::abc(
			vec![ShiftedValueIndex {
				value_index: ValueIndex(0),
				shift_variant: ShiftVariant::Sll32,
				amount: 32,
			}],
			vec![ShiftedValueIndex::plain(ValueIndex(8))],
			vec![ShiftedValueIndex::plain(ValueIndex(8))],
		));

		match cs.validate().unwrap_err() {
			ConstraintSystemError::ShiftAmountTooLarge {
				constraint_kind,
				constraint_index,
				operand_name,
				shift_amount,
				max_amount,
			} => {
				assert_eq!(constraint_kind, ConstraintKind::And);
				assert_eq!(constraint_index, 0);
				assert_eq!(operand_name, "a");
				assert_eq!(shift_amount, 32);
				assert_eq!(max_amount, 32);
			}
			other => panic!("Expected ShiftAmountTooLarge, got: {:?}", other),
		}
	}

	#[test]
	fn public_segment_round_trips_through_its_inout_values() {
		// Invariant: dropping everything but the inout values loses nothing.
		//
		// Fixture state: the test shape, whose public segment is
		//
		//     [ 1 42 0xDEADBEEF _ | i0 i1 _ _ ]
		let cs = test_shape();
		let inout = [Word::from_u64(0x1111), Word::from_u64(0x2222)];

		let public = cs.public_segment(&inout).unwrap();
		assert_eq!(
			public,
			[
				Word::from_u64(1),
				Word::from_u64(42),
				Word::from_u64(0xDEADBEEF),
				Word::ZERO,
				Word::from_u64(0x1111),
				Word::from_u64(0x2222),
				Word::ZERO,
				Word::ZERO,
			]
		);
		assert_eq!(cs.inout_values(&public), &inout);
	}

	#[test]
	fn public_segment_rejects_the_wrong_number_of_inout_values() {
		// Invariant: a segment is only rebuilt from inout values that fit the shape.
		//
		// Fixture state: the test shape, which declares two inout values, given one.
		let cs = test_shape();

		match cs.public_segment(&[Word::ONE]).unwrap_err() {
			ConstraintSystemError::IncorrectInoutLength { expected, actual } => {
				assert_eq!(expected, 2);
				assert_eq!(actual, 1);
			}
			other => panic!("Expected IncorrectInoutLength, got: {other:?}"),
		}
	}

	#[test]
	fn test_roundtrip_cs_and_witnesses_reconstruct_valuevec() {
		let cs = test_shape();

		// Build a value vector: the constants and padding the shape fixes, an instance's inout
		// values, and a deterministic pattern for the hidden segment.
		let inout = (0..cs.n_inout)
			.map(|i| Word::from_u64(0xA5A5_5A5A ^ (i as u64 * 0x9E37_79B9)))
			.collect::<Vec<_>>();
		let public = cs.public_segment(&inout).unwrap();
		let private = (0..cs.n_hidden_words())
			.map(|i| Word::from_u64(0x5A5A_A5A5 ^ (i as u64 * 0x9E37_79B9)))
			.collect::<Vec<_>>();
		let values = ValueVec::new_from_data(&public, &private);

		// Serialize all three artifacts. The public one carries the inout values alone, because
		// the constants and the padding around them are already in the constraint system.
		let mut buf_cs = Vec::new();
		cs.serialize(&mut buf_cs).unwrap();

		let mut buf_inout = Vec::new();
		ValuesRef::new(cs.inout_values(values.public()))
			.serialize(&mut buf_inout)
			.unwrap();

		let mut buf_non_pub = Vec::new();
		ValuesRef::new(values.non_public())
			.serialize(&mut buf_non_pub)
			.unwrap();

		// Deserialize everything back
		let cs2 = ConstraintSystem::deserialize(&mut buf_cs.as_slice()).unwrap();
		let inout2 = ValuesData::deserialize(&mut buf_inout.as_slice()).unwrap();
		let non_pub2 = ValuesData::deserialize(&mut buf_non_pub.as_slice()).unwrap();
		assert_eq!(cs2.n_inout, inout2.len());
		assert_eq!(cs2.n_hidden_words(), non_pub2.len());

		// Reconstruct ValueVec from deserialized pieces
		let reconstructed =
			ValueVec::new_from_data(&cs2.public_segment(&inout2).unwrap(), &non_pub2);

		assert_eq!(reconstructed.combined_witness(), values.combined_witness());
	}
}
