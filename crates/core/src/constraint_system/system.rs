// Copyright 2025 Irreducible Inc.
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::{AndConstraint, MulConstraint, Operand, ShiftVariant, ValueVec, ValueVecLayout};
use crate::{error::ConstraintSystemError, word::Word};

/// The ConstraintSystem is the core data structure in Binius64 that defines the computational
/// constraints to be proven in zero-knowledge. It represents a system of equations over 64-bit
/// words that must be satisfied by a valid values vector [`ValueVec`].
///
/// # Clone
///
/// While this type is cloneable it may be expensive to do so since the constraint systems often
/// can have millions of constraints.
#[derive(Debug, Clone)]
pub struct ConstraintSystem {
	/// Description of the value vector layout expected by this constraint system.
	pub value_vec_layout: ValueVecLayout,
	/// The constants that this constraint system defines.
	///
	/// Those constants will be going to be available for constraints in the value vector. Those
	/// are known to both prover and verifier.
	pub constants: Vec<Word>,
	/// List of AND constraints that must be satisfied by the values vector.
	pub and_constraints: Vec<AndConstraint>,
	/// List of MUL constraints that must be satisfied by the values vector.
	pub mul_constraints: Vec<MulConstraint>,
}

impl ConstraintSystem {
	/// Serialization format version for compatibility checking
	pub const SERIALIZATION_VERSION: u32 = 2;
}

impl ConstraintSystem {
	/// Creates a new constraint system.
	pub fn new(
		constants: Vec<Word>,
		value_vec_layout: ValueVecLayout,
		and_constraints: Vec<AndConstraint>,
		mul_constraints: Vec<MulConstraint>,
	) -> Self {
		assert_eq!(constants.len(), value_vec_layout.n_const);
		ConstraintSystem {
			constants,
			value_vec_layout,
			and_constraints,
			mul_constraints,
		}
	}

	/// Ensures that this constraint system is well-formed and ready for proving.
	///
	/// Specifically checks that:
	///
	/// - the value vec layout is [valid][`ValueVecLayout::validate`].
	/// - every [shifted value index][super::ShiftedValueIndex] is canonical.
	/// - referenced values indices are in the range.
	/// - constraints do not reference values in the padding area.
	/// - shifts amounts are valid.
	pub fn validate(&self) -> Result<(), ConstraintSystemError> {
		tracing::debug_span!("Validating constraint system");

		// Validate the value vector layout
		self.value_vec_layout.validate()?;

		for i in 0..self.and_constraints.len() {
			validate_operand(&self.and_constraints[i].a, &self.value_vec_layout, "and", i, "a")?;
			validate_operand(&self.and_constraints[i].b, &self.value_vec_layout, "and", i, "b")?;
			validate_operand(&self.and_constraints[i].c, &self.value_vec_layout, "and", i, "c")?;
		}
		for i in 0..self.mul_constraints.len() {
			validate_operand(&self.mul_constraints[i].a, &self.value_vec_layout, "mul", i, "a")?;
			validate_operand(&self.mul_constraints[i].b, &self.value_vec_layout, "mul", i, "b")?;
			validate_operand(&self.mul_constraints[i].lo, &self.value_vec_layout, "mul", i, "lo")?;
			validate_operand(&self.mul_constraints[i].hi, &self.value_vec_layout, "mul", i, "hi")?;
		}

		return Ok(());

		fn validate_operand(
			operand: &Operand,
			value_vec_layout: &ValueVecLayout,
			constraint_type: &'static str,
			constraint_index: usize,
			operand_name: &'static str,
		) -> Result<(), ConstraintSystemError> {
			for term in operand {
				// check canonicity. SLL is the canonical form of the operand.
				if term.amount == 0 && term.shift_variant != ShiftVariant::Sll {
					return Err(ConstraintSystemError::NonCanonicalShift {
						constraint_type,
						constraint_index,
						operand_name,
					});
				}
				if term.amount >= 64 {
					return Err(ConstraintSystemError::ShiftAmountTooLarge {
						constraint_type,
						constraint_index,
						operand_name,
						shift_amount: term.amount as usize,
					});
				}
				// Check if the value index is out of bounds.
				if value_vec_layout.is_committed_oob(term.value_index) {
					return Err(ConstraintSystemError::OutOfRangeValueIndex {
						constraint_type,
						constraint_index,
						operand_name,
						value_index: term.value_index.0,
						total_len: value_vec_layout.committed_total_len,
					});
				}
				// No value should refer to padding.
				if value_vec_layout.is_padding(term.value_index) {
					return Err(ConstraintSystemError::PaddingValueIndex {
						constraint_type,
						constraint_index,
						operand_name,
					});
				}
			}
			Ok(())
		}
	}

	/// [Validates][`Self::validate`] and prepares this constraint system for proving/verifying.
	///
	/// This function performs the following:
	/// 1. Validates the value vector layout (including public input checks)
	/// 2. Validates the constraints.
	/// 3. Pads the AND and MUL constraints to the next po2 size
	pub fn validate_and_prepare(&mut self) -> Result<(), ConstraintSystemError> {
		self.validate()?;

		// Require all constraint types to have a power-of-two count. An empty MUL constraint set is
		// kept at zero (rather than padded to a single dummy constraint) so the prover and verifier
		// can skip the IntMul reduction entirely — see `IOPProver::prove` / `IOPVerifier::verify`.
		let and_target_size = self.and_constraints.len().next_power_of_two();
		let mul_target_size = if self.mul_constraints.is_empty() {
			0
		} else {
			self.mul_constraints.len().next_power_of_two()
		};

		self.and_constraints
			.resize_with(and_target_size, AndConstraint::default);
		self.mul_constraints
			.resize_with(mul_target_size, MulConstraint::default);

		Ok(())
	}

	#[cfg(test)]
	fn add_and_constraint(&mut self, and_constraint: AndConstraint) {
		self.and_constraints.push(and_constraint);
	}

	#[cfg(test)]
	fn add_mul_constraint(&mut self, mul_constraint: MulConstraint) {
		self.mul_constraints.push(mul_constraint);
	}

	/// Returns the number of AND constraints in the system.
	pub const fn n_and_constraints(&self) -> usize {
		self.and_constraints.len()
	}

	/// Returns the number of MUL  constraints in the system.
	pub const fn n_mul_constraints(&self) -> usize {
		self.mul_constraints.len()
	}

	/// The total length of the [`ValueVec`] expected by this constraint system.
	pub const fn value_vec_len(&self) -> usize {
		self.value_vec_layout.committed_total_len
	}

	/// Create a new [`ValueVec`] with the size expected by this constraint system.
	pub fn new_value_vec(&self) -> ValueVec {
		ValueVec::new(self.value_vec_layout.clone())
	}
}

impl SerializeBytes for ConstraintSystem {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		Self::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

		self.value_vec_layout.serialize(&mut write_buf)?;
		self.constants.serialize(&mut write_buf)?;
		self.and_constraints.serialize(&mut write_buf)?;
		self.mul_constraints.serialize(write_buf)
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

		let value_vec_layout = ValueVecLayout::deserialize(&mut read_buf)?;
		let constants = Vec::<Word>::deserialize(&mut read_buf)?;
		let and_constraints = Vec::<AndConstraint>::deserialize(&mut read_buf)?;
		let mul_constraints = Vec::<MulConstraint>::deserialize(read_buf)?;

		if constants.len() != value_vec_layout.n_const {
			return Err(SerializationError::InvalidConstruction {
				name: "ConstraintSystem::constants",
			});
		}

		Ok(ConstraintSystem {
			value_vec_layout,
			constants,
			and_constraints,
			mul_constraints,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constraint_system::{ShiftedValueIndex, ValueIndex, ValuesData};

	pub(crate) fn create_test_constraint_system() -> ConstraintSystem {
		let constants = vec![
			Word::from_u64(1),
			Word::from_u64(42),
			Word::from_u64(0xDEADBEEF),
		];

		let value_vec_layout = ValueVecLayout {
			n_const: 3,
			n_inout: 2,
			n_witness: 10,
			n_internal: 3,
			offset_inout: 4,         // Must be power of 2 and >= n_const
			offset_witness: 8,       // Must be power of 2 and >= offset_inout + n_inout
			committed_total_len: 16, // Must be power of 2 and >= offset_witness + n_witness
			n_scratch: 0,
		};

		let and_constraints = vec![
			AndConstraint::plain_abc(
				vec![ValueIndex(0), ValueIndex(1)],
				vec![ValueIndex(2)],
				vec![ValueIndex(3), ValueIndex(4)],
			),
			AndConstraint::abc(
				vec![ShiftedValueIndex::sll(ValueIndex(0), 5)],
				vec![ShiftedValueIndex::srl(ValueIndex(1), 10)],
				vec![ShiftedValueIndex::sar(ValueIndex(2), 15)],
			),
		];

		let mul_constraints = vec![MulConstraint {
			a: vec![ShiftedValueIndex::plain(ValueIndex(0))],
			b: vec![ShiftedValueIndex::plain(ValueIndex(1))],
			hi: vec![ShiftedValueIndex::plain(ValueIndex(2))],
			lo: vec![ShiftedValueIndex::plain(ValueIndex(3))],
		}];

		ConstraintSystem::new(constants, value_vec_layout, and_constraints, mul_constraints)
	}

	#[test]
	fn test_constraint_system_serialization_round_trip() {
		let original = create_test_constraint_system();

		let mut buf = Vec::new();
		original.serialize(&mut buf).unwrap();

		let deserialized = ConstraintSystem::deserialize(&mut buf.as_slice()).unwrap();

		// Check version
		assert_eq!(ConstraintSystem::SERIALIZATION_VERSION, 2);

		// Check value_vec_layout
		assert_eq!(original.value_vec_layout, deserialized.value_vec_layout);

		// Check constants
		assert_eq!(original.constants.len(), deserialized.constants.len());
		for (orig, deser) in original.constants.iter().zip(deserialized.constants.iter()) {
			assert_eq!(orig, deser);
		}

		// Check and_constraints
		assert_eq!(original.and_constraints.len(), deserialized.and_constraints.len());

		// Check mul_constraints
		assert_eq!(original.mul_constraints.len(), deserialized.mul_constraints.len());
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
	fn test_constraint_system_constants_length_mismatch() {
		// Create valid components but with mismatched constants length
		let value_vec_layout = ValueVecLayout {
			n_const: 5, // Expect 5 constants
			n_inout: 2,
			n_witness: 10,
			n_internal: 3,
			offset_inout: 8,
			offset_witness: 16,
			committed_total_len: 32,
			n_scratch: 0,
		};

		let constants = vec![Word::from_u64(1), Word::from_u64(2)]; // Only 2 constants
		let and_constraints: Vec<AndConstraint> = vec![];
		let mul_constraints: Vec<MulConstraint> = vec![];

		// Serialize components manually
		let mut buf = Vec::new();
		ConstraintSystem::SERIALIZATION_VERSION
			.serialize(&mut buf)
			.unwrap();
		value_vec_layout.serialize(&mut buf).unwrap();
		constants.serialize(&mut buf).unwrap();
		and_constraints.serialize(&mut buf).unwrap();
		mul_constraints.serialize(&mut buf).unwrap();

		let result = ConstraintSystem::deserialize(&mut buf.as_slice());
		assert!(result.is_err());
		match result.unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "ConstraintSystem::constants");
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
		let test_data_path = std::path::Path::new("test_data/constraint_system_v2.bin");

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
		// We now have v2 format with n_scratch field
		// The v1 file is no longer compatible, so we test with v2
		let binary_data = include_bytes!("../../test_data/constraint_system_v2.bin");

		let deserialized = ConstraintSystem::deserialize(&mut binary_data.as_slice()).unwrap();

		assert_eq!(deserialized.value_vec_layout.n_const, 3);
		assert_eq!(deserialized.value_vec_layout.n_inout, 2);
		assert_eq!(deserialized.value_vec_layout.n_witness, 10);
		assert_eq!(deserialized.value_vec_layout.n_internal, 3);
		assert_eq!(deserialized.value_vec_layout.offset_inout, 4);
		assert_eq!(deserialized.value_vec_layout.offset_witness, 8);
		assert_eq!(deserialized.value_vec_layout.committed_total_len, 16);
		assert_eq!(deserialized.value_vec_layout.n_scratch, 0);

		assert_eq!(deserialized.constants.len(), 3);
		assert_eq!(deserialized.constants[0].as_u64(), 1);
		assert_eq!(deserialized.constants[1].as_u64(), 42);
		assert_eq!(deserialized.constants[2].as_u64(), 0xDEADBEEF);

		assert_eq!(deserialized.and_constraints.len(), 2);
		assert_eq!(deserialized.mul_constraints.len(), 1);

		// Verify that the version is what we expect
		// This is implicitly checked during deserialization, but we can also verify
		// the file starts with the correct version bytes
		let version_bytes = &binary_data[0..4]; // First 4 bytes should be version
		let expected_version_bytes = 2u32.to_le_bytes(); // Version 2 in little-endian
		assert_eq!(
			version_bytes, expected_version_bytes,
			"Binary file version mismatch. If you made breaking changes, increment ConstraintSystem::SERIALIZATION_VERSION"
		);
	}

	#[test]
	fn test_validate_rejects_padding_references() {
		let mut cs = ConstraintSystem::new(
			vec![Word::from_u64(1)],
			ValueVecLayout {
				n_const: 1,
				n_inout: 1,
				n_witness: 2,
				n_internal: 2,
				offset_inout: 4,
				offset_witness: 8,
				committed_total_len: 16,
				n_scratch: 0,
			},
			vec![],
			vec![],
		);

		// Add constraint that references padding (index 2 is padding between const and inout)
		cs.add_and_constraint(AndConstraint::plain_abc(
			vec![ValueIndex(0)], // valid constant
			vec![ValueIndex(2)], // PADDING!
			vec![ValueIndex(8)], // valid witness
		));

		let result = cs.validate_and_prepare();
		assert!(result.is_err(), "Should reject constraint referencing padding");

		match result.unwrap_err() {
			ConstraintSystemError::PaddingValueIndex {
				constraint_type, ..
			} => {
				assert_eq!(constraint_type, "and");
			}
			other => panic!("Expected PaddingValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_accepts_non_padding_references() {
		let mut cs = ConstraintSystem::new(
			vec![Word::from_u64(1), Word::from_u64(2)],
			ValueVecLayout {
				n_const: 2,
				n_inout: 2,
				n_witness: 4,
				n_internal: 4,
				offset_inout: 2,
				offset_witness: 4,
				committed_total_len: 16,
				n_scratch: 0,
			},
			vec![],
			vec![],
		);

		// Add constraint that only references valid non-padding indices
		cs.add_and_constraint(AndConstraint::plain_abc(
			vec![ValueIndex(0), ValueIndex(1)], // constants
			vec![ValueIndex(2), ValueIndex(3)], // inout
			vec![ValueIndex(4), ValueIndex(5)], // witness
		));

		cs.add_mul_constraint(MulConstraint {
			a: vec![ShiftedValueIndex::plain(ValueIndex(6))], // witness
			b: vec![ShiftedValueIndex::plain(ValueIndex(7))], // witness
			hi: vec![ShiftedValueIndex::plain(ValueIndex(8))], // internal
			lo: vec![ShiftedValueIndex::plain(ValueIndex(9))], // internal
		});

		let result = cs.validate_and_prepare();
		assert!(
			result.is_ok(),
			"Should accept constraints with only valid references: {:?}",
			result
		);
	}

	#[test]
	fn test_validate_rejects_out_of_range_indices() {
		let mut cs = ConstraintSystem::new(
			vec![Word::from_u64(1)],
			ValueVecLayout {
				n_const: 1,
				n_inout: 1,
				n_witness: 2,
				n_internal: 2,
				offset_inout: 4,
				offset_witness: 8,
				committed_total_len: 16,
				n_scratch: 0,
			},
			vec![],
			vec![],
		);

		// Add AND constraint that references an out-of-range index
		cs.add_and_constraint(AndConstraint::plain_abc(
			vec![ValueIndex(0)],  // valid constant
			vec![ValueIndex(16)], // OUT OF RANGE! (total_len is 16, so max valid index is 15)
			vec![ValueIndex(8)],  // valid witness
		));

		let result = cs.validate_and_prepare();
		assert!(result.is_err(), "Should reject constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_type,
				operand_name,
				value_index,
				total_len,
				..
			} => {
				assert_eq!(constraint_type, "and");
				assert_eq!(operand_name, "b");
				assert_eq!(value_index, 16);
				assert_eq!(total_len, 16);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_rejects_out_of_range_in_mul_constraint() {
		let mut cs = ConstraintSystem::new(
			vec![Word::from_u64(1), Word::from_u64(2)],
			ValueVecLayout {
				n_const: 2,
				n_inout: 2,
				n_witness: 4,
				n_internal: 4,
				offset_inout: 2,
				offset_witness: 4,
				committed_total_len: 16,
				n_scratch: 0,
			},
			vec![],
			vec![],
		);

		// Add MUL constraint with out-of-range index in 'hi' operand
		cs.add_mul_constraint(MulConstraint {
			a: vec![ShiftedValueIndex::plain(ValueIndex(0))], // valid
			b: vec![ShiftedValueIndex::plain(ValueIndex(1))], // valid
			hi: vec![ShiftedValueIndex::plain(ValueIndex(100))], // WAY out of range!
			lo: vec![ShiftedValueIndex::plain(ValueIndex(3))], // valid
		});

		let result = cs.validate_and_prepare();
		assert!(result.is_err(), "Should reject MUL constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_type,
				operand_name,
				value_index,
				total_len,
				..
			} => {
				assert_eq!(constraint_type, "mul");
				assert_eq!(operand_name, "hi");
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
		let mut cs = ConstraintSystem::new(
			vec![Word::from_u64(1)],
			ValueVecLayout {
				n_const: 1,
				n_inout: 1,
				n_witness: 2,
				n_internal: 2,
				offset_inout: 4,
				offset_witness: 8,
				committed_total_len: 16,
				n_scratch: 0,
			},
			vec![],
			vec![],
		);

		// Index 20 is out of range (>= 16)
		// If it were in range, indices 2-3 and 6-7 would be padding
		cs.add_and_constraint(AndConstraint::plain_abc(
			vec![ValueIndex(0)],
			vec![ValueIndex(20)], // out of range
			vec![ValueIndex(8)],
		));

		let result = cs.validate_and_prepare();
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
	fn test_roundtrip_cs_and_witnesses_reconstruct_valuevec_with_scratch() {
		// Layout with non-zero scratch. Public = 8, total committed = 16, scratch = 5
		let layout = ValueVecLayout {
			n_const: 2,
			n_inout: 3,
			n_witness: 4,
			n_internal: 3,
			offset_inout: 4,   // >= n_const and power of two
			offset_witness: 8, // >= offset_inout + n_inout and power of two
			committed_total_len: 16,
			n_scratch: 5, // non-zero scratch
		};

		let constants = vec![Word::from_u64(11), Word::from_u64(22)];
		let cs = ConstraintSystem::new(constants, layout.clone(), vec![], vec![]);

		// Build a ValueVec and fill both committed and scratch with non-zero data
		let mut values = cs.new_value_vec();
		let full_len = layout.committed_total_len + layout.n_scratch;
		for i in 0..full_len {
			// Deterministic pattern
			let val = Word::from_u64(0xA5A5_5A5A ^ (i as u64 * 0x9E37_79B9));
			values.set(i, val);
		}

		// Split into public and non-public witnesses and serialize all artifacts
		let public_data = ValuesData::from(values.public());
		let non_public_data = ValuesData::from(values.non_public());

		let mut buf_cs = Vec::new();
		cs.serialize(&mut buf_cs).unwrap();

		let mut buf_pub = Vec::new();
		public_data.serialize(&mut buf_pub).unwrap();

		let mut buf_non_pub = Vec::new();
		non_public_data.serialize(&mut buf_non_pub).unwrap();

		// Deserialize everything back
		let cs2 = ConstraintSystem::deserialize(&mut buf_cs.as_slice()).unwrap();
		let pub2 = ValuesData::deserialize(&mut buf_pub.as_slice()).unwrap();
		let non_pub2 = ValuesData::deserialize(&mut buf_non_pub.as_slice()).unwrap();

		// Reconstruct ValueVec from deserialized pieces
		let reconstructed =
			ValueVec::new_from_data(cs2.value_vec_layout, pub2.into_owned(), non_pub2.into_owned())
				.unwrap();

		// Ensure committed part matches exactly
		assert_eq!(reconstructed.combined_witness(), values.combined_witness());

		// Scratch is not serialized; reconstructed scratch should be zero-filled
		let scratch_start = layout.committed_total_len;
		let scratch_end = scratch_start + layout.n_scratch;
		for i in scratch_start..scratch_end {
			assert_eq!(reconstructed.get(i), Word::ZERO, "scratch index {i} should be zero");
		}
	}
}
