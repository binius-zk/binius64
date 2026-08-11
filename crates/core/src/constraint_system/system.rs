// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::iter;

use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	serialization::{DeserializeBytes, SerializationError, SerializeBytes},
};
use bytes::{Buf, BufMut};

use super::{
	AndConstraint, BmulConstraint, ConstraintKind, ImulConstraint, Operand, ValueIndex,
	ValueSegment, ValueVec, ZeroConstraint,
};
use crate::{
	error::{ConstraintSystemError, VerificationError},
	word::Word,
};

/// Which of the two value-vector segments holds the inout values.
///
/// The constants are always public and the private values always hidden, so this is the only
/// freedom in where the segment boundary falls. A proving protocol picks the placement that suits
/// how its verifier learns the inout words, and passes it to every accessor that reports a segment
/// length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InoutSegment {
	/// The inout values are public: the verifier knows every one of them, so the reduction reads
	/// them as shared data and nothing about them is committed.
	Public,
	/// The inout values are hidden: they are committed with the private values.
	///
	/// This is what a data-parallel protocol needs. Each instance chooses its own inout words, so
	/// they are not one set of shared values the verifier can evaluate.
	Hidden,
}

/// The ConstraintSystem is the core data structure in Binius64 that defines the computational
/// constraints to be proven in zero-knowledge. It represents a system of equations over 64-bit
/// words that must be satisfied by a valid values vector [`ValueVec`].
///
/// # Value vector shape
///
/// The constraints reference words of a value vector partitioned into two segments. The public
/// segment holds the words the verifier evaluates itself; the hidden segment holds the words the
/// prover commits. The constants are always public and the private values always hidden; the inout
/// values sit in whichever segment the proving protocol places them in, which every segment-length
/// accessor takes as an [`InoutSegment`]. Each group of values is followed by padding, so under
/// [`InoutSegment::Public`] the value vector runs
///
/// ```text
/// [ constants | inout | pad ][ private | pad ]
///  \--- public segment ---/  \- hidden segment -/
/// ```
///
/// Both segments are padded by the proving protocol rather than by the system: the public segment
/// up to a power of two, and the hidden segment up to at least the public length. The system
/// stores the value counts and derives the padded lengths from them.
///
/// A constraint names a word by its [`ValueSegment`] and its position within that segment, so
/// the padding is unaddressable: an index reaches only the values of its own segment, and where
/// those values sit in the vector is the layout's business rather than the constraint's.
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
	/// The number of input/output values, which are public but chosen per instance.
	pub n_inout: usize,
	/// The number of private values, which only the prover knows.
	pub n_private: usize,
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
	pub const SERIALIZATION_VERSION: u32 = 9;

	/// Returns the number of constants.
	pub const fn n_const(&self) -> usize {
		self.constants.len()
	}

	/// Returns the index of the first inout value.
	pub const fn offset_inout(&self) -> usize {
		self.n_const()
	}

	/// Returns the number of public values: the constants and the inout values.
	pub const fn n_public_values(&self) -> usize {
		self.n_const() + self.n_inout
	}

	/// Returns the number of words in the public segment.
	///
	/// This is the constants, followed by the inout values when they are placed there.
	pub const fn n_public_words(&self, inout: InoutSegment) -> usize {
		match inout {
			InoutSegment::Public => self.n_public_values(),
			InoutSegment::Hidden => self.n_const(),
		}
	}

	/// Returns the number of word-index variables the public segment spans.
	///
	/// The word count need not be a power of two; the reductions read the words past it as zero.
	pub const fn log_public_words(&self, inout: InoutSegment) -> usize {
		log2_ceil_usize(self.n_public_words(inout))
	}

	/// Returns the number of words in the hidden segment.
	///
	/// This is the private values, preceded by the inout values when they are placed there.
	pub const fn n_hidden_words(&self, inout: InoutSegment) -> usize {
		match inout {
			InoutSegment::Public => self.n_private,
			InoutSegment::Hidden => self.n_inout + self.n_private,
		}
	}

	/// Returns the number of word-index variables the hidden segment spans.
	pub const fn log_witness_words(&self, inout: InoutSegment) -> usize {
		log2_ceil_usize(self.n_hidden_words(inout))
	}

	/// Returns the number of word-index variables the shift reduction runs over.
	///
	/// The reduction addresses both segments with one set of word-index challenges, so it needs
	/// as many as the wider of the two spans. The narrower segment reads the extra coordinates as
	/// zero.
	pub const fn log_segment_words(&self, inout: InoutSegment) -> usize {
		if self.log_public_words(inout) > self.log_witness_words(inout) {
			self.log_public_words(inout)
		} else {
			self.log_witness_words(inout)
		}
	}

	/// Returns the number of values the given segment holds, excluding the padding after them.
	///
	/// The scratch segment holds no values a constraint may name, so it reports zero: every index
	/// into it is out of range as far as this system is concerned.
	pub const fn segment_len(&self, segment: ValueSegment) -> usize {
		match segment {
			ValueSegment::Constant => self.n_const(),
			ValueSegment::InOut => self.n_inout,
			ValueSegment::Private => self.n_private,
			ValueSegment::Scratch => 0,
		}
	}

	/// Returns the position of the word a [`ValueIndex`] names within the value vector.
	///
	/// This is the address the proving protocol reads the word at: the constants, then the inout
	/// values, then the private ones. Where the segment boundary falls does not enter, so the
	/// address is the same under either [`InoutSegment`] placement. Scratch words are not part of a
	/// constraint system, so a scratch index lands past the last word — [`Self::validate`] rejects
	/// any constraint naming one.
	pub const fn word_offset(&self, index: ValueIndex) -> usize {
		let segment_start = match index.segment() {
			ValueSegment::Constant => 0,
			ValueSegment::InOut => self.offset_inout(),
			ValueSegment::Private => self.n_public_values(),
			ValueSegment::Scratch => self.value_vec_len(),
		};
		segment_start + index.index() as usize
	}

	/// Builds a value vector from the inout values and the private values.
	///
	/// The constants come from the system itself, so a caller supplies only what varies per
	/// instance — the same split [`Self::validate`] enforces and the verifier takes.
	pub fn value_vec_from_data(&self, inout: &[Word], private: &[Word]) -> ValueVec {
		let public = [self.constants.as_slice(), inout].concat();
		ValueVec::new_from_data(self.n_const(), &public, private)
	}

	/// Ensures that this constraint system is well-formed and ready for proving.
	///
	/// Specifically checks that:
	///
	/// - every [shifted value index][super::ShiftedValueIndex] is canonical.
	/// - referenced value indices are within their segment.
	/// - constraints do not reference scratch values.
	/// - shifts amounts are valid.
	pub fn validate(&self) -> Result<(), ConstraintSystemError> {
		tracing::debug_span!("Validating constraint system");

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

	/// Checks that a value vector satisfies this constraint system.
	///
	/// Specifically checks that:
	///
	/// - the value vector opens the declared constants to their declared words.
	/// - every constraint holds, in kind order: zero, then AND, then IMUL, then BMUL.
	///
	/// Operands are evaluated one word at a time, directly off the value vector.
	/// That makes this the reference the prover's packed evaluation is checked against.
	///
	/// # Errors
	///
	/// Reports the first failure found, in the order listed above.
	/// A reported constraint position counts within that constraint's own kind.
	pub fn verify(&self, values: &ValueVec) -> Result<(), VerificationError> {
		// Constraints read constants through the value vector.
		// A vector opening one to the wrong word satisfies a different system than declared.
		for (index, &constant) in self.constants.iter().enumerate() {
			let value_index = index as u32;
			let actual = values[ValueIndex::constant(value_index)];
			if actual != constant {
				return Err(VerificationError::ConstantMismatch {
					value_index,
					expected: constant.as_u64(),
					actual: actual.as_u64(),
				});
			}
		}

		// Each kind is numbered from zero, so a position is only meaningful with its kind.
		// The violation carries that kind, which is what the reported message prints.
		for (constraint_index, constraint) in self.zero_constraints.iter().enumerate() {
			constraint
				.verify(values)
				.map_err(|source| VerificationError::Unsatisfied {
					constraint_index,
					source,
				})?;
		}
		for (constraint_index, constraint) in self.and_constraints.iter().enumerate() {
			constraint
				.verify(values)
				.map_err(|source| VerificationError::Unsatisfied {
					constraint_index,
					source,
				})?;
		}
		for (constraint_index, constraint) in self.imul_constraints.iter().enumerate() {
			constraint
				.verify(values)
				.map_err(|source| VerificationError::Unsatisfied {
					constraint_index,
					source,
				})?;
		}
		for (constraint_index, constraint) in self.bmul_constraints.iter().enumerate() {
			constraint
				.verify(values)
				.map_err(|source| VerificationError::Unsatisfied {
					constraint_index,
					source,
				})?;
		}

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
			// check canonicity. SLL is the canonical form of the identity.
			if !term.shift.is_canonical() {
				return Err(ConstraintSystemError::NonCanonicalShift {
					constraint_kind,
					constraint_index,
					operand_name,
				});
			}
			// Half-word (*32) variants cap at 32, full-width at 64. `Shift::new` and the
			// deserializer both enforce this, but the fields are public, so a hand-built term can
			// still carry an amount the variant cannot represent.
			let max_amount = term.shift.variant.max_amount();
			if usize::from(term.shift.amount) >= max_amount {
				return Err(ConstraintSystemError::ShiftAmountTooLarge {
					constraint_kind,
					constraint_index,
					operand_name,
					shift_amount: term.shift.amount as usize,
					max_amount,
				});
			}
			// Scratch words are uncommitted temporaries of the circuit that produced this system,
			// so no constraint may name one.
			let segment = term.value_index.segment();
			if !segment.is_referenceable() {
				return Err(ConstraintSystemError::ScratchValueIndex {
					constraint_kind,
					constraint_index,
					operand_name,
				});
			}
			// An index is checked against its own segment, so it can only name a declared value.
			// Padding follows the values of a segment, and no index reaches it.
			let segment_len = self.segment_len(segment);
			if term.value_index.index() as usize >= segment_len {
				return Err(ConstraintSystemError::OutOfRangeValueIndex {
					constraint_kind,
					constraint_index,
					operand_name,
					segment,
					value_index: term.value_index.index(),
					segment_len,
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
		self.n_public_values() + self.n_private
	}
}

impl SerializeBytes for ConstraintSystem {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		Self::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

		self.constants.serialize(&mut write_buf)?;
		self.n_inout.serialize(&mut write_buf)?;
		self.n_private.serialize(&mut write_buf)?;
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
		let n_inout = usize::deserialize(&mut read_buf)?;
		let n_private = usize::deserialize(&mut read_buf)?;
		let zero_constraints = Vec::<ZeroConstraint>::deserialize(&mut read_buf)?;
		let and_constraints = Vec::<AndConstraint>::deserialize(&mut read_buf)?;
		let imul_constraints = Vec::<ImulConstraint>::deserialize(&mut read_buf)?;
		let bmul_constraints = Vec::<BmulConstraint>::deserialize(read_buf)?;

		Ok(ConstraintSystem {
			constants,
			n_inout,
			n_private,
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
	use crate::{
		constraint_system::{Shift, ShiftVariant, ShiftedValueIndex, ValuesData},
		error::ConstraintViolation,
	};

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
			n_inout: 2,
			n_private: 6,
			zero_constraints: vec![],
			and_constraints: vec![],
			imul_constraints: vec![],
			bmul_constraints: vec![],
		}
	}

	pub(crate) fn create_test_constraint_system() -> ConstraintSystem {
		ConstraintSystem {
			zero_constraints: vec![ZeroConstraint::plain([
				ValueIndex::constant(0),
				ValueIndex::inout(0),
				ValueIndex::private(0),
			])],
			and_constraints: vec![
				AndConstraint::plain_abc(
					vec![ValueIndex::constant(0), ValueIndex::constant(1)],
					vec![ValueIndex::constant(2)],
					vec![ValueIndex::inout(0), ValueIndex::inout(1)],
				),
				AndConstraint::abc(
					vec![ShiftedValueIndex::sll(ValueIndex::constant(0), 5)],
					vec![ShiftedValueIndex::srl(ValueIndex::constant(1), 10)],
					vec![ShiftedValueIndex::sar(ValueIndex::constant(2), 15)],
				),
			],
			imul_constraints: vec![ImulConstraint([
				vec![ShiftedValueIndex::plain(ValueIndex::constant(0))],
				vec![ShiftedValueIndex::plain(ValueIndex::constant(1))],
				vec![ShiftedValueIndex::plain(ValueIndex::constant(2))],
				vec![ShiftedValueIndex::plain(ValueIndex::private(0))],
			])],
			bmul_constraints: vec![BmulConstraint([
				vec![ShiftedValueIndex::plain(ValueIndex::constant(0))],
				vec![ShiftedValueIndex::plain(ValueIndex::constant(1))],
				vec![ShiftedValueIndex::plain(ValueIndex::constant(2))],
				vec![ShiftedValueIndex::plain(ValueIndex::inout(0))],
				vec![ShiftedValueIndex::plain(ValueIndex::inout(1))],
				vec![ShiftedValueIndex::sll(ValueIndex::constant(0), 5)],
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
		assert_eq!(ConstraintSystem::SERIALIZATION_VERSION, 9);

		// Check the value vector shape
		assert_eq!(original.constants, deserialized.constants);
		assert_eq!(original.n_inout, deserialized.n_inout);
		assert_eq!(original.n_private, deserialized.n_private);

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
		let test_data_path = std::path::Path::new("test_data/constraint_system_v9.bin");

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
		// The v9 format drops the padding counts, which are derived from the value counts now.
		// Older files carry them as explicit fields, so they are no longer compatible.
		let binary_data = include_bytes!("../../test_data/constraint_system_v9.bin");

		let deserialized = ConstraintSystem::deserialize(&mut binary_data.as_slice()).unwrap();

		assert_eq!(deserialized.n_const(), 3);
		assert_eq!(deserialized.n_inout, 2);
		assert_eq!(deserialized.n_private, 6);

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
		let expected_version_bytes = 9u32.to_le_bytes(); // Version 9 in little-endian
		assert_eq!(
			version_bytes, expected_version_bytes,
			"Binary file version mismatch. If you made breaking changes, increment ConstraintSystem::SERIALIZATION_VERSION"
		);
	}

	#[test]
	fn test_log_witness_words() {
		let cs = |n_private: usize| ConstraintSystem {
			n_private,
			..test_shape()
		};
		// Typical: more private values than public words, rounded up to a power of two.
		assert_eq!(cs(60).log_witness_words(InoutSegment::Public), 6);
		// Exact power-of-two private count.
		assert_eq!(cs(32).log_witness_words(InoutSegment::Public), 5);
	}

	#[test]
	fn segment_lengths_are_the_value_counts() {
		// Three constants and two inout values are five public words; six private values are six
		// hidden words. Neither is padded.
		let cs = test_shape();
		assert_eq!(cs.n_public_values(), 5);
		assert_eq!(cs.n_public_words(InoutSegment::Public), 5);
		assert_eq!(cs.n_hidden_words(InoutSegment::Public), 6);
		assert_eq!(cs.value_vec_len(), 11);

		// The spans are the counts rounded up, and the reduction runs over the wider of the two.
		assert_eq!(cs.log_public_words(InoutSegment::Public), 3);
		assert_eq!(cs.log_witness_words(InoutSegment::Public), 3);
		assert_eq!(cs.log_segment_words(InoutSegment::Public), 3);

		// A hidden segment wider than the public one sets the span.
		let wide = ConstraintSystem {
			n_private: 60,
			..test_shape()
		};
		assert_eq!(wide.log_public_words(InoutSegment::Public), 3);
		assert_eq!(wide.log_witness_words(InoutSegment::Public), 6);
		assert_eq!(wide.log_segment_words(InoutSegment::Public), 6);

		// And a public segment wider than the hidden one sets it instead — the case the old
		// hidden-segment padding existed to rule out.
		let public_heavy = ConstraintSystem {
			n_inout: 200,
			n_private: 4,
			..test_shape()
		};
		assert_eq!(public_heavy.log_public_words(InoutSegment::Public), 8);
		assert_eq!(public_heavy.log_witness_words(InoutSegment::Public), 2);
		assert_eq!(public_heavy.log_segment_words(InoutSegment::Public), 8);
	}

	#[test]
	fn hidden_inout_moves_the_segment_boundary() {
		// The same shape as above, read with the inout values in the hidden segment: three
		// constants are the whole public segment, and the two inout values join the six private
		// ones.
		let cs = test_shape();
		assert_eq!(cs.n_public_words(InoutSegment::Hidden), 3);
		assert_eq!(cs.n_hidden_words(InoutSegment::Hidden), 8);

		// The value vector is the same either way, so the word addresses are too.
		assert_eq!(cs.value_vec_len(), 11);
		assert_eq!(cs.word_offset(ValueIndex::inout(0)), 3);
		assert_eq!(cs.word_offset(ValueIndex::private(0)), 5);
	}

	#[test]
	fn test_validate_rejects_scratch_references() {
		let mut cs = test_shape();

		// Scratch words are the evaluating circuit's uncommitted temporaries, so a system that
		// names one references a word that was never committed.
		cs.and_constraints.push(AndConstraint::plain_abc(
			vec![ValueIndex::constant(0)],
			vec![ValueIndex::scratch(0)], // SCRATCH!
			vec![ValueIndex::private(0)],
		));

		match cs.validate().unwrap_err() {
			ConstraintSystemError::ScratchValueIndex {
				constraint_kind, ..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::And);
			}
			other => panic!("Expected ScratchValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_checks_each_segment_against_its_own_length() {
		// The shape holds 3 constants, 2 inout values and 6 private values. Index 3 is out of
		// range in the constant segment while naming a perfectly valid private word, which is
		// what makes the check segment-relative rather than global.
		let mut cs = test_shape();
		cs.and_constraints.push(AndConstraint::plain_abc(
			vec![ValueIndex::constant(3)],
			vec![ValueIndex::inout(0)],
			vec![ValueIndex::private(3)],
		));

		match cs.validate().unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				segment,
				value_index,
				segment_len,
				..
			} => {
				assert_eq!(segment, ValueSegment::Constant);
				assert_eq!(value_index, 3);
				assert_eq!(segment_len, 3);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_accepts_non_padding_references() {
		let mut cs = test_shape();

		// Add constraints that only reference valid non-padding indices
		cs.and_constraints.push(AndConstraint::plain_abc(
			vec![ValueIndex::constant(0), ValueIndex::constant(1)], // constants
			vec![ValueIndex::inout(0), ValueIndex::inout(1)],       // inout
			vec![ValueIndex::private(0), ValueIndex::private(1)],   // private
		));

		cs.imul_constraints.push(ImulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex::private(2))], // a
			vec![ShiftedValueIndex::plain(ValueIndex::private(3))], // b
			vec![ShiftedValueIndex::plain(ValueIndex::private(4))], // lo
			vec![ShiftedValueIndex::plain(ValueIndex::private(5))], // hi
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
			.push(ZeroConstraint::plain([ValueIndex::constant(0), ValueIndex::private(100)]));

		match cs.validate().unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				segment_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::Zero);
				assert_eq!(operand_name, "val");
				assert_eq!(value_index, 100);
				assert_eq!(segment_len, 6);
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

		cs.zero_constraints =
			vec![ZeroConstraint::plain([ValueIndex::constant(0), ValueIndex::private(0)]); 3];
		assert_eq!(cs.log_zero_constraints(), Some(2));

		let and = AndConstraint::plain_abc(
			vec![ValueIndex::constant(0)],
			vec![ValueIndex::inout(0)],
			vec![ValueIndex::private(0)],
		);
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
			vec![ValueIndex::constant(0)], // valid constant
			vec![ValueIndex::private(6)],  // OUT OF RANGE! the private segment holds 6 values
			vec![ValueIndex::private(0)],  // valid private value
		));

		let result = cs.validate();
		assert!(result.is_err(), "Should reject constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				segment_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::And);
				assert_eq!(operand_name, "b");
				assert_eq!(value_index, 6);
				assert_eq!(segment_len, 6);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_rejects_out_of_range_in_imul_constraint() {
		let mut cs = test_shape();

		// Add IMUL constraint with out-of-range index in 'hi' operand
		cs.imul_constraints.push(ImulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex::constant(0))], // a: valid
			vec![ShiftedValueIndex::plain(ValueIndex::constant(1))], // b: valid
			vec![ShiftedValueIndex::plain(ValueIndex::private(0))],  // lo: valid
			vec![ShiftedValueIndex::plain(ValueIndex::private(100))], // hi: WAY out of range!
		]));

		let result = cs.validate();
		assert!(result.is_err(), "Should reject IMUL constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				segment_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::Imul);
				assert_eq!(operand_name, "hi");
				assert_eq!(value_index, 100);
				assert_eq!(segment_len, 6);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_rejects_out_of_range_in_bmul_constraint() {
		let mut cs = test_shape();

		// Add BMUL constraint with out-of-range index in 'c_hi' operand
		cs.bmul_constraints.push(BmulConstraint([
			vec![ShiftedValueIndex::plain(ValueIndex::constant(0))], // a_lo: valid const
			vec![ShiftedValueIndex::plain(ValueIndex::inout(0))],    // a_hi: valid inout
			vec![ShiftedValueIndex::plain(ValueIndex::inout(1))],    // b_lo: valid inout
			vec![ShiftedValueIndex::plain(ValueIndex::private(0))],  // b_hi: valid private
			vec![ShiftedValueIndex::plain(ValueIndex::private(1))],  // c_lo: valid private
			vec![ShiftedValueIndex::plain(ValueIndex::private(100))], // c_hi: WAY out of range!
		]));

		let result = cs.validate();
		assert!(result.is_err(), "Should reject BMUL constraint with out-of-range index");

		match result.unwrap_err() {
			ConstraintSystemError::OutOfRangeValueIndex {
				constraint_kind,
				operand_name,
				value_index,
				segment_len,
				..
			} => {
				assert_eq!(constraint_kind, ConstraintKind::Bmul);
				assert_eq!(operand_name, "c_hi");
				assert_eq!(value_index, 100);
				assert_eq!(segment_len, 6);
			}
			other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
		}
	}

	#[test]
	fn test_validate_rejects_half_word_shift_amount_out_of_range() {
		let mut cs = test_shape();

		// A half-word (*32) shift may only use amounts < 32.
		// 32 is out of range even though it is below the full-width bound of 64.
		cs.and_constraints.push(AndConstraint::abc(
			vec![ShiftedValueIndex {
				value_index: ValueIndex::constant(0),
				// Built raw: `Shift::new` would reject this amount, and `validate` is what is
				// under test here.
				shift: Shift {
					variant: ShiftVariant::Sll32,
					amount: 32,
				},
			}],
			vec![ShiftedValueIndex::plain(ValueIndex::private(0))],
			vec![ShiftedValueIndex::plain(ValueIndex::private(0))],
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
	fn test_roundtrip_cs_and_witnesses_reconstruct_valuevec() {
		let cs = test_shape();

		// Build a value vector and fill every per-instance word with a deterministic pattern. The
		// constants come from the system, so they are not supplied here.
		let inout = (0..cs.n_inout)
			.map(|i| Word::from_u64(0xA5A5_5A5A ^ (i as u64 * 0x9E37_79B9)))
			.collect::<Vec<_>>();
		let private = (0..cs.n_private)
			.map(|i| Word::from_u64(0x5A5A_A5A5 ^ (i as u64 * 0x9E37_79B9)))
			.collect::<Vec<_>>();
		let values = cs.value_vec_from_data(&inout, &private);

		// Serialize only what varies per instance, alongside the system itself
		let inout_data = ValuesData::from(values.inout());
		let non_public_data = ValuesData::from(values.non_public());

		let mut buf_cs = Vec::new();
		cs.serialize(&mut buf_cs).unwrap();

		let mut buf_pub = Vec::new();
		inout_data.serialize(&mut buf_pub).unwrap();

		let mut buf_non_pub = Vec::new();
		non_public_data.serialize(&mut buf_non_pub).unwrap();

		// Deserialize everything back
		let cs2 = ConstraintSystem::deserialize(&mut buf_cs.as_slice()).unwrap();
		let pub2 = ValuesData::deserialize(&mut buf_pub.as_slice()).unwrap();
		let non_pub2 = ValuesData::deserialize(&mut buf_non_pub.as_slice()).unwrap();
		assert_eq!(cs2.n_inout, pub2.len());
		assert_eq!(cs2.n_private, non_pub2.len());

		// Reconstruct ValueVec from deserialized pieces
		let reconstructed = cs2.value_vec_from_data(&pub2, &non_pub2);

		assert_eq!(reconstructed.combined_witness(), values.combined_witness());
	}

	/// A system whose only constraints are `n` zero constraints, each reading one hidden word.
	///
	///     [ _ _ _ _ _ _ _ _ ][ v_0 .. v_(n-1) ... ]
	///       0 ...        7     8 ...
	fn zero_constraint_system(n: usize) -> ConstraintSystem {
		ConstraintSystem {
			constants: vec![],
			n_inout: 0,
			n_private: 8,
			zero_constraints: (0..n)
				.map(|i| ZeroConstraint::plain([ValueIndex::private(i as u32)]))
				.collect(),
			and_constraints: vec![],
			imul_constraints: vec![],
			bmul_constraints: vec![],
		}
	}

	#[test]
	fn verify_accepts_a_value_vector_satisfying_every_constraint() {
		let cs = zero_constraint_system(3);
		let values = cs.value_vec_from_data(&[Word::ZERO; 8], &[Word::ZERO; 8]);

		assert!(cs.verify(&values).is_ok());
	}

	#[test]
	fn verify_reports_the_index_of_the_first_unsatisfied_constraint() {
		// Constraints 0 and 2 hold; only constraint 1 reads a nonzero word.
		let cs = zero_constraint_system(3);
		let mut private = [Word::ZERO; 8];
		private[1] = Word::from_u64(0xabc);
		let values = cs.value_vec_from_data(&[Word::ZERO; 8], &private);

		let err = cs.verify(&values).unwrap_err();

		// The message names the kind, the position and the failing arithmetic.
		// Printing the error alone is therefore enough to locate the constraint.
		assert_eq!(err.to_string(), "zero #1 is unsatisfied: 0000000000000abc != 0");

		match err {
			VerificationError::Unsatisfied {
				constraint_index,
				source,
			} => {
				assert_eq!(constraint_index, 1);
				assert_eq!(source.kind(), ConstraintKind::Zero);
				match source {
					ConstraintViolation::Zero { val } => assert_eq!(val, 0xabc),
					other => panic!("wrong violation: {other:?}"),
				}
			}
			other => panic!("wrong error: {other:?}"),
		}
	}

	#[test]
	fn verify_rejects_a_value_vector_that_opens_a_constant_to_the_wrong_word() {
		// The vector opens the third constant to a different word than the system declares.
		// Constraints read constants through the vector, so this is a different system.
		let cs = ConstraintSystem {
			zero_constraints: vec![],
			..test_shape()
		};
		// `value_vec_from_data` sources the constants from the system, so it cannot open one to
		// the wrong word — the vector is built directly to inject the disagreement. A vector the
		// circuit filled can still carry one, which is what `verify` guards against.
		let mut public = [Word::ZERO; 8];
		public[0] = Word::from_u64(1);
		public[1] = Word::from_u64(42);
		public[2] = Word::from_u64(0xBAADF00D);
		let values = ValueVec::new_from_data(cs.n_const(), &public, &[Word::ZERO; 8]);

		match cs.verify(&values).unwrap_err() {
			VerificationError::ConstantMismatch {
				value_index,
				expected,
				actual,
			} => {
				assert_eq!(value_index, 2);
				assert_eq!(expected, 0xDEADBEEF);
				assert_eq!(actual, 0xBAADF00D);
			}
			other => panic!("wrong error: {other:?}"),
		}
	}
}
