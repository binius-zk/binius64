// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use super::{ConstraintSystem, ValueIndex, ValueSegment};
use crate::word::Word;

/// Description of a layout of the value vector for a particular circuit.
///
/// This is the compiler's view of the value vector: it names every section the circuit allocates,
/// including the ones a [`ConstraintSystem`] has no interest in — the split of the private segment
/// into declared witness and gate-created internal values, and the scratch tail used only while
/// evaluating the circuit.
///
/// The sections are stored back to back, with no padding between them:
///
/// ```text
/// [ constants | inout ][ witness | internal ][ scratch ]
///  \-- public values -/ \--- private values -/
/// ```
///
/// The proving protocol pads the two committed segments to the widths its reductions need, which
/// [`ConstraintSystem`] derives; none of that padding is stored here.
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

	/// The number of scratch values at the end of the value vec.
	pub n_scratch: usize,
}

impl ValueVecLayout {
	/// Returns the word at which the inout values start.
	pub const fn offset_inout(&self) -> usize {
		self.n_const
	}

	/// Returns the word at which the private values start, which is where the public ones end.
	pub const fn offset_witness(&self) -> usize {
		self.n_const + self.n_inout
	}

	/// Returns the number of private values: the declared witness values followed by the
	/// gate-created internal ones, which share the segment.
	pub const fn n_private(&self) -> usize {
		self.n_witness + self.n_internal
	}

	/// Returns the combined number of public and private values, excluding scratch.
	///
	/// This is the length of the value vector prefix that constraint operands can reference.
	pub const fn combined_len(&self) -> usize {
		self.offset_witness() + self.n_private()
	}

	/// Returns the flat position of the word a [`ValueIndex`] names, counting the scratch tail.
	pub const fn word_offset(&self, index: ValueIndex) -> usize {
		let segment_start = match index.segment() {
			ValueSegment::Constant => 0,
			ValueSegment::InOut => self.offset_inout(),
			ValueSegment::Private => self.offset_witness(),
			ValueSegment::Scratch => self.combined_len(),
		};
		segment_start + index.index() as usize
	}

	/// Returns the constraint system shape this layout realizes.
	///
	/// The returned system has no constraints; the caller fills them in.
	///
	/// # Panics
	///
	/// Panics if the constant count does not match the layout's.
	pub fn constraint_system_shape(&self, constants: Vec<Word>) -> ConstraintSystem {
		assert!(constants.len() == self.n_const, "constants must match the layout's n_const");
		ConstraintSystem {
			constants,
			n_inout: self.n_inout,
			n_private: self.n_private(),
			zero_constraints: Vec::new(),
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::{super::InoutSegment, *};

	/// A layout of two constants, two inout values and eight private values.
	fn test_layout() -> ValueVecLayout {
		ValueVecLayout {
			n_const: 2,    // constants at words 0-1
			n_inout: 2,    // inout at words 2-3
			n_witness: 4,  // witness at words 4-7
			n_internal: 4, // internal at words 8-11
			n_scratch: 3,  // scratch at words 12-14
		}
	}

	#[test]
	fn sections_are_stored_back_to_back() {
		let layout = test_layout();
		assert_eq!(layout.offset_inout(), 2);
		assert_eq!(layout.offset_witness(), 4);
		assert_eq!(layout.n_private(), 8);
		assert_eq!(layout.combined_len(), 12);

		// Every section is reached by resolving an index of its own segment, with no gap between
		// them. The internal values continue the private segment past the witness ones.
		let offsets = [
			ValueIndex::constant(0),
			ValueIndex::inout(0),
			ValueIndex::private(0),
			ValueIndex::private(4),
			ValueIndex::scratch(0),
		]
		.map(|index| layout.word_offset(index));
		assert_eq!(offsets, [0, 2, 4, 8, 12]);
	}

	#[test]
	fn constraint_system_shape_carries_the_value_counts() {
		let layout = test_layout();
		let cs = layout.constraint_system_shape(vec![Word::ONE, Word::ALL_ONE]);

		assert_eq!(cs.n_const(), 2);
		assert_eq!(cs.n_inout, 2);
		// The witness and internal values share the private segment.
		assert_eq!(cs.n_private, 8);
		assert_eq!(cs.n_public_values(), layout.offset_witness());

		// The system pads the segments the protocol commits; the layout stores neither padding.
		assert_eq!(cs.n_public_words(InoutSegment::Public), 4);
		assert_eq!(cs.n_hidden_words(InoutSegment::Public), 8);
	}
}
