// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use super::{ConstraintSystem, ValueIndex, ValueSegment};
use crate::word::Word;

/// Description of a layout of the value vector for a particular circuit.
///
/// This is the compiler's view of the value vector: it names every section the circuit allocates,
/// including the ones a [`ConstraintSystem`] has no interest in — the split of the hidden segment
/// into declared witness and gate-created internal values, and the scratch tail used only while
/// evaluating the circuit. The section sizes it shares with the constraint system are stored
/// redundantly in both.
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
	/// Returns the number of words in the public segment: the constants and inout values,
	/// including padding up to the power-of-two segment length.
	pub const fn n_public_words(&self) -> usize {
		self.offset_witness
	}

	/// Returns the combined number of public and hidden words, excluding scratch.
	///
	/// This is the length of the value vector prefix that constraint operands can reference.
	pub const fn combined_len(&self) -> usize {
		self.offset_witness + self.n_hidden_words
	}

	/// Returns the flat position of the word a [`ValueIndex`] names, counting the scratch tail.
	///
	/// The witness and internal values share the private segment, in that order, so it starts
	/// where the witness values do.
	pub const fn word_offset(&self, index: ValueIndex) -> usize {
		let segment_start = match index.segment() {
			ValueSegment::Constant => 0,
			ValueSegment::InOut => self.offset_inout,
			ValueSegment::Private => self.offset_witness,
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
	/// Panics if the layout's padded section offsets disagree with the ones the returned system
	/// derives from its value counts, which would leave the two views of the same value vector
	/// addressing different words.
	pub fn constraint_system_shape(&self, constants: Vec<Word>) -> ConstraintSystem {
		assert!(constants.len() == self.n_const, "constants must match the layout's n_const");
		let system = ConstraintSystem {
			constants,
			n_inout: self.n_inout,
			n_private: self.n_witness + self.n_internal,
			zero_constraints: Vec::new(),
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};
		assert_eq!(
			system.offset_inout(),
			self.offset_inout,
			"the layout and the system must place the inout values at the same word"
		);
		assert_eq!(
			system.n_public_words(),
			self.offset_witness,
			"the layout and the system must pad the public segment to the same length"
		);
		assert_eq!(
			system.n_hidden_words(),
			self.n_hidden_words,
			"the layout and the system must pad the hidden segment to the same length"
		);
		system
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A layout of two constants, two inout values and eight private values.
	///
	/// Four public values pad to a four-word public segment; the eight private values exceed that,
	/// so the hidden segment holds them unpadded.
	fn test_layout() -> ValueVecLayout {
		ValueVecLayout {
			n_const: 2,    // constants at indices 0-1
			n_inout: 2,    // inout at indices 2-3
			n_witness: 4,  // witness at indices 4-7
			n_internal: 4, // internal at indices 8-11
			offset_inout: 2,
			offset_witness: 4,
			n_hidden_words: 8,
			n_scratch: 3,
		}
	}

	#[test]
	fn constraint_system_shape_carries_the_value_counts() {
		let layout = test_layout();
		let cs = layout.constraint_system_shape(vec![Word::ONE, Word::ALL_ONE]);

		assert_eq!(cs.n_const(), 2);
		assert_eq!(cs.n_inout, 2);
		// The witness and internal values share the private segment.
		assert_eq!(cs.n_private, 8);

		// The system derives the same padded sections the layout lays out.
		assert_eq!(cs.offset_inout(), layout.offset_inout);
		assert_eq!(cs.n_public_words(), layout.n_public_words());
		assert_eq!(cs.n_hidden_words(), layout.n_hidden_words);
		assert_eq!(cs.value_vec_len(), layout.combined_len());
	}

	#[test]
	#[should_panic(expected = "pad the public segment to the same length")]
	fn constraint_system_shape_rejects_a_layout_it_cannot_reproduce() {
		// The system derives a four-word public segment from the four public values, so a layout
		// that pads it to eight describes a different value vector.
		let layout = ValueVecLayout {
			offset_witness: 8,
			..test_layout()
		};
		let _ = layout.constraint_system_shape(vec![Word::ONE, Word::ALL_ONE]);
	}
}
