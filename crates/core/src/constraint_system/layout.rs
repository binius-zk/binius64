// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use super::ConstraintSystem;
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

	/// Returns the constraint system shape this layout realizes.
	///
	/// The returned system has no constraints; the caller fills them in.
	pub fn constraint_system_shape(&self, constants: Vec<Word>) -> ConstraintSystem {
		assert_eq!(constants.len(), self.n_const);
		ConstraintSystem {
			constants,
			n_const_pad: self.offset_inout - self.n_const,
			n_inout: self.n_inout,
			n_inout_pad: self.offset_witness - self.offset_inout - self.n_inout,
			n_private: self.n_witness + self.n_internal,
			n_private_pad: self.n_hidden_words - self.n_witness - self.n_internal,
			zero_constraints: Vec::new(),
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn constraint_system_shape_splits_sections_into_values_and_padding() {
		let layout = ValueVecLayout {
			n_const: 2,         // constants at indices 0-1
			n_inout: 2,         // inout at indices 8-9
			n_witness: 4,       // witness at indices 16-19
			n_internal: 4,      // internal at indices 20-23
			offset_inout: 8,    // padding after the constants at indices 2-7
			offset_witness: 16, // padding after the inout values at indices 10-15
			n_hidden_words: 16, // padding after the internal values at indices 24-31
			n_scratch: 3,
		};

		let cs = layout.constraint_system_shape(vec![Word::ONE, Word::ALL_ONE]);
		assert_eq!(cs.n_const(), 2);
		assert_eq!(cs.n_const_pad, 6);
		assert_eq!(cs.n_inout, 2);
		assert_eq!(cs.n_inout_pad, 6);
		assert_eq!(cs.n_private, 8);
		assert_eq!(cs.n_private_pad, 8);

		// The derived sections reproduce the layout's offsets and lengths.
		assert_eq!(cs.offset_inout(), layout.offset_inout);
		assert_eq!(cs.n_public_words(), layout.n_public_words());
		assert_eq!(cs.n_hidden_words(), layout.n_hidden_words);
		assert_eq!(cs.value_vec_len(), layout.combined_len());
		cs.validate_shape().unwrap();
	}
}
