// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use binius_core::{ConstraintSystem, ValueIndex, ValueVecLayout, Word};
use cranelift_entity::SecondaryMap;

use crate::compiler::Wire;

pub struct Assignment {
	pub wire_mapping: SecondaryMap<Wire, ValueIndex>,
	pub value_vec_layout: ValueVecLayout,
	pub constants: Vec<Word>,
}

/// A structure that provides you assignments of value indices for wires and get a
/// [`ValueVecLayout`].
pub struct Alloc {
	w_const: Vec<(Wire, Word)>,
	w_inout: Vec<Wire>,
	w_witness: Vec<Wire>,
	w_internal: Vec<Wire>,
	/// Uncommitted values, each paired with the slot it occupies within the scratch segment.
	///
	/// Two values whose lifetimes do not overlap may be given the same slot.
	w_scratch: Vec<(Wire, u32)>,
	/// Length of the scratch segment.
	///
	/// This is at most the number of values placed in it, and less whenever slots are shared.
	n_scratch_slots: usize,
}

impl Alloc {
	/// Creates an allocator whose scratch segment is the given number of words long.
	pub const fn new(n_scratch_slots: usize) -> Self {
		Self {
			w_const: Vec::new(),
			w_inout: Vec::new(),
			w_witness: Vec::new(),
			w_internal: Vec::new(),
			w_scratch: Vec::new(),
			n_scratch_slots,
		}
	}

	pub fn add_constant(&mut self, wire: Wire, value: Word) {
		self.w_const.push((wire, value));
	}

	pub fn add_inout(&mut self, wire: Wire) {
		self.w_inout.push(wire);
	}

	pub fn add_witness(&mut self, wire: Wire) {
		self.w_witness.push(wire);
	}

	pub fn add_internal(&mut self, wire: Wire) {
		self.w_internal.push(wire);
	}

	/// Places an uncommitted value at the given slot of the scratch segment.
	///
	/// # Arguments
	///
	/// * `wire` - the value to place.
	/// * `slot` - its index within the segment, which another value may also hold.
	pub fn add_scratch(&mut self, wire: Wire, slot: u32) {
		// A slot past the declared length would place the value outside the value vector.
		debug_assert!((slot as usize) < self.n_scratch_slots);
		self.w_scratch.push((wire, slot));
	}

	pub fn into_assignment(self) -> Assignment {
		// `ValueVec` expects the wires to be in a certain order. Specifically:
		//
		// 1. const
		// 2. inout
		// 3. witness
		// 4. internal
		// 5. scratch
		//
		// So we create a mapping between a `Wire` to the final `ValueIndex`.

		let mut wire_mapping = SecondaryMap::new();

		let n_const = self.w_const.len();
		let n_inout = self.w_inout.len();
		let n_witness = self.w_witness.len();
		let n_internal = self.w_internal.len();
		let n_scratch = self.n_scratch_slots;

		// Constants keep the order in which they were added, which is wire-creation order.
		// The gate graph seeds the all-one constant first (see `GateGraph::new`).
		// So it is the first constant here and lands at value index 0.

		// First, allocate the indices for the public section of the value vec. The public section
		// consists of constant wires followed by inout wires.
		//
		// Next, we align the current index to the next power of 2 so the witness section starts at
		// a power-of-two offset.
		//
		// Finally, allocate wires for witness values and internal wires. The committed length is
		// not padded to a power of two; the prover zero-pads the committed witness when it builds
		// the witness polynomial.
		let mut cur_index: u32 = 0;
		let mut constants = Vec::with_capacity(n_const);
		for (wire, value) in self.w_const {
			wire_mapping[wire] = ValueIndex(cur_index);
			constants.push(value);
			cur_index += 1;
		}
		let offset_inout = cur_index as usize;
		for wire in self.w_inout {
			wire_mapping[wire] = ValueIndex(cur_index);
			cur_index += 1;
		}
		// Ensure the public section meets the minimum size requirement
		cur_index = cur_index.max(ConstraintSystem::MIN_WORDS_PER_SEGMENT as u32);
		cur_index = cur_index.next_power_of_two();
		let offset_witness = cur_index as usize;
		for wire in self.w_witness.into_iter().chain(self.w_internal) {
			wire_mapping[wire] = ValueIndex(cur_index);
			cur_index += 1;
		}

		// Pad the hidden segment to at least the public segment length, so
		// `log_witness_words >= log_public_words` (see `ConstraintSystem::validate_shape`).
		cur_index = cur_index.max(2 * offset_witness as u32);

		let n_hidden_words = cur_index as usize - offset_witness;

		// Each uncommitted value lands at its own slot within the segment.
		// Two values given the same slot share one index.
		// That is sound only because their lifetimes do not overlap.
		//
		// The segment is the tail of the value vector.
		// So the running index is not advanced past it.
		let scratch_base = cur_index;
		for (wire, slot) in self.w_scratch {
			wire_mapping[wire] = ValueIndex(scratch_base + slot);
		}

		let value_vec_layout = ValueVecLayout {
			n_const,
			n_inout,
			n_witness,
			n_internal,
			offset_inout,
			offset_witness,
			n_hidden_words,
			n_scratch,
		};

		Assignment {
			wire_mapping,
			value_vec_layout,
			constants,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_value_vec_alloc_ordering() {
		// Test that the allocator correctly orders wires according to:
		// 1. const
		// 2. inout
		// 3. witness
		// 4. internal
		// 5. scratch (handled separately)

		let mut alloc = Alloc::new(2);

		// Add wires in a deliberately mixed order to test section ordering.
		let witness1 = Wire::from_u32(0);
		let const1 = Wire::from_u32(1);
		let internal1 = Wire::from_u32(2);
		let inout1 = Wire::from_u32(3);
		let witness2 = Wire::from_u32(4);
		let const2 = Wire::from_u32(5);
		let inout2 = Wire::from_u32(6);
		let witness3 = Wire::from_u32(7);
		let const3 = Wire::from_u32(8);
		let scratch1 = Wire::from_u32(9);
		let scratch2 = Wire::from_u32(10);

		// Add them in mixed order
		alloc.add_witness(witness1);
		alloc.add_constant(const1, Word(42));
		alloc.add_internal(internal1);
		alloc.add_inout(inout1);
		alloc.add_witness(witness2);
		alloc.add_constant(const2, Word(100));
		alloc.add_inout(inout2);
		alloc.add_witness(witness3);
		alloc.add_constant(const3, Word(200));
		alloc.add_scratch(scratch1, 0);
		alloc.add_scratch(scratch2, 1);

		// Build the assignment
		let assignment = alloc.into_assignment();

		// Constants come first, in the order they were added.
		assert_eq!(assignment.wire_mapping[const1], ValueIndex(0));
		assert_eq!(assignment.wire_mapping[const2], ValueIndex(1));
		assert_eq!(assignment.wire_mapping[const3], ValueIndex(2));

		// The constants vector preserves insertion order.
		assert_eq!(assignment.constants, vec![Word(42), Word(100), Word(200)]);

		// Inout wires should come after constants
		let inout1_idx = assignment.wire_mapping[inout1];
		let inout2_idx = assignment.wire_mapping[inout2];
		assert!(inout1_idx.0 > assignment.wire_mapping[const3].0);
		assert!(inout2_idx.0 > inout1_idx.0);

		// Witness wires should start at a power-of-two index
		let witness1_idx = assignment.wire_mapping[witness1];
		assert!(witness1_idx.0.is_power_of_two(), "witness values must start with a po2 index");

		// Verify witness wires come after inout and maintain relative order
		let witness2_idx = assignment.wire_mapping[witness2];
		let witness3_idx = assignment.wire_mapping[witness3];
		assert!(witness1_idx.0 > inout2_idx.0);
		assert!(witness2_idx.0 > witness1_idx.0);
		assert!(witness3_idx.0 > witness2_idx.0);

		// Internal wires come after witness wires
		let internal1_idx = assignment.wire_mapping[internal1];
		assert!(internal1_idx.0 > witness3_idx.0);

		// Scratch wires should be in scratch mapping with high bit set
		let scratch1_idx = assignment.wire_mapping[scratch1];
		let scratch2_idx = assignment.wire_mapping[scratch2];
		assert!(scratch1_idx.0 > internal1_idx.0);
		assert!(scratch2_idx.0 > scratch1_idx.0);

		// Verify the value_vec_layout
		assert_eq!(assignment.value_vec_layout.n_const, 3);
		assert_eq!(assignment.value_vec_layout.n_inout, 2);
		assert_eq!(assignment.value_vec_layout.n_witness, 3);
		assert_eq!(assignment.value_vec_layout.n_internal, 1);
		assert_eq!(assignment.value_vec_layout.offset_inout, 3);
		assert_eq!(assignment.value_vec_layout.offset_witness, witness1_idx.0 as usize);
		// The witness segment is padded from 4 words up to the public segment length (8).
		assert_eq!(assignment.value_vec_layout.n_hidden_words, 8);
	}

	#[test]
	fn test_minimum_segment_size() {
		// Test that the public section meets the minimum size requirement
		let mut alloc = Alloc::new(0);

		// Add just one constant
		let const1 = Wire::from_u32(0);
		alloc.add_constant(const1, Word(42));

		// Add one witness
		let witness1 = Wire::from_u32(1);
		alloc.add_witness(witness1);

		let assignment = alloc.into_assignment();

		// Even with just one constant, the witness should start at MIN_WORDS_PER_SEGMENT
		// (which should be a power of 2)
		let witness_idx = assignment.wire_mapping[witness1];
		assert!(witness_idx.0 >= ConstraintSystem::MIN_WORDS_PER_SEGMENT as u32);
		assert!(witness_idx.0.is_power_of_two());
	}
}
