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
		// Each wire is named by the segment it belongs to and its position within that segment,
		// so the four groups are numbered independently and nothing here has to reason about
		// where a segment lands in the value vector.
		//
		// The witness and internal wires share the private segment, the witness wires first.

		let mut wire_mapping = SecondaryMap::new();

		let n_const = self.w_const.len();
		let n_inout = self.w_inout.len();
		let n_witness = self.w_witness.len();
		let n_internal = self.w_internal.len();
		let n_scratch = self.n_scratch_slots;

		// Constants keep the order in which they were added, which is wire-creation order.
		// The gate graph seeds the all-one constant first (see `GateGraph::new`).
		// So it is the first constant here and lands at constant index 0.
		let mut constants = Vec::with_capacity(n_const);
		for (index, (wire, value)) in self.w_const.into_iter().enumerate() {
			wire_mapping[wire] = ValueIndex::constant(index as u32);
			constants.push(value);
		}
		for (index, wire) in self.w_inout.into_iter().enumerate() {
			wire_mapping[wire] = ValueIndex::inout(index as u32);
		}
		for (index, wire) in self
			.w_witness
			.into_iter()
			.chain(self.w_internal)
			.enumerate()
		{
			wire_mapping[wire] = ValueIndex::private(index as u32);
		}

		// Each uncommitted value lands at its own slot within the scratch segment.
		// Two values given the same slot share one index.
		// That is sound only because their lifetimes do not overlap.
		for (wire, slot) in self.w_scratch {
			wire_mapping[wire] = ValueIndex::scratch(slot);
		}

		// The layout places the segments in the value vector. The public section holds the
		// constants followed by the inout values, padded up to a power-of-two length that meets
		// the minimum segment size. The hidden section follows, padded to at least the public
		// length so that `log_witness_words >= log_public_words` (see
		// `ConstraintSystem::validate_shape`).
		let offset_inout = n_const;
		let offset_witness = (n_const + n_inout)
			.max(ConstraintSystem::MIN_WORDS_PER_SEGMENT)
			.next_power_of_two();
		let n_hidden_words = (n_witness + n_internal).max(offset_witness);

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
		let mapping = &assignment.wire_mapping;

		// Each segment is numbered from zero, in the order its wires were added.
		assert_eq!(mapping[const1], ValueIndex::constant(0));
		assert_eq!(mapping[const2], ValueIndex::constant(1));
		assert_eq!(mapping[const3], ValueIndex::constant(2));
		assert_eq!(mapping[inout1], ValueIndex::inout(0));
		assert_eq!(mapping[inout2], ValueIndex::inout(1));

		// The witness wires take the front of the private segment and the internal ones follow.
		assert_eq!(mapping[witness1], ValueIndex::private(0));
		assert_eq!(mapping[witness2], ValueIndex::private(1));
		assert_eq!(mapping[witness3], ValueIndex::private(2));
		assert_eq!(mapping[internal1], ValueIndex::private(3));

		// A scratch wire keeps the slot it was placed at.
		assert_eq!(mapping[scratch1], ValueIndex::scratch(0));
		assert_eq!(mapping[scratch2], ValueIndex::scratch(1));

		// The constants vector preserves insertion order.
		assert_eq!(assignment.constants, vec![Word(42), Word(100), Word(200)]);

		// The layout is what places those segments in the value vector.
		let layout = &assignment.value_vec_layout;
		assert_eq!(layout.n_const, 3);
		assert_eq!(layout.n_inout, 2);
		assert_eq!(layout.n_witness, 3);
		assert_eq!(layout.n_internal, 1);
		assert_eq!(layout.offset_inout, 3);
		// Five public values padded up to the next power of two.
		assert_eq!(layout.offset_witness, 8);
		// The hidden segment is padded from 4 words up to the public segment length.
		assert_eq!(layout.n_hidden_words, 8);

		// Resolving the indices through the layout reproduces the section order the value vector
		// lays the words out in.
		let offsets = [
			const1, const2, const3, inout1, inout2, witness1, witness2, witness3, internal1,
			scratch1, scratch2,
		]
		.map(|wire| layout.word_offset(mapping[wire]));
		assert_eq!(offsets, [0, 1, 2, 3, 4, 8, 9, 10, 11, 16, 17]);
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

		// Even with just one constant, the public segment is padded to MIN_WORDS_PER_SEGMENT, so
		// that is where the hidden segment starts.
		let offset_witness = assignment.value_vec_layout.offset_witness;
		assert!(offset_witness >= ConstraintSystem::MIN_WORDS_PER_SEGMENT);
		assert!(offset_witness.is_power_of_two());
	}
}
