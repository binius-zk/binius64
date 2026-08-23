// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::constraint_system::{ConstraintSystem, Operand, Shift};

use super::{key::ConstraintIndex, operation::Operation};

/// One key still being assembled while its segment is built.
///
/// Constraint indices are collected directly into a vector here, rather than as a range into a
/// shared flattened vector like the final form uses.
/// That flattened layout is not known until every word's keys have been collected.
pub(super) struct BuilderKey {
	/// The shift sequence this key's word is referenced under, inner shift first.
	pub shift_seq: [Shift; 2],
	/// The constraint kind this key's constraints belong to.
	pub operation: Operation,
	/// The constraint indices collected so far for this key.
	pub constraint_indices: Vec<ConstraintIndex>,
}

/// One builder key list per word of the constraint system, indexed by word position.
pub(super) struct BuilderKeyLists(Vec<Vec<BuilderKey>>);

impl BuilderKeyLists {
	/// An empty list for every one of `word_count` words.
	pub(super) fn new(word_count: usize) -> Self {
		Self((0..word_count).map(|_| Vec::new()).collect())
	}

	/// Splits the words from `public_word_count` onward into a second set of lists.
	///
	/// This is what separates the public segment, kept here, from the hidden one, returned.
	pub(super) fn split_off(&mut self, public_word_count: usize) -> Self {
		Self(self.0.split_off(public_word_count))
	}

	/// The underlying per-word lists, ready for a segment to be built from them.
	pub(super) fn into_inner(self) -> Vec<Vec<BuilderKey>> {
		self.0
	}

	/// Records one operand's shifted-word references into the builder keys of the words they
	/// touch.
	///
	/// # Arguments
	///
	/// - `operation`: the constraint kind these constraints belong to.
	/// - `operand_index`: this operand's position in the constraint.
	/// - `operand_values`: this operand's shifted-word references, one list per constraint.
	/// - `cs`: resolves a value index to its segment-relative word position.
	fn update_with_operand(
		&mut self,
		operation: Operation,
		operand_index: usize,
		operand_values: impl Iterator<Item = impl AsRef<Operand>>,
		cs: &ConstraintSystem,
	) {
		for (constraint_idx, operand_value) in operand_values.enumerate() {
			// Each operand value is a Vec<ShiftedValueIndex> - multiple shifted word references
			for term in operand_value.as_ref() {
				// The lists are indexed by word position, so resolve the term's segment-relative
				// index against the segment starts.
				let builder_keys = &mut self.0[cs.word_offset(term.value_index)];
				let shift_seq = term.shift_seq;

				// Find existing builder key or create a new one for this (operation, sequence) pair
				let constraint_index = ConstraintIndex {
					operand_index: operand_index as u8,
					constraint_index: constraint_idx as u32,
				};
				if let Some(builder_key) = builder_keys
					.iter_mut()
					.find(|key| key.shift_seq == shift_seq && key.operation == operation)
				{
					builder_key.constraint_indices.push(constraint_index);
				} else {
					builder_keys.push(BuilderKey {
						shift_seq,
						operation,
						constraint_indices: vec![constraint_index],
					});
				}
			}
		}
	}

	/// Records every operand of every constraint of one operation into the builder keys of the
	/// words they touch.
	///
	/// Operands are indexed by their position in the constraint's operand array.
	/// That is also the order the shift reduction batches them in.
	pub(super) fn update_with_constraints<C, const ARITY: usize>(
		&mut self,
		operation: Operation,
		constraints: &[C],
		cs: &ConstraintSystem,
	) where
		C: AsRef<[Operand; ARITY]>,
	{
		for operand_index in 0..ARITY {
			self.update_with_operand(
				operation,
				operand_index,
				constraints
					.iter()
					.map(|constraint| &constraint.as_ref()[operand_index]),
				cs,
			);
		}
	}
}
