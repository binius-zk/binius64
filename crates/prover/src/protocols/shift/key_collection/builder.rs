// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::constraint_system::{ConstraintSystem, InoutSegment, Operand, Shift};

use super::{
	collection::KeyCollection, key::ConstraintIndex, key_segment::KeySegment, operation::Operation,
};

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
struct BuilderKeyLists(Vec<Vec<BuilderKey>>);

impl BuilderKeyLists {
	/// An empty list for every one of `word_count` words.
	fn new(word_count: usize) -> Self {
		Self((0..word_count).map(|_| Vec::new()).collect())
	}

	/// Splits the words from `public_word_count` onward into a second set of lists.
	///
	/// This is what separates the public segment, kept here, from the hidden one, returned.
	fn split_off(&mut self, public_word_count: usize) -> Self {
		Self(self.0.split_off(public_word_count))
	}

	/// The underlying per-word lists, ready for a segment to be built from them.
	fn into_inner(self) -> Vec<Vec<BuilderKey>> {
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
	fn update_with_constraints<C, const ARITY: usize>(
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

/// Walks a constraint system once and builds the prover's dense key collection.
///
/// # Arguments
///
/// - `cs`: the constraint system to walk.
/// - `inout`: the split point between the public and hidden key segments.
pub fn build_key_collection(cs: &ConstraintSystem, inout: InoutSegment) -> KeyCollection {
	let mut builder_key_lists = BuilderKeyLists::new(cs.value_vec_len());

	// Update the builder keys lists with respect to each operand of each operation.
	builder_key_lists.update_with_constraints(Operation::Zero, &cs.zero_constraints, cs);
	builder_key_lists.update_with_constraints(Operation::BitwiseAnd, &cs.and_constraints, cs);
	builder_key_lists.update_with_constraints(Operation::IntegerMul, &cs.imul_constraints, cs);
	builder_key_lists.update_with_constraints(Operation::BinMul, &cs.bmul_constraints, cs);

	// Split the builder keys lists at the public segment boundary and build one `KeySegment`
	// per half.
	let hidden_lists = builder_key_lists.split_off(cs.n_public_words(inout));
	KeyCollection {
		public: KeySegment::build(builder_key_lists.into_inner()),
		hidden: KeySegment::build(hidden_lists.into_inner()),
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{
		constraint_system::{AndConstraint, ShiftedValueIndex, ValueIndex},
		word::Word,
	};
	use binius_utils::serialization::{DeserializeBytes, SerializeBytes};
	use binius_verifier::protocols::shift::SHIFT_COUNT;

	use super::*;

	/// A shift sequence carrying one shift, which the canonical form places in the inner slot.
	fn single(shift: Shift) -> [Shift; 2] {
		[shift, Shift::IDENTITY]
	}

	/// A constraint system with a handful of distinct shifts, differing between the two segments.
	///
	/// The public segment references `Sll(0)` and `Slr(3)`.
	/// The hidden one references `Sll(0)`, `Sar(7)` and `Rotr(1)`.
	/// Every outer slot is the identity, so the sequences sort by their inner shift alone.
	fn shifted_constraint_system() -> ConstraintSystem {
		// The system has four constants and no inout values, so the public segment is the
		// constants and the hidden one is the private values.
		let public = ValueIndex::constant(1);
		let hidden = ValueIndex::private(1);
		ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([
				vec![
					ShiftedValueIndex::plain(public),
					ShiftedValueIndex::srl(public, 3),
				],
				vec![ShiftedValueIndex::sar(hidden, 7)],
				vec![
					ShiftedValueIndex::rotr(hidden, 1),
					ShiftedValueIndex::plain(hidden),
				],
			])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}

	#[test]
	fn dense_shift_encoding_covers_the_sequences_its_segment_uses() {
		let key_collection =
			build_key_collection(&shifted_constraint_system(), InoutSegment::Public);

		let public_sequences = key_collection
			.public
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(public_sequences, [single(Shift::IDENTITY), single(Shift::srl(3))]);

		let hidden_sequences = key_collection
			.hidden
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(
			hidden_sequences,
			[
				single(Shift::IDENTITY),
				single(Shift::sar(7)),
				single(Shift::rotr(1)),
			]
		);

		// The point of the encoding: a segment names far fewer sequences than the space holds, and
		// the space is now the square of one slot's alphabet.
		assert!(key_collection.hidden.dense_shift_enc.len() < SHIFT_COUNT * SHIFT_COUNT);
	}

	#[test]
	fn dense_shift_encoding_distinguishes_sequences_sharing_an_inner_shift() {
		// Two terms sharing an inner shift but differing outside must land on distinct indices.
		// Keyed on the inner shift alone they would collide and accumulate into one row.
		let hidden = ValueIndex::private(1);
		let cs = ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([
				vec![
					ShiftedValueIndex::new(hidden, [Shift::srl(3), Shift::sll(3)]),
					ShiftedValueIndex::new(hidden, [Shift::srl(3), Shift::sll(5)]),
					ShiftedValueIndex::srl(hidden, 3),
				],
				Vec::new(),
				Vec::new(),
			])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};

		let key_collection = build_key_collection(&cs, InoutSegment::Public);
		let sequences = key_collection
			.hidden
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(
			sequences,
			[
				single(Shift::srl(3)),
				[Shift::srl(3), Shift::sll(3)],
				[Shift::srl(3), Shift::sll(5)],
			]
		);

		// The word's three keys therefore hold three distinct dense indices.
		let mut indices = key_collection
			.hidden
			.word_keys(1)
			.iter()
			.map(|key| key.dense_shift_idx)
			.collect::<Vec<_>>();
		indices.sort_unstable();
		assert_eq!(indices, [0, 1, 2]);
	}

	#[test]
	fn keys_index_their_segments_dense_encoding() {
		let key_collection =
			build_key_collection(&shifted_constraint_system(), InoutSegment::Public);

		// The shift sequences a word's keys name, as its own segment's encoding recovers them.
		let word_sequences = |segment: &KeySegment, word: usize| {
			let mut sequences = segment
				.word_keys(word)
				.iter()
				.map(|key| {
					segment
						.dense_shift_enc
						.iter()
						.nth(key.dense_shift_idx as usize)
						.unwrap()
				})
				.collect::<Vec<_>>();
			sequences.sort();
			sequences
		};

		// Value index 1 is the second public word; value index 5 the second hidden one.
		assert_eq!(
			word_sequences(&key_collection.public, 1),
			[single(Shift::IDENTITY), single(Shift::srl(3))]
		);
		assert_eq!(
			word_sequences(&key_collection.hidden, 1),
			[
				single(Shift::IDENTITY),
				single(Shift::sar(7)),
				single(Shift::rotr(1)),
			]
		);
	}

	#[test]
	fn dense_shift_encoding_survives_serialization() {
		let key_collection =
			build_key_collection(&shifted_constraint_system(), InoutSegment::Public);

		let mut buf = Vec::new();
		key_collection.serialize(&mut buf).unwrap();
		let deserialized = KeyCollection::deserialize(buf.as_slice()).unwrap();

		for (segment, deserialized) in [
			(&key_collection.public, &deserialized.public),
			(&key_collection.hidden, &deserialized.hidden),
		] {
			assert_eq!(
				segment.dense_shift_enc.iter().collect::<Vec<_>>(),
				deserialized.dense_shift_enc.iter().collect::<Vec<_>>()
			);
		}
	}
}
