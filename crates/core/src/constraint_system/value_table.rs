// Copyright 2026 The Binius Developers

use std::ops::Deref;

use super::{ValueIndex, ValueVec, ValueVecLayout};
use crate::word::Word;

/// The witness for a batch of `2^k` independent instances of one circuit, in wire-major order.
///
/// Each wire's values across all instances are grouped together, one wire per row (wire-major):
///
/// ```text
///                 instance 0   instance 1   ...   instance K-1
///   wire 0       [   w        |   w        | ... |   w        ]   <- one row
///   wire 1       [   w        |   w        | ... |   w        ]
///        ...
/// ```
///
/// Each row is one wire and each column is one instance.
/// So a batched interpreter can advance every instance of a wire in a single pass.
///
/// This is the batched counterpart of [`ValueVec`]: one instance read back out with
/// [`Self::instance_value_vec`] is bit-for-bit the value vector populating that instance on its own
/// would produce.
///
/// This table is specialized for the M4 accelerator setting, where the value vector splits with
/// the inout values on the hidden side ([`InoutSegment::Hidden`](super::InoutSegment::Hidden)):
///
/// 1. **Inout wires are committed.** Every instance chooses its own inout words, so they are not
///    one set of shared values a verifier can evaluate. They are committed with the private values
///    instead, at the front of the hidden segment.
/// 2. **Constants are not stored.** The constants live once on the constraint system as a
///    `Vec<Word>`, not replicated per instance. Only the hidden segment — the inout, witness and
///    internal words — is stored here. Reading an instance back takes the constants as an argument.
///
/// So the stored data holds exactly the hidden segment of every instance: `n_hidden_words` rows by
/// `2^log_instances` columns.
///
/// The committed inout words are not tied to any value the verifier knows, so the statement a
/// proof over this table makes is that *some* inout values complete every instance.
///
/// Populating one is the circuit frontend's business, since it takes a circuit to evaluate — see
/// `Circuit::populate_batch`.
///
/// `Data` is the buffer backing the words. It defaults to [`Vec<Word>`], but the prover populates
/// a table straight into a buffer drawn from its pool, so any `Deref<Target = [Word]>` will do.
#[derive(Clone, Debug)]
pub struct ValueTable<Data = Vec<Word>> {
	/// The per-instance value layout, shared by every instance in the batch.
	layout: ValueVecLayout,
	/// The base-2 logarithm of the instance count.
	log_instances: usize,
	/// The number of hidden-word rows the proving protocol commits per instance.
	///
	/// This is the inout values followed by the private ones, which is
	/// [`ConstraintSystem::n_hidden_words`](super::ConstraintSystem::n_hidden_words) under
	/// [`InoutSegment::Hidden`](super::InoutSegment::Hidden).
	n_hidden_words: usize,
	/// The hidden words of every wire, in wire-major order.
	///
	/// Row `r` (for `r` in `0..n_hidden_words`) holds the `2^log_instances` values of hidden wire
	/// `r` — inout value `r`, then private value `r - n_inout` — one per instance. The rows are
	/// laid out contiguously, so the length is `n_hidden_words << log_instances`.
	data: Data,
}

impl<Data: Deref<Target = [Word]>> ValueTable<Data> {
	/// Builds a table from the hidden words of every instance, in wire-major order.
	///
	/// `data` is what [`Self::as_words`] returns: the rows of the hidden segment laid out
	/// contiguously, each holding one wire's value in every instance. The row count follows from
	/// the layout, so it is not passed separately.
	///
	/// This is the seam the frontend populates through; nothing in this crate can evaluate a
	/// circuit to produce the words.
	///
	/// # Panics
	///
	/// Panics if `data` is not `n_hidden_words << log_instances` words long.
	pub fn from_hidden_words(layout: ValueVecLayout, log_instances: usize, data: Data) -> Self {
		let n_hidden_words = layout.combined_len() - layout.offset_inout();
		assert_eq!(
			data.len(),
			n_hidden_words << log_instances,
			"the data must hold one row per hidden word, one column per instance"
		);
		Self {
			layout,
			log_instances,
			n_hidden_words,
			data,
		}
	}

	/// The base-2 logarithm of the number of instances.
	pub const fn log_instances(&self) -> usize {
		self.log_instances
	}

	/// The number of instances in the batch.
	pub const fn n_instances(&self) -> usize {
		1usize << self.log_instances
	}

	/// The per-instance value layout shared by every instance.
	pub const fn layout(&self) -> &ValueVecLayout {
		&self.layout
	}

	/// The number of hidden-word rows per instance, including the protocol's zero padding.
	pub const fn n_hidden_words(&self) -> usize {
		self.n_hidden_words
	}

	/// The whole batch as one flat, wire-major word buffer.
	///
	/// Row `r` (hidden wire `r`) occupies `data[r << log_instances .. (r + 1) << log_instances]`,
	/// holding that wire's value in every instance.
	pub fn as_words(&self) -> &[Word] {
		&self.data
	}

	/// Reconstructs one instance as a standalone single-instance value vector.
	///
	/// Because the constants are not stored in the table, the caller supplies them (they live on
	/// the constraint system). The result is bit-for-bit what populating this instance on its own
	/// would produce, so it can be fed directly to single-instance constraint checking.
	///
	/// # Panics
	///
	/// Panics if the index is not below the instance count, or if `constants` does not match the
	/// layout's constant count.
	pub fn instance_value_vec(&self, instance: usize, constants: &[Word]) -> ValueVec {
		assert!(instance < self.n_instances(), "instance index out of range");
		assert_eq!(
			constants.len(),
			self.layout.n_const,
			"constants length must match the layout's constant count"
		);

		// The constants are the whole public segment; the table stores everything after them.
		let mut values = ValueVec::new(&self.layout);
		for (i, &constant) in constants.iter().enumerate() {
			values[ValueIndex::constant(i as u32)] = constant;
		}

		// Gather this instance's column of hidden values across every row, which the value vector
		// holds from the first inout word onwards.
		for row in 0..self.n_hidden_words {
			*values.word_mut((self.layout.offset_inout() + row) as u32) =
				self.data[(row << self.log_instances) + instance];
		}

		values
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A layout of one constant, two inout values and one private value.
	fn layout() -> ValueVecLayout {
		ValueVecLayout {
			n_const: 1,
			n_inout: 2,
			n_witness: 1,
			n_internal: 0,
			n_scratch: 0,
		}
	}

	/// A two-instance table whose hidden word `r` holds `0x10 * r + instance`.
	fn table() -> ValueTable {
		let n_hidden_words = layout().combined_len() - layout().offset_inout();
		let data = (0..n_hidden_words)
			.flat_map(|row| {
				(0..2).map(move |instance| Word::from_u64(0x10 * row as u64 + instance))
			})
			.collect();
		ValueTable::from_hidden_words(layout(), 1, data)
	}

	#[test]
	fn the_row_count_follows_from_the_layout() {
		let table = table();
		assert_eq!(table.log_instances(), 1);
		assert_eq!(table.n_instances(), 2);
		// Three hidden words: two inout values and one private one.
		assert_eq!(table.n_hidden_words(), 3);
		assert_eq!(table.as_words().len(), 6);
	}

	#[test]
	#[should_panic(expected = "one row per hidden word")]
	fn a_buffer_of_the_wrong_length_is_rejected() {
		ValueTable::from_hidden_words(layout(), 1, vec![Word::ZERO; 5]);
	}

	// An instance is the column at its index, behind the constants the table does not store.
	#[test]
	fn an_instance_reads_back_as_its_own_column() {
		let table = table();
		let constants = [Word::from_u64(0xc0)];

		for instance in 0..2 {
			let values = table.instance_value_vec(instance, &constants);
			assert_eq!(values[ValueIndex::constant(0)], constants[0]);
			assert_eq!(values[ValueIndex::inout(0)], Word::from_u64(instance as u64));
			assert_eq!(values[ValueIndex::inout(1)], Word::from_u64(0x10 + instance as u64));
			assert_eq!(values[ValueIndex::private(0)], Word::from_u64(0x20 + instance as u64));
		}
	}

	#[test]
	#[should_panic(expected = "instance index out of range")]
	fn reading_past_the_last_instance_panics() {
		table().instance_value_vec(2, &[Word::from_u64(0xc0)]);
	}

	#[test]
	#[should_panic(expected = "constants length must match")]
	fn the_wrong_number_of_constants_is_rejected() {
		table().instance_value_vec(0, &[]);
	}
}
