// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::ConstraintSystem;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::config::LOG_WORDS_PER_ELEM;

/// The committed-multilinear shape for a batch of `2^log_instances` instances.
///
/// The batch witness is the hidden words of every instance, in wire-major order:
/// - one row per hidden word of a single instance.
/// - one column per instance.
///
/// The constants are shared by every instance and are not committed.
/// Only the hidden segment is committed, so the constants stay on the constraint system.
/// The instance count is already a power of two, so the instance axis needs no padding.
/// Only the hidden-word count is rounded up.
///
/// ```text
///                 instance 0   instance 1   ...   instance K-1
///   hidden word 0 [   w        |   w        | ... |   w        ]
///   hidden word 1 [   w        |   w        | ... |   w        ]
///         ...
///   rows padded up to 2^log_hidden_words with zeros
///
///   high coordinates -> hidden word index
///   low  coordinates -> instance index
/// ```
///
/// The prover packs this from the wire-major table.
/// The verifier derives the same shape from the constraint system.
/// Deriving it the same way keeps the committed buffer and the oracle the same size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchCommitLayout {
	/// The base-2 logarithm of the instance count.
	pub log_instances: usize,
	/// The base-2 logarithm of one instance's hidden-word count, rounded up.
	///
	/// The hidden-word rows are padded up to this many with zeros.
	pub log_hidden_words: usize,
	/// The base-2 logarithm of the total committed word count across all instances.
	pub log_witness_words: usize,
	/// The base-2 logarithm of the committed field-element count.
	///
	/// Two 64-bit words pack into one field element.
	/// So this is `log_witness_words` minus the words-per-element logarithm.
	pub log_witness_elems: usize,
}

impl BatchCommitLayout {
	/// Builds the layout for `2^log_instances` instances.
	///
	/// # Arguments
	///
	/// - `hidden_words`: hidden words of one instance, before power-of-two padding.
	/// - `log_instances`: base-2 logarithm of the instance count.
	pub fn new(hidden_words: usize, log_instances: usize) -> Self {
		// Pad one instance's hidden-word rows up to a power-of-two count.
		// Floor at the words-per-element log so the committed buffer is a whole number of elements.
		let log_hidden_words = log2_ceil_usize(hidden_words).max(LOG_WORDS_PER_ELEM);

		// The hidden words ride the high coordinates and the instances the low.
		// So the total word log is additive.
		let log_witness_words = log_hidden_words + log_instances;

		// Two words share one field element.
		let log_witness_elems = log_witness_words - LOG_WORDS_PER_ELEM;

		Self {
			log_instances,
			log_hidden_words,
			log_witness_words,
			log_witness_elems,
		}
	}

	/// Builds the layout from a constraint system and an instance count.
	///
	/// # Arguments
	///
	/// - `cs`: the single-instance constraint system shared by every instance.
	/// - `log_instances`: base-2 logarithm of the instance count.
	pub fn for_constraint_system(cs: &ConstraintSystem, log_instances: usize) -> Self {
		// Only the hidden segment is committed.
		// The shared constants are known to the verifier, so they stay off the oracle.
		Self::new(cs.value_vec_layout.n_hidden_words, log_instances)
	}

	/// The number of hidden-word rows one instance occupies after power-of-two padding.
	pub const fn padded_hidden_words(&self) -> usize {
		1 << self.log_hidden_words
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn power_of_two_hidden_count_needs_no_padding() {
		// Fixture state: 8 hidden words/instance (already a power of two), 2^3 = 8 instances.
		//
		//     log_hidden_words  = log2(8)        = 3
		//     log_witness_words = 3 + 3          = 6   (64 words total)
		//     log_witness_elems = 6 - 1          = 5   (32 field elements, 2 words each)
		let layout = BatchCommitLayout::new(8, 3);

		assert_eq!(layout.log_hidden_words, 3);
		assert_eq!(layout.padded_hidden_words(), 8);
		assert_eq!(layout.log_witness_words, 6);
		assert_eq!(layout.log_witness_elems, 5);
	}

	#[test]
	fn non_power_of_two_hidden_count_rounds_up() {
		// Fixture state: 6 hidden words/instance, 2^2 = 4 instances.
		//
		// Mutation: 6 is not a power of two, so the hidden rows pad up to 8.
		//
		//     log_hidden_words  = ceil(log2(6))  = 3   (8 padded rows)
		//     log_witness_words = 3 + 2          = 5   (32 words total)
		//     log_witness_elems = 5 - 1          = 4   (16 field elements)
		let layout = BatchCommitLayout::new(6, 2);

		assert_eq!(layout.log_hidden_words, 3);
		assert_eq!(layout.padded_hidden_words(), 8);
		assert_eq!(layout.log_witness_words, 5);
		assert_eq!(layout.log_witness_elems, 4);
	}

	#[test]
	fn tiny_hidden_count_floors_at_words_per_element() {
		// Fixture state: 1 hidden word/instance, a single instance (2^0).
		//
		// Invariant: the committed buffer is never smaller than one field element.
		// So the padded hidden-word count floors at the words-per-element count (2 words).
		//
		//     log_hidden_words = max(log2(1), LOG_WORDS_PER_ELEM) = max(0, 1) = 1
		let layout = BatchCommitLayout::new(1, 0);

		assert_eq!(layout.log_hidden_words, LOG_WORDS_PER_ELEM);
		assert_eq!(layout.padded_hidden_words(), 1 << LOG_WORDS_PER_ELEM);
		assert_eq!(layout.log_witness_elems, 0);
	}
}
