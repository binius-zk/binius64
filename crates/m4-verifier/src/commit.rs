// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::ConstraintSystem;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::config::LOG_WORDS_PER_ELEM;

/// The committed-multilinear shape for a batch of `2^log_instances` instances.
///
/// The batch witness is a 2-D table.
/// - One row per instance.
/// - One column per committed word of a single instance.
///
/// Each instance is padded to a power-of-two word count.
/// The padded instances are then concatenated and committed as one multilinear.
/// So the instance index becomes the high-order coordinates.
/// One instance then occupies a contiguous sub-cube.
///
/// ```text
///     instance 0        instance 1        instance K-1
///   [ words | pad ]   [ words | pad ]   [ words | pad ]
///
///   each block is 2^log_instance_words words
///   high coordinates -> instance index
///   low  coordinates -> word index in one instance
/// ```
///
/// The prover packs this from the table.
/// The verifier derives the same shape from the constraint system.
/// Deriving it the same way keeps the committed buffer and the oracle the same size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchCommitLayout {
	/// The base-2 logarithm of the instance count.
	pub log_instances: usize,
	/// The base-2 logarithm of the padded word count of one instance.
	///
	/// One instance is padded up to this many words.
	/// This puts the instance index on the high-order coordinates.
	pub log_instance_words: usize,
	/// The base-2 logarithm of the total padded word count across all instances.
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
	/// - `instance_words`: committed words of one instance, before power-of-two padding.
	/// - `log_instances`: base-2 logarithm of the instance count.
	pub fn new(instance_words: usize, log_instances: usize) -> Self {
		// Pad one instance up to a power-of-two word count.
		// Floor at the words-per-element log so an instance is a whole number of elements.
		// Then packing two words per element never straddles an instance boundary.
		let log_instance_words = log2_ceil_usize(instance_words).max(LOG_WORDS_PER_ELEM);

		// The instance index rides on the high coordinates, so the total log is additive.
		let log_witness_words = log_instance_words + log_instances;

		// Two words share one field element.
		let log_witness_elems = log_witness_words - LOG_WORDS_PER_ELEM;

		Self {
			log_instances,
			log_instance_words,
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
		// One instance's committed length drives the per-instance word count.
		Self::new(cs.value_vec_len(), log_instances)
	}

	/// The number of words one instance occupies after power-of-two padding.
	pub const fn padded_instance_words(&self) -> usize {
		1 << self.log_instance_words
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn power_of_two_instance_needs_no_padding() {
		// Fixture state: 8 committed words/instance (already a power of two), 2^3 = 8 instances.
		//
		//     log_instance_words = log2(8)        = 3
		//     log_witness_words  = 3 + 3          = 6   (64 words total)
		//     log_witness_elems  = 6 - 1          = 5   (32 field elements, 2 words each)
		let layout = BatchCommitLayout::new(8, 3);

		assert_eq!(layout.log_instance_words, 3);
		assert_eq!(layout.padded_instance_words(), 8);
		assert_eq!(layout.log_witness_words, 6);
		assert_eq!(layout.log_witness_elems, 5);
	}

	#[test]
	fn non_power_of_two_instance_rounds_up() {
		// Fixture state: 6 committed words/instance, 2^2 = 4 instances.
		//
		// Mutation: 6 is not a power of two, so one instance is padded up to 8 words.
		//
		//     log_instance_words = ceil(log2(6))  = 3   (8 padded words)
		//     log_witness_words  = 3 + 2          = 5   (32 words total)
		//     log_witness_elems  = 5 - 1          = 4   (16 field elements)
		let layout = BatchCommitLayout::new(6, 2);

		assert_eq!(layout.log_instance_words, 3);
		assert_eq!(layout.padded_instance_words(), 8);
		assert_eq!(layout.log_witness_words, 5);
		assert_eq!(layout.log_witness_elems, 4);
	}

	#[test]
	fn tiny_instance_floors_at_words_per_element() {
		// Fixture state: 1 committed word/instance, a single instance (2^0).
		//
		// Invariant: an instance is never smaller than one field element.
		// So the per-instance word count floors at the words-per-element count (2 words).
		//
		//     log_instance_words = max(log2(1), LOG_WORDS_PER_ELEM) = max(0, 1) = 1
		let layout = BatchCommitLayout::new(1, 0);

		assert_eq!(layout.log_instance_words, LOG_WORDS_PER_ELEM);
		assert_eq!(layout.padded_instance_words(), 1 << LOG_WORDS_PER_ELEM);
		assert_eq!(layout.log_witness_elems, 0);
	}
}
