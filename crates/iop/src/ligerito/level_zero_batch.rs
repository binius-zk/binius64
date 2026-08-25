// Copyright 2026 The Binius Developers

//! What level 0 folds on top of one transparent message's own lanes.

use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::channel::OracleSpec;

/// The messages one ladder opens together, as level 0's proximity test sees them.
///
/// Level 0 folds a tensor over more than its own lane index.
/// A batch adds the index of the message inside it.
/// A masked message adds the lane holding its mask.
/// Both widen the row union the proximity test runs against, and neither is bought back by queries.
///
/// The message lengths are deliberately absent.
/// A shorter message carries fewer lanes, which only ever narrows the union.
/// So the count and the presence of a mask are the whole of what the ceiling depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelZeroBatch {
	/// How many separately committed messages level 0 opens together.
	n_oracles: usize,
	/// Whether any of those messages is interleaved with a mask.
	is_masked: bool,
}

impl LevelZeroBatch {
	/// A batch of `n_oracles` messages, `is_masked` recording whether any of them carries a mask.
	///
	/// ## Preconditions
	///
	/// * `n_oracles` is positive.
	pub const fn new(n_oracles: usize, is_masked: bool) -> Self {
		assert!(n_oracles > 0, "precondition: a ladder opens at least one message");
		Self {
			n_oracles,
			is_masked,
		}
	}

	/// One message, committed in the clear.
	///
	/// This is the shape a ladder is searched and reported against.
	pub const fn single() -> Self {
		Self::new(1, false)
	}

	/// The batch a channel opening these oracles together presents to level 0.
	///
	/// ## Preconditions
	///
	/// * `specs` is non-empty.
	pub fn from_specs(specs: &[OracleSpec]) -> Self {
		Self::new(specs.len(), specs.iter().any(|spec| spec.is_zk))
	}

	/// How many separately committed messages level 0 opens together.
	pub const fn n_oracles(&self) -> usize {
		self.n_oracles
	}

	/// Whether any message in the batch is interleaved with a mask.
	pub const fn is_masked(&self) -> bool {
		self.is_masked
	}

	/// log2 the factor this batch multiplies level 0's row union by.
	///
	/// ```text
	///     one message,   no mask  ->  2^log_lanes rows          extra 0
	///     k messages,    no mask  ->  2^log_lanes * k rows      extra ceil(log2 k)
	///     k messages, any masked  ->  2^log_lanes * k * 2 rows  extra ceil(log2 k) + 1
	/// ```
	///
	/// A batch where only some messages are masked is charged the full extra bit.
	/// Its union is at most twice the unmasked one, and the bound is what the ceiling reads.
	pub fn log_extra_rows(&self) -> usize {
		log2_ceil_usize(self.n_oracles) + usize::from(self.is_masked)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn one_transparent_message_widens_nothing() {
		// Fixture state: the shape every ladder is searched against.
		let batch = LevelZeroBatch::single();
		assert_eq!(batch.n_oracles(), 1);
		assert!(!batch.is_masked());
		assert_eq!(batch.log_extra_rows(), 0);
	}

	#[test]
	fn a_mask_costs_the_same_bit_a_doubling_of_the_batch_costs() {
		// Invariant: a mask is one more interleaved lane, so it doubles the union exactly as one
		// more message does. The two levers therefore have to be charged at the same rate.
		//
		//     1 message,  masked -> extra 1
		//     2 messages, clear  -> extra 1
		assert_eq!(LevelZeroBatch::new(1, true).log_extra_rows(), 1);
		assert_eq!(LevelZeroBatch::new(2, false).log_extra_rows(), 1);

		// Both at once, and the two charges add.
		assert_eq!(LevelZeroBatch::new(2, true).log_extra_rows(), 2);

		// A count that is not a power of two rounds up, since the tensor spans a whole hypercube.
		assert_eq!(LevelZeroBatch::new(5, false).log_extra_rows(), 3);
		assert_eq!(LevelZeroBatch::new(5, true).log_extra_rows(), 4);
	}

	#[test]
	fn a_batch_reads_its_shape_off_the_oracles_it_opens() {
		// Fixture state: three oracles of three lengths, the middle one masked.
		let specs = [
			OracleSpec::new(8),
			OracleSpec::new_zk(7),
			OracleSpec::new(5),
		];
		let batch = LevelZeroBatch::from_specs(&specs);
		assert_eq!(batch.n_oracles(), 3);
		// One masked message is enough to widen the union, so the flag is a disjunction.
		assert!(batch.is_masked());
		assert_eq!(batch.log_extra_rows(), 3);

		// The same three lengths with nothing masked keep the plain batching charge.
		let clear = [OracleSpec::new(8), OracleSpec::new(7), OracleSpec::new(5)];
		assert_eq!(LevelZeroBatch::from_specs(&clear).log_extra_rows(), 2);
	}

	#[test]
	#[should_panic(expected = "precondition: a ladder opens at least one message")]
	fn a_batch_of_no_messages_is_refused() {
		// A level 0 with no codeword has no rows to fold, so it has no union to price.
		LevelZeroBatch::new(0, false);
	}
}
