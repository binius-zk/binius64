// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The bit-axis lookup tables, one per byte of a word.

use std::{array, iter};

use binius_core::word::Word;
use binius_field::{BinaryField, Divisible, util::expand_subset_sums_array};

use crate::bit_matrix::WEIGHTS_PER_TABLE;

/// Subset sums of a word's bit weights, eight bit positions at a time.
///
/// One table per byte of a word, each holding all 256 subset sums of that byte's eight weights.
/// A byte of the word then indexes those eight weights' whole contribution in one load.
///
/// This is the bit-axis counterpart of the row tables the word-axis folder holds, and the two are
/// built the same way.
///
/// Storing the tables as a fixed-length array rather than a heap-allocated one is worth 9 to 18
/// percent over the generic bytewise lookup in the field crate, which is why this specialization
/// exists at all.
///
/// This uses the [Method of Four Russians]: precompute one lookup table per byte position, then
/// combine the word's bytes.
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
#[derive(Debug)]
pub struct BitWeightTables<F> {
	/// One table per byte of a word, holding that byte's eight weights' subset sums.
	tables: [[F; 1 << WEIGHTS_PER_TABLE]; Word::BYTES],
}

impl<F: BinaryField> BitWeightTables<F> {
	/// Builds the tables from one weight per bit position of a word.
	///
	/// # Panics
	///
	/// Panics unless there is exactly one weight per bit position.
	pub fn new(weights: &[F]) -> Self {
		assert_eq!(weights.len(), Word::BITS);

		// Byte `i` of a word carries bit positions `8i` through `8i + 7`, so its table covers
		// exactly those eight weights.
		let tables = array::from_fn(|byte| {
			let group: [F; WEIGHTS_PER_TABLE] = weights
				[byte * WEIGHTS_PER_TABLE..(byte + 1) * WEIGHTS_PER_TABLE]
				.try_into()
				.expect("a word has Word::BYTES groups of WEIGHTS_PER_TABLE weights");
			expand_subset_sums_array(group)
		});

		Self { tables }
	}

	/// The inner product of a word's bits with the weights.
	///
	/// A clear bit reads as zero and a set bit as one, so the sum runs over the set bits only.
	#[inline]
	pub fn fold(&self, word: Word) -> F {
		// Each byte selects its group's whole contribution, and the eight groups partition the
		// word's bit positions, so summing them is the full inner product.
		iter::zip(Divisible::<u8>::ref_iter(&word.0), &self.tables)
			.map(|(byte, table)| table[byte as usize])
			.fold(F::ZERO, |acc, contribution| acc + contribution)
	}
}
