// Copyright 2026 The Binius Developers

//! The bit-axis fold every target can run: the subset-sum tables, a word at a time.

use binius_core::word::Word;
use binius_field::BinaryField;

use super::WORDS_PER_BATCH;
use crate::fold_word::lookup::BitWeightTables;

/// The bit-axis fold: one field element per word, from one weight per bit position.
#[derive(Debug)]
pub struct BitFold<F>(BitWeightTables<F>);

impl<F: BinaryField> BitFold<F> {
	/// Builds the fold from one weight per bit position of a word.
	///
	/// # Panics
	///
	/// Panics unless there is exactly one weight per bit position.
	pub fn new(weights: &[F]) -> Self {
		Self(BitWeightTables::new(weights))
	}

	/// Folds a full batch, one field element per word, in word order.
	#[inline]
	pub fn fold_batch(&self, words: &[Word; WORDS_PER_BATCH]) -> [F; WORDS_PER_BATCH] {
		words.map(|word| self.0.fold(word))
	}

	/// Folds a short batch, leaving the entries past its end zero.
	///
	/// # Panics
	///
	/// Panics if `words` is longer than one batch.
	#[inline]
	pub fn fold_prefix(&self, words: &[Word]) -> [F; WORDS_PER_BATCH] {
		super::fold_zero_padded(words, |batch| self.fold_batch(batch))
	}
}
