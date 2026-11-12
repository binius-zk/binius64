// Copyright 2026 The Binius Developers

//! The bit-axis fold on a target with AVX-512 and GFNI.
//!
//! The block-matrix kernel needs the field to be the 128-bit GHASH field, which is a property of
//! the type parameter rather than of the target. So this decides once, when the fold is built, and
//! the losing implementation's tables are never materialized.

use std::any::TypeId;

use binius_core::word::Word;
use binius_field::{BinaryField, Ghash128b};

use super::{WORDS_PER_BATCH, gfni::BitFoldMats};
use crate::fold_word::lookup::BitWeightTables;

/// The bit-axis fold: one field element per word, from one weight per bit position.
#[derive(Debug)]
pub struct BitFold<F>(Repr<F>);

/// Which representation the fold chose when it was built.
///
/// The lint that flags a wide variant cannot size the table variant, because its field is a type
/// parameter.
///
/// Wherever the block variant is reachable the tables are 32 KiB, so the blocks are the small
/// variant, not the large one.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum Repr<F> {
	/// The table-free block matrix, 1 KiB, over the 128-bit GHASH field.
	Blocks(BitFoldMats),
	/// The subset-sum tables, 32 KiB, over any other binary field.
	Tables(BitWeightTables<F>),
}

impl<F: BinaryField> BitFold<F> {
	/// Builds the fold from one weight per bit position of a word.
	///
	/// # Panics
	///
	/// Panics unless there is exactly one weight per bit position.
	pub fn new(weights: &[F]) -> Self {
		Self(as_ghash(weights).map_or_else(
			|| Repr::Tables(BitWeightTables::new(weights)),
			|ghash| Repr::Blocks(BitFoldMats::new(ghash)),
		))
	}

	/// Folds a full batch, one field element per word, in word order.
	#[inline]
	pub fn fold_batch(&self, words: &[Word; WORDS_PER_BATCH]) -> [F; WORDS_PER_BATCH] {
		match &self.0 {
			Repr::Blocks(blocks) => from_ghash(blocks.fold_batch(words)),
			Repr::Tables(tables) => words.map(|word| tables.fold(word)),
		}
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

/// Views a scalar slice as GHASH elements, when the scalar type is the 128-bit GHASH field.
///
/// Comparing type identifiers is the standard proof that two type parameters name one type.
///
/// The comparison folds to a constant once the scalar type is known, so it costs nothing at
/// runtime.
fn as_ghash<F: BinaryField>(weights: &[F]) -> Option<&[Ghash128b]> {
	(TypeId::of::<F>() == TypeId::of::<Ghash128b>()).then(|| {
		// SAFETY: the type identifiers just compared equal, so the two slice types have
		// identical layout, alignment and validity.
		unsafe { std::slice::from_raw_parts(weights.as_ptr().cast(), weights.len()) }
	})
}

/// Reads a batch of GHASH elements back as `F`.
///
/// # Panics
///
/// Panics unless the scalar type is the 128-bit GHASH field.
///
/// The block representation is only ever built for that field, so nothing reachable can panic.
#[inline]
fn from_ghash<F: BinaryField>(folded: [Ghash128b; WORDS_PER_BATCH]) -> [F; WORDS_PER_BATCH] {
	assert_eq!(TypeId::of::<F>(), TypeId::of::<Ghash128b>());
	// SAFETY: the type identifiers just compared equal, so the two array types are one type.
	//
	// A copying transmute is used only because the compiler cannot see that equality through the
	// type parameter; the two sizes match exactly.
	unsafe { std::mem::transmute_copy(&folded) }
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Rijndael8b};
	use binius_math::test_utils::random_scalars;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	/// The block kernel against the lookup fold, which is the reference for this whole module.
	///
	/// Both fold the same GF(2)-linear map, so the two must agree on every bit of every word.
	fn check_batch(weights: &[Ghash128b], words: &[Word; WORDS_PER_BATCH]) {
		let tables = BitWeightTables::new(weights);
		let expected = words.map(|word| tables.fold(word));
		assert_eq!(BitFoldMats::new(weights).fold_batch(words), expected);
	}

	/// The GHASH field must actually reach the block kernel, or every property below would be
	/// comparing the lookup fold with itself.
	#[test]
	fn the_ghash_field_takes_the_block_path() {
		let weights = [Ghash128b::ZERO; Word::BITS];
		assert!(matches!(BitFold::new(&weights).0, Repr::Blocks(_)));
	}

	/// Any other binary field has no block kernel, so it must fall back to the tables.
	#[test]
	fn another_field_takes_the_lookup_path() {
		let weights = [Rijndael8b::ZERO; Word::BITS];
		assert!(matches!(BitFold::new(&weights).0, Repr::Tables(_)));
	}

	/// One weight per bit position, one bit position set: the fold of a word is then the XOR of
	/// the basis elements its set bits pick out, which pins the block layout one column at a time.
	#[test]
	fn every_basis_column_lands_where_the_lookup_fold_puts_it() {
		for bit in 0..Word::BITS {
			for out_bit in 0..128 {
				let mut weights = [Ghash128b::ZERO; Word::BITS];
				weights[bit] = Ghash128b::from(1u128 << out_bit);
				check_batch(&weights, &std::array::from_fn(|n| Word::from_u64(1 << (n % 64))));
			}
		}
	}

	/// The words that make every bit of the input either always set or always clear.
	#[test]
	fn the_extreme_words_fold_the_same_way() {
		let mut rng = StdRng::seed_from_u64(0);
		let weights = random_scalars::<Ghash128b>(&mut rng, Word::BITS);
		for word in [Word::ZERO, Word::ALL_ONE, Word::MSB_ONE, Word::ONE] {
			check_batch(&weights, &[word; WORDS_PER_BATCH]);
		}
		// A batch that mixes them, so a cross-word mistake in the transpose network shows up.
		check_batch(
			&weights,
			&std::array::from_fn(|n| {
				if n % 2 == 0 {
					Word::ZERO
				} else {
					Word::ALL_ONE
				}
			}),
		);
	}

	proptest! {
		#[test]
		fn the_block_fold_matches_the_lookup_fold(seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let weights = random_scalars::<Ghash128b>(&mut rng, Word::BITS);
			let words = std::array::from_fn(|_| Word::from_u64(rng.random()));
			check_batch(&weights, &words);
		}

		/// A short batch reads its missing words as zero, and a zero word folds to zero.
		#[test]
		fn a_short_batch_zero_fills(n_words in 0..=WORDS_PER_BATCH, seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let weights = random_scalars::<Ghash128b>(&mut rng, Word::BITS);
			let words = (0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();

			let mut padded = [Word::ZERO; WORDS_PER_BATCH];
			padded[..n_words].copy_from_slice(&words);

			let fold = BitFold::<Ghash128b>::new(&weights);
			prop_assert_eq!(fold.fold_prefix(&words), fold.fold_batch(&padded));
		}
	}
}
