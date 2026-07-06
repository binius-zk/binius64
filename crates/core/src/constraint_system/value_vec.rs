// Copyright 2025 Irreducible Inc.
use std::ops::{Deref, DerefMut, Index, IndexMut};

use bytemuck::{Pod, Zeroable};

use super::{ShiftedValueIndex, ValueIndex, ValueVecLayout};
use crate::{error::ConstraintSystemError, word::Word};

/// A 16-byte-aligned pair of words, the storage block of the aligned word buffer.
///
/// - A word is 8 bytes, so a plain vector of words only lands on a 16-byte boundary half the time.
/// - Two words inside a 16-byte-aligned block force every allocation onto that boundary.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct WordPair([Word; 2]);

// SAFETY: the two impls below assert that any bit pattern is a valid `WordPair`.
// - Each word is plain-old-data, so every field is plain-old-data.
// - Two 8-byte words exactly fill the 16-byte size, leaving no padding bytes.
unsafe impl Zeroable for WordPair {}
unsafe impl Pod for WordPair {}

/// A heap-allocated buffer of words whose first element is 16-byte aligned.
///
/// The value vector is copied in bulk on the prover's hot path:
/// - cloned wholesale,
/// - packed into field elements,
/// - sliced out into owned vectors.
///
/// A 16-byte-aligned start keeps each of those copies on the aligned SIMD `memcpy` path.
/// An 8-byte-aligned plain vector would instead pay a misalignment prologue half the time.
///
/// Storage groups two words per block, so capacity rounds up to an even count.
/// The valid word count is tracked separately from the block count.
/// An odd count leaves the last block's second word as zeroed, unused padding.
#[derive(Clone, Debug)]
struct AlignedWords {
	/// Backing store of 16-byte-aligned blocks, one block per two words.
	blocks: Vec<WordPair>,
	/// Number of valid words, at most twice the block count.
	len: usize,
}

impl AlignedWords {
	/// Allocates a 16-byte-aligned buffer of `len` zeroed words.
	fn zeroed(len: usize) -> Self {
		Self {
			// Round up to whole blocks; the macro zero-fills, so every word starts at zero.
			blocks: vec![WordPair([Word::ZERO; 2]); len.div_ceil(2)],
			len,
		}
	}
}

impl Deref for AlignedWords {
	type Target = [Word];

	fn deref(&self) -> &[Word] {
		// Reinterpret the aligned blocks as twice as many words.
		// Slice to the valid count, dropping the padding word of an odd buffer.
		&bytemuck::cast_slice(&self.blocks)[..self.len]
	}
}

impl DerefMut for AlignedWords {
	fn deref_mut(&mut self) -> &mut [Word] {
		// Same reinterpretation as the shared view, but handing out mutable words.
		&mut bytemuck::cast_slice_mut(&mut self.blocks)[..self.len]
	}
}

/// The vector of values used in constraint evaluation and proof generation.
///
/// `ValueVec` is the concrete instantiation of values that satisfy (or should satisfy) a
/// [`ConstraintSystem`](super::ConstraintSystem). It follows the layout defined by
/// [`ValueVecLayout`] and serves as the primary data structure for both constraint evaluation and
/// polynomial commitment.
///
/// Between these sections, there may be padding regions to satisfy alignment requirements.
///
/// The words live in a buffer that starts on a 16-byte boundary.
/// That keeps the frequent bulk copies of the vector on the aligned SIMD `memcpy` path.
#[derive(Clone, Debug)]
pub struct ValueVec {
	/// Section offsets and counts that partition the words below.
	layout: ValueVecLayout,
	/// The committed words followed by the scratch tail, 16-byte aligned.
	data: AlignedWords,
}

impl ValueVec {
	/// Creates a new value vector with the given layout.
	///
	/// The values are filled with zeros.
	pub fn new(layout: ValueVecLayout) -> ValueVec {
		let size = layout.committed_total_len + layout.n_scratch;
		ValueVec {
			layout,
			data: AlignedWords::zeroed(size),
		}
	}

	/// Creates a new value vector with the given layout and data.
	///
	/// The data is checked to have the correct length.
	pub fn new_from_data(
		layout: ValueVecLayout,
		public: Vec<Word>,
		private: Vec<Word>,
	) -> Result<ValueVec, ConstraintSystemError> {
		let committed_len = public.len() + private.len();
		if committed_len != layout.committed_total_len {
			return Err(ConstraintSystemError::ValueVecLenMismatch {
				expected: layout.committed_total_len,
				actual: committed_len,
			});
		}

		// Full buffer = committed words + scratch tail.
		let full_len = layout.committed_total_len + layout.n_scratch;
		// Fresh 16-byte-aligned buffer; the scratch tail past the committed words stays zeroed.
		let mut data = AlignedWords::zeroed(full_len);
		// Public words occupy the front of the committed region.
		data[..public.len()].copy_from_slice(&public);
		// Private words follow, filling the rest of the committed region.
		data[public.len()..committed_len].copy_from_slice(&private);

		Ok(ValueVec { layout, data })
	}

	/// The total size of the committed portion of the vector (excluding scratch).
	pub const fn size(&self) -> usize {
		self.layout.committed_total_len
	}

	/// Returns the public portion of the values vector.
	pub fn public(&self) -> &[Word] {
		&self.data[..self.layout.offset_witness]
	}

	/// Return all non-public values (witness + internal) without scratch space.
	pub fn non_public(&self) -> &[Word] {
		&self.data[self.layout.offset_witness..self.layout.committed_total_len]
	}

	/// Returns the witness portion of the values vector.
	pub fn witness(&self) -> &[Word] {
		let start = self.layout.offset_witness;
		let end = start + self.layout.n_witness;
		&self.data[start..end]
	}

	/// Returns the combined values vector.
	pub fn combined_witness(&self) -> &[Word] {
		let start = 0;
		let end = self.layout.committed_total_len;
		&self.data[start..end]
	}

	/// Evaluates an operand against this witness.
	///
	/// An operand is the XOR of its shifted-value terms.
	/// An empty operand evaluates to the zero word, the XOR identity.
	#[inline]
	pub fn eval_operand(&self, operand: &[ShiftedValueIndex]) -> Word {
		// Fold each shifted term into the running XOR, starting from the identity.
		operand
			.iter()
			.fold(Word::ZERO, |acc, term| acc ^ term.eval(self))
	}
}

impl Index<ValueIndex> for ValueVec {
	type Output = Word;

	fn index(&self, index: ValueIndex) -> &Self::Output {
		&self.data[index.0 as usize]
	}
}

impl IndexMut<ValueIndex> for ValueVec {
	fn index_mut(&mut self, index: ValueIndex) -> &mut Self::Output {
		&mut self.data[index.0 as usize]
	}
}

#[cfg(test)]
mod tests {
	use proptest::{collection, prelude::any, prop_assert_eq, proptest};

	use super::*;

	#[test]
	fn split_values_vec_and_combine() {
		let values = ValueVec::new(ValueVecLayout {
			n_const: 2,
			n_inout: 2,
			n_witness: 2,
			n_internal: 2,
			offset_inout: 2,
			offset_witness: 4,
			committed_total_len: 8,
			n_scratch: 0,
		});

		let public = values.public();
		let non_public = values.non_public();
		let combined =
			ValueVec::new_from_data(values.layout.clone(), public.to_vec(), non_public.to_vec())
				.unwrap();
		assert_eq!(combined.combined_witness(), values.combined_witness());
	}

	// The property that makes the optimization work: the first word sits on a 16-byte boundary.
	fn assert_16_byte_aligned(words: &[Word]) {
		assert_eq!(words.as_ptr() as usize % 16, 0);
	}

	#[test]
	fn zeroed_is_aligned_zero_filled_and_correct_length() {
		// Cases:
		//   0      -> empty buffer, no blocks
		//   1, 3   -> odd, so the last block's second word is padding
		//   2, 16  -> even, every block fully used
		//   17     -> odd and spans many blocks
		for len in [0, 1, 2, 3, 16, 17] {
			let words = AlignedWords::zeroed(len);
			// The view reports the requested word count, not the rounded-up block capacity.
			assert_eq!(words.len(), len);
			// Alignment must hold for every length, including the empty buffer.
			assert_16_byte_aligned(&words);
			// A freshly allocated buffer is entirely zero.
			assert!(words.iter().all(|&w| w == Word::ZERO));
		}
	}

	#[test]
	fn deref_mut_writes_are_visible_through_deref() {
		// Length 5 is odd, so the last block holds one valid word and one padding word.
		let mut words = AlignedWords::zeroed(5);
		// Write 1..=5 through the mutable view; this must not touch the padding word.
		for (i, w) in words.iter_mut().enumerate() {
			*w = Word::from_u64(i as u64 + 1);
		}
		// The shared view reads back exactly the five words written.
		assert_eq!(
			&*words,
			&[
				Word::from_u64(1),
				Word::from_u64(2),
				Word::from_u64(3),
				Word::from_u64(4),
				Word::from_u64(5),
			]
		);
	}

	proptest! {
		#[test]
		fn value_vec_preserves_words_and_alignment(
			public in collection::vec(any::<u64>(), 4..32usize),
			n_witness in 0..32usize,
			n_scratch in 0..16usize,
		) {
			// Public words come straight from the strategy; private words use a recognizable pattern.
			let public: Vec<Word> = public.into_iter().map(Word).collect();
			let private: Vec<Word> = (0..n_witness).map(|i| Word::from_u64(0xdead_0000 + i as u64)).collect();

			// The public section is padded to a power of two.
			// The witness section follows it, then the scratch tail.
			//
			//     [0, offset_witness)                    -> public  (power of two)
			//     [offset_witness, committed_total_len)  -> witness
			//     [committed_total_len, +n_scratch)      -> scratch
			let offset_witness = public.len().next_power_of_two();
			let committed_total_len = offset_witness + private.len();
			let layout = ValueVecLayout {
				n_const: 0,
				n_inout: public.len(),
				n_witness: private.len(),
				n_internal: 0,
				offset_inout: 0,
				offset_witness,
				committed_total_len,
				n_scratch,
			};

			// The public input must fill its whole power-of-two section, so zero-pad it.
			let mut public_padded = public;
			public_padded.resize(offset_witness, Word::ZERO);

			let vv = ValueVec::new_from_data(layout, public_padded.clone(), private.clone()).unwrap();

			// Alignment survives construction for any word count.
			assert_16_byte_aligned(vv.combined_witness());
			// Both sections read back byte-for-byte what went in.
			prop_assert_eq!(vv.public(), &public_padded[..]);
			prop_assert_eq!(vv.witness(), &private[..]);

			// The scratch tail past the committed words is zeroed and addressable.
			for i in committed_total_len..committed_total_len + n_scratch {
				prop_assert_eq!(vv[ValueIndex(i as u32)], Word::ZERO);
			}
		}
	}
}
