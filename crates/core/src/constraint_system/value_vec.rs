// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::ops::{Deref, DerefMut, Index, IndexMut};

use bytemuck::{Pod, Zeroable};

use super::{ShiftedValueIndex, ValueIndex, ValueSegment, ValueVecLayout};
use crate::word::Word;

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
/// [`ConstraintSystem`](super::ConstraintSystem). It is the primary data structure for both
/// constraint evaluation and polynomial commitment.
///
/// The words are the public values followed by the private ones, stored back to back with no
/// padding. A vector built with [`Self::new`] carries the circuit's scratch tail past them; one
/// built with [`Self::new_from_data`] ends with the private values.
///
/// The words live in a buffer that starts on a 16-byte boundary.
/// That keeps the frequent bulk copies of the vector on the aligned SIMD `memcpy` path.
///
/// # Addressing
///
/// A [`ValueIndex`] names a word by its segment and its position within that segment, and the
/// vector stores the two counts that place each segment in the buffer.
///
/// These are *storage* positions, and they are deliberately not the positions the proving
/// protocol reads a word at: the protocol pads the public segment to a power of two and the
/// hidden segment to at least that width, so it addresses the same word further along. Its
/// addresses come from
/// [`ConstraintSystem::word_offset`](super::ConstraintSystem::word_offset) instead. Nothing needs
/// both, because the prover pads each segment as it packs it into field elements.
#[derive(Clone, Debug)]
pub struct ValueVec {
	/// The number of constants, which is where the inout values start.
	n_const: usize,
	/// The number of public values, which is where the private ones start.
	n_public_values: usize,
	/// The number of private values.
	n_private: usize,
	/// The values followed by the scratch tail, 16-byte aligned.
	data: AlignedWords,
}

impl ValueVec {
	/// Creates a zero-filled value vector holding the sections of the given circuit layout,
	/// including its scratch tail.
	pub fn new(layout: &ValueVecLayout) -> ValueVec {
		ValueVec {
			n_const: layout.n_const,
			n_public_values: layout.offset_witness(),
			n_private: layout.n_private(),
			data: AlignedWords::zeroed(layout.combined_len() + layout.n_scratch),
		}
	}

	/// Creates a value vector from the words of its public and private values.
	///
	/// The vector has no scratch tail; scratch words only exist while a circuit is evaluated.
	///
	/// `n_const` splits the public words into the constants and the inout values, which is what
	/// resolves an [`InOut`](super::ValueSegment::InOut) index. Rebuilding a vector from
	/// serialized segments therefore needs the system describing them, so prefer
	/// [`ConstraintSystem::value_vec_from_data`](super::ConstraintSystem::value_vec_from_data),
	/// which passes it for you.
	pub fn new_from_data(n_const: usize, public: &[Word], private: &[Word]) -> ValueVec {
		// Fresh 16-byte-aligned buffer holding the public words followed by the private ones.
		let mut data = AlignedWords::zeroed(public.len() + private.len());
		data[..public.len()].copy_from_slice(public);
		data[public.len()..].copy_from_slice(private);

		ValueVec {
			n_const,
			n_public_values: public.len(),
			n_private: private.len(),
			data,
		}
	}

	/// Returns one word by its flat position, counting the scratch tail.
	///
	/// This is the view for the few readers that address whole segments rather than named values:
	/// the evaluation form, whose bytecode holds one register per position, and the batch witness,
	/// which copies a segment across including its padding. Everything else names words by
	/// [`ValueIndex`], which cannot reach a padding word.
	#[inline]
	pub fn word(&self, offset: u32) -> Word {
		self.data[offset as usize]
	}

	/// Returns a mutable reference to one word by its flat position, counting the scratch tail.
	///
	/// This is the mutable counterpart of [`Self::word`], which documents when to reach for it.
	#[inline]
	pub fn word_mut(&mut self, offset: u32) -> &mut Word {
		&mut self.data[offset as usize]
	}

	/// The flat position of the word a [`ValueIndex`] names.
	///
	/// A vector built by [`Self::new_from_data`] has no scratch tail, so a scratch index lands
	/// past the last word and panics rather than reading a committed one.
	#[inline]
	const fn word_offset(&self, index: ValueIndex) -> usize {
		let segment_start = match index.segment() {
			ValueSegment::Constant => 0,
			ValueSegment::InOut => self.n_const,
			ValueSegment::Private => self.n_public_values,
			ValueSegment::Scratch => self.size(),
		};
		segment_start + index.index() as usize
	}

	/// The number of values the vector holds, excluding scratch.
	pub const fn size(&self) -> usize {
		self.n_public_values + self.n_private
	}

	/// Returns the public values: the constants followed by the inout values.
	///
	/// These are the words as the circuit declares them, unpadded. The prover pads them up to the
	/// public segment width as it packs them.
	pub fn public(&self) -> &[Word] {
		&self.data[..self.n_public_values]
	}

	/// Returns the private values, unpadded and without scratch space.
	pub fn non_public(&self) -> &[Word] {
		&self.data[self.n_public_values..self.size()]
	}

	/// Returns the combined values vector.
	pub fn combined_witness(&self) -> &[Word] {
		&self.data[..self.size()]
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
		&self.data[self.word_offset(index)]
	}
}

impl IndexMut<ValueIndex> for ValueVec {
	fn index_mut(&mut self, index: ValueIndex) -> &mut Self::Output {
		let offset = self.word_offset(index);
		&mut self.data[offset]
	}
}

#[cfg(test)]
mod tests {
	use proptest::{collection, prelude::any, prop_assert_eq, proptest};

	use super::*;

	#[test]
	fn split_values_vec_and_combine() {
		let layout = ValueVecLayout {
			n_const: 2,
			n_inout: 2,
			n_witness: 2,
			n_internal: 2,
			n_scratch: 0,
		};
		let values = ValueVec::new(&layout);

		let public = values.public();
		let non_public = values.non_public();
		let combined = ValueVec::new_from_data(layout.n_const, public, non_public);
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

			// The sections sit back to back, then the scratch tail:
			//
			//     [0, public.len())                            -> public
			//     [public.len(), public.len() + private.len()) -> private
			//     then the scratch tail
			let layout = ValueVecLayout {
				n_const: 0,
				n_inout: public.len(),
				n_witness: private.len(),
				n_internal: 0,
				n_scratch,
			};

			// A vector built from the layout carries the scratch tail; one built from its
			// segments holds only those words.
			let zeroed = ValueVec::new(&layout);
			let vv = ValueVec::new_from_data(layout.n_const, &public, &private);

			// Alignment survives construction for any word count.
			assert_16_byte_aligned(vv.combined_witness());
			// Both sections read back byte-for-byte what went in.
			prop_assert_eq!(vv.public(), &public[..]);
			prop_assert_eq!(vv.non_public(), &private[..]);

			// The scratch tail past the committed words is zeroed and addressable.
			for slot in 0..n_scratch {
				prop_assert_eq!(zeroed[ValueIndex::scratch(slot as u32)], Word::ZERO);
			}
		}
	}
}
