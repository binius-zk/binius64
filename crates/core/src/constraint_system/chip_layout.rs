// Copyright 2026 The Binius Developers

//! Placement of chip blocks in the hidden segment a composite system commits.
//!
//! A chip is one constraint system proved over `2^k` instances at once.
//! A composite system holds a main circuit and the chips it calls.
//! All of it commits as one hidden segment:
//!
//! ```text
//! hidden segment, 2^N words:
//!
//!     main private words
//!     chip 0 block          aligned to its own length
//!     chip 1 block          aligned to its own length
//!     padding
//! ```
//!
//! Every block is a power of two words long.
//! Every block starts at a multiple of its own length.
//!
//! That alignment buys one identity:
//!
//! ```text
//! block_mle(p) == hidden_mle(p || bits(block_index))
//! ```
//!
//! A chip's reduction ends with a claim on its own block.
//! The identity rewrites it as a claim on the whole segment, at a point with fixed high bits.
//! So every chip's claim, and the main circuit's own, opens the same commitment.

use binius_utils::checked_arithmetics::log2_ceil_usize;

/// One chip's block of the hidden segment.
///
/// The block holds every instance of every hidden word of one chip, wire-major:
///
/// ```text
///     wire 0:  instance 0, instance 1, ..., instance 2^k - 1
///     wire 1:  instance 0, instance 1, ..., instance 2^k - 1
///     ...
/// ```
///
/// The instance index takes the low coordinates, the wire index the high ones.
/// So a word's address is affine in the instance, with a power-of-two stride.
/// That is what lets a uniform chip system and a non-uniform caller name the same word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChipBlock {
	/// The chip's hidden words per instance, as a base-2 logarithm.
	///
	/// The true count is rounded up to this power of two.
	/// The rows past it hold zeros.
	log_words_per_instance: usize,
	/// The base-2 logarithm of the instance count.
	log_instances: usize,
	/// The block's first word, counted from the start of the hidden segment.
	///
	/// Always a multiple of [`Self::n_words`].
	offset: usize,
}

impl ChipBlock {
	/// The base-2 logarithm of the block's length in words.
	pub const fn log_words(&self) -> usize {
		self.log_words_per_instance + self.log_instances
	}

	/// The block's length in words.
	pub const fn n_words(&self) -> usize {
		1 << self.log_words()
	}

	/// The block's first word, counted from the start of the hidden segment.
	pub const fn offset(&self) -> usize {
		self.offset
	}

	/// The base-2 logarithm of the instance count.
	pub const fn log_instances(&self) -> usize {
		self.log_instances
	}

	/// The base-2 logarithm of the padded hidden-word count of one instance.
	pub const fn log_words_per_instance(&self) -> usize {
		self.log_words_per_instance
	}

	/// The block's index among the equal-sized slots the hidden segment splits into.
	///
	/// The block is aligned, so this is exact.
	/// Its bits are the high coordinates that lift a claim on this block to the whole segment.
	pub const fn block_index(&self) -> usize {
		self.offset >> self.log_words()
	}

	/// The hidden-segment word holding one instance of one of the chip's hidden wires.
	///
	/// # Panics
	///
	/// Panics unless `wire` is below the padded per-instance word count.
	/// Panics unless `instance` is below the instance count.
	/// A word outside the block belongs to another chip.
	pub const fn word(&self, wire: usize, instance: usize) -> usize {
		assert!(wire < 1 << self.log_words_per_instance, "wire is outside the block");
		assert!(instance < 1 << self.log_instances, "instance is outside the block");
		self.offset + (wire << self.log_instances) + instance
	}
}

/// The hidden segment of a composite system.
///
/// The main circuit's private words come first, so their addresses survive adding a chip.
/// One block per chip follows, placed largest first.
/// That order bounds the total padding by the largest block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChipLayout {
	/// The main circuit's own private words, at the start of the segment.
	n_main_private: usize,
	/// One block per chip, indexed by chip, in the order the chips were given.
	blocks: Vec<ChipBlock>,
	/// The base-2 logarithm of the whole segment's length in words.
	log_hidden_words: usize,
}

impl ChipLayout {
	/// Lays out the hidden segment for a main circuit and its chips.
	///
	/// Each chip contributes `(hidden_words_per_instance, log_instances)`.
	/// The word count is rounded up to a power of two.
	/// The instance count is already one.
	///
	/// Each block goes at the next offset that is a multiple of its own length.
	/// Taking the largest first keeps every gap inside an earlier one.
	pub fn new(n_main_private: usize, chips: impl IntoIterator<Item = (usize, usize)>) -> Self {
		let mut blocks = chips
			.into_iter()
			.map(|(words_per_instance, log_instances)| ChipBlock {
				log_words_per_instance: log2_ceil_usize(words_per_instance),
				log_instances,
				// Assigned below, once the placement order is known.
				offset: 0,
			})
			.collect::<Vec<_>>();

		// A block's own alignment is the only thing that can leave a gap.
		// So placing the coarsest alignment first keeps every later gap inside an earlier one.
		let mut order = (0..blocks.len()).collect::<Vec<_>>();
		order.sort_by_key(|&chip| std::cmp::Reverse(blocks[chip].log_words()));

		let mut cursor = n_main_private;
		for chip in order {
			let n_words = blocks[chip].n_words();
			// Round the cursor up to this block's own length.
			cursor = cursor.next_multiple_of(n_words);
			blocks[chip].offset = cursor;
			cursor += n_words;
		}

		Self {
			n_main_private,
			blocks,
			log_hidden_words: log2_ceil_usize(cursor),
		}
	}

	/// The main circuit's own private words, at the start of the segment.
	pub const fn n_main_private(&self) -> usize {
		self.n_main_private
	}

	/// One block per chip, indexed by chip.
	pub fn blocks(&self) -> &[ChipBlock] {
		&self.blocks
	}

	/// The block of one chip.
	///
	/// # Panics
	///
	/// Panics if `chip` is not a chip of this layout.
	pub fn block(&self, chip: usize) -> &ChipBlock {
		&self.blocks[chip]
	}

	/// The base-2 logarithm of the segment's length in words.
	pub const fn log_hidden_words(&self) -> usize {
		self.log_hidden_words
	}

	/// The segment's length in words, including the padding after the last block.
	pub const fn n_hidden_words(&self) -> usize {
		1 << self.log_hidden_words
	}

	/// The words the layout places, excluding the padding alignment leaves behind.
	///
	/// The difference against [`Self::n_hidden_words`] is what the commitment spends on padding.
	pub fn n_placed_words(&self) -> usize {
		self.n_main_private + self.blocks.iter().map(ChipBlock::n_words).sum::<usize>()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_layout_with_no_chips_is_just_the_main_words() {
		let layout = ChipLayout::new(100, []);

		assert_eq!(layout.n_main_private(), 100);
		assert!(layout.blocks().is_empty());
		// 100 words round up to 128.
		assert_eq!(layout.log_hidden_words(), 7);
		assert_eq!(layout.n_placed_words(), 100);
	}

	#[test]
	fn a_block_starts_after_the_main_words_at_its_own_alignment() {
		// 6 hidden words per instance round up to 8; 4 instances make a 32-word block.
		// Main holds 20 words, so the block starts at 32, the first multiple of 32 past 20.
		let layout = ChipLayout::new(20, [(6, 2)]);
		let block = layout.block(0);

		assert_eq!(block.log_words_per_instance(), 3);
		assert_eq!(block.log_instances(), 2);
		assert_eq!(block.n_words(), 32);
		assert_eq!(block.offset(), 32);
		assert_eq!(block.block_index(), 1);
		assert_eq!(layout.log_hidden_words(), 6);
	}

	#[test]
	fn a_word_is_wire_major_with_the_instance_lowest() {
		// 2 wires per instance, 4 instances: wire 0's instances first, then wire 1's.
		let layout = ChipLayout::new(0, [(2, 2)]);
		let block = layout.block(0);

		assert_eq!(block.word(0, 0), 0);
		assert_eq!(block.word(0, 3), 3);
		assert_eq!(block.word(1, 0), 4);
		assert_eq!(block.word(1, 3), 7);
	}

	#[test]
	#[should_panic(expected = "instance is outside the block")]
	fn a_word_past_the_instance_count_is_rejected() {
		ChipLayout::new(0, [(2, 2)]).block(0).word(0, 4);
	}

	#[test]
	#[should_panic(expected = "wire is outside the block")]
	fn a_word_past_the_padded_wire_count_is_rejected() {
		// 3 wires pad to 4, so wire 4 is past the block even though the chip has only 3.
		ChipLayout::new(0, [(3, 1)]).block(0).word(4, 0);
	}

	/// Every block must be aligned to its own length, non-overlapping, and inside the segment.
	///
	/// Alignment is what makes a claim on one block a claim on the whole segment.
	/// So it is the invariant the protocol rests on.
	fn assert_layout_invariants(layout: &ChipLayout) {
		let mut placed = layout
			.blocks()
			.iter()
			.map(|block| (block.offset(), block.n_words()))
			.collect::<Vec<_>>();
		placed.sort_unstable();

		let mut end_of_previous = layout.n_main_private();
		for (offset, n_words) in placed {
			assert_eq!(
				offset % n_words,
				0,
				"block at {offset} is not aligned to its {n_words} words"
			);
			assert!(offset >= end_of_previous, "block at {offset} overlaps the one before it");
			end_of_previous = offset + n_words;
		}
		assert!(end_of_previous <= layout.n_hidden_words(), "the last block runs past the segment");
	}

	#[test]
	fn blocks_are_placed_largest_first_and_stay_indexed_by_chip() {
		// Chip 0 is 8 words, chip 1 is 64, chip 2 is 16. Placement order is 1, 2, 0, but the
		// accessors still index by chip.
		let layout = ChipLayout::new(0, [(4, 1), (8, 3), (8, 1)]);

		assert_eq!(layout.block(0).n_words(), 8);
		assert_eq!(layout.block(1).n_words(), 64);
		assert_eq!(layout.block(2).n_words(), 16);

		// The largest block takes offset 0, then the 16-word block, then the 8-word one.
		assert_eq!(layout.block(1).offset(), 0);
		assert_eq!(layout.block(2).offset(), 64);
		assert_eq!(layout.block(0).offset(), 80);
		assert_layout_invariants(&layout);
	}

	#[test]
	fn padding_stays_under_the_largest_block() {
		// Awkward sizes and an awkward main count, so every placement needs a gap.
		let layout = ChipLayout::new(37, [(5, 4), (3, 0), (17, 2), (1, 6)]);
		assert_layout_invariants(&layout);

		let largest = layout
			.blocks()
			.iter()
			.map(ChipBlock::n_words)
			.max()
			.expect("the layout has chips");
		let padding = layout.n_hidden_words() - layout.n_placed_words();
		assert!(
			padding < 2 * largest,
			"padding {padding} should stay under twice the largest block {largest}"
		);
	}

	#[test]
	fn a_single_instance_chip_is_a_block_of_one_instance() {
		// `log_instances == 0` is a chip called once: the block is just its padded wire count.
		let layout = ChipLayout::new(0, [(5, 0)]);
		let block = layout.block(0);

		assert_eq!(block.log_instances(), 0);
		assert_eq!(block.n_words(), 8);
		assert_eq!(block.word(4, 0), 4);
		assert_layout_invariants(&layout);
	}
}
