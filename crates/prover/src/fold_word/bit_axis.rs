// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Contracting the bit axis: one field element per word.

use std::{array, iter, mem::MaybeUninit};

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_math::FieldBuffer;
use binius_utils::rayon::prelude::*;

use super::{
	kernel::{BitFold, WORDS_PER_BATCH},
	output::PackedOutput,
};

/// Minimum words one parallel task folds along the bit axis.
///
/// A batch is the unit of work here, and a batch is a fixed 64 words.
/// Left unbounded, the split reaches one task per batch, and the handoff costs more than the fold.
///
/// A floor also caps how far a loop can divide.
/// A list of `n` words splits into at most `n / floor` tasks.
/// So the floor must stay below the shortest list this fold runs at, divided by the core count.
/// Otherwise the split stops before the cores are full.
///
/// The shared task-size budgets in the utilities crate are calibrated for wider items.
/// They land high enough here to breach that cap on a mid-sized list.
const MIN_WORDS_PER_TASK: usize = 1 << 12;

/// A reusable folder over a fixed vector of bit-index scalars.
///
/// The one-shot function above rebuilds its kernel on every call.
/// A caller folding several word-lists against the same scalar vector builds it once here, and
/// reuses it across folds.
///
/// The word axis has a folder of its own, built the same way.
#[derive(Debug)]
pub struct BitAxisFolder<F: BinaryField> {
	/// The fold built from the bit-index scalars, shared by every fold call.
	kernel: BitFold<F>,
}

impl<F: BinaryField> BitAxisFolder<F> {
	/// Builds the folding transform for `vec`.
	///
	/// ## Preconditions
	/// * `vec` contains exactly one scalar per bit of a word
	pub fn new(vec: &[F]) -> Self {
		Self {
			kernel: BitFold::new(vec),
		}
	}

	/// Folds `words`, mapping each word to the inner product of its bits with the scalar vector.
	///
	/// The one-shot function at the top of this module states the exact contract.
	pub fn fold<P, A>(&self, alloc: &A, words: &[Word]) -> FieldBuffer<P, A::Vec<P>>
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		let slots_per_batch = slots_per_batch::<P>();
		let mut out = PackedOutput::for_words(alloc, words.len());

		// Partition the words into whole batches and a short tail.
		//
		//     words:  [ batch 0 | batch 1 | ... | batch n-1 | tail (< WORDS_PER_BATCH) ]
		let n_batches = words.len() / WORDS_PER_BATCH;
		let (words_aligned, words_remaining) = words.split_at(n_batches * WORDS_PER_BATCH);

		let slots = out.chunk_slots(n_batches * slots_per_batch);
		let slot_groups = slots.par_chunks_exact_mut(slots_per_batch);
		let word_batches = words_aligned.par_chunks_exact(WORDS_PER_BATCH);
		assert_eq!(slot_groups.len(), word_batches.len());

		(slot_groups, word_batches)
			.into_par_iter()
			// One item is one batch, so the floor converts from words to items.
			.with_min_len(MIN_WORDS_PER_TASK.div_ceil(WORDS_PER_BATCH))
			.for_each(|(slot_group, word_batch)| {
				write_batch(slot_group, self.kernel.fold_batch(as_batch(word_batch)));
			});

		// Safety: the loop above writes every one of the slots handed out exactly once, because
		// the two iterators are equal in length and each item writes its whole group.
		unsafe { out.commit_chunks(n_batches * slots_per_batch) };

		// The tail is under one batch, so it folds as a single zero-padded batch.
		if !words_remaining.is_empty() {
			push_tail(&mut out, self.kernel.fold_prefix(words_remaining), words_remaining.len());
		}

		out.finish()
	}

	/// Folds the two stored BitAnd operand columns and their derived AND column in one pass.
	///
	/// # Overview
	///
	/// The BitAnd zerocheck folds three columns of the constraint `A & B = C`.
	/// On a satisfying witness the third column equals the AND of the first two.
	/// So this fold reads only the two stored columns and derives the third in registers:
	///
	/// ```text
	///     stream A ──┬──> fold ──> folded A
	///     stream B ──┼──> fold ──> folded B
	///                └──> A & B ──> fold ──> folded C   (no third input stream)
	/// ```
	///
	/// # Returns
	///
	/// Three folded buffers, in order:
	/// - the first operand column, folded as a single column would be.
	/// - the second operand column, folded the same way.
	/// - the word-by-word AND of the two columns, folded the same way.
	///
	/// The AND column is derived in registers and never written to memory.
	///
	/// # Performance
	///
	/// - Two input streams instead of three.
	/// - Two register ANDs per word pair replace one memory stream.
	/// - The kernel's working set stays hot across all three outputs.
	///
	/// # Preconditions
	///
	/// * The two word-lists have equal length.
	pub fn fold_bitand_operands<P, A>(
		&self,
		alloc: &A,
		a_words: &[Word],
		b_words: &[Word],
	) -> [FieldBuffer<P, A::Vec<P>>; 3]
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		assert_eq!(a_words.len(), b_words.len());

		// Padding contract, mirrored from the single-column fold:
		// the high words up to the next power of two read as zero.
		// `0 & 0 = 0`, so the derived column stays consistent over that padding.
		//
		// One output buffer per folded column, filled through spare capacity.
		let [mut a_out, mut b_out, mut c_out] =
			array::from_fn(|_| PackedOutput::for_words(alloc, a_words.len()));

		// Phase 1: partition the inputs into whole batches and a short tail.
		//
		//     words:  [ batch 0 | batch 1 | ... | batch n-1 | tail (< WORDS_PER_BATCH) ]
		let slots_per_batch = slots_per_batch::<P>();
		let n_batches = a_words.len() / WORDS_PER_BATCH;
		let (a_aligned, a_remaining) = a_words.split_at(n_batches * WORDS_PER_BATCH);
		let (b_aligned, b_remaining) = b_words.split_at(n_batches * WORDS_PER_BATCH);

		let n_slots = n_batches * slots_per_batch;
		let a_slots = a_out
			.chunk_slots(n_slots)
			.par_chunks_exact_mut(slots_per_batch);
		let b_slots = b_out
			.chunk_slots(n_slots)
			.par_chunks_exact_mut(slots_per_batch);
		let c_slots = c_out
			.chunk_slots(n_slots)
			.par_chunks_exact_mut(slots_per_batch);

		// Phase 2: fold the aligned batches in parallel.
		// Each task owns one batch of both inputs and writes one slot group per output.
		(
			a_slots,
			b_slots,
			c_slots,
			a_aligned.par_chunks_exact(WORDS_PER_BATCH),
			b_aligned.par_chunks_exact(WORDS_PER_BATCH),
		)
			.into_par_iter()
			// One item is one batch of each output, so the floor converts from words.
			.with_min_len(MIN_WORDS_PER_TASK.div_ceil(WORDS_PER_BATCH))
			.for_each(|(a_group, b_group, c_group, a_batch, b_batch)| {
				let a_batch = as_batch(a_batch);
				let b_batch = as_batch(b_batch);
				// Derive the third column in registers, so it is never read from memory.
				let c_batch = array::from_fn(|i| a_batch[i] & b_batch[i]);

				// One kernel per column, all three against the same shared fold.
				write_batch(a_group, self.kernel.fold_batch(a_batch));
				write_batch(b_group, self.kernel.fold_batch(b_batch));
				write_batch(c_group, self.kernel.fold_batch(&c_batch));
			});

		// Safety: the loop above writes every one of the slots handed out by each output exactly
		// once, because the iterators are equal in length and each item writes its whole group.
		unsafe {
			a_out.commit_chunks(n_slots);
			b_out.commit_chunks(n_slots);
			c_out.commit_chunks(n_slots);
		}

		// Phase 3: fold the short tail, which is under one batch of each column.
		let n_remaining = a_remaining.len();
		if n_remaining > 0 {
			let mut c_remaining = [Word::ZERO; WORDS_PER_BATCH];
			for (c, (&a, &b)) in iter::zip(&mut c_remaining, iter::zip(a_remaining, b_remaining)) {
				*c = a & b;
			}
			push_tail(&mut a_out, self.kernel.fold_prefix(a_remaining), n_remaining);
			push_tail(&mut b_out, self.kernel.fold_prefix(b_remaining), n_remaining);
			let c_remaining = &c_remaining[..n_remaining];
			push_tail(&mut c_out, self.kernel.fold_prefix(c_remaining), n_remaining);
		}

		// Phase 4: each output zero-pads itself up to the power-of-two capacity.
		[a_out, b_out, c_out].map(PackedOutput::finish)
	}
}

/// Packed elements one batch of folded words fills.
///
/// The batch is a power of two and at least as wide as any packing of a 128-bit scalar, so it
/// divides evenly and one batch never straddles a slot.
#[inline]
const fn slots_per_batch<P: PackedField>() -> usize {
	const {
		assert!(
			WORDS_PER_BATCH.is_multiple_of(P::WIDTH),
			"a batch must be a whole number of packed elements"
		);
	}
	WORDS_PER_BATCH / P::WIDTH
}

/// Views a chunk the batch loop produced as a fixed-size batch.
///
/// The parallel iterator is built from `par_chunks_exact`, so every chunk it yields has exactly
/// this length; the conversion cannot fail.
#[inline]
fn as_batch(chunk: &[Word]) -> &[Word; WORDS_PER_BATCH] {
	chunk
		.try_into()
		.expect("par_chunks_exact yields chunks of exactly WORDS_PER_BATCH words")
}

/// Packs one batch of folded scalars into its slot group.
///
/// # Panics
///
/// Panics unless the group holds exactly the batch's worth of packed elements.
#[inline]
fn write_batch<P: PackedField>(slots: &mut [MaybeUninit<P>], folded: [P::Scalar; WORDS_PER_BATCH]) {
	assert_eq!(slots.len(), slots_per_batch::<P>());
	for (slot, scalars) in iter::zip(slots, folded.chunks_exact(P::WIDTH)) {
		slot.write(P::from_scalars(scalars.iter().copied()));
	}
}

/// Appends the packed elements the short tail folds to.
///
/// The batched fold zero-fills past the tail's end, and `from_scalars` zero-fills the last packed
/// element, so both kinds of padding read as zero — which is what the buffer's contract promises
/// for words past the end of the list.
#[inline]
fn push_tail<P: PackedField, V: VecLike<P>>(
	out: &mut PackedOutput<P, V>,
	folded: [P::Scalar; WORDS_PER_BATCH],
	n_words: usize,
) {
	for scalars in folded[..n_words].chunks(P::WIDTH) {
		out.push(P::from_scalars(scalars.iter().copied()));
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, PackedGhash2x128b, arch::OptimalPackedB128};
	use binius_math::test_utils::random_scalars;
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use binius_verifier::config::B128;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::fold_word::CHUNK_SIZE;

	/// Contracts the bit axis, leaving one element per word.
	///
	/// A list shorter than a power of two reads its high words as zero, so the padded slots
	/// fold to zero and the buffer is the next power of two long.
	fn reference_fold_bit_axis<F: Field, P: PackedField<Scalar = F>>(
		words: &[Word],
		weights: &[F],
	) -> FieldBuffer<P> {
		assert_eq!(weights.len(), Word::BITS);

		let log_n = log2_ceil_usize(words.len());
		let scalars = (0..1 << log_n)
			.map(|i| {
				// Absent high words are zero, and a zero word has no set bits to weight.
				words.get(i).map_or(F::ZERO, |word| {
					(0..Word::BITS)
						.filter(|bit| (word.as_u64() >> bit) & 1 == 1)
						.map(|bit| weights[bit])
						.sum()
				})
			})
			.collect::<Vec<_>>();

		FieldBuffer::from_values(&scalars)
	}

	/// Word counts spanning every regime both folds branch on.
	///
	/// The folds split their input into whole chunks and a short tail, so the interesting lengths
	/// sit around those boundaries rather than at round powers of two.
	fn any_n_words() -> impl Strategy<Value = usize> {
		0..=4 * CHUNK_SIZE
	}

	fn words_of(n: usize, seed: u64) -> Vec<Word> {
		let mut rng = StdRng::seed_from_u64(seed);
		(0..n).map(|_| Word::from_u64(rng.random())).collect()
	}

	// The bit-axis folds split their input at a multiple of the packing width, so the width decides
	// which branches run at all:
	//
	//     width 1 : every word is its own chunk, so the short tail is never taken
	//     width 2 : an odd word count leaves a tail, and the buffer above it is zero-padded
	//
	// The optimal packing is one scalar wide on some targets, so pinning only that would leave the
	// tail and padding paths untested there. Every bit-axis property runs at both widths.
	fn check_bit_axis_fold<P: PackedField<Scalar = B128>>(words: &[Word], weights: &[B128]) {
		assert_eq!(
			BitAxisFolder::new(weights).fold::<P, _>(&GlobalAllocator, words),
			reference_fold_bit_axis(words, weights),
			"bit-axis fold differs at P::WIDTH = {}, {} words",
			P::WIDTH,
			words.len()
		);
	}

	fn check_fused_bitand_fold<P: PackedField<Scalar = B128>>(
		a_words: &[Word],
		b_words: &[Word],
		weights: &[B128],
	) {
		let c_words = iter::zip(a_words, b_words)
			.map(|(&a, &b)| a & b)
			.collect::<Vec<_>>();
		let folder = BitAxisFolder::new(weights);

		let [a, b, c] = folder.fold_bitand_operands::<P, _>(&GlobalAllocator, a_words, b_words);
		let width = P::WIDTH;

		assert_eq!(a, folder.fold(&GlobalAllocator, a_words), "a differs at width {width}");
		assert_eq!(b, folder.fold(&GlobalAllocator, b_words), "b differs at width {width}");
		assert_eq!(c, folder.fold(&GlobalAllocator, &c_words), "c differs at width {width}");
	}

	proptest! {
		#[test]
		fn bit_axis_fold_matches_the_definition(n_words in any_n_words(), seed: u64) {
			// Every length, not just the powers of two the old fixture used. A non-power-of-two
			// list exercises the tail element and the zero padding above it, which is the path
			// the fixture never reached.
			let words = words_of(n_words, seed);
			let mut rng = StdRng::seed_from_u64(seed ^ 1);
			let weights = random_scalars::<B128>(&mut rng, Word::BITS);

			check_bit_axis_fold::<OptimalPackedB128>(&words, &weights);
			check_bit_axis_fold::<PackedGhash2x128b>(&words, &weights);
		}

		#[test]
		fn fused_bitand_fold_matches_three_separate_folds(n_words in any_n_words(), seed: u64) {
			// The AND-reduction fold derives its third column in registers rather than reading it.
			// That must equal folding a materialized third column.
			//
			//     fused(A, B)  ==  [ fold(A), fold(B), fold(A & B) ]
			let a_words = words_of(n_words, seed);
			let b_words = words_of(n_words, seed ^ 0xff);

			let mut rng = StdRng::seed_from_u64(seed ^ 4);
			let weights = random_scalars::<B128>(&mut rng, Word::BITS);

			check_fused_bitand_fold::<OptimalPackedB128>(&a_words, &b_words, &weights);
			check_fused_bitand_fold::<PackedGhash2x128b>(&a_words, &b_words, &weights);
		}

	}
}
