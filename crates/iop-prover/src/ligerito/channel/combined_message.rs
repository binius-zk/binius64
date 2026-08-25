// Copyright 2026 The Binius Developers

//! The single message a batched ladder folds, assembled from the committed ones.

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldSlice, FieldVec};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};

/// The single level-0 message a batched Ligerito opening folds.
///
/// Each committed message enters scaled by its batching coefficient.
/// A message shorter than the longest one is zero-extended over the coordinates it lacks.
///
/// ```text
///     PI = sum_i e_i * pi_i,   pi_i zero-extended to the longest message's variables
/// ```
///
/// This is the message the ladder's level 0 folds.
/// Its encoding is the same combination of the committed codewords.
/// That is what lets one query position serve every one of them.
pub(super) struct CombinedMessage<P: PackedField, A: Allocator> {
	/// The running sum, one entry per variable assignment of the longest message.
	buffer: FieldVec<P, A>,
}

impl<P: PackedField, A: Allocator> CombinedMessage<P, A> {
	/// An empty combination over `log_len` variables, which every message is added into.
	pub(super) fn zeros_in(alloc: &A, log_len: usize) -> Self {
		Self {
			buffer: FieldBuffer::zeros_in(alloc, log_len),
		}
	}

	/// Adds one committed message, scaled by its batching coefficient.
	///
	/// The message lands on the low entries and the rest keep whatever they already hold.
	/// That is what zero-extending a shorter multilinear means.
	///
	/// ## Preconditions
	///
	/// * `message` is no longer than the combination.
	pub(super) fn add_scaled(&mut self, message: FieldSlice<'_, P>, coefficient: P::Scalar) {
		assert!(
			message.log_len() <= self.buffer.log_len(),
			"precondition: a message of 2^{} entries exceeds the combination's 2^{}",
			message.log_len(),
			self.buffer.log_len()
		);

		let scale = P::broadcast(coefficient);
		if message.log_len() >= P::LOG_WIDTH {
			// Whole packed elements line up, so the scaled add is a prefix of the destination.
			let n_packed = 1usize << (message.log_len() - P::LOG_WIDTH);
			self.buffer.as_mut()[..n_packed]
				.par_iter_mut()
				.zip(message.as_ref())
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|(entry, addend)| *entry += scale * *addend);
		} else {
			// The message is narrower than one packed element, so only its low lanes are live and
			// the rest of that element must contribute nothing.
			let live = 1usize << message.log_len();
			let lanes = P::WIDTH.min(1usize << self.buffer.log_len());
			let padded = P::from_scalars((0..lanes).map(|lane| {
				if lane < live {
					message.get(lane)
				} else {
					P::Scalar::ZERO
				}
			}));
			self.buffer.as_mut()[0] += scale * padded;
		}
	}

	/// The assembled message, ready for the ladder to fold.
	pub(super) fn into_buffer(self) -> FieldVec<P, A> {
		self.buffer
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{
		Field, Ghash128b as B128, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
		PackedBinaryGhash4x128b, Random,
	};
	use binius_math::test_utils::random_field_buffer;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	/// The definition the packed path is checked against, one scalar at a time.
	fn add_scaled_by_scalars<P: PackedField>(
		dst: &mut FieldBuffer<P>,
		src: &FieldBuffer<P>,
		coefficient: P::Scalar,
	) {
		for index in 0..1usize << src.log_len() {
			dst.set(index, dst.get(index) + coefficient * src.get(index));
		}
	}

	/// Every message shape must land exactly where the scalar definition puts it.
	fn check_all_shapes<P: PackedField<Scalar = B128>>() {
		let mut rng = StdRng::seed_from_u64(0);
		for log_len in 0..=4 {
			for log_msg in 0..=log_len {
				// Fixture state: one combination over 2^log_len entries, two messages added in.
				let first = random_field_buffer::<P>(&mut rng, log_msg);
				let second = random_field_buffer::<P>(&mut rng, log_len);
				let coefficients = [B128::random(&mut rng), B128::random(&mut rng)];

				let mut expected = FieldBuffer::<P>::zeros(log_len);
				add_scaled_by_scalars(&mut expected, &first, coefficients[0]);
				add_scaled_by_scalars(&mut expected, &second, coefficients[1]);

				let mut combined = CombinedMessage::zeros_in(&GlobalAllocator, log_len);
				combined.add_scaled(first.as_view(), coefficients[0]);
				combined.add_scaled(second.as_view(), coefficients[1]);
				let actual = combined.into_buffer();

				for index in 0..1usize << log_len {
					assert_eq!(
						actual.get(index),
						expected.get(index),
						"P::LOG_WIDTH={}, log_msg={log_msg}, log_len={log_len}, index={index}",
						P::LOG_WIDTH,
					);
				}
			}
		}
	}

	// Packed widths from one scalar per element up to four, so the sub-element path is exercised
	// against a reference that never packs.
	#[test]
	fn a_combination_matches_the_scalar_definition_at_every_width() {
		check_all_shapes::<PackedBinaryGhash1x128b>();
		check_all_shapes::<PackedBinaryGhash2x128b>();
		check_all_shapes::<PackedBinaryGhash4x128b>();
	}

	#[test]
	#[should_panic(expected = "exceeds the combination's")]
	fn a_message_longer_than_the_combination_is_refused() {
		// The combination spans 2^3 entries and the message claims 2^4, so it has nowhere to land.
		let message = random_field_buffer::<B128>(&mut StdRng::seed_from_u64(0), 4);
		let mut combined = CombinedMessage::zeros_in(&GlobalAllocator, 3);
		combined.add_scaled(message.as_view(), B128::ONE);
	}
}
