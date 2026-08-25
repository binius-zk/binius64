// Copyright 2026 The Binius Developers

//! The fresh randomness one zero-knowledge oracle is committed beside.

use binius_compute::Allocator;
use binius_field::{PackedField, Random};
use binius_math::{FieldBuffer, FieldSlice, FieldVec, multilinear::Multilinear};
use binius_utils::{
	buffer::VecLike,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};
use rand::Rng;

/// A uniform multilinear of one oracle's own length, drawn before any challenge exists.
///
/// The mask is committed in a lane beside the message it blinds, in the same level-0 codeword.
/// Once the masking challenge is drawn, the two are blended into the message the ladder folds:
///
/// ```text
///     pi' = (1 - gamma) * pi + gamma * omega
/// ```
///
/// Nothing downstream removes the mask again.
/// So every value the ladder derives from the blend carries it, the cleartext residual included.
pub(super) struct Mask<P: PackedField, A: Allocator> {
	/// The uniform values, one per variable assignment of the message this mask blinds.
	values: FieldVec<P, A>,
}

impl<P: PackedField, A: Allocator> Mask<P, A> {
	/// Draws a uniform mask over `log_len` variables.
	///
	/// The randomness is the caller's, so a channel with no zero-knowledge oracle never reads any.
	pub(super) fn draw(alloc: &A, log_len: usize, rng: &mut impl Rng) -> Self {
		let mut values = FieldBuffer::zeros_in(alloc, log_len);

		// A buffer shorter than one packed element must leave its dead lanes zero, which is the
		// invariant every consumer of a field buffer relies on. So the two cases fill differently.
		if log_len >= P::LOG_WIDTH {
			for word in values.as_mut() {
				*word = P::random(&mut *rng);
			}
		} else {
			values.as_mut()[0] =
				P::from_scalars((0..1 << log_len).map(|_| P::Scalar::random(&mut *rng)));
		}

		Self { values }
	}

	/// The buffer level 0 commits: the message padded out, then this mask padded out beside it.
	///
	/// The mask occupies the whole upper half, so it is the *highest* variable that selects it.
	/// The lane index of a codeword is the bit-reversed high variables of the committed buffer.
	/// So the mask lands in lane bit zero, which is the fold point's leading coordinate.
	///
	/// ```text
	///     committed = [ pad(pi) | pad(omega) ]      one variable more than the padded message
	/// ```
	///
	/// Each half is zero-padded on its own.
	/// Padding the concatenation instead would put the mask where the encoder reads zeros.
	///
	/// ## Preconditions
	///
	/// * `message` has this mask's own length.
	/// * `log_padded_len` is at least that length.
	pub(super) fn interleaved_with(
		&self,
		alloc: &A,
		message: FieldSlice<'_, P>,
		log_padded_len: usize,
	) -> FieldVec<P, A> {
		assert_eq!(
			message.log_len(),
			self.values.log_len(),
			"precondition: a mask blinds a message of its own length"
		);
		assert!(
			log_padded_len >= message.log_len(),
			"precondition: the padded length must cover the message"
		);

		let padded_message =
			FieldBuffer::from_view_in(alloc, message).zero_extend_in(alloc, log_padded_len);
		let padded_mask = FieldBuffer::from_view_in(alloc, self.values.as_view())
			.zero_extend_in(alloc, log_padded_len);

		// Whole packed words concatenate directly. A pair of halves narrower than one word shares
		// that word instead, so the two are repacked lane by lane.
		let values = if log_padded_len >= P::LOG_WIDTH {
			let packed_len = 1usize << (log_padded_len - P::LOG_WIDTH);
			let mut values = alloc.alloc::<P>(2 * packed_len);
			values.extend_from_slice(padded_message.as_ref());
			values.extend_from_slice(padded_mask.as_ref());
			values
		} else {
			let mut values = alloc.alloc::<P>(1);
			values.push(P::from_scalars(std::iter::chain(
				padded_message.iter_scalars(),
				padded_mask.iter_scalars(),
			)));
			values
		};

		FieldBuffer::new(log_padded_len + 1, values)
	}

	/// What this mask pairs to against a transparent multilinear.
	///
	/// The verifier moves the oracle's claim along the line between this value and the claim.
	/// It reaches the wire before the challenge that blends the two.
	/// So it cannot be chosen against that challenge.
	///
	/// ## Preconditions
	///
	/// * `transparent` has this mask's own length.
	pub(super) fn pair(&self, transparent: FieldSlice<'_, P>) -> P::Scalar {
		self.values.par_inner_product(transparent)
	}

	/// Blends this mask into the message it blinds, in place, at the masking challenge.
	///
	/// ```text
	///     pi[j] <- pi[j] + gamma * (omega[j] - pi[j])
	/// ```
	///
	/// ## Preconditions
	///
	/// * `message` has this mask's own length.
	pub(super) fn blend(&self, message: &mut FieldVec<P, A>, gamma: P::Scalar) {
		assert_eq!(
			message.log_len(),
			self.values.log_len(),
			"precondition: a mask blinds a message of its own length"
		);

		let gamma = P::broadcast(gamma);
		(message.as_mut(), self.values.as_ref())
			.into_par_iter()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(entry, mask)| *entry += gamma * (*mask - *entry));
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{
		Field, Ghash128b as B128, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
		PackedBinaryGhash4x128b, arithmetic_traits::InvertOrZero,
	};
	use binius_math::test_utils::random_scalars;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	/// The interleaving and the blend, checked one scalar at a time at every packing width.
	fn check_all_shapes<P: PackedField<Scalar = B128>>() {
		let mut rng = StdRng::seed_from_u64(0);
		for log_len in 0..=3 {
			for log_padded_len in log_len..=4 {
				// Fixture state: one message, one mask of its own length, one padded width.
				// The message is packed from scalars, so the lanes past its length are zero, which
				// is what every buffer the channel is handed guarantees.
				let message =
					FieldBuffer::<P>::from_values(&random_scalars::<B128>(&mut rng, 1 << log_len));
				let mask = Mask::draw(&GlobalAllocator, log_len, &mut rng);
				let committed =
					mask.interleaved_with(&GlobalAllocator, message.as_view(), log_padded_len);

				// The committed buffer is the two padded halves, message first.
				//
				//     [0 .. 2^log_len)                 the message
				//     [2^log_len .. 2^log_padded_len)  zeros
				//     [2^log_padded_len .. + 2^log_len) the mask
				assert_eq!(committed.log_len(), log_padded_len + 1);
				let padded = 1usize << log_padded_len;
				let live = 1usize << log_len;
				for index in 0..padded {
					let expected = if index < live {
						message.get(index)
					} else {
						B128::ZERO
					};
					assert_eq!(committed.get(index), expected, "message half at {index}");
					let expected = if index < live {
						mask.values.get(index)
					} else {
						B128::ZERO
					};
					assert_eq!(committed.get(padded + index), expected, "mask half at {index}");
				}

				// The blend interpolates the line through the message and the mask.
				let gamma = B128::random(&mut rng);
				let mut blended = FieldBuffer::from_view_in(&GlobalAllocator, message.as_view());
				mask.blend(&mut blended, gamma);
				for index in 0..live {
					let expected =
						message.get(index) + gamma * (mask.values.get(index) - message.get(index));
					assert_eq!(blended.get(index), expected, "blend at {index}");
				}
			}
		}
	}

	// Packed widths from one scalar per element up to four, so the shared-word path is exercised
	// against a reference that never packs. The narrow shapes also pin that a mask leaves the dead
	// lanes of its own buffer zero: a live value there would show up in the committed padding.
	#[test]
	fn the_committed_buffer_and_the_blend_match_the_scalar_definition() {
		check_all_shapes::<PackedBinaryGhash1x128b>();
		check_all_shapes::<PackedBinaryGhash2x128b>();
		check_all_shapes::<PackedBinaryGhash4x128b>();
	}

	/// Every mask of one message is a mask of another, and the two blends are the same buffer.
	///
	/// This is the hiding argument in one line, at the point the mask enters the protocol.
	/// For a nonzero challenge the blend is a translation of the mask space:
	///
	///     (1 - g) * pi + g * omega  =  (1 - g) * rho + g * omega'
	///     omega' = omega + ((1 - g) / g) * (pi - rho)
	///
	/// A translation is a bijection.
	/// So a uniform mask for one witness is a uniform mask for the other.
	/// The ladder derives everything from the blend, so it sees one distribution either way.
	/// The cleartext residual is derived from the blend.
	#[test]
	fn every_mask_of_one_message_is_a_mask_of_another() {
		type P = PackedBinaryGhash1x128b;

		// Fixture state: two unrelated messages of 2^4 entries, one mask, one challenge.
		let mut rng = StdRng::seed_from_u64(4);
		let len = 1usize << 4;
		let first = FieldBuffer::<P>::from_values(&random_scalars::<B128>(&mut rng, len));
		let second = FieldBuffer::<P>::from_values(&random_scalars::<B128>(&mut rng, len));
		let mask = Mask::draw(&GlobalAllocator, 4, &mut rng);
		let gamma = B128::random(&mut rng);
		assert_ne!(gamma, B128::ZERO, "the blend carries no mask at a zero challenge");

		// The mask the second message needs to land on the first one's blend.
		let scale = (B128::ONE - gamma) * gamma.invert_or_zero();
		let remapped = (0..len)
			.map(|index| mask.values.get(index) + scale * (first.get(index) - second.get(index)))
			.collect::<Vec<_>>();
		let remapped = Mask::<P, GlobalAllocator> {
			values: FieldBuffer::from_values_in(&GlobalAllocator, &remapped),
		};

		let mut blended_first = FieldBuffer::from_view_in(&GlobalAllocator, first.as_view());
		mask.blend(&mut blended_first, gamma);
		let mut blended_second = FieldBuffer::from_view_in(&GlobalAllocator, second.as_view());
		remapped.blend(&mut blended_second, gamma);

		assert_eq!(blended_first.as_ref(), blended_second.as_ref());
	}

	#[test]
	fn two_draws_from_one_generator_differ() {
		// A mask that repeated would blind the second oracle with the first one's randomness.
		let mut rng = StdRng::seed_from_u64(2);
		let first = Mask::<PackedBinaryGhash1x128b, _>::draw(&GlobalAllocator, 4, &mut rng);
		let second = Mask::<PackedBinaryGhash1x128b, _>::draw(&GlobalAllocator, 4, &mut rng);
		assert_ne!(first.values.as_ref(), second.values.as_ref());
	}

	#[test]
	#[should_panic(expected = "precondition: a mask blinds a message of its own length")]
	fn a_mask_of_the_wrong_length_is_refused() {
		// A mask shorter than its message would leave the tail of that message in the clear.
		let mut rng = StdRng::seed_from_u64(3);
		let message = FieldBuffer::<PackedBinaryGhash1x128b>::from_values(&random_scalars::<B128>(
			&mut rng, 16,
		));
		let mask = Mask::draw(&GlobalAllocator, 3, &mut rng);
		mask.interleaved_with(&GlobalAllocator, message.as_view(), 4);
	}
}
