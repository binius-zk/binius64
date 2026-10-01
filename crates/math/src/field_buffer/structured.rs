// Copyright 2026 The Binius Developers

//! A field buffer whose nonzero values may be confined to one aligned block.

use std::ops::Deref;

use binius_compute::{Allocator, VecLike};
use binius_field::PackedField;

use super::{FieldBuffer, FieldSliceMut};

/// A field buffer that may hold explicit values for only one aligned block, and zero elsewhere.
///
/// Passing the structure along lets a consumer that understands it skip the zeros. A consumer
/// that does not calls [`Self::materialize`].
#[derive(Debug, Clone)]
pub enum StructuredBuffer<P: PackedField, Data: Deref<Target = [P]>> {
	/// Every value held explicitly.
	Buffer(FieldBuffer<P, Data>),
	/// `inner` placed in block `index` of `2^log_n_blocks` equal blocks, and zero everywhere else.
	ZeroPadded {
		inner: Box<Self>,
		log_n_blocks: usize,
		/// Index in the range `0..1 << log_n_blocks`.
		index: usize,
	},
}

impl<P: PackedField, Data: Deref<Target = [P]>> StructuredBuffer<P, Data> {
	/// Returns the base-2 logarithm of the number of field elements.
	pub fn log_len(&self) -> usize {
		match self {
			Self::Buffer(buffer) => buffer.log_len(),
			Self::ZeroPadded {
				inner,
				log_n_blocks,
				..
			} => inner.log_len() + log_n_blocks,
		}
	}

	/// Writes every value into `dst`, which must hold zeros outside the explicit block.
	///
	/// # Panics
	///
	/// Panics if `dst` is not the same size as `self`, or any `ZeroPadded` index is out of range.
	fn write_into(self, mut dst: FieldSliceMut<'_, P>) {
		match self {
			Self::Buffer(buffer) => {
				assert_eq!(buffer.log_len(), dst.log_len()); // precondition
				if buffer.log_len() < P::LOG_WIDTH {
					dst.as_mut()[0] = P::from_scalars(buffer.iter_scalars());
				} else {
					dst.as_mut().copy_from_slice(buffer.as_ref());
				}
			}
			Self::ZeroPadded {
				inner,
				log_n_blocks,
				index,
			} => {
				let mut block = dst.chunk_mut(dst.log_len() - log_n_blocks, index);
				inner.write_into(block.chunk());
			}
		}
	}
}

impl<P: PackedField, Data: VecLike<P>> StructuredBuffer<P, Data> {
	/// Writes out every value, zeros included, as one buffer.
	///
	/// A buffer already holding every value is returned with no copy.
	pub fn materialize<A>(self, alloc: &A) -> FieldBuffer<P, Data>
	where
		A: Allocator<Vec<P> = Data>,
	{
		match self {
			Self::Buffer(buffer) => buffer,
			padded => {
				let mut buffer = FieldBuffer::zeros_in(alloc, padded.log_len());
				padded.write_into(buffer.as_mut_view());
				buffer
			}
		}
	}
}

impl<P: PackedField, Data: Deref<Target = [P]>> From<FieldBuffer<P, Data>>
	for StructuredBuffer<P, Data>
{
	fn from(buffer: FieldBuffer<P, Data>) -> Self {
		Self::Buffer(buffer)
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, PackedField, PackedGhash1x128b, PackedGhash4x128b};
	use rand::{SeedableRng, rngs::StdRng};

	use super::StructuredBuffer;
	use crate::{FieldBuffer, test_utils::random_field_buffer};

	/// Materializing a nested zero-padding matches placing the values by hand.
	fn check<P: PackedField>(log_inner: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let inner = random_field_buffer::<P>(&mut rng, log_inner);

		// The inner buffer at block 1 of 2, and that at block 2 of 4.
		let structured = StructuredBuffer::ZeroPadded {
			inner: Box::new(StructuredBuffer::ZeroPadded {
				inner: Box::new(inner.clone().into()),
				log_n_blocks: 1,
				index: 1,
			}),
			log_n_blocks: 2,
			index: 2,
		};
		assert_eq!(structured.log_len(), log_inner + 3);

		let offset = (2 * 2 + 1) << log_inner;
		let expected = FieldBuffer::<P>::from_values(
			&(0..1 << (log_inner + 3))
				.map(|i| {
					if (offset..offset + (1 << log_inner)).contains(&i) {
						inner.get(i - offset)
					} else {
						P::Scalar::ZERO
					}
				})
				.collect::<Vec<_>>(),
		);
		assert_eq!(structured.materialize(&GlobalAllocator), expected);
	}

	#[test]
	fn materialize_matches_naive_placement() {
		for log_inner in 0..4 {
			check::<PackedGhash1x128b>(log_inner);
			check::<PackedGhash4x128b>(log_inner);
		}
	}
}
