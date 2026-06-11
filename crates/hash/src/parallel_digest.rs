// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, marker::PhantomData, mem::MaybeUninit};

use binius_utils::{
	FixedSizeSerializeBytes, SerializeBytes,
	rayon::{
		iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
		slice::ParallelSliceMut,
	},
};
use bytes::BytesMut;
use digest::{Digest, FixedOutputReset, Output, block_api::BlockSizeUser};

use crate::HashBuffer;

/// An object that efficiently computes `N` instances of a cryptographic hash function
/// in parallel.
///
/// This trait is useful when there is a more efficient way of calculating multiple digests at once,
/// e.g. using SIMD instructions. It is supposed that this trait is implemented directly for some
/// digest and some fixed `N` and passed as an implementation of the `ParallelDigest` trait which
/// hides the `N` value.
pub trait MultiDigest<const N: usize>: Clone {
	/// The corresponding non-parallelized hash function.
	type Digest: Digest;

	/// Create new hasher instance with empty state.
	fn new() -> Self;

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		let mut hasher = Self::new();
		hasher.update([data.as_ref(); N]);
		hasher
	}

	/// Process data, updating the internal state.
	/// The number of rows in `data` must be equal to `parallel_instances()`.
	fn update(&mut self, data: [&[u8]; N]);

	/// Process input data in a chained manner.
	#[must_use]
	fn chain_update(self, data: [&[u8]; N]) -> Self {
		let mut hasher = self;
		hasher.update(data);
		hasher
	}

	/// Write result into provided array and consume the hasher instance.
	fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]);

	/// Write result into provided array and reset the hasher instance.
	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]);

	/// Reset hasher instance to its initial state.
	fn reset(&mut self);

	/// Compute hash of `data`.
	/// All slices in the `data` must have the same length.
	///
	/// # Panics
	/// Panics if data contains slices of different lengths.
	fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<Output<Self::Digest>>; N]);
}

pub trait ParallelDigest: Send {
	/// The corresponding non-parallelized hash function.
	type Digest: Digest;

	/// Create new hasher instance with empty state.
	fn new() -> Self;

	/// Calculate the digest of multiple hashes by processing a parallel iterator of iterators.
	///
	/// The source parameter provides a parallel iterator where:
	/// - Each element of the outer iterator maps to one leaf/digest in the output
	/// - Each element contains an inner iterator of items that will be serialized and concatenated
	///   to form that leaf's content
	///
	/// # Panics
	/// All items must be able to serialize with SerializationMode::Native without error, or this
	/// method will panic.
	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	);

	/// Like [`digest`](Self::digest), but specialized for the case where every leaf is built from
	/// exactly `n_items_per_input` items of a [`FixedSizeSerializeBytes`] type, so that each leaf
	/// has the same, compile-time-derivable byte length.
	///
	/// This extra structure lets implementations skip per-leaf length bookkeeping (and, for short
	/// leaves, the message padding) that [`digest`](Self::digest) must redo every time. The default
	/// implementation simply forwards to [`digest`](Self::digest).
	///
	/// # Panics
	/// Each iterator in `source` must yield exactly `n_items_per_input` items, and all items must
	/// serialize without error, or this method may panic.
	fn digest_with_const_len<I: IntoIterator<Item: FixedSizeSerializeBytes>>(
		&self,
		n_items_per_input: usize,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		let _ = n_items_per_input;
		self.digest(source, out);
	}
}

/// A wrapper that implements the `ParallelDigest` trait for a `MultiDigest` implementation.
#[derive(Clone)]
pub struct ParallelMultidigestImpl<D: MultiDigest<N>, const N: usize>(D);

impl<D: MultiDigest<N> + Default, const N: usize> Default for ParallelMultidigestImpl<D, N> {
	fn default() -> Self {
		Self(D::default())
	}
}

impl<D: MultiDigest<N, Digest: Send> + Send + Sync, const N: usize> ParallelDigest
	for ParallelMultidigestImpl<D, N>
{
	type Digest = D::Digest;

	fn new() -> Self {
		Self(D::new())
	}

	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		let buffers = array::from_fn::<_, N, _>(|_| BytesMut::new());
		source.chunks(N).zip(out.par_chunks_mut(N)).for_each_with(
			buffers,
			|buffers, (data, out_chunk)| {
				let mut hasher = self.0.clone();
				for (mut buf, chunk) in buffers.iter_mut().zip(data) {
					buf.clear();
					for item in chunk {
						item.serialize(&mut buf)
							.expect("pre-condition: items must serialize without error")
					}
				}
				let data = array::from_fn(|i| buffers[i].as_ref());
				hasher.update(data);

				if out_chunk.len() == N {
					hasher
						.finalize_into_reset(out_chunk.try_into().expect("chunk size is correct"));
				} else {
					let mut result = array::from_fn::<_, N, _>(|_| MaybeUninit::uninit());
					hasher.finalize_into(&mut result);
					for (out, res) in out_chunk.iter_mut().zip(result) {
						out.write(unsafe { res.assume_init() });
					}
				}
			},
		);
	}
}

/// Adapts a sequential [`Digest`] into a [`ParallelDigest`] that hashes one leaf per element of a
/// parallel iterator.
///
/// Each Rayon work-item is seeded with a single hasher (via `for_each_with`) which is recycled in
/// place with `finalize_reset` between leaves, rather than cloning a fresh hasher per leaf. This
/// requires `D: FixedOutputReset`.
pub struct ParallelDigestAdapter<D>(PhantomData<D>);

impl<D> Default for ParallelDigestAdapter<D> {
	fn default() -> Self {
		Self(PhantomData)
	}
}

impl<D> ParallelDigest for ParallelDigestAdapter<D>
where
	D: Digest + FixedOutputReset + BlockSizeUser + Send + Sync + Clone,
{
	type Digest = D;

	fn new() -> Self {
		Self(PhantomData)
	}

	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		source
			.zip(out.par_iter_mut())
			.for_each_with(D::new(), |hasher, (items, out)| {
				{
					let mut buffer = HashBuffer::new(hasher);
					for item in items {
						item.serialize(&mut buffer)
							.expect("pre-condition: items must serialize without error")
					}
				}
				out.write(hasher.finalize_reset());
			});
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_utils::rayon::iter::IntoParallelRefIterator;
	use digest::{
		FixedOutput, HashMarker, OutputSizeUser, Reset, Update,
		consts::{U1, U32},
	};
	use itertools::izip;
	use rand::prelude::*;

	use super::*;

	#[derive(Clone, Default)]
	struct MockDigest {
		state: u8,
	}

	impl HashMarker for MockDigest {}

	impl Update for MockDigest {
		fn update(&mut self, data: &[u8]) {
			for &byte in data {
				self.state ^= byte;
			}
		}
	}

	impl Reset for MockDigest {
		fn reset(&mut self) {
			self.state = 0;
		}
	}

	impl OutputSizeUser for MockDigest {
		type OutputSize = U32;
	}

	impl BlockSizeUser for MockDigest {
		type BlockSize = U1;
	}

	impl FixedOutput for MockDigest {
		fn finalize_into(self, out: &mut Output<Self>) {
			out[0] = self.state;
			for byte in &mut out[1..] {
				*byte = 0;
			}
		}
	}

	#[derive(Clone, Default)]
	struct MockMultiDigest {
		digests: [MockDigest; 4],
	}

	impl MultiDigest<4> for MockMultiDigest {
		type Digest = MockDigest;

		fn new() -> Self {
			Self::default()
		}

		fn update(&mut self, data: [&[u8]; 4]) {
			for (digest, &chunk) in self.digests.iter_mut().zip(data.iter()) {
				digest::Digest::update(digest, chunk);
			}
		}

		fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; 4]) {
			for (digest, out) in self.digests.into_iter().zip(out.iter_mut()) {
				let mut output = digest::Output::<Self::Digest>::default();
				digest::Digest::finalize_into(digest, &mut output);
				*out = MaybeUninit::new(output);
			}
		}

		fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; 4]) {
			for (digest, out) in self.digests.iter_mut().zip(out.iter_mut()) {
				let mut digest_copy = MockDigest::default();
				std::mem::swap(digest, &mut digest_copy);
				*out = MaybeUninit::new(digest_copy.finalize());
			}
			self.reset();
		}

		fn reset(&mut self) {
			for digest in &mut self.digests {
				*digest = MockDigest::default();
			}
		}

		fn digest(data: [&[u8]; 4], out: &mut [MaybeUninit<Output<Self::Digest>>; 4]) {
			let mut hasher = Self::default();
			hasher.update(data);
			hasher.finalize_into(out);
		}
	}

	fn generate_mock_data(n_hashes: usize, chunk_size: usize) -> Vec<Vec<u8>> {
		let mut rng = StdRng::seed_from_u64(0);

		(0..n_hashes)
			.map(|_| {
				let mut chunk = vec![0; chunk_size];
				rng.fill_bytes(&mut chunk);
				chunk
			})
			.collect()
	}

	fn check_parallel_digest_consistency<
		D: ParallelDigest<Digest: BlockSizeUser + Send + Sync + Clone>,
	>(
		data: Vec<Vec<u8>>,
	) {
		let parallel_digest = D::new();
		let mut parallel_results = repeat_with(MaybeUninit::<Output<D::Digest>>::uninit)
			.take(data.len())
			.collect::<Vec<_>>();
		parallel_digest.digest(data.par_iter(), &mut parallel_results);

		let serial_results = data.iter().map(<D::Digest as Digest>::digest);

		for (parallel, serial) in izip!(parallel_results, serial_results) {
			assert_eq!(unsafe { parallel.assume_init() }, serial);
		}
	}

	#[test]
	fn test_empty_data() {
		let data = generate_mock_data(0, 16);
		check_parallel_digest_consistency::<ParallelMultidigestImpl<MockMultiDigest, 4>>(data);
	}

	#[test]
	fn test_non_empty_data() {
		for n_hashes in [1, 2, 4, 8, 9] {
			let data = generate_mock_data(n_hashes, 16);
			check_parallel_digest_consistency::<ParallelMultidigestImpl<MockMultiDigest, 4>>(data);
		}
	}

	#[test]
	fn test_adapter_matches_serial_sha256() {
		use sha2::Sha256;

		for n_hashes in [0, 1, 2, 4, 8, 9, 100] {
			let data = generate_mock_data(n_hashes, 16);

			let adapter = ParallelDigestAdapter::<Sha256>::new();
			let mut results = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
				.take(data.len())
				.collect::<Vec<_>>();
			adapter.digest(data.par_iter(), &mut results);

			for (result, leaf) in results.into_iter().zip(&data) {
				assert_eq!(unsafe { result.assume_init() }, <Sha256 as Digest>::digest(leaf));
			}
		}
	}
}
