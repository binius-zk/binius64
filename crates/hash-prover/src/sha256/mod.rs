// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Batched SHA-256 leaf hashing and inner-node compression.
//!
//! A round cannot start until the one before it lands, so a single chain stalls on itself.
//! Both paths here hash many independent messages per call, which fills those stalls.
//!
//! The portable submodule holds the multi-lane kernel every batched path runs on.
//! The architecture-specific kernels it dispatches to are private submodules:
//!
//! ```text
//!     avx512   x86-64, sixteen lanes held transposed across 512-bit registers
//!     sha_ni   x86-64, independent chains interleaved over the SHA extension
//!     neon     aarch64, the same over the ARMv8 crypto extension
//! ```
//!
//! Each is compiled in only when the target has the feature.
//! Each is also pinned equal to the portable path by its tests.
//!
//! The spec constants live here, since every kernel reads them.
//!
//! Reference: FIPS 180-4, section 6.2.

use std::mem::MaybeUninit;

use binius_hash::{Sha256Compression, Sha256HashSuite};
use binius_utils::{
	FixedSizeSerializeBytes, SerializeBytes,
	rayon::{
		iter::{IndexedParallelIterator, ParallelIterator},
		slice::{ParallelSlice, ParallelSliceMut},
		task_size::{WorkPerItem, min_len_for_work},
	},
};
use bytemuck::must_cast;
use portable::LANES;
use sha2::{Sha256, digest::Output};

use crate::{
	parallel_compression::ParallelPseudoCompression,
	parallel_digest::{ParallelDigest, ParallelDigestAdapter},
	suite::ParallelHashSuite,
};

pub mod portable;

/// Interleaved chains over the x86-64 SHA extension, at any lane count.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "sha",
	target_feature = "sse2",
	target_feature = "ssse3",
	target_feature = "sse4.1"
))]
mod sha_ni;

/// Transposed 16-lane kernel, the fastest measured path on x86-64.
///
/// It is also the only wide option on an AVX-512 machine with no SHA extension, such as
/// Skylake-SP, Cascade Lake, or Cooper Lake.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "avx512f",
	target_feature = "avx512bw"
))]
mod avx512;

/// Interleaved chains over the ARMv8 crypto extension, at four lanes.
#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
mod neon;

/// Bytes in one SHA-256 message block.
const BLOCK_LEN: usize = 64;

/// Bytes in one SHA-256 digest.
const DIGEST_LEN: usize = 32;

/// SHA-256 initial hash values, the starting state of an unkeyed hash.
///
/// The fractional parts of the square roots of the first eight primes, per FIPS 180-4 section
/// 5.3.3.
const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The 64 SHA-256 round constants.
///
/// The fractional parts of the cube roots of the first 64 primes, per FIPS 180-4 section 4.2.2.
const K: [u32; 64] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The largest message, in bytes, that still fits with its padding in a single 64-byte block.
///
/// The padding is one `0x80` terminator and the eight-byte big-endian bit length:
///
/// ```text
///     64 - 1 - 8 = 55
/// ```
const SINGLE_BLOCK_MAX_LEN: usize = BLOCK_LEN - 1 - 8;

/// Minimum batches per task, so one task still covers the work floor the cost model sets.
///
/// The floor counts compressions, but a batched loop hands out whole batches.
/// Applying it to batches directly would inflate the task by the batch width.
/// A layer smaller than that inflated floor would then run as a single task.
#[inline]
fn min_batches_per_task(batch: usize) -> usize {
	min_len_for_work(WorkPerItem::HashCompression).div_ceil(batch)
}

/// Serializes eight state words into a standard SHA-256 digest.
///
/// FIPS 180-4 section 6.2.2 emits the state most significant byte first.
#[inline]
fn be_digest(state: &[u32; 8]) -> Output<Sha256> {
	let mut digest = Output::<Sha256>::default();
	for (chunk, word) in digest.chunks_exact_mut(4).zip(state) {
		chunk.copy_from_slice(&word.to_be_bytes());
	}
	digest
}

/// Batched SHA-256, on top of the sequential pair the verifier side defines.
impl ParallelHashSuite for Sha256HashSuite {
	type ParLeafHash = ParallelSha256Digest;
	type ParCompression = ParallelSha256Compression;
}

/// Folds a batch of node pairs with a single block compression.
///
/// A pair is `left || right`, two 32-byte digests, so the message is exactly one block.
///
/// The output length sets how many pairs are folded and must not exceed the batch width.
/// A partial batch leaves the unused high lanes zero, and nothing reads their output.
#[inline]
fn compress_node_pairs<const N: usize>(
	initial_state: &[u32; 8],
	inputs: &[Output<Sha256>],
	out: &mut [MaybeUninit<Output<Sha256>>],
) {
	// Pack each pair into one 64-byte message block: bytes 0..32 left child, 32..64 right child.
	let mut blocks = [[0u8; BLOCK_LEN]; N];
	for (block, pair) in blocks.iter_mut().zip(inputs.chunks_exact(2)) {
		block[..DIGEST_LEN].copy_from_slice(&pair[0]);
		block[DIGEST_LEN..].copy_from_slice(&pair[1]);
	}

	// Every lane starts from the same domain-separated state.
	let mut states = [*initial_state; N];
	portable::compress256_multi(&mut states, &blocks);

	for (slot, state) in out.iter_mut().zip(states) {
		slot.write(must_cast::<[u32; 8], [u8; DIGEST_LEN]>(state).into());
	}
}

/// Parallel SHA-256 two-to-one compression for the inner nodes of a Merkle tree.
///
/// Batches of independent compressions run through the multi-lane kernel.
/// Every output byte equals compressing that node on its own with the scalar function.
#[derive(Debug, Clone, Default)]
pub struct ParallelSha256Compression {
	/// The scalar two-to-one compression whose output the batched path reproduces exactly.
	compression: Sha256Compression,
}

impl ParallelPseudoCompression<Output<Sha256>, 2> for ParallelSha256Compression {
	type Compression = Sha256Compression;

	fn compression(&self) -> &Self::Compression {
		&self.compression
	}

	fn parallel_compress(
		&self,
		inputs: &[Output<Sha256>],
		out: &mut [MaybeUninit<Output<Sha256>>],
	) {
		assert_eq!(inputs.len(), 2 * out.len(), "Input length must be N * output length");

		// One batch is `LANES` parent nodes, fed by `2 * LANES` child digests.
		// A trailing batch shorter than `LANES` runs the kernel on its valid lanes only.
		inputs
			.par_chunks(2 * LANES)
			.zip(out.par_chunks_mut(LANES))
			.with_min_len(min_batches_per_task(LANES))
			.for_each(|(pairs, out_batch)| {
				compress_node_pairs::<LANES>(self.compression.initial_state(), pairs, out_batch);
			});
	}
}

/// Hashes leaves that fit, together with their padding, in a single 64-byte block.
///
/// Every leaf is the same length, so the padding suffix is shared and built once.
/// Each leaf then overwrites only the message prefix.
///
/// A task reuses its own block buffers across batches, keeping the serialize pass off the
/// allocator.
///
/// # Panics
///
/// Panics if a leaf does not serialize to its expected length.
fn digest_single_block_leaves<const N: usize, I>(
	leaf_len: usize,
	source: impl IndexedParallelIterator<Item = I>,
	out: &mut [MaybeUninit<Output<Sha256>>],
) where
	I: IntoIterator<Item: SerializeBytes>,
{
	debug_assert!(leaf_len <= SINGLE_BLOCK_MAX_LEN, "pre-condition: the leaf fits in one block");

	// The `0x80` terminator right after the message, zeros, then the 64-bit big-endian bit length.
	let mut template = [0u8; BLOCK_LEN];
	template[leaf_len] = 0x80;
	template[BLOCK_LEN - 8..].copy_from_slice(&((leaf_len as u64) * 8).to_be_bytes());

	source
		.chunks(N)
		.zip(out.par_chunks_mut(N))
		.with_min_len(min_batches_per_task(N))
		.for_each_with([template; N], |blocks, (leaves, out_batch)| {
			// Overwrite each lane's message prefix.
			// The padding suffix stays untouched.
			for (block, items) in blocks.iter_mut().zip(leaves) {
				let mut cursor = &mut block[..leaf_len];
				for item in items {
					item.serialize(&mut cursor)
						.expect("pre-condition: items must serialize without error");
				}
				debug_assert!(cursor.is_empty(), "pre-condition: each leaf serializes to leaf_len");
			}

			// A trailing batch leaves the high lanes holding an earlier leaf, whose digest is
			// computed and then dropped, since `out_batch` is shorter than the batch.
			let mut states = [IV; N];
			portable::compress256_multi(&mut states, blocks);
			for (slot, state) in out_batch.iter_mut().zip(states) {
				slot.write(be_digest(&state));
			}
		});
}

/// Hashes leaves too long for a single block, a batch at a time.
///
/// Every leaf is the same length, which is what lets one batch share a block count.
///
/// # Panics
///
/// Panics if a leaf does not serialize to its expected length.
fn digest_multi_block_leaves<const N: usize, I>(
	leaf_len: usize,
	source: impl IndexedParallelIterator<Item = I>,
	out: &mut [MaybeUninit<Output<Sha256>>],
) where
	I: IntoIterator<Item: SerializeBytes>,
{
	use bytes::BytesMut;

	source
		.chunks(N)
		.zip(out.par_chunks_mut(N))
		.with_min_len(min_batches_per_task(N))
		.for_each_with(
			std::array::from_fn::<_, N, _>(|_| BytesMut::new()),
			|bufs, (leaves, out_batch)| {
				let n_leaves = leaves.len();
				for (buf, items) in bufs.iter_mut().zip(leaves) {
					// Reuse the capacity this task's earlier batches already grew.
					buf.clear();
					for item in items {
						item.serialize(&mut *buf)
							.expect("pre-condition: items must serialize without error");
					}
					assert_eq!(
						buf.len(),
						leaf_len,
						"pre-condition: each leaf serializes to leaf_len"
					);
				}

				// All lanes must share a length, so pad an unfilled high lane with zeros.
				// Its digest is computed and then dropped.
				for buf in &mut bufs[n_leaves..] {
					buf.resize(leaf_len, 0);
				}

				let inputs: [&[u8]; N] = std::array::from_fn(|i| bufs[i].as_ref());
				for (slot, digest) in out_batch.iter_mut().zip(portable::sha256_multi(inputs)) {
					slot.write(digest.into());
				}
			},
		);
}

/// Batches fixed-length SHA-256 leaves through the multi-lane kernel.
///
/// With a fixed leaf length, the leaf size picks the route:
///
/// ```text
///     up to 55 bytes : one batched block compression, padding folded into the block
///     longer         : a batched multi-block hash, one shared block count
/// ```
///
/// Without a fixed length a batch cannot share a block count, so that path falls back to
/// hashing one leaf at a time.
#[derive(Debug, Clone, Default)]
pub struct ParallelSha256Digest;

impl ParallelDigest for ParallelSha256Digest {
	type Digest = Sha256;

	fn new() -> Self {
		Self
	}

	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Sha256>>],
	) {
		ParallelDigestAdapter::<Sha256>::new().digest(source, out);
	}

	fn digest_with_const_len<I: IntoIterator<Item: FixedSizeSerializeBytes>>(
		&self,
		n_items_per_input: usize,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Sha256>>],
	) {
		// Every leaf serializes to the same fixed byte length.
		let leaf_len = n_items_per_input * <I::Item as FixedSizeSerializeBytes>::BYTE_SIZE;

		if leaf_len <= SINGLE_BLOCK_MAX_LEN {
			digest_single_block_leaves::<LANES, I>(leaf_len, source, out);
		} else {
			digest_multi_block_leaves::<LANES, I>(leaf_len, source, out);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_utils::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
	use digest::Digest;
	use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::parallel_compression::ParallelCompressionAdaptor;

	#[test]
	fn test_batch_floor_counts_compressions_not_batches() {
		// Invariant: batching must not change how much work one task takes on.
		//
		// The floor is stated per compression, so widening the batch has to divide it.
		// Applying it to batches unconverted is what silently serializes a whole layer.
		let per_compression = min_len_for_work(WorkPerItem::HashCompression);
		assert_eq!(min_batches_per_task(1), per_compression);

		for batch in [2usize, 4, 8, 16] {
			let batches = min_batches_per_task(batch);
			// A task covers at least the floor, and overshoots by less than one batch.
			assert!(batches * batch >= per_compression, "batch {batch} undershoots the floor");
			assert!(
				(batches - 1) * batch < per_compression,
				"batch {batch} overshoots by a whole batch"
			);
		}
	}

	#[test]
	fn test_parallel_compression_matches_adaptor() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the batched path equals per-node scalar compression byte for byte.
		//
		// Node counts crossing every regime of the batching, at LANES of 4, 8, or 16:
		//
		//     1, 2, 3     -> tail only (the top Merkle layers)
		//     4, 8, 16    -> exactly one full batch at some tuned width
		//     5, 9, 17    -> one full batch plus a tail
		//     64, 1000    -> many batches, the second not a multiple of any width
		for n_nodes in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 64, 1000] {
			// Two random child digests per output node.
			let inputs: Vec<Output<Sha256>> = repeat_with(|| {
				let mut digest = Output::<Sha256>::default();
				rng.fill_bytes(&mut digest);
				digest
			})
			.take(2 * n_nodes)
			.collect();

			// Compress with the batched path.
			let mut got = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
				.take(n_nodes)
				.collect::<Vec<_>>();
			ParallelSha256Compression::default().parallel_compress(&inputs, &mut got);

			// Compress every node one at a time through the scalar function as the reference.
			let mut want = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
				.take(n_nodes)
				.collect::<Vec<_>>();
			ParallelCompressionAdaptor::new(Sha256Compression::default())
				.parallel_compress(&inputs, &mut want);

			for (i, (got_i, want_i)) in got.iter().zip(&want).enumerate() {
				// SAFETY: the compression calls above initialize every output slot.
				let (got_i, want_i) =
					unsafe { (got_i.assume_init_ref(), want_i.assume_init_ref()) };
				assert_eq!(got_i, want_i, "mismatch at node {i} of {n_nodes}");
			}
		}
	}

	/// Hashes a run of fixed-length leaves and pins every one to the reference digest.
	fn check_const_len_leaves(rng: &mut StdRng, n_items: usize, n_leaves: usize) {
		// `u128` serializes to 16 little-endian bytes, so the leaf is `16 * n_items` bytes.
		let leaves: Vec<Vec<u128>> = (0..n_leaves)
			.map(|_| (0..n_items).map(|_| rng.random()).collect())
			.collect();

		let mut got = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
			.take(n_leaves)
			.collect::<Vec<_>>();
		ParallelSha256Digest::new().digest_with_const_len(
			n_items,
			leaves.par_iter().map(|leaf| leaf.iter().copied()),
			&mut got,
		);

		for (i, (slot, leaf)) in got.into_iter().zip(&leaves).enumerate() {
			let mut bytes = Vec::new();
			for &item in leaf {
				bytes.extend_from_slice(&item.to_le_bytes());
			}
			// SAFETY: the digest call above initializes every output slot.
			let got_i = unsafe { slot.assume_init() };
			assert_eq!(
				got_i,
				<Sha256 as Digest>::digest(&bytes),
				"leaf {i} of {n_leaves}, {n_items} items"
			);
		}
	}

	#[test]
	fn test_const_len_leaves_match_serial() {
		let mut rng = StdRng::seed_from_u64(0);

		// Leaf lengths straddle the single-block boundary of 55 bytes:
		//
		//     1, 2, 3 items -> 16, 32, 48 bytes -> one block, padding folded in
		//     4, 8 items    -> 64, 128 bytes    -> the batched multi-block route
		//
		// Leaf counts straddle every tuned batch width, and 1 leaves a batch mostly unfilled,
		// which is the case that has to pad the idle lanes rather than read stale bytes.
		for n_items in [1, 2, 3, 4, 8] {
			for n_leaves in [1, 4, 7, 8, 9, 16, 17, 48, 50] {
				check_const_len_leaves(&mut rng, n_items, n_leaves);
			}
		}
	}

	#[test]
	fn test_variable_len_leaves_match_serial() {
		let mut rng = StdRng::seed_from_u64(1);
		// Without a fixed length a batch cannot share a block count.
		// So this routes to the path that hashes one leaf at a time.
		// Pin that path too, since callers reach it.
		let n_leaves = 50;
		let leaves: Vec<Vec<u128>> = (0..n_leaves)
			.map(|_| (0..4).map(|_| rng.random()).collect())
			.collect();

		let mut got = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
			.take(n_leaves)
			.collect::<Vec<_>>();
		ParallelSha256Digest::new()
			.digest(leaves.par_iter().map(|leaf| leaf.iter().copied()), &mut got);

		for (slot, leaf) in got.into_iter().zip(&leaves) {
			let mut bytes = Vec::new();
			for &item in leaf {
				bytes.extend_from_slice(&item.to_le_bytes());
			}
			// SAFETY: the digest call above initializes every output slot.
			assert_eq!(unsafe { slot.assume_init() }, <Sha256 as Digest>::digest(&bytes));
		}
	}
}
