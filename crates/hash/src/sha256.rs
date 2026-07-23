// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! SHA-256 compression function for use in Merkle tree constructions.

use std::mem::MaybeUninit;

use binius_utils::{
	FixedSizeSerializeBytes, SerializeBytes,
	rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
};
use bytemuck::{bytes_of_mut, must_cast};
use digest::Digest;
use sha2::{Sha256, block_api::compress256, digest::Output};

use super::{
	binary_merkle_tree::HashSuite,
	compress::CompressionFunction,
	parallel_compression::ParallelPseudoCompression,
	parallel_digest::{ParallelDigest, ParallelDigestAdapter},
	sha256_x4::compress256_x4,
};

/// Hashes every leaf through the four-way interleaved SHA-256 kernel.
///
/// The leaves serialize into one contiguous buffer, then hash four at a time.
/// The serialize pass is cheap next to the compression work it feeds.
///
/// The caller guarantees the leaf count is a nonzero multiple of four.
/// So every group of four is full.
#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
fn digest_with_const_len_x4<I: IntoIterator<Item: FixedSizeSerializeBytes>>(
	n_items_per_input: usize,
	source: impl IndexedParallelIterator<Item = I>,
	out: &mut [MaybeUninit<Output<Sha256>>],
) {
	use binius_utils::rayon::slice::{ParallelSlice, ParallelSliceMut};

	let leaf_len = n_items_per_input * <I::Item as FixedSizeSerializeBytes>::BYTE_SIZE;

	// Serialize each leaf's bytes into its slot of a contiguous buffer, one leaf per task.
	let mut leaf_bytes = vec![0u8; out.len() * leaf_len];
	source
		.zip(leaf_bytes.par_chunks_mut(leaf_len))
		.for_each(|(items, dst)| {
			let mut cursor = &mut dst[..];
			for item in items {
				item.serialize(&mut cursor)
					.expect("pre-condition: items serialize without error");
			}
			debug_assert!(cursor.is_empty(), "pre-condition: each leaf serializes to leaf_len");
		});

	// Hash four adjacent leaves at once, writing the four digests into their output slots.
	out.par_chunks_mut(4)
		.zip(leaf_bytes.par_chunks(4 * leaf_len))
		.for_each(|(out4, bytes4)| {
			let inputs: [&[u8]; 4] =
				std::array::from_fn(|i| &bytes4[i * leaf_len..(i + 1) * leaf_len]);
			let digests = crate::sha256_x4::sha256_x4(inputs);
			for (slot, digest) in out4.iter_mut().zip(digests) {
				let mut hash = Output::<Sha256>::default();
				hash.copy_from_slice(&digest);
				slot.write(hash);
			}
		});
}

/// SHA-256 initial hash values, used as the starting state for a raw block compression.
const SHA256_IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The largest leaf, in bytes, that still fits (together with SHA-256 padding) in a single
/// 64-byte block: one byte for the `0x80` terminator and eight for the big-endian bit length.
const SINGLE_BLOCK_MAX_LEN: usize = 64 - 1 - 8;

/// A two-to-one compression function for SHA-256 digests.
#[derive(Debug, Clone)]
pub struct Sha256Compression {
	initial_state: [u32; 8],
}

impl Default for Sha256Compression {
	fn default() -> Self {
		let initial_state_bytes = Sha256::digest(b"BINIUS SHA-256 COMPRESS");
		let mut initial_state = [0u32; 8];
		bytes_of_mut(&mut initial_state).copy_from_slice(&initial_state_bytes);
		Self { initial_state }
	}
}

impl CompressionFunction<Output<Sha256>, 2> for Sha256Compression {
	fn compress(&self, input: [Output<Sha256>; 2]) -> Output<Sha256> {
		let mut ret = self.initial_state;
		let mut block = [0u8; 64];
		block[..32].copy_from_slice(input[0].as_slice());
		block[32..].copy_from_slice(input[1].as_slice());
		compress256(&mut ret, &[block]);
		must_cast::<[u32; 8], [u8; 32]>(ret).into()
	}
}

/// SHA-256 [`HashSuite`]: SHA-256 leaves and a SHA-256 compression function for inner nodes.
#[derive(Debug, Clone, Default)]
pub struct Sha256HashSuite;

impl HashSuite for Sha256HashSuite {
	type LeafHash = Sha256;
	type Compression = Sha256Compression;
	type ParLeafHash = ParallelSha256Digest;
	type ParCompression = ParallelSha256Compression;
}

/// Parallel SHA-256 two-to-one compression for the inner nodes of a Merkle tree.
///
/// Groups of four independent compressions run through the interleaved four-way block kernel.
/// A trailing group smaller than four compresses one node at a time.
/// Every output byte equals compressing each node on its own with the scalar function.
#[derive(Debug, Clone, Default)]
pub struct ParallelSha256Compression {
	/// The scalar two-to-one compression whose output the grouped path reproduces exactly.
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
		use binius_utils::rayon::slice::{ParallelSlice, ParallelSliceMut};

		assert_eq!(inputs.len(), 2 * out.len(), "Input length must be N * output length");

		// Split into full groups of four compressions and a short tail.
		//
		//     inputs:  [ 8 digests | 8 digests | ... | tail (< 8) ]
		//     out:     [ 4 nodes   | 4 nodes   | ... | tail (< 4) ]
		let n_groups = out.len() / 4;
		let (input_groups, input_tail) = inputs.split_at(8 * n_groups);
		let (out_groups, out_tail) = out.split_at_mut(4 * n_groups);

		// Each group packs four sibling pairs into four one-block messages.
		// One interleaved call then advances all four states at once.
		input_groups
			.par_chunks_exact(8)
			.zip(out_groups.par_chunks_exact_mut(4))
			.for_each(|(pairs, out4)| {
				// Pack each sibling pair into one 64-byte message block: left child then right.
				let mut blocks = [[0u8; 64]; 4];
				for (block, pair) in blocks.iter_mut().zip(pairs.chunks_exact(2)) {
					block[..32].copy_from_slice(&pair[0]);
					block[32..].copy_from_slice(&pair[1]);
				}

				// All four lanes start from the same fixed initial state.
				let mut states = [self.compression.initial_state; 4];
				compress256_x4(&mut states, [&blocks[0], &blocks[1], &blocks[2], &blocks[3]]);

				// Serialize each advanced state in native word order, as the scalar path does.
				for (slot, state) in out4.iter_mut().zip(states) {
					slot.write(must_cast::<[u32; 8], [u8; 32]>(state).into());
				}
			});

		// The tail (at most three nodes) compresses one pair at a time.
		for (slot, pair) in out_tail.iter_mut().zip(input_tail.chunks_exact(2)) {
			slot.write(self.compression.compress([pair[0], pair[1]]));
		}
	}
}

/// A [`ParallelDigest`] for SHA-256 that specializes
/// [`digest_with_const_len`](ParallelDigest::digest_with_const_len) for short, fixed-length
/// leaves.
///
/// When every leaf serializes to at most `SINGLE_BLOCK_MAX_LEN` bytes, the whole leaf — message,
/// padding, and length suffix — fits in one 64-byte block, so the digest is a single call to the
/// raw [`compress256`] block function starting from the SHA-256 IV. This skips the `update`/
/// `finalize` bookkeeping that the generic [`ParallelDigestAdapter`] performs per leaf.
///
/// Longer leaves fall back to [`ParallelDigestAdapter`].
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
		// On aarch64 with the SHA extension, hash four leaves at once with the interleaved kernel.
		// It needs full groups of four, which every power-of-two leaf count of at least four meets.
		#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
		if out.len() >= 4 && out.len().is_multiple_of(4) {
			digest_with_const_len_x4(n_items_per_input, source, out);
			return;
		}

		let leaf_len = n_items_per_input * <I::Item as FixedSizeSerializeBytes>::BYTE_SIZE;
		if leaf_len > SINGLE_BLOCK_MAX_LEN {
			self.digest(source, out);
			return;
		}

		// Precompute the padding suffix once: a `0x80` terminator immediately after the message,
		// then zeros, then the 64-bit big-endian message bit length. Because `leaf_len` is constant
		// for every leaf, this suffix is identical across leaves; each leaf only overwrites the
		// `leaf_len`-byte message prefix.
		let mut block_template = [0u8; 64];
		block_template[leaf_len] = 0x80;
		block_template[56..64].copy_from_slice(&((leaf_len as u64) * 8).to_be_bytes());

		source
			.zip(out.par_iter_mut())
			.for_each_with(block_template, |block, (items, out)| {
				// Overwrite the message prefix; the padding suffix stays untouched.
				let mut cursor = &mut block[..leaf_len];
				let mut n_items = 0;
				for item in items {
					item.serialize(&mut cursor)
						.expect("pre-condition: items must serialize without error");
					n_items += 1;
				}
				debug_assert_eq!(n_items, n_items_per_input);
				debug_assert!(cursor.is_empty(), "pre-condition: each leaf serializes to leaf_len");

				let mut state = SHA256_IV;
				compress256(&mut state, std::slice::from_ref(&*block));

				// SHA-256 emits its state words in big-endian byte order.
				let mut digest = Output::<Sha256>::default();
				for (chunk, word) in digest.chunks_exact_mut(4).zip(state) {
					chunk.copy_from_slice(&word.to_be_bytes());
				}
				out.write(digest);
			});
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_utils::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
	use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::parallel_compression::ParallelCompressionAdaptor;

	#[test]
	fn test_parallel_sha256_compression_matches_adaptor() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the grouped four-way path equals per-node scalar compression byte for byte.
		//
		// Node counts crossing every regime of the grouping:
		//
		//     1, 2, 3   → tail only (the top Merkle layers)
		//     4         → exactly one full group
		//     5, 7      → full group plus tail
		//     8, 64     → several full groups (64 = a wide tree layer)
		for n_nodes in [1usize, 2, 3, 4, 5, 7, 8, 64] {
			// Two random child digests per output node.
			let inputs: Vec<Output<Sha256>> = repeat_with(|| {
				let mut digest = Output::<Sha256>::default();
				rng.fill_bytes(&mut digest);
				digest
			})
			.take(2 * n_nodes)
			.collect();

			// Compress with the grouped four-way path.
			let grouped = ParallelSha256Compression::default();
			let mut got = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
				.take(n_nodes)
				.collect::<Vec<_>>();
			grouped.parallel_compress(&inputs, &mut got);

			// Compress every node one at a time through the scalar function as the reference.
			let adaptor = ParallelCompressionAdaptor::new(Sha256Compression::default());
			let mut want = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
				.take(n_nodes)
				.collect::<Vec<_>>();
			adaptor.parallel_compress(&inputs, &mut want);

			for (i, (got_i, want_i)) in got.iter().zip(&want).enumerate() {
				// Safety: the compression calls above initialize every output slot.
				let (got_i, want_i) =
					unsafe { (got_i.assume_init_ref(), want_i.assume_init_ref()) };
				assert_eq!(got_i, want_i, "mismatch at node {i} of {n_nodes}");
			}
		}
	}

	/// Checks that the specialized digest matches `Sha256::digest` over the serialized leaf bytes,
	/// covering both the single-block fast path and the multi-block fallback.
	#[test]
	fn test_parallel_sha256_matches_serial() {
		let mut rng = StdRng::seed_from_u64(0);
		// `u128` serializes to 16 little-endian bytes, so leaf lengths are 16, 32, 48 (single
		// block) and 64 (> SINGLE_BLOCK_MAX_LEN, exercises the fallback).
		for n_items_per_input in [1, 2, 3, 4] {
			let n_leaves = 50;
			let leaves: Vec<Vec<u128>> = (0..n_leaves)
				.map(|_| {
					(0..n_items_per_input)
						.map(|_| rng.random::<u128>())
						.collect()
				})
				.collect();

			let digest = ParallelSha256Digest::new();
			let mut results = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
				.take(n_leaves)
				.collect::<Vec<_>>();
			digest.digest_with_const_len(
				n_items_per_input,
				leaves.par_iter().map(|leaf| leaf.iter().copied()),
				&mut results,
			);

			for (result, leaf) in results.into_iter().zip(&leaves) {
				let mut bytes = Vec::new();
				for &item in leaf {
					bytes.extend_from_slice(&item.to_le_bytes());
				}
				assert_eq!(unsafe { result.assume_init() }, <Sha256 as Digest>::digest(&bytes));
			}
		}
	}
}
