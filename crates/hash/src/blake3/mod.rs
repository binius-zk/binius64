// Copyright 2026 The Binius Developers

//! Blake3 hash and compression functions for use in Merkle tree constructions.
//!
//! [`portable`] holds the multi-lane kernel both parallel paths run on.
//! The arch-specific kernels it dispatches to are private submodules here:
//! - `neon` — hand-written message transpose and block compression for ARM64.
//! - `avx512` — hand-written message transpose for x86-64.
//!
//! Each is compiled in only when the target has the feature, and each is pinned
//! equal to the portable path in [`portable`]'s tests.
//!
//! The Blake3 spec constants live here too, since every kernel reads them.

use digest::Output;
use portable::{PortableBlake3ParallelCompression, PortableBlake3ParallelDigest};

use super::{binary_merkle_tree::HashSuite, compress::CompressionFunction};

pub mod portable;

/// Hand-written vector kernels for the message load and the block compression.
///
/// They stand in for the byte-wise loader and the lane loops in [`portable`].
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

/// Hand-written vector transpose for the message load, used in place of the byte-wise loader.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;

/// Blake3 domain-separation flag marking the first block of a chunk.
const CHUNK_START: u32 = 1 << 0;

/// Blake3 domain-separation flag marking the last block of a chunk.
const CHUNK_END: u32 = 1 << 1;

/// Blake3 domain-separation flag marking the last block of the whole tree.
const ROOT: u32 = 1 << 3;

/// Blake3 initial chaining value: the eight IV words, identical to the SHA-256 IV.
const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Blake3 message permutation applied between rounds.
///
/// The single fixed schedule from section 2.2 of the Blake3 spec, Table 2.
const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// The 7-round count of the Blake3 keyed permutation.
const N_ROUNDS: usize = 7;

/// Message word that each of a round's 16 slots reads, one row per round.
///
/// Blake3 advances the message between rounds by one fixed permutation.
/// Applying that permutation `r` times gives the words round `r` reads:
///
/// ```text
///     row 0:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15    <- the message in order
///     row 1:   2  6  3 10  7  0  4 13  1 11 12  5  9 14 15  8    <- permuted once
///     row 2:   3  4 10 12 13  2  7 14  6  5  9  0 11 15  8  1    <- permuted twice
///     ...
/// ```
///
/// # Why compose it here
///
/// Permuting the words a round reads is the same as permuting the words themselves.
/// Settling that at compile time leaves the 16 message words loaded once and never moved.
const MSG_SCHEDULE: [[usize; 16]; N_ROUNDS] = {
	let mut schedule = [[0usize; 16]; N_ROUNDS];

	// Row 0: the first round consumes the message in its natural order.
	let mut w = 0;
	while w < 16 {
		schedule[0][w] = w;
		w += 1;
	}

	// Row r: reading the row above through the permutation applies it one more time.
	let mut r = 1;
	while r < N_ROUNDS {
		let mut w = 0;
		while w < 16 {
			schedule[r][w] = schedule[r - 1][MSG_PERMUTATION[w]];
			w += 1;
		}
		r += 1;
	}

	schedule
};

/// A two-to-one compression function that hashes the concatenation of its inputs with Blake3.
#[derive(Debug, Clone, Default)]
pub struct Blake3Compression;

impl CompressionFunction<Output<blake3::Hasher>, 2> for Blake3Compression {
	fn compress(&self, input: [Output<blake3::Hasher>; 2]) -> Output<blake3::Hasher> {
		let mut hasher = blake3::Hasher::new();
		hasher.update(input[0].as_slice());
		hasher.update(input[1].as_slice());
		(*hasher.finalize().as_bytes()).into()
	}
}

/// Blake3 [`HashSuite`]: Blake3 leaves and a Blake3 compression function for inner nodes.
///
/// Both parallel compute paths use the portable auto-vectorized kernel, not the scalar loop:
/// - Leaves within one 1024-byte chunk are hashed by the batch kernel.
/// - Larger leaves fall back to the scalar adapter that walks the tree.
/// - Every inner-node level folds its node pairs through the batched two-to-one compression.
///
/// The batch width is fixed at 16 lanes for both paths:
/// - The throughput sweet spot on NEON in the portable-kernel benchmark.
/// - The width the AVX2 / AVX-512 vectorizer fills.
/// - 4 and 8 lanes both measure slower.
#[derive(Debug, Clone, Default)]
pub struct Blake3HashSuite;

impl HashSuite for Blake3HashSuite {
	type LeafHash = blake3::Hasher;
	type Compression = Blake3Compression;
	type ParLeafHash = PortableBlake3ParallelDigest<16>;
	type ParCompression = PortableBlake3ParallelCompression<16>;
}

#[cfg(test)]
mod tests {
	use std::{iter::repeat_with, mem::MaybeUninit};

	use binius_utils::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{ParallelDigest, parallel_digest::ParallelDigestAdapter};

	/// Checks that the compression function matches `blake3::hash` of the concatenated inputs.
	#[test]
	fn test_blake3_compression_matches_reference() {
		let mut rng = StdRng::seed_from_u64(0);
		let left: [u8; 32] = rng.random();
		let right: [u8; 32] = rng.random();

		let compressed = Blake3Compression.compress([left.into(), right.into()]);

		let mut concatenated = [0u8; 64];
		concatenated[..32].copy_from_slice(&left);
		concatenated[32..].copy_from_slice(&right);
		let expected = blake3::hash(&concatenated);

		assert_eq!(compressed.as_slice(), expected.as_bytes());
	}

	/// Checks that the parallel leaf digest matches `blake3::hash` over the serialized leaf bytes.
	#[test]
	fn test_parallel_blake3_matches_serial() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_leaves = 50;
		// `u128` serializes to 16 little-endian bytes.
		let leaves: Vec<Vec<u128>> = (0..n_leaves)
			.map(|_| (0..4).map(|_| rng.random()).collect())
			.collect();

		let digest = <ParallelDigestAdapter<blake3::Hasher> as ParallelDigest>::new();
		let mut results = repeat_with(MaybeUninit::<Output<blake3::Hasher>>::uninit)
			.take(n_leaves)
			.collect::<Vec<_>>();
		digest.digest(leaves.par_iter().map(|leaf| leaf.iter().copied()), &mut results);

		for (result, leaf) in results.into_iter().zip(&leaves) {
			let mut bytes = Vec::new();
			for &item in leaf {
				bytes.extend_from_slice(&item.to_le_bytes());
			}
			let expected = blake3::hash(&bytes);
			assert_eq!(unsafe { result.assume_init() }.as_slice(), expected.as_bytes());
		}
	}

	#[test]
	fn test_portable_leaf_hash_matches_scalar_reference() {
		// The suite's parallel leaf path is the portable vectorized kernel.
		//
		// Pin it equal to the scalar adapter and to `blake3::hash` across the routing boundary.
		let mut rng = StdRng::seed_from_u64(1);
		let n_leaves = 50;

		// Leaves are `u8` items (BYTE_SIZE = 1), so `leaf_len` bytes == `leaf_len` items.
		let mut check = |leaf_len: usize| {
			let leaves: Vec<Vec<u8>> = (0..n_leaves)
				.map(|_| (0..leaf_len).map(|_| rng.random()).collect())
				.collect();

			// The scalar adapter that walks the Blake3 tree — the reference path.
			let mut scalar = repeat_with(MaybeUninit::<Output<blake3::Hasher>>::uninit)
				.take(n_leaves)
				.collect::<Vec<_>>();
			ParallelDigestAdapter::<blake3::Hasher>::default().digest_with_const_len(
				leaf_len,
				leaves.par_iter().map(|leaf| leaf.iter().copied()),
				&mut scalar,
			);

			// The suite's parallel leaf path — the portable batch kernel.
			let mut portable = repeat_with(MaybeUninit::<Output<blake3::Hasher>>::uninit)
				.take(n_leaves)
				.collect::<Vec<_>>();
			<Blake3HashSuite as HashSuite>::ParLeafHash::default().digest_with_const_len(
				leaf_len,
				leaves.par_iter().map(|leaf| leaf.iter().copied()),
				&mut portable,
			);

			// Invariant: both reproduce `blake3::hash`, so their leaf digests match.
			for ((s, p), leaf) in scalar.into_iter().zip(portable).zip(&leaves) {
				let expected = blake3::hash(leaf);
				let (s, p) = unsafe { (s.assume_init(), p.assume_init()) };
				assert_eq!(s.as_slice(), expected.as_bytes(), "scalar, leaf_len {leaf_len}");
				assert_eq!(p.as_slice(), expected.as_bytes(), "portable, leaf_len {leaf_len}");
			}
		};

		// Straddle the 1024-byte routing boundary:
		// - 0, 1, 63, 100, 1000, 1024 : within one chunk -> portable batch route.
		// - 1025, 4096                : multi-chunk       -> scalar adapter fallback.
		for leaf_len in [0, 1, 63, 100, 1000, 1024, 1025, 4096] {
			check(leaf_len);
		}
	}
}
