// Copyright 2026 The Binius Developers

//! Blake3 hash and compression functions for use in Merkle tree constructions.

use digest::Output;

use super::{
	binary_merkle_tree::HashSuite,
	compress::{CompressionFunction, PseudoCompressionFunction},
	parallel_compression::ParallelCompressionAdaptor,
	parallel_digest::ParallelDigestAdapter,
};

/// A two-to-one compression function that hashes the concatenation of its inputs with Blake3.
#[derive(Debug, Clone, Default)]
pub struct Blake3Compression;

impl PseudoCompressionFunction<Output<blake3::Hasher>, 2> for Blake3Compression {
	fn compress(&self, input: [Output<blake3::Hasher>; 2]) -> Output<blake3::Hasher> {
		let mut hasher = blake3::Hasher::new();
		hasher.update(input[0].as_slice());
		hasher.update(input[1].as_slice());
		(*hasher.finalize().as_bytes()).into()
	}
}

impl CompressionFunction<Output<blake3::Hasher>, 2> for Blake3Compression {}

/// Blake3 [`HashSuite`]: Blake3 leaves and a Blake3 compression function for inner nodes.
#[derive(Debug, Clone, Default)]
pub struct Blake3HashSuite;

impl HashSuite for Blake3HashSuite {
	type LeafHash = blake3::Hasher;
	type Compression = Blake3Compression;
	type ParLeafHash = ParallelDigestAdapter<blake3::Hasher>;
	type ParCompression = ParallelCompressionAdaptor<Blake3Compression>;
}

#[cfg(test)]
mod tests {
	use std::{iter::repeat_with, mem::MaybeUninit};

	use binius_utils::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::ParallelDigest;

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
			.map(|_| (0..4).map(|_| rng.random::<u128>()).collect())
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
}
