// Copyright 2026 The Binius Developers

//! Blake3 leaf hash and two-to-one compression for Merkle tree constructions.
//!
//! The compression is a plain Blake3 hash of the two children concatenated.
//! The batched kernels a prover wants live in the prover-side crate, not here.

use digest::Output;

use crate::{compress::CompressionFunction, suite::HashSuite};

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

/// Blake3 leaves and a Blake3 compression function for inner nodes.
#[derive(Debug, Clone, Default)]
pub struct Blake3HashSuite;

impl HashSuite for Blake3HashSuite {
	type LeafHash = blake3::Hasher;
	type Compression = Blake3Compression;
}

#[cfg(test)]
mod tests {
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;

	#[test]
	fn test_compression_matches_reference_hash() {
		let mut rng = StdRng::seed_from_u64(0);
		let left: [u8; 32] = rng.random();
		let right: [u8; 32] = rng.random();

		let compressed = Blake3Compression.compress([left.into(), right.into()]);

		// Invariant: the compression is the reference hash of the two children concatenated.
		let mut concatenated = [0u8; 64];
		concatenated[..32].copy_from_slice(&left);
		concatenated[32..].copy_from_slice(&right);
		assert_eq!(compressed.as_slice(), blake3::hash(&concatenated).as_bytes());
	}
}
