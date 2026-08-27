// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! SHA-256 leaf hash and two-to-one compression for Merkle tree constructions.
//!
//! One block compression per node, straight through the reference implementation.
//! The batched kernels a prover wants live in the prover-side crate, not here.

use bytemuck::{bytes_of_mut, must_cast};
use digest::Digest;
use sha2::{Sha256, block_api::compress256, digest::Output};

use crate::{compress::CompressionFunction, suite::HashSuite};

/// Bytes in one SHA-256 message block.
const BLOCK_LEN: usize = 64;

/// Bytes in one SHA-256 digest.
const DIGEST_LEN: usize = 32;

/// A two-to-one compression function for SHA-256 digests.
///
/// One raw block compression of `left || right`, from a domain-separated initial state.
/// This is not a full hash of the pair: there is no padding block and no length suffix.
#[derive(Debug, Clone)]
pub struct Sha256Compression {
	/// Domain-separating initial state, standing in for the SHA-256 IV.
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

impl Sha256Compression {
	/// The domain-separated state every compression starts from.
	///
	/// A batched implementation has to seed its lanes with exactly these words to agree with
	/// this one, so the state is part of the public contract rather than an internal detail.
	pub const fn initial_state(&self) -> &[u32; 8] {
		&self.initial_state
	}
}

impl CompressionFunction<Output<Sha256>, 2> for Sha256Compression {
	fn compress(&self, input: [Output<Sha256>; 2]) -> Output<Sha256> {
		// The two 32-byte children fill one 64-byte block exactly.
		let mut block = [0u8; BLOCK_LEN];
		block[..DIGEST_LEN].copy_from_slice(input[0].as_slice());
		block[DIGEST_LEN..].copy_from_slice(input[1].as_slice());

		let mut state = self.initial_state;
		compress256(&mut state, std::slice::from_ref(&block));

		// Native word order, not the big-endian digest order.
		// The output only has to be a fixed 32-byte function of the pair.
		must_cast::<[u32; 8], [u8; DIGEST_LEN]>(state).into()
	}
}

/// SHA-256 leaves and a SHA-256 compression function for inner nodes.
#[derive(Debug, Clone, Default)]
pub struct Sha256HashSuite;

impl HashSuite for Sha256HashSuite {
	type LeafHash = Sha256;
	type Compression = Sha256Compression;
}

#[cfg(test)]
mod tests {
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;

	#[test]
	fn test_compression_matches_known_answer() {
		// The compression emits its state in native word order, not the big-endian digest order.
		// A Merkle path committed under one order does not verify under the other, so the
		// convention is pinned here by a vector computed from FIPS 180-4 directly.
		//
		// Left child all zero, right child the bytes 0..32, under the domain-separated state.
		let left: Output<Sha256> = [0u8; DIGEST_LEN].into();
		let right: Output<Sha256> = std::array::from_fn::<u8, DIGEST_LEN, _>(|i| i as u8).into();

		let got = Sha256Compression::default().compress([left, right]);

		let want = "4731c4e3a3190d19dace68db5752af1b4ecf26305e75e85db86217662bbeff74";
		let got_hex: String = got.iter().map(|b| format!("{b:02x}")).collect();
		assert_eq!(got_hex, want, "the compression byte order changed");
	}

	#[test]
	fn test_compression_is_one_block_from_the_domain_state() {
		let mut rng = StdRng::seed_from_u64(0);
		let compression = Sha256Compression::default();

		// Invariant: the compression is one raw block of `left || right` from the fixed state.
		for _ in 0..64 {
			let mut left = Output::<Sha256>::default();
			let mut right = Output::<Sha256>::default();
			rng.fill_bytes(&mut left);
			rng.fill_bytes(&mut right);

			let got = compression.compress([left, right]);

			let mut block = [0u8; BLOCK_LEN];
			block[..DIGEST_LEN].copy_from_slice(&left);
			block[DIGEST_LEN..].copy_from_slice(&right);
			let mut want = *compression.initial_state();
			compress256(&mut want, std::slice::from_ref(&block));

			assert_eq!(got.as_slice(), must_cast::<[u32; 8], [u8; DIGEST_LEN]>(want));
		}
	}
}
