// Copyright 2026 The Binius Developers

#![warn(rustdoc::missing_crate_level_docs)]

//! Batched hash and Merkle tree construction for a Binius prover.
//!
//! Proving folds whole Merkle layers at once, so every hash here takes a batch of independent
//! messages per call and runs it through whatever vector kernel the target has.
//!
//! Nothing a verifier links against depends on this crate.
//! Every batched path is pinned byte-for-byte to the sequential one it stands in for, which is
//! what lets the two sides of a proof commit to the same tree.

pub mod binary_merkle_tree;
pub mod blake3;
pub mod parallel_compression;
pub mod parallel_digest;
pub mod sha256;
pub mod suite;

pub use binary_merkle_tree::BinaryMerkleTree;
pub use blake3::{PortableBlake3ParallelCompression, PortableBlake3ParallelDigest};
pub use parallel_compression::{ParallelCompressionAdaptor, ParallelPseudoCompression};
pub use parallel_digest::{
	MultiDigest, ParallelDigest, ParallelDigestAdapter, ParallelMultidigestImpl,
};
pub use sha256::{ParallelSha256Compression, ParallelSha256Digest};
pub use suite::ParallelHashSuite;
