// Copyright 2026 The Binius Developers

#![warn(rustdoc::missing_crate_level_docs)]

//! Cryptographic hash functions and compression functions for Binius.
//!
//! This crate provides hash function implementations used throughout the Binius proof system,
//! such as standard hash functions (SHA-256).

pub mod binary_merkle_tree;
pub mod blake3;
pub mod compress;
pub mod parallel_compression;
pub mod parallel_digest;
mod serialization;
pub mod sha256;

pub use compress::CompressionFunction;
pub use parallel_compression::{ParallelCompressionAdaptor, ParallelPseudoCompression};
pub use parallel_digest::{
	MultiDigest, ParallelDigest, ParallelDigestAdapter, ParallelMultidigestImpl,
};
pub use serialization::*;
pub use sha256::ParallelSha256Digest;

/// The standard digest is SHA-256.
pub type StdDigest = sha2::Sha256;
pub type StdCompression = sha256::Sha256Compression;
pub type StdHashSuite = sha256::Sha256HashSuite;
