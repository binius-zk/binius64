// Copyright 2026 The Binius Developers

#![warn(rustdoc::missing_crate_level_docs)]
// A verifier's security must not rest on hand-written vector code, so this crate carries none.
// The one exception is marked at its definition: filling a hash buffer implements an unsafe
// trait from the buffer crate.
#![deny(unsafe_code)]

//! Hash and compression functions a Binius verifier depends on.
//!
//! Everything here is sequential and reference-driven: one block compression per node, one
//! digest per leaf, no vector kernels and no thread pool.
//!
//! The batched and architecture-specific kernels a prover wants live in the prover-side
//! crate, so nothing a verifier links against depends on them.

pub mod blake3;
pub mod compress;
mod serialization;
pub mod sha256;
pub mod suite;

pub use blake3::{Blake3Compression, Blake3HashSuite};
pub use compress::CompressionFunction;
pub use serialization::*;
pub use sha256::{Sha256Compression, Sha256HashSuite};
pub use suite::HashSuite;

/// The standard digest is SHA-256.
pub type StdDigest = sha2::Sha256;

/// The standard two-to-one compression pairs with the standard digest.
pub type StdCompression = sha256::Sha256Compression;

/// The standard hash suite pairs the standard digest with the standard compression.
pub type StdHashSuite = sha256::Sha256HashSuite;
