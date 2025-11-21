// Copyright 2025 Irreducible Inc.

pub mod compress;
mod serialization;

// Vision permutations with states of 4 and 6 Ghash elements
pub mod vision_4;
pub mod vision_6;

pub use compress::{CompressionFunction, PseudoCompressionFunction};
pub use serialization::*;

/// The standard digest is SHA-256.
pub type StdDigest = sha2::Sha256;
pub type StdCompression = compress::sha256::Sha256Compression;
