// Copyright 2026 The Binius Developers

//! FIPS 202 SHA-3 hash functions: SHA3-256, SHA3-384, and SHA3-512.
//!
//! SHA-3 is built from the same 1600-bit Keccak permutation and multi-rate padding rule as the
//! Ethereum-style Keccak hash already in this crate.
//!
//! The only difference is a two-bit domain-separation suffix that FIPS 202 appends to the message
//! before padding.
//!
//! For a byte-aligned message this changes the first padding byte from `0x01` to `0x06`.
//!
//! SHA3-224 is intentionally not provided.
//!
//! Its 224-bit digest does not divide evenly into 64-bit words.
//!
//! This crate represents every digest as a whole number of 64-bit words.
//!
//! Each hash function fixes its own rate and capacity.
//!
//! The capacity is twice the digest length.
//!
//! The rate is whatever remains of the 1600-bit permutation width.
//!
//! The digest never exceeds the rate for any of these three hash functions.
//!
//! That means the digest is always read directly out of the state after the last block's
//! permutation, with no extra squeezing permutation ever needed.
//!
//! | Function  | Rate (bytes) | Capacity (bits) | Digest (words) |
//! |-----------|--------------|------------------|-----------------|
//! | SHA3-256  | 136          | 512              | 4               |
//! | SHA3-384  | 104          | 768              | 6               |
//! | SHA3-512  | 72           | 1024             | 8               |

pub mod fixed_length;
pub mod varlen;

/// The two-bit domain-separation suffix FIPS 202 appends to the message before padding.
///
/// For a byte-aligned message this folds into the first padding byte as `0x06`.
///
/// The original Keccak padding would place `0x01` in that same position instead.
const SHA3_DELIMITER_BYTE: u64 = 0x06;

/// Rate of SHA3-256, in bytes.
///
/// The 1600-bit permutation width minus twice the digest length, converted to bytes.
pub const SHA3_256_RATE_BYTES: usize = 136;

/// Rate of SHA3-384, in bytes.
///
/// The 1600-bit permutation width minus twice the digest length, converted to bytes.
pub const SHA3_384_RATE_BYTES: usize = 104;

/// Rate of SHA3-512, in bytes.
///
/// The 1600-bit permutation width minus twice the digest length, converted to bytes.
pub const SHA3_512_RATE_BYTES: usize = 72;

/// SHA3-256 digest length, in 64-bit words.
pub const SHA3_256_DIGEST_WORDS: usize = 4;

/// SHA3-384 digest length, in 64-bit words.
pub const SHA3_384_DIGEST_WORDS: usize = 6;

/// SHA3-512 digest length, in 64-bit words.
pub const SHA3_512_DIGEST_WORDS: usize = 8;
