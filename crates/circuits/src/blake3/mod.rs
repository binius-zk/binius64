// Copyright 2025 Irreducible Inc.

//! BLAKE3 circuit gadgets.
//!
//! This module provides circuit primitives for the BLAKE3 hash function. The primitives
//! are exposed as free functions that take input wires and return output wires — no
//! wrapping structs.
//!
//! The entry points are:
//! - [`blake3_compress`] — single-block compression primitive.
//! - [`blake3_fixed`](fn@blake3_fixed) — single-chunk hash gadget for messages of
//!   compile-time-known length up to 1024 bytes (to be added).

pub mod compress;

pub use compress::blake3_compress;

/// BLAKE3 initial chaining value. Same as the SHA-256 IV.
pub const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Message schedule for each of the 7 rounds of the BLAKE3 compression function.
///
/// Matches the `MSG_SCHEDULE` constant in the [reference implementation].
///
/// [reference implementation]: https://github.com/BLAKE3-team/BLAKE3/blob/master/src/portable.rs
pub const MSG_SCHEDULE: [[usize; 16]; 7] = [
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
	[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
	[3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
	[10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
	[12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
	[9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
	[11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

// Domain separation flags.
pub const CHUNK_START: u32 = 1 << 0;
pub const CHUNK_END: u32 = 1 << 1;
pub const PARENT: u32 = 1 << 2;
pub const ROOT: u32 = 1 << 3;
pub const KEYED_HASH: u32 = 1 << 4;
pub const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
pub const DERIVE_KEY_MATERIAL: u32 = 1 << 6;
