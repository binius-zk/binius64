// Copyright 2025 Irreducible Inc.
//! Protocol-level constants for the Binius64 constraint system.

/// The minimum number of words per segment.
///
/// This is the minimum size requirement for public input segments in the constraint system.
pub const MIN_WORDS_PER_SEGMENT: usize = 2;

/// The number of bits in a byte.
pub const BYTE_BITS: usize = 8;
