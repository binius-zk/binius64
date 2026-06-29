// Copyright 2025 Irreducible Inc.
//! Protocol-level constants for the Binius64 constraint system.

use binius_utils::checked_arithmetics::checked_log_2;

/// The protocol proves constraint systems over 64-bit words.
pub const WORD_SIZE_BYTES: usize = 8;

/// The protocol proves constraint systems over 64-bit words.
pub const WORD_SIZE_BITS: usize = WORD_SIZE_BYTES * 8;

/// log2 of [`WORD_SIZE_BITS`].
pub const LOG_WORD_SIZE_BITS: usize = checked_log_2(WORD_SIZE_BITS);

/// The minimum number of words per segment.
///
/// This is the minimum size requirement for public input segments in the constraint system.
pub const MIN_WORDS_PER_SEGMENT: usize = 2;

/// The number of bits in a byte.
pub const BYTE_BITS: usize = 8;

/// log2 of [`BYTE_BITS`].
pub const LOG_BYTE_BITS: usize = checked_log_2(BYTE_BITS);
