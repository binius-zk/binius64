// Copyright 2024-2025 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::arch::packed_ghash_256::PackedBinaryGhash2x128b;

// 512 bit value that just contains four 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryGhash4x128b = [PackedBinaryGhash2x128b; 2]);
