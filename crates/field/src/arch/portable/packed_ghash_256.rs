// Copyright 2024-2025 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::arch::packed_ghash_128::PackedBinaryGhash1x128b;

// 256 bit value that just contains two 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryGhash2x128b = [PackedBinaryGhash1x128b; 2]);
