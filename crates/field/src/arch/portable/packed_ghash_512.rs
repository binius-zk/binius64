// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::scaled_arithmetic::Scaled4xWideMul;
use crate::arch::strategies::ScaledStrategy;

/// Widening-multiply wrapper used by the `PackedBinaryGhash4x128b` packing.
pub type GhashWideMul4x<T> = Scaled4xWideMul<T>;

/// Square strategy for the `PackedBinaryGhash4x128b` packing.
pub type GhashSquare4x = ScaledStrategy;

/// Invert strategy for the `PackedBinaryGhash4x128b` packing.
pub type GhashInvert4x = ScaledStrategy;
