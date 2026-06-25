// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::scaled_arithmetic::Scaled2xWideMul;
use crate::arch::strategies::ScaledStrategy;

/// Widening-multiply wrapper used by the `PackedBinaryGhash2x128b` packing.
pub type GhashWideMul2x<T> = Scaled2xWideMul<T>;

/// Square strategy for the `PackedBinaryGhash2x128b` packing.
pub type GhashSquare2x = ScaledStrategy;

/// Invert strategy for the `PackedBinaryGhash2x128b` packing.
pub type GhashInvert2x = ScaledStrategy;
