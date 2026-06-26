// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::scaled_arithmetic::{Scaled, Scaled2xWideMul};

/// Widening-multiply wrapper used by the `PackedBinaryGhash2x128b` packing.
pub type GhashWideMul2x<T> = Scaled2xWideMul<T>;

/// Square wrapper for the `PackedBinaryGhash2x128b` packing.
pub type GhashSquare2x<T> = Scaled<T>;

/// Invert wrapper for the `PackedBinaryGhash2x128b` packing.
pub type GhashInvert2x<T> = Scaled<T>;
