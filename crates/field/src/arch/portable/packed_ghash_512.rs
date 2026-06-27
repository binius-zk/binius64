// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::scaled_arithmetic::Scaled;
use crate::arch::{Divide, M128};

/// Widening-multiply wrapper used by the `PackedBinaryGhash4x128b` packing: divide into four
/// `M128` lanes and apply the width-1 GHASH `WideMul` to each, deferring reduction per lane.
pub type GhashWideMul4x<T> = Divide<M128, T, 4>;

/// Square wrapper for the `PackedBinaryGhash4x128b` packing.
pub type GhashSquare4x<T> = Scaled<T>;

/// Invert wrapper for the `PackedBinaryGhash4x128b` packing.
pub type GhashInvert4x<T> = Scaled<T>;
