// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::scaled_arithmetic::Scaled;
use crate::arch::{Divide, M128};

/// Widening-multiply wrapper used by the `PackedBinaryGhash2x128b` packing: divide into two
/// `M128` lanes and apply the width-1 GHASH `WideMul` to each, deferring reduction per lane.
pub type GhashWideMul2x<T> = Divide<M128, T, 2>;

/// Square wrapper for the `PackedBinaryGhash2x128b` packing.
pub type GhashSquare2x<T> = Scaled<T>;

/// Invert wrapper for the `PackedBinaryGhash2x128b` packing.
pub type GhashInvert2x<T> = Scaled<T>;
