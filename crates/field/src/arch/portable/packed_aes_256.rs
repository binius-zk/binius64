// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub type AesWideMul32x<T> = super::scaled_arithmetic::Scaled2xWideMul<T>;
pub type AesSquare32x = crate::arch::ScaledStrategy;
pub type AesInvert32x = crate::arch::ScaledStrategy;
