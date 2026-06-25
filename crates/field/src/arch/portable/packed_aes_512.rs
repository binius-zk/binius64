// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub type AesWideMul64x<T> = super::scaled_arithmetic::Scaled4xWideMul<T>;
pub type AesSquare64x<T> = crate::arch::Scaled<T>;
pub type AesInvert64x = crate::arch::ScaledStrategy;
