// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub type AesWideMul32x<T> = crate::arch::Divide<u8, T, 32>;
pub type AesSquare32x<T> = crate::arch::Scaled<T>;
pub type AesInvert32x<T> = crate::arch::Scaled<T>;
