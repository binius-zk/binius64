// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub type AesWideMul64x<T> = crate::arch::Divide<u8, T, 64>;
pub type AesSquare64x<T> = crate::arch::Scaled<T>;
pub type AesInvert64x<T> = crate::arch::Scaled<T>;
