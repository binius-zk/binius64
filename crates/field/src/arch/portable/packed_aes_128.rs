// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub type AesWideMul16x<T> = crate::arch::ElementwiseWideMul<T>;
// Square/invert one byte at a time via the 1×8b `BytewiseLookup` strategy — no packed arithmetic
// defined in terms of scalar arithmetic.
pub type AesSquare16x<T> = crate::arch::Divide<u8, T>;
pub type AesInvert16x<T> = crate::arch::Divide<u8, T>;
