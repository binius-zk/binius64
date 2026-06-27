// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

// Divide into 16 `u8` lanes (the 1×8b AES packing) for all three ops — no packed arithmetic
// defined in terms of scalar arithmetic.
pub type AesWideMul16x<T> = crate::arch::Divide<u8, T, 16>;
pub type AesSquare16x<T> = crate::arch::Divide<u8, T, 16>;
pub type AesInvert16x<T> = crate::arch::Divide<u8, T, 16>;
