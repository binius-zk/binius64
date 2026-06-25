// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub type AesWideMul16x<T> = crate::arch::ElementwiseWideMul<T>;
pub type AesSquare16x = crate::arch::PairwiseTableStrategy;
pub type AesInvert16x = crate::arch::PairwiseTableStrategy;
