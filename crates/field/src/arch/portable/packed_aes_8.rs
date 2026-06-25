// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub type AesWideMul1x<T> = crate::arch::portable::pairwise_table_arithmetic::AesLookupWideMul<T>;
pub type AesSquare1x = crate::arch::PairwiseTableStrategy;
pub type AesInvert1x = crate::arch::PairwiseTableStrategy;
