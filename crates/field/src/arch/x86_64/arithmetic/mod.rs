// Copyright 2026 The Binius Developers

pub mod ghash;
// The GHASH² strategy here is only selected where a 256-bit carry-less multiply exists;
// otherwise the portable sliced strategy is used.
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2"))]
pub mod ghash_sq;
