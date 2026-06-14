// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub(crate) mod divisible;
pub(crate) mod maskable;
mod scaled;
mod small_uint;
mod underlier_impls;
mod underlier_type;
mod underlier_with_bit_ops;

pub use divisible::*;
pub use maskable::*;
pub use scaled::ScaledUnderlier;
pub use small_uint::*;
pub use underlier_type::*;
// The re-exported items are bit-op helpers used only by the SIMD arch backends (and tests), so
// on targets without a SIMD backend (e.g. portable wasm32) nothing consumes them through this
// glob.
#[allow(unused_imports)]
pub(crate) use underlier_with_bit_ops::*;
