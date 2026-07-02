// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use cfg_if::cfg_if;

mod aes_arithmetic;
mod arch_optimal;
pub mod portable;
mod strategies;

cfg_if! {
	if #[cfg(all(target_arch = "x86_64"))] {
		mod x86_64;
		pub use x86_64::{packed_ghash_128, packed_ghash_256, packed_ghash_512, M128, M256, M512, m256_from_u128s};
		pub use x86_64::packed_ghash_128::{GhashWideMul1x, GhashSquare1x, GhashInvert1x};
		pub use x86_64::packed_ghash_256::{GhashWideMul2x, GhashSquare2x, GhashInvert2x};
		pub use x86_64::packed_ghash_512::{GhashWideMul4x, GhashSquare4x, GhashInvert4x};
	} else if #[cfg(target_arch = "aarch64")] {
		mod aarch64;
		pub use aarch64::{packed_ghash_128, M128, M256, M512, m256_from_u128s};
		pub use aarch64::packed_ghash_128::{GhashWideMul1x, GhashSquare1x, GhashInvert1x};
		pub use portable::{packed_ghash_256, packed_ghash_512};
		pub use portable::packed_ghash_256::{GhashWideMul2x, GhashSquare2x, GhashInvert2x};
		pub use portable::packed_ghash_512::{GhashWideMul4x, GhashSquare4x, GhashInvert4x};
	} else if #[cfg(target_arch = "wasm32")] {
		mod wasm32;
		pub use wasm32::{packed_ghash_128, packed_ghash_256};
		pub use wasm32::packed_ghash_128::{GhashWideMul1x, GhashSquare1x, GhashInvert1x};
		pub use portable::{M128, M256, M512, m256_from_u128s, packed_ghash_512};
		pub use portable::packed_ghash_256::{GhashWideMul2x, GhashSquare2x, GhashInvert2x};
		pub use portable::packed_ghash_512::{GhashWideMul4x, GhashSquare4x, GhashInvert4x};
	} else {
		pub use portable::{M128, M256, M512, m256_from_u128s, packed_ghash_128, packed_ghash_256, packed_ghash_512};
		pub use portable::packed_ghash_128::{GhashWideMul1x, GhashSquare1x, GhashInvert1x};
		pub use portable::packed_ghash_256::{GhashWideMul2x, GhashSquare2x, GhashInvert2x};
		pub use portable::packed_ghash_512::{GhashWideMul4x, GhashSquare4x, GhashInvert4x};
	}
}

pub use aes_arithmetic::PackedAesArithmetic;
pub use arch_optimal::*;
pub(crate) use portable::packed_arithmetic::{interleave_mask_even, interleave_with_mask};
pub use portable::{
	arithmetic::itoh_tsujii::invert_b128, packed::PackedPrimitiveType, scaled_arithmetic::Scaled,
};
pub use strategies::*;
