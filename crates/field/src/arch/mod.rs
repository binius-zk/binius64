// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use cfg_if::cfg_if;

mod arch_optimal;
pub mod portable;
mod shared;
mod strategies;

cfg_if! {
	if #[cfg(all(target_arch = "x86_64"))] {
		mod x86_64;
		pub use x86_64::{packed_128, packed_256, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, packed_ghash_128, packed_ghash_256, packed_ghash_512, M128, M256};
	} else if #[cfg(target_arch = "aarch64")] {
		mod aarch64;
		pub use aarch64::{packed_128, packed_aes_128, packed_ghash_128, M128};
		pub use portable::{packed_256::{self, M256}, packed_512, packed_aes_256, packed_aes_512, packed_ghash_256, packed_ghash_512};
	} else if #[cfg(target_arch = "wasm32")] {
		mod wasm32;
		pub use wasm32::{packed_ghash_128, packed_ghash_256};
		pub use portable::{m128::M128, packed_128, packed_256::{self, M256}, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, packed_ghash_512};
	} else {
		pub use portable::{m128::M128, packed_128, packed_256::{self, M256}, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, packed_ghash_128, packed_ghash_256, packed_ghash_512};
	}
}

pub use arch_optimal::*;
pub(crate) use portable::packed_arithmetic::{interleave_mask_even, interleave_with_mask};
pub use portable::{
	arithmetic::itoh_tsujii::invert_b128, packed::PackedPrimitiveType, packed_1, packed_2,
	packed_4, packed_8, packed_16, packed_32, packed_64, packed_aes_8, packed_aes_16,
	packed_aes_32, packed_aes_64,
};
pub use strategies::*;
