// Copyright 2023-2025 Irreducible Inc.

use cfg_if::cfg_if;

pub mod arithmetic;
pub mod packed_ghash_sq_256;

#[cfg(target_feature = "gfni")]
mod gfni;

mod simd;

cfg_if! {
	if #[cfg(target_feature = "sse2")] {
		mod m128;

		pub use m128::M128;
		pub mod packed_aes_128;
		pub mod packed_ghash_128;
	} else {
		pub use super::portable::m128::M128;
		pub use super::portable::packed_aes_128;
		pub use super::portable::packed_ghash_128;
	}
}

cfg_if! {
	if #[cfg(target_feature = "avx2")] {
		mod m256;
		pub use m256::{M256, m256_from_u128s};
		pub mod packed_aes_256;
		pub mod packed_ghash_256;

	} else {
		pub use super::portable::m256::{M256, m256_from_u128s};
		pub use super::portable::packed_aes_256;
		pub use super::portable::packed_ghash_256;
	}
}

cfg_if! {
	if #[cfg(target_feature = "avx512f")] {
		mod m512;
		pub use m512::M512;
		pub mod packed_aes_512;
		pub mod packed_ghash_512;
	} else {
		pub use super::portable::m512::M512;
		pub use super::portable::packed_aes_512;
		pub use super::portable::packed_ghash_512;
	}
}
