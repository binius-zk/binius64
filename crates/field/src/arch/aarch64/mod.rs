// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_feature = "neon", target_feature = "aes"))] {
		pub(super) mod m128;
		pub(super) mod simd_arithmetic;

		pub mod arithmetic;

		pub mod packed_aes_128;
		pub mod packed_ghash_128;

		pub use m128::M128;
		pub use super::portable::m256::{M256, m256_from_u128s};
		pub use super::portable::m512::M512;
	} else {
		pub use super::portable::packed_aes_128;
		pub use super::portable::packed_ghash_128;

		pub use super::portable::m128::M128;
		pub use super::portable::m256::{M256, m256_from_u128s};
		pub use super::portable::m512::M512;
	}
}
