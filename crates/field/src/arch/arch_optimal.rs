// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
		pub const OPTIMAL_ALIGNMENT: usize = 512;

		pub type OptimalPackedB1 = crate::PackedBinaryField512x1b;
		pub type OptimalPackedB128 = crate::PackedBinaryGhash4x128b;
		pub type OptimalB128 = crate::BinaryField128bGhash;

	} else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
		pub const OPTIMAL_ALIGNMENT: usize = 256;

		pub type OptimalPackedB1 = crate::PackedBinaryField256x1b;
		pub type OptimalPackedB128 = crate::PackedBinaryGhash2x128b;
		pub type OptimalB128 = crate::BinaryField128bGhash;
	} else if #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))] {
		pub const OPTIMAL_ALIGNMENT: usize = 128;

		pub type OptimalPackedB1 = crate::PackedBinaryField128x1b;
		pub type OptimalPackedB128 = crate::PackedBinaryGhash1x128b;
		pub type OptimalB128 = crate::BinaryField128bGhash;
	} else if #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_feature = "aes"))] {
		pub const OPTIMAL_ALIGNMENT: usize = 128;

		pub type OptimalPackedB1 = crate::PackedBinaryField128x1b;
		pub type OptimalPackedB128 = crate::PackedBinaryGhash1x128b;
		pub type OptimalB128 = crate::BinaryField128bGhash;
	} else {
		pub const OPTIMAL_ALIGNMENT: usize = 128;

		pub type OptimalPackedB1 = crate::PackedBinaryField128x1b;
		pub type OptimalPackedB128 = crate::PackedBinaryGhash1x128b;
		pub type OptimalB128 = crate::BinaryField128bGhash;
	}
}
