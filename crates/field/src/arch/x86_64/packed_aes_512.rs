// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		pub type AesWideMul64x<T> = super::gfni::gfni_arithmetics::GfniWideMul<T>;
		pub type AesSquare64x = crate::arch::ReuseMultiplyStrategy;
		pub type AesInvert64x = crate::arch::GfniStrategy;
	} else {
		pub type AesWideMul64x<T> = crate::arch::ElementwiseWideMul<T>;
		pub type AesSquare64x = crate::arch::PairwiseTableStrategy;
		pub type AesInvert64x = crate::arch::PairwiseTableStrategy;
	}
}
