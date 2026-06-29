// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		pub type AesWideMul32x<T> = super::gfni::gfni_arithmetics::GfniWideMul<T>;
		pub type AesSquare32x<T> = crate::arch::ReuseMultiply<T>;
		pub type AesInvert32x<T> = super::gfni::gfni_arithmetics::Gfni<T>;
	} else {
		// Divide into 32 `u8` lanes (the 1×8b AES packing) for all three ops.
		pub type AesWideMul32x<T> = crate::arch::Divide<u8, T, 32>;
		pub type AesSquare32x<T> = crate::arch::Divide<u8, T, 32>;
		pub type AesInvert32x<T> = crate::arch::Divide<u8, T, 32>;
	}
}
