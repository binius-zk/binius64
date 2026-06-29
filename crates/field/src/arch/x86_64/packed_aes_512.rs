// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		pub type AesWideMul64x<T> = super::gfni::gfni_arithmetics::GfniWideMul<T>;
		pub type AesSquare64x<T> = crate::arch::ReuseMultiply<T>;
		pub type AesInvert64x<T> = super::gfni::gfni_arithmetics::Gfni<T>;
	} else {
		// Divide into 64 `u8` lanes (the 1×8b AES packing) for all three ops.
		pub type AesWideMul64x<T> = crate::arch::Divide<u8, T, 64>;
		pub type AesSquare64x<T> = crate::arch::Divide<u8, T, 64>;
		pub type AesInvert64x<T> = crate::arch::Divide<u8, T, 64>;
	}
}
