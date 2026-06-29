// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		pub type AesWideMul16x<T> = super::gfni::gfni_arithmetics::GfniWideMul<T>;
		pub type AesSquare16x<T> = crate::arch::ReuseMultiply<T>;
		pub type AesInvert16x<T> = super::gfni::gfni_arithmetics::Gfni<T>;
	} else {
		// Divide into 16 `u8` lanes (the 1×8b AES packing) for all three ops.
		pub type AesWideMul16x<T> = crate::arch::Divide<u8, T, 16>;
		pub type AesSquare16x<T> = crate::arch::Divide<u8, T, 16>;
		pub type AesInvert16x<T> = crate::arch::Divide<u8, T, 16>;
	}
}
