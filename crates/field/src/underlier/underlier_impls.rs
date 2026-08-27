// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::traits::Underlier;
use crate::arch::{interleave_mask_even, interleave_with_mask};

macro_rules! impl_underlier {
	($name:ty, $($mask_idx:literal),+) => {
		impl Underlier for $name {
			const LOG_BITS: usize =
				binius_utils::checked_arithmetics::checked_log_2(Self::BITS as _);

			const ZERO: Self = 0;
			const ONE: Self = 1;
			const ONES: Self = Self::MAX;

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				const MASKS: &[$name] = &[
					$(interleave_mask_even!($name, $mask_idx)),+
				];
				interleave_with_mask(self, other, log_block_len, MASKS)
			}
		}
	};
}

impl_underlier!(u8, 0, 1, 2);
impl_underlier!(u16, 0, 1, 2, 3);
impl_underlier!(u32, 0, 1, 2, 3, 4);
impl_underlier!(u64, 0, 1, 2, 3, 4, 5);
impl_underlier!(u128, 0, 1, 2, 3, 4, 5, 6);
