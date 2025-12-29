// Copyright 2024-2025 Irreducible Inc.

use super::packed_arithmetic::{
	UnderlierWithBitConstants, interleave_mask_even, interleave_with_mask,
};
use crate::underlier::{U1, U2, U4, UnderlierType};

impl UnderlierWithBitConstants for U1 {
	fn interleave(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
		panic!("interleave not supported for U1");
	}
}

impl UnderlierWithBitConstants for U2 {
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[U2] = &[U2::new(interleave_mask_even!(u8, 0))];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl UnderlierWithBitConstants for U4 {
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[U4] = &[
			U4::new(interleave_mask_even!(u8, 0)),
			U4::new(interleave_mask_even!(u8, 1)),
		];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl UnderlierWithBitConstants for u8 {
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[u8] = &[
			interleave_mask_even!(u8, 0),
			interleave_mask_even!(u8, 1),
			interleave_mask_even!(u8, 2),
		];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl UnderlierWithBitConstants for u16 {
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[u16] = &[
			interleave_mask_even!(u16, 0),
			interleave_mask_even!(u16, 1),
			interleave_mask_even!(u16, 2),
			interleave_mask_even!(u16, 3),
		];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl UnderlierWithBitConstants for u32 {
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[u32] = &[
			interleave_mask_even!(u32, 0),
			interleave_mask_even!(u32, 1),
			interleave_mask_even!(u32, 2),
			interleave_mask_even!(u32, 3),
			interleave_mask_even!(u32, 4),
		];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl UnderlierWithBitConstants for u64 {
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[u64] = &[
			interleave_mask_even!(u64, 0),
			interleave_mask_even!(u64, 1),
			interleave_mask_even!(u64, 2),
			interleave_mask_even!(u64, 3),
			interleave_mask_even!(u64, 4),
			interleave_mask_even!(u64, 5),
		];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl UnderlierWithBitConstants for u128 {
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[u128] = &[
			interleave_mask_even!(u128, 0),
			interleave_mask_even!(u128, 1),
			interleave_mask_even!(u128, 2),
			interleave_mask_even!(u128, 3),
			interleave_mask_even!(u128, 4),
			interleave_mask_even!(u128, 5),
			interleave_mask_even!(u128, 6),
		];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}
