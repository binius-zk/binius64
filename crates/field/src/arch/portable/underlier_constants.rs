// Copyright 2024-2025 Irreducible Inc.

use super::packed_arithmetic::{
	UnderlierWithBitConstants, interleave_mask_even, interleave_mask_odd,
};
use crate::underlier::{U1, U2, U4, UnderlierType};

impl UnderlierWithBitConstants for U1 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[];
}

impl UnderlierWithBitConstants for U2 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[Self::new(interleave_mask_even!(u8, 0))];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[Self::new(interleave_mask_odd!(u8, 0))];
}

impl UnderlierWithBitConstants for U4 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		Self::new(interleave_mask_even!(u8, 0)),
		Self::new(interleave_mask_even!(u8, 1)),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		Self::new(interleave_mask_odd!(u8, 0)),
		Self::new(interleave_mask_odd!(u8, 1)),
	];
}

impl UnderlierWithBitConstants for u8 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(Self, 0),
		interleave_mask_even!(Self, 1),
		interleave_mask_even!(Self, 2),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(Self, 0),
		interleave_mask_odd!(Self, 1),
		interleave_mask_odd!(Self, 2),
	];
}

impl UnderlierWithBitConstants for u16 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(Self, 0),
		interleave_mask_even!(Self, 1),
		interleave_mask_even!(Self, 2),
		interleave_mask_even!(Self, 3),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(Self, 0),
		interleave_mask_odd!(Self, 1),
		interleave_mask_odd!(Self, 2),
		interleave_mask_odd!(Self, 3),
	];
}

impl UnderlierWithBitConstants for u32 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(Self, 0),
		interleave_mask_even!(Self, 1),
		interleave_mask_even!(Self, 2),
		interleave_mask_even!(Self, 3),
		interleave_mask_even!(Self, 4),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(Self, 0),
		interleave_mask_odd!(Self, 1),
		interleave_mask_odd!(Self, 2),
		interleave_mask_odd!(Self, 3),
		interleave_mask_odd!(Self, 4),
	];
}

impl UnderlierWithBitConstants for u64 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(Self, 0),
		interleave_mask_even!(Self, 1),
		interleave_mask_even!(Self, 2),
		interleave_mask_even!(Self, 3),
		interleave_mask_even!(Self, 4),
		interleave_mask_even!(Self, 5),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(Self, 0),
		interleave_mask_odd!(Self, 1),
		interleave_mask_odd!(Self, 2),
		interleave_mask_odd!(Self, 3),
		interleave_mask_odd!(Self, 4),
		interleave_mask_odd!(Self, 5),
	];
}

impl UnderlierWithBitConstants for u128 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(Self, 0),
		interleave_mask_even!(Self, 1),
		interleave_mask_even!(Self, 2),
		interleave_mask_even!(Self, 3),
		interleave_mask_even!(Self, 4),
		interleave_mask_even!(Self, 5),
		interleave_mask_even!(Self, 6),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(Self, 0),
		interleave_mask_odd!(Self, 1),
		interleave_mask_odd!(Self, 2),
		interleave_mask_odd!(Self, 3),
		interleave_mask_odd!(Self, 4),
		interleave_mask_odd!(Self, 5),
		interleave_mask_odd!(Self, 6),
	];
}
