// Copyright 2024-2025 Irreducible Inc.

use super::{
	small_uint::{U1, U2, U4},
	underlier_type::{CastFrom, UnderlierType},
};
use crate::arch::{interleave_mask_even, interleave_with_mask};

macro_rules! impl_underlier_type {
	($name:ty, $($mask_idx:literal),+) => {
		impl UnderlierType for $name {
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

impl_underlier_type!(u8, 0, 1, 2);
impl_underlier_type!(u16, 0, 1, 2, 3);
impl_underlier_type!(u32, 0, 1, 2, 3, 4);
impl_underlier_type!(u64, 0, 1, 2, 3, 4, 5);
impl_underlier_type!(u128, 0, 1, 2, 3, 4, 5, 6);

macro_rules! impl_num_cast {
	(@pair U1, U2) => {impl_num_cast!(@small_u_from_small_u U1, U2);};
	(@pair U1, U4) => {impl_num_cast!(@small_u_from_small_u U1, U4);};
	(@pair U2, U4) => {impl_num_cast!(@small_u_from_small_u U2, U4);};
	(@pair U1, $bigger:ty) => {impl_num_cast!(@small_u_from_u U1, $bigger);};
	(@pair U2, $bigger:ty) => {impl_num_cast!(@small_u_from_u U2, $bigger);};
	(@pair U4, $bigger:ty) => {impl_num_cast!(@small_u_from_u U4, $bigger);};
	(@pair $smaller:ident, $bigger:ident) => {
		impl CastFrom<$bigger> for $smaller {
			#[inline(always)]
			fn cast_from(val: $bigger) -> Self {
				val as _
			}
		}

		impl CastFrom<$smaller> for $bigger {
			#[inline(always)]
			fn cast_from(val: $smaller) -> Self {
				val as _
			}
		}
	};
	(@small_u_from_small_u $smaller:ty, $bigger:ty) => {
		impl CastFrom<$bigger> for $smaller {
			#[inline(always)]
			fn cast_from(val: $bigger) -> Self {
				Self::new(val.val()) & Self::ONES
			}
		}

		impl CastFrom<$smaller> for $bigger {
			#[inline(always)]
			fn cast_from(val: $smaller) -> Self {
				Self::new(val.val())
			}
		}
	};
	(@small_u_from_u $smaller:ty, $bigger:ty) => {
		impl CastFrom<$bigger> for $smaller {
			#[inline(always)]
			fn cast_from(val: $bigger) -> Self {
				Self::new(val as u8) & Self::ONES
			}
		}

		impl CastFrom<$smaller> for $bigger {
			#[inline(always)]
			fn cast_from(val: $smaller) -> Self {
				val.val() as _
			}
		}
	};
	($_:ty,) => {};
	(,) => {};
	(all_pairs) => {};
	(all_pairs $_:ty) => {};
	(all_pairs $_:ty,) => {};
	(all_pairs $smaller:ident, $head:ident, $($tail:ident,)*) => {
		impl_num_cast!(@pair $smaller, $head);
		impl_num_cast!(all_pairs $smaller, $($tail,)*);
	};
	($smaller:ident, $($tail:ident,)+) => {
		impl_num_cast!(all_pairs $smaller, $($tail,)+);
		impl_num_cast!($($tail,)+);
	};
}

impl_num_cast!(U1, U2, U4, u8, u16, u32, u64, u128,);
