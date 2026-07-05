// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

#[cfg(test)]
use std::ops::Shl;

#[cfg(test)]
use super::underlier_type::UnderlierType;
#[cfg(test)]
use crate::Divisible;

#[cfg(test)]
#[allow(unused)]
pub(crate) fn single_element_mask_bits<T: UnderlierType + Shl<usize, Output = T>>(
	bits_count: usize,
) -> T {
	use binius_utils::checked_arithmetics::checked_log_2;

	if bits_count == T::BITS {
		!T::ZERO
	} else {
		let mut result = T::ONE;
		for height in 0..checked_log_2(bits_count) {
			result |= result << (1usize << height)
		}

		result
	}
}

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, bits, proptest};

	use super::{
		super::small_uint::{U1, U2, U4},
		*,
	};

	#[test]
	fn test_from_fn() {
		assert_eq!(u32::from_fn(|_| U1::new(0)), 0);
		assert_eq!(u32::from_fn(|i| U1::new((i % 2) as u8)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U1::new(1)), u32::MAX);

		assert_eq!(u32::from_fn(|_| U2::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U2::new(1)), 0x55555555);
		assert_eq!(u32::from_fn(|_| U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U2::new(3)), u32::MAX);
		assert_eq!(u32::from_fn(|i| U2::new((i % 4) as u8)), 0xe4e4e4e4);

		assert_eq!(u32::from_fn(|_| U4::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U4::new(1)), 0x11111111);
		assert_eq!(u32::from_fn(|_| U4::new(8)), 0x88888888);
		assert_eq!(u32::from_fn(|_| U4::new(31)), 0xffffffff);
		assert_eq!(u32::from_fn(|i| U4::new(i as u8)), 0x76543210);

		assert_eq!(u32::from_fn(|_| 0u8), 0);
		assert_eq!(u32::from_fn(|_| 0xabu8), 0xabababab);
		assert_eq!(u32::from_fn(|_| 255u8), 0xffffffff);
		assert_eq!(u32::from_fn(|i| i as u8), 0x03020100);
	}

	#[test]
	fn test_broadcast_subvalue() {
		assert_eq!(u32::broadcast_subvalue(U1::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U1::new(1)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(U2::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U2::new(1)), 0x55555555);
		assert_eq!(u32::broadcast_subvalue(U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::broadcast_subvalue(U2::new(3)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(U4::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U4::new(1)), 0x11111111);
		assert_eq!(u32::broadcast_subvalue(U4::new(8)), 0x88888888);
		assert_eq!(u32::broadcast_subvalue(U4::new(31)), 0xffffffff);

		assert_eq!(u32::broadcast_subvalue(0u8), 0);
		assert_eq!(u32::broadcast_subvalue(0xabu8), 0xabababab);
		assert_eq!(u32::broadcast_subvalue(255u8), 0xffffffff);
	}

	#[test]
	fn test_divisible_get_u32() {
		let value = 0xab12cd34u32;

		assert_eq!(Divisible::<U1>::get(&value, 0), U1::new(0));
		assert_eq!(Divisible::<U1>::get(&value, 1), U1::new(0));
		assert_eq!(Divisible::<U1>::get(&value, 2), U1::new(1));
		assert_eq!(Divisible::<U1>::get(&value, 31), U1::new(1));

		assert_eq!(Divisible::<U2>::get(&value, 0), U2::new(0));
		assert_eq!(Divisible::<U2>::get(&value, 1), U2::new(1));
		assert_eq!(Divisible::<U2>::get(&value, 2), U2::new(3));
		assert_eq!(Divisible::<U2>::get(&value, 15), U2::new(2));

		assert_eq!(Divisible::<U4>::get(&value, 0), U4::new(4));
		assert_eq!(Divisible::<U4>::get(&value, 1), U4::new(3));
		assert_eq!(Divisible::<U4>::get(&value, 2), U4::new(13));
		assert_eq!(Divisible::<U4>::get(&value, 7), U4::new(10));

		assert_eq!(Divisible::<u8>::get(&value, 0), 0x34u8);
		assert_eq!(Divisible::<u8>::get(&value, 1), 0xcdu8);
		assert_eq!(Divisible::<u8>::get(&value, 2), 0x12u8);
		assert_eq!(Divisible::<u8>::get(&value, 3), 0xabu8);
	}

	proptest! {
		#[test]
		fn test_divisible_set_1b(mut init_val in any::<u32>(), i in 0usize..31, val in bits::u8::masked(1)) {
			Divisible::<U1>::set(&mut init_val, i, U1::new(val));
			assert_eq!(Divisible::<U1>::get(&init_val, i), U1::new(val));
		}

		#[test]
		fn test_divisible_set_2b(mut init_val in any::<u32>(), i in 0usize..15, val in bits::u8::masked(3)) {
			Divisible::<U2>::set(&mut init_val, i, U2::new(val));
			assert_eq!(Divisible::<U2>::get(&init_val, i), U2::new(val));
		}

		#[test]
		fn test_divisible_set_4b(mut init_val in any::<u32>(), i in 0usize..7, val in bits::u8::masked(7)) {
			Divisible::<U4>::set(&mut init_val, i, U4::new(val));
			assert_eq!(Divisible::<U4>::get(&init_val, i), U4::new(val));
		}

		#[test]
		fn test_divisible_set_8b(mut init_val in any::<u32>(), i in 0usize..3, val in bits::u8::masked(15)) {
			Divisible::<u8>::set(&mut init_val, i, val);
			assert_eq!(Divisible::<u8>::get(&init_val, i), val);
		}
	}
}
