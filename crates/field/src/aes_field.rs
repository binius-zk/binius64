// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::{Debug, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};

use super::binary_field::{BinaryField, BinaryField1b, binary_field, impl_field_extension};
use crate::{ExtensionField, Field, underlier::U1};

// These fields represent a tower based on AES GF(2^8) field (GF(256)/x^8+x^4+x^3+x+1)
// that is isomorphically included into binary tower, i.e.:
//  - AESTowerField16b is GF(2^16) / (x^2 + x * x_2 + 1) where `x_2` is 0x10 from
// BinaryField8b isomorphically projected to Rijndael8b.
//  - AESTowerField32b is GF(2^32) / (x^2 + x * x_3 + 1), where `x_3` is 0x1000 from
//    AESTowerField16b.
//  ...
// `1 << 5` is the lowest single-bit element of trace 1; `1 << 7` is the only other one.
binary_field!(pub Rijndael8b(u8), 0xD0, 1 << 5);

impl Rijndael8b {
	pub const fn new(value: u8) -> Self {
		Self(value)
	}
}

unsafe impl Pod for Rijndael8b {}

impl_field_extension!(BinaryField1b(U1) < @3 => Rijndael8b(u8));

#[cfg(test)]
mod tests {
	use binius_utils::{DeserializeBytes, SerializeBytes, bytes::BytesMut};
	use proptest::{arbitrary::any, proptest};
	use rand::prelude::*;

	use super::*;
	use crate::{Random, binary_field::tests::is_binary_field_valid_generator};

	fn check_square(f: impl Field) {
		assert_eq!(f.square(), f * f);
	}

	proptest! {
		#[test]
		fn test_square_8(a in any::<u8>()) {
			check_square(Rijndael8b::from(a));
		}
	}

	fn check_invert(f: impl Field) {
		let inversed = f.invert_or_zero();
		if f.is_zero() {
			assert!(inversed.is_zero());
		} else {
			assert_eq!(inversed * f, Field::ONE);
		}
	}

	proptest! {
		#[test]
		fn test_invert_8(a in any::<u8>()) {
			check_invert(Rijndael8b::from(a));
		}
	}

	fn check_mul_by_one<F: Field>(f: F) {
		assert_eq!(F::ONE * f, f);
		assert_eq!(f * F::ONE, f);
	}

	fn check_commutative<F: Field>(f_1: F, f_2: F) {
		assert_eq!(f_1 * f_2, f_2 * f_1);
	}

	fn check_associativity_and_lineraity<F: Field>(f_1: F, f_2: F, f_3: F) {
		assert_eq!(f_1 * (f_2 * f_3), (f_1 * f_2) * f_3);
		assert_eq!(f_1 * (f_2 + f_3), f_1 * f_2 + f_1 * f_3);
	}

	fn check_mul<F: Field>(f_1: F, f_2: F, f_3: F) {
		check_mul_by_one(f_1);
		check_mul_by_one(f_2);
		check_mul_by_one(f_3);

		check_commutative(f_1, f_2);
		check_commutative(f_1, f_3);
		check_commutative(f_2, f_3);

		check_associativity_and_lineraity(f_1, f_2, f_3);
		check_associativity_and_lineraity(f_1, f_3, f_2);
		check_associativity_and_lineraity(f_2, f_1, f_3);
		check_associativity_and_lineraity(f_2, f_3, f_1);
		check_associativity_and_lineraity(f_3, f_1, f_2);
		check_associativity_and_lineraity(f_3, f_2, f_1);
	}

	proptest! {
		#[test]
		fn test_mul_8(a in any::<u8>(), b in any::<u8>(), c in any::<u8>()) {
			check_mul(Rijndael8b::from(a), Rijndael8b::from(b), Rijndael8b::from(c));
		}
	}

	#[test]
	fn test_multiplicative_generators() {
		assert!(is_binary_field_valid_generator::<Rijndael8b>());
	}

	#[test]
	fn test_serialization() {
		let mut buffer = BytesMut::new();
		let mut rng = StdRng::seed_from_u64(0);
		let aes8 = Rijndael8b::random(&mut rng);

		SerializeBytes::serialize(&aes8, &mut buffer).unwrap();

		let mut read_buffer = buffer.freeze();

		assert_eq!(Rijndael8b::deserialize(&mut read_buffer).unwrap(), aes8);
	}
}
