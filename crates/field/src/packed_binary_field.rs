// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Mul;

use crate::{
	BinaryField1b,
	arch::{M128, M256, M512, PackedPrimitiveType},
	arithmetic_traits::{InvertOrZero, Square, WideMul},
	underlier::{U1, U2, U4, UnderlierType},
};

// Type aliases for the `BinaryField1b` packings. The underlier determines the width; `M128`/`M256`/
// `M512` resolve to the architecture-appropriate type (SIMD where available, scaled otherwise).
pub type PackedBinaryField1x1b = PackedPrimitiveType<U1, BinaryField1b>;
pub type PackedBinaryField2x1b = PackedPrimitiveType<U2, BinaryField1b>;
pub type PackedBinaryField4x1b = PackedPrimitiveType<U4, BinaryField1b>;
pub type PackedBinaryField8x1b = PackedPrimitiveType<u8, BinaryField1b>;
pub type PackedBinaryField16x1b = PackedPrimitiveType<u16, BinaryField1b>;
pub type PackedBinaryField32x1b = PackedPrimitiveType<u32, BinaryField1b>;
pub type PackedBinaryField64x1b = PackedPrimitiveType<u64, BinaryField1b>;
pub type PackedBinaryField128x1b = PackedPrimitiveType<M128, BinaryField1b>;
pub type PackedBinaryField256x1b = PackedPrimitiveType<M256, BinaryField1b>;
pub type PackedBinaryField512x1b = PackedPrimitiveType<M512, BinaryField1b>;

// Every `BinaryField1b` packing shares the same arithmetic, which is available for any underlier:
// addition is bitwise XOR (provided generically for all `PackedPrimitiveType` in `packed.rs`) and
// multiplication is bitwise AND. Squaring and inversion are the identity, since `0` and `1` are
// each their own square and inverse. A single blanket impl over `U` therefore replaces the
// per-type definitions that the `define_packed_binary_field` macro used to generate.
impl<U: UnderlierType> Mul for PackedPrimitiveType<U, BinaryField1b> {
	type Output = Self;

	#[inline]
	#[allow(clippy::suspicious_arithmetic_impl)]
	fn mul(self, rhs: Self) -> Self {
		(self.0 & rhs.0).into()
	}
}

impl<U: UnderlierType> Square for PackedPrimitiveType<U, BinaryField1b> {
	#[inline]
	fn square(self) -> Self {
		self
	}
}

impl<U: UnderlierType> InvertOrZero for PackedPrimitiveType<U, BinaryField1b> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		self
	}
}

impl<U: UnderlierType> WideMul for PackedPrimitiveType<U, BinaryField1b> {
	type Output = Self;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self {
		a * b
	}

	#[inline]
	fn reduce(wide: Self) -> Self {
		wide
	}
}

/// Common code to test different multiply, square and invert implementations
#[cfg(test)]
pub mod test_utils {
	use proptest::{
		arbitrary::{Arbitrary, any},
		strategy::{BoxedStrategy, Strategy},
	};

	use crate::{
		Field, PackedField,
		arch::{M128, M256, M512},
		underlier::WithUnderlier,
	};

	// Proptest generates primitive underliers itself; a SIMD underlier borrows the strategy of the
	// `u128` array it converts from, so `any::<P::Underlier>()` resolves at every packing width.
	impl Arbitrary for M128 {
		type Parameters = ();
		type Strategy = BoxedStrategy<Self>;

		fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
			any::<u128>().prop_map(Self::from).boxed()
		}
	}

	impl Arbitrary for M256 {
		type Parameters = ();
		type Strategy = BoxedStrategy<Self>;

		fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
			any::<[u128; 2]>().prop_map(Self::from).boxed()
		}
	}

	impl Arbitrary for M512 {
		type Parameters = ();
		type Strategy = BoxedStrategy<Self>;

		fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
			any::<[u128; 4]>().prop_map(Self::from).boxed()
		}
	}

	/// Every lane of the product is the product of the operands' lanes.
	pub fn check_mul<P: PackedField + WithUnderlier>(a: P::Underlier, b: P::Underlier) {
		let (a, b) = (P::from_underlier(a), P::from_underlier(b));

		let c = a * b;
		for i in 0..P::WIDTH {
			assert_eq!(c.get(i), a.get(i) * b.get(i));
		}
	}

	/// Every lane of the square is its own lane multiplied by itself.
	pub fn check_square<P: PackedField + WithUnderlier>(a: P::Underlier) {
		let a = P::from_underlier(a);

		let c = a.square();
		for i in 0..P::WIDTH {
			assert_eq!(c.get(i), a.get(i) * a.get(i));
		}
	}

	/// A non-zero lane inverts to its multiplicative inverse, and a zero lane inverts to zero.
	pub fn check_invert_or_zero<P: PackedField + WithUnderlier>(a: P::Underlier) {
		let a = P::from_underlier(a);

		let c = a.invert_or_zero();
		for i in 0..P::WIDTH {
			if a.get(i).is_zero() {
				assert!(c.get(i).is_zero());
			} else {
				assert_eq!(a.get(i) * c.get(i), P::Scalar::ONE);
			}
		}
	}

	/// One deferred product, reduced immediately, equals the plain multiply.
	pub fn check_wide_mul<P: PackedField + WithUnderlier>(a: P::Underlier, b: P::Underlier) {
		let (a, b) = (P::from_underlier(a), P::from_underlier(b));

		assert_eq!(P::reduce(P::wide_mul(a, b)), a * b);
	}

	/// Two deferred products summed and reduced once equal the sum of the plain multiplies.
	pub fn check_wide_mul_linearity<P: PackedField + WithUnderlier>(
		a1: P::Underlier,
		b1: P::Underlier,
		a2: P::Underlier,
		b2: P::Underlier,
	) {
		let (a1, b1) = (P::from_underlier(a1), P::from_underlier(b1));
		let (a2, b2) = (P::from_underlier(a2), P::from_underlier(b2));

		// The sum reaches wide values no single product produces, so this exercises the reduction
		// over its full accumulated domain.
		let sum = P::wide_mul(a1, b1) + P::wide_mul(a2, b2);
		assert_eq!(P::reduce(sum), a1 * b1 + a2 * b2);
	}

	/// Check the packed arithmetic of `$ty` lane-by-lane against its own scalar field.
	macro_rules! packed_field_tests {
		($mod:ident, $ty:ty) => {
			mod $mod {
				use proptest::{prelude::any, proptest};
				use $crate::packed_binary_field::test_utils::{
					check_invert_or_zero, check_mul, check_square, check_wide_mul,
					check_wide_mul_linearity,
				};

				use super::*;

				// The underlier is the packing's raw bit pattern, so one strategy fits every width.
				type U = <$ty as $crate::underlier::WithUnderlier>::Underlier;

				proptest! {
					#[test]
					fn mul(a in any::<U>(), b in any::<U>()) {
						check_mul::<$ty>(a, b);
					}

					#[test]
					fn square(a in any::<U>()) {
						check_square::<$ty>(a);
					}

					#[test]
					fn invert_or_zero(a in any::<U>()) {
						check_invert_or_zero::<$ty>(a);
					}

					#[test]
					fn wide_mul(a in any::<U>(), b in any::<U>()) {
						check_wide_mul::<$ty>(a, b);
					}

					#[test]
					fn wide_mul_linearity(
						a1 in any::<U>(), b1 in any::<U>(),
						a2 in any::<U>(), b2 in any::<U>(),
					) {
						check_wide_mul_linearity::<$ty>(a1, b1, a2, b2);
					}
				}
			}
		};
	}

	pub(crate) use packed_field_tests;

	pub fn check_interleave<P: PackedField + WithUnderlier>(
		lhs: P::Underlier,
		rhs: P::Underlier,
		log_block_len: usize,
	) {
		let lhs = P::from_underlier(lhs);
		let rhs = P::from_underlier(rhs);
		let (a, b) = lhs.interleave(rhs, log_block_len);
		let block_len = 1 << log_block_len;
		for i in (0..P::WIDTH).step_by(block_len * 2) {
			for j in 0..block_len {
				assert_eq!(a.get(i + j), lhs.get(i + j));
				assert_eq!(a.get(i + j + block_len), rhs.get(i + j));

				assert_eq!(b.get(i + j), lhs.get(i + j + block_len));
				assert_eq!(b.get(i + j + block_len), rhs.get(i + j + block_len));
			}
		}
	}

	pub fn check_interleave_all_heights<P: PackedField + WithUnderlier>(
		lhs: P::Underlier,
		rhs: P::Underlier,
	) {
		for log_block_len in 0..P::LOG_WIDTH {
			check_interleave::<P>(lhs, rhs, log_block_len);
		}
	}

	pub fn check_unzip<P: PackedField + WithUnderlier>(
		lhs: P::Underlier,
		rhs: P::Underlier,
		log_block_len: usize,
	) {
		let lhs = P::from_underlier(lhs);
		let rhs = P::from_underlier(rhs);
		let block_len = 1 << log_block_len;
		let (a, b) = lhs.unzip(rhs, log_block_len);
		for i in (0..P::WIDTH / 2).step_by(block_len) {
			for j in 0..block_len {
				assert_eq!(
					a.get(i + j),
					lhs.get(2 * i + j),
					"i: {}, j: {}, log_block_len: {}, P: {:?}",
					i,
					j,
					log_block_len,
					P::zero()
				);
				assert_eq!(
					b.get(i + j),
					lhs.get(2 * i + j + block_len),
					"i: {}, j: {}, log_block_len: {}, P: {:?}",
					i,
					j,
					log_block_len,
					P::zero()
				);
			}
		}

		for i in (0..P::WIDTH / 2).step_by(block_len) {
			for j in 0..block_len {
				assert_eq!(
					a.get(i + j + P::WIDTH / 2),
					rhs.get(2 * i + j),
					"i: {}, j: {}, log_block_len: {}, P: {:?}",
					i,
					j,
					log_block_len,
					P::zero()
				);
				assert_eq!(b.get(i + j + P::WIDTH / 2), rhs.get(2 * i + j + block_len));
			}
		}
	}

	pub fn check_transpose_all_heights<P: PackedField + WithUnderlier>(
		lhs: P::Underlier,
		rhs: P::Underlier,
	) {
		for log_block_len in 0..P::LOG_WIDTH {
			check_unzip::<P>(lhs, rhs, log_block_len);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{fmt::Debug, iter::repeat_with};

	use binius_utils::{
		DeserializeBytes, FixedSizeSerializeBytes, SerializeBytes, bytes::BytesMut,
	};
	use proptest::prelude::*;
	use rand::prelude::*;
	use test_utils::check_interleave_all_heights;

	use super::{test_utils::packed_field_tests, *};
	use crate::{
		Divisible, PackedAESBinaryField1x8b, PackedAESBinaryField16x8b, PackedAESBinaryField32x8b,
		PackedAESBinaryField64x8b, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
		PackedBinaryGhash4x128b, PackedField, Random,
		test_utils::check_transpose_all_heights,
		underlier::{U2, U4},
	};

	fn test_add_packed<P: PackedField + From<u128>>(a_val: u128, b_val: u128) {
		let a = P::from(a_val);
		let b = P::from(b_val);
		let c = a + b;
		for i in 0..P::WIDTH {
			assert_eq!(c.get(i), a.get(i) + b.get(i));
		}
	}

	fn test_mul_packed<P: PackedField>(a: P, b: P) {
		let c = a * b;
		for i in 0..P::WIDTH {
			assert_eq!(c.get(i), a.get(i) * b.get(i));
		}
	}

	fn test_mul_packed_random<P: PackedField>() {
		let mut rng = StdRng::seed_from_u64(0);
		test_mul_packed(P::random(&mut rng), P::random(&mut rng))
	}

	fn test_set_then_get<P: PackedField>() {
		let mut rng = StdRng::seed_from_u64(0);
		let mut elem = P::random(&mut rng);

		let scalars = repeat_with(|| P::Scalar::random(&mut rng))
			.take(P::WIDTH)
			.collect::<Vec<_>>();

		for (i, val) in scalars.iter().enumerate() {
			elem.set(i, *val);
		}
		for (i, val) in scalars.iter().enumerate() {
			assert_eq!(elem.get(i), *val);
		}
	}

	fn test_serialize_then_deserialize<P: PackedField + DeserializeBytes + SerializeBytes>() {
		let mut buffer = BytesMut::new();
		let mut rng = StdRng::seed_from_u64(0);
		let packed = P::random(&mut rng);
		packed.serialize(&mut buffer).unwrap();

		let mut read_buffer = buffer.freeze();

		assert_eq!(P::deserialize(&mut read_buffer).unwrap(), packed);
	}

	#[test]
	fn test_set_then_get_128b() {
		test_set_then_get::<PackedBinaryGhash1x128b>();
		test_set_then_get::<PackedBinaryGhash2x128b>();
		test_set_then_get::<PackedBinaryGhash4x128b>();
	}

	#[test]
	fn test_serialize_then_deserialize_128b() {
		test_serialize_then_deserialize::<PackedBinaryGhash1x128b>();
		test_serialize_then_deserialize::<PackedBinaryGhash2x128b>();
		test_serialize_then_deserialize::<PackedBinaryGhash4x128b>();
	}

	#[test]
	fn test_serialize_deserialize_different_packing_width() {
		let mut rng = StdRng::seed_from_u64(0);

		let packed0 = PackedBinaryGhash1x128b::random(&mut rng);
		let packed1 = PackedBinaryGhash1x128b::random(&mut rng);

		let mut buffer = BytesMut::new();
		packed0.serialize(&mut buffer).unwrap();
		packed1.serialize(&mut buffer).unwrap();

		let mut read_buffer = buffer.freeze();
		let packed01 = PackedBinaryGhash2x128b::deserialize(&mut read_buffer).unwrap();

		assert!(
			packed01
				.iter()
				.zip([packed0, packed1])
				.all(|(x, y)| x == y.get(0))
		);
	}

	// TODO: Generate lots more proptests using macros
	proptest! {
		#[test]
		fn test_add_packed_128x1b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryField128x1b>(a_val, b_val)
		}

		#[test]
		fn test_add_packed_16x8b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedAESBinaryField16x8b>(a_val, b_val)
		}

		#[test]
		fn test_add_packed_1x128b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryGhash1x128b>(a_val, b_val)
		}
	}

	#[test]
	fn test_mul_packed_256x1b() {
		test_mul_packed_random::<PackedBinaryField256x1b>()
	}

	#[test]
	fn test_mul_packed_32x8b() {
		test_mul_packed_random::<PackedAESBinaryField32x8b>()
	}

	#[test]
	fn test_mul_packed_2x128b() {
		test_mul_packed_random::<PackedBinaryGhash2x128b>()
	}

	#[test]
	fn test_iter_size_hint() {
		assert_valid_iterator_with_exact_size_hint::<PackedBinaryField128x1b>();
	}

	fn assert_valid_iterator_with_exact_size_hint<P: PackedField>() {
		assert_eq!(P::default().iter().size_hint(), (P::WIDTH, Some(P::WIDTH)));
		assert_eq!(P::default().into_iter().size_hint(), (P::WIDTH, Some(P::WIDTH)));
		assert_eq!(P::default().iter().count(), P::WIDTH);
		assert_eq!(P::default().into_iter().count(), P::WIDTH);
	}

	packed_field_tests!(packed_8x1b, PackedBinaryField8x1b);
	packed_field_tests!(packed_16x1b, PackedBinaryField16x1b);
	packed_field_tests!(packed_32x1b, PackedBinaryField32x1b);
	packed_field_tests!(packed_64x1b, PackedBinaryField64x1b);
	packed_field_tests!(packed_128x1b, PackedBinaryField128x1b);
	packed_field_tests!(packed_256x1b, PackedBinaryField256x1b);
	packed_field_tests!(packed_512x1b, PackedBinaryField512x1b);

	proptest! {
		#[test]
		fn test_interleave_2b(a_val in 0u8..3, b_val in 0u8..3) {
			check_interleave_all_heights::<PackedBinaryField2x1b>(U2::new(a_val), U2::new(b_val));
		}

		#[test]
		fn test_interleave_4b(a_val in 0u8..16, b_val in 0u8..16) {
			check_interleave_all_heights::<PackedBinaryField4x1b>(U4::new(a_val), U4::new(b_val));
		}

		#[test]
		fn test_interleave_8b(a_val in 0u8.., b_val in 0u8..) {
			check_interleave_all_heights::<PackedBinaryField8x1b>(a_val, b_val);
			check_interleave_all_heights::<PackedAESBinaryField1x8b>(a_val, b_val);
		}

		#[test]
		fn test_interleave_16b(a_val in 0u16.., b_val in 0u16..) {
			check_interleave_all_heights::<PackedBinaryField16x1b>(a_val, b_val);
		}

		#[test]
		fn test_interleave_32b(a_val in 0u32.., b_val in 0u32..) {
			check_interleave_all_heights::<PackedBinaryField32x1b>(a_val, b_val);
		}

		#[test]
		fn test_interleave_64b(a_val in 0u64.., b_val in 0u64..) {
			check_interleave_all_heights::<PackedBinaryField64x1b>(a_val, b_val);
		}

		#[test]
		#[allow(clippy::useless_conversion)] // this warning depends on the target platform
		fn test_interleave_128b(a_val in 0u128.., b_val in 0u128..) {
			check_interleave_all_heights::<PackedBinaryField128x1b>(a_val.into(), b_val.into());
			check_interleave_all_heights::<PackedAESBinaryField16x8b>(a_val.into(), b_val.into());
			check_interleave_all_heights::<PackedBinaryGhash1x128b>(a_val.into(), b_val.into());
		}

		#[test]
		fn test_interleave_256b(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
			check_interleave_all_heights::<PackedBinaryField256x1b>(a_val.into(), b_val.into());
			check_interleave_all_heights::<PackedAESBinaryField32x8b>(a_val.into(), b_val.into());
			check_interleave_all_heights::<PackedBinaryGhash2x128b>(a_val.into(), b_val.into());
		}

		#[test]
		fn test_interleave_512b(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
			check_interleave_all_heights::<PackedBinaryField512x1b>(a_val.into(), b_val.into());
			check_interleave_all_heights::<PackedAESBinaryField64x8b>(a_val.into(), b_val.into());
			check_interleave_all_heights::<PackedBinaryGhash4x128b>(a_val.into(), b_val.into());
		}

		#[test]
		fn check_transpose_2b(a_val in 0u8..3, b_val in 0u8..3) {
			check_transpose_all_heights::<PackedBinaryField2x1b>(U2::new(a_val), U2::new(b_val));
		}

		#[test]
		fn check_transpose_4b(a_val in 0u8..16, b_val in 0u8..16) {
			check_transpose_all_heights::<PackedBinaryField4x1b>(U4::new(a_val), U4::new(b_val));
		}

		#[test]
		fn check_transpose_8b(a_val in 0u8.., b_val in 0u8..) {
			check_transpose_all_heights::<PackedBinaryField8x1b>(a_val, b_val);
			check_transpose_all_heights::<PackedAESBinaryField1x8b>(a_val, b_val);
		}

		#[test]
		fn check_transpose_16b(a_val in 0u16.., b_val in 0u16..) {
			check_transpose_all_heights::<PackedBinaryField16x1b>(a_val, b_val);
		}

		#[test]
		fn check_transpose_32b(a_val in 0u32.., b_val in 0u32..) {
			check_transpose_all_heights::<PackedBinaryField32x1b>(a_val, b_val);
		}

		#[test]
		fn check_transpose_64b(a_val in 0u64.., b_val in 0u64..) {
			check_transpose_all_heights::<PackedBinaryField64x1b>(a_val, b_val);
		}

		#[test]
		#[allow(clippy::useless_conversion)] // this warning depends on the target platform
		fn check_transpose_128b(a_val in 0u128.., b_val in 0u128..) {
			check_transpose_all_heights::<PackedBinaryField128x1b>(a_val.into(), b_val.into());
			check_transpose_all_heights::<PackedAESBinaryField16x8b>(a_val.into(), b_val.into());
			check_transpose_all_heights::<PackedBinaryGhash1x128b>(a_val.into(), b_val.into());
		}

		#[test]
		fn check_transpose_256b(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
			check_transpose_all_heights::<PackedBinaryField256x1b>(a_val.into(), b_val.into());
			check_transpose_all_heights::<PackedAESBinaryField32x8b>(a_val.into(), b_val.into());
			check_transpose_all_heights::<PackedBinaryGhash2x128b>(a_val.into(), b_val.into());
		}

		#[test]
		fn check_transpose_512b(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
			check_transpose_all_heights::<PackedBinaryField512x1b>(a_val.into(), b_val.into());
			check_transpose_all_heights::<PackedAESBinaryField64x8b>(a_val.into(), b_val.into());
			check_transpose_all_heights::<PackedBinaryGhash4x128b>(a_val.into(), b_val.into());
		}
	}

	// The generic `SerializeBytes`/`DeserializeBytes` impls on `PackedPrimitiveType` round-trip
	// across both integer underliers and (where applicable) SIMD underliers.
	#[test]
	fn test_serialize_roundtrip() {
		fn check_roundtrip<
			P: PackedField + SerializeBytes + DeserializeBytes + PartialEq + Debug,
		>(
			rng: &mut StdRng,
		) {
			let value = P::random(rng);
			let mut buf = BytesMut::new();
			value.serialize(&mut buf).unwrap();
			let deserialized = P::deserialize(buf.freeze()).unwrap();
			assert_eq!(value, deserialized);
		}

		let mut rng = StdRng::seed_from_u64(0);
		check_roundtrip::<PackedBinaryField8x1b>(&mut rng);
		check_roundtrip::<PackedBinaryField64x1b>(&mut rng);
		check_roundtrip::<PackedBinaryField128x1b>(&mut rng);
		check_roundtrip::<PackedBinaryGhash1x128b>(&mut rng);
	}

	// `FixedSizeSerializeBytes` propagates from the underlier. Integer-backed packed fields (here,
	// `u8`- and `u64`-backed on every arch) report the underlier's byte size.
	#[test]
	fn test_fixed_size_byte_size() {
		assert_eq!(<PackedBinaryField8x1b as FixedSizeSerializeBytes>::BYTE_SIZE, 1);
		assert_eq!(<PackedBinaryField64x1b as FixedSizeSerializeBytes>::BYTE_SIZE, 8);
	}
}
