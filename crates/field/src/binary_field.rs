// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::{Debug, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{
	DeserializeBytes, FixedSizeSerializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
};
use bytemuck::Zeroable;

use super::{UnderlierType, WithUnderlier, extension::ExtensionField};
use crate::{Field, underlier::U1};

/// A finite field with characteristic 2.
pub trait BinaryField:
	ExtensionField<BinaryField1b> + WithUnderlier<Underlier: UnderlierType>
{
	const N_BITS: usize = Self::ORDER_EXPONENT;
}

/// Generates a binary field type over an underlier `$typ`.
///
/// The default form derives the field's arithmetic from its width-one packing.
/// The `custom_arithmetic` form omits that, for a field that defines its own arithmetic.
macro_rules! binary_field {
	// Default: the field's arithmetic is its width-one packing's arithmetic.
	($vis:vis $name:ident($typ:ty), $gen:expr) => {
		binary_field!(@base $vis $name($typ), $gen);
		binary_field!(@arithmetic_via_packed $name, $typ);
	};
	// The field provides its own `Mul`/`Square`/`InvertOrZero`/`WideMul` separately.
	(custom_arithmetic $vis:vis $name:ident($typ:ty), $gen:expr) => {
		binary_field!(@base $vis $name($typ), $gen);
	};

	(@base $vis:vis $name:ident($typ:ty), $gen:expr) => {
		#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, bytemuck::TransparentWrapper)]
		#[repr(transparent)]
		$vis struct $name(pub(crate) $typ);

		// NOTE: `new` is intentionally NOT generated here. Each field defines its own `new` so it
		// can take an ergonomic constructor type independent of the underlier (e.g.
		// `BinaryField128bGhash::new` takes `u128` even though its underlier is `M128`).
		impl $name {
			pub const fn val(self) -> $typ {
				self.0
			}
		}

		unsafe impl $crate::underlier::WithUnderlier for $name {
			type Underlier = $typ;
		}

		impl Neg for $name {
			type Output = Self;

			fn neg(self) -> Self::Output {
				self
			}
		}

		impl Add<Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn add(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Add<&Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn add(self, rhs: &Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn sub(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<&Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn sub(self, rhs: &Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Mul<&Self> for $name {
			type Output = Self;

			fn mul(self, rhs: &Self) -> Self::Output {
				self * *rhs
			}
		}

		impl AddAssign<Self> for $name {
			fn add_assign(&mut self, rhs: Self) {
				*self = *self + rhs;
			}
		}

		impl AddAssign<&Self> for $name {
			fn add_assign(&mut self, rhs: &Self) {
				*self = *self + *rhs;
			}
		}

		impl SubAssign<Self> for $name {
			fn sub_assign(&mut self, rhs: Self) {
				*self = *self - rhs;
			}
		}

		impl SubAssign<&Self> for $name {
			fn sub_assign(&mut self, rhs: &Self) {
				*self = *self - *rhs;
			}
		}

		impl MulAssign<Self> for $name {
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
			}
		}

		impl MulAssign<&Self> for $name {
			fn mul_assign(&mut self, rhs: &Self) {
				*self = *self * rhs;
			}
		}

		impl Sum<Self> for $name {
			fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::ZERO, |acc, x| acc + x)
			}
		}

		impl<'a> Sum<&'a Self> for $name {
			fn sum<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
				iter.fold(Self::ZERO, |acc, x| acc + x)
			}
		}

		impl Product<Self> for $name {
			fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::ONE, |acc, x| acc * x)
			}
		}

		impl<'a> Product<&'a Self> for $name {
			fn product<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
				iter.fold(Self::ONE, |acc, x| acc * x)
			}
		}


		impl Field for $name {
			const ZERO: Self = $name(<$typ as $crate::underlier::UnderlierType>::ZERO);
			const ONE: Self = $name(<$typ as $crate::underlier::UnderlierType>::ONE);
			const CHARACTERISTIC: usize = 2;
			const ORDER_EXPONENT: usize = <$typ as $crate::underlier::UnderlierType>::BITS;
			const MULTIPLICATIVE_GENERATOR: $name = $name($gen);

			fn double(&self) -> Self {
				Self::ZERO
			}
		}

		// A field element divides into exactly one element of itself. This makes the field a
		// degenerate packed field of width one (see the `PackedField` impl below).
		impl $crate::Divisible<$name> for $name {
			const LOG_N: usize = 0;

			#[inline]
			fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $name> + Send + Clone {
				std::iter::once(value)
			}

			#[inline]
			fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $name> + Send + Clone + '_ {
				std::iter::once(*value)
			}

			#[inline]
			fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $name> + Send + Clone + '_ {
				slice.iter().copied()
			}

			#[inline]
			unsafe fn get_unchecked(&self, _index: usize) -> $name {
				*self
			}

			#[inline]
			unsafe fn set_unchecked(&mut self, _index: usize, val: $name) {
				*self = val;
			}

			#[inline]
			fn broadcast(val: $name) -> Self {
				val
			}

			#[inline]
			fn from_iter(mut iter: impl Iterator<Item = $name>) -> Self {
				iter.next().unwrap_or(Self::ZERO)
			}
		}

		// As a width-one packed field, a field element is masked by its single selector: kept
		// when selected, otherwise zeroed. Uses the same underlier bitmask strategy as
		// PackedPrimitiveType so the mask type and AND operation are consistent.
		impl $crate::Maskable<$name> for $name {
			type Mask = $typ;

			#[inline]
			fn make_mask(mut selectors: impl Iterator<Item = bool>) -> $typ {
				<$typ as $crate::underlier::UnderlierType>::fill_with_bit(
					u8::from(selectors.next().unwrap_or(false)),
				)
			}

			#[inline]
			fn select(&self, mask: &$typ) -> Self {
				Self(self.0 & *mask)
			}
		}

		impl $crate::PackedField for $name {
			#[inline]
			fn iter(&self) -> impl Iterator<Item = Self::Scalar> + Send + Clone + '_ {
				std::iter::once(*self)
			}

			#[inline]
			fn into_iter(self) -> impl Iterator<Item = Self::Scalar> + Send + Clone {
				std::iter::once(self)
			}

			#[inline]
			fn iter_slice(slice: &[Self]) -> impl Iterator<Item = Self::Scalar> + Send + Clone + '_ {
				slice.iter().copied()
			}

			fn interleave(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
				panic!("cannot interleave when WIDTH = 1");
			}

			fn unzip(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
				panic!("cannot transpose when WIDTH = 1");
			}

			#[inline]
			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				f(0)
			}

			#[inline]
			unsafe fn spread_unchecked(self, _log_block_len: usize, _block_idx: usize) -> Self {
				self
			}
		}

		impl ::rand::distr::Distribution<$name> for ::rand::distr::StandardUniform {
			fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> $name {
				$name(::rand::distr::StandardUniform.sample(rng))
			}
		}

		impl Display for $name {
			fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
				write!(f, "0x{repr:0>width$x}", repr=self.val(), width=Self::N_BITS.max(4) / 4)
			}
		}

		impl Debug for $name {
			fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
				let structure_name = std::any::type_name::<$name>().split("::").last().expect("exist");

				write!(f, "{}({})",structure_name, self)
			}
		}

		impl BinaryField for $name {}

		impl From<$typ> for $name {
			fn from(val: $typ) -> Self {
				return Self(val)
			}
		}

		impl From<$name> for $typ {
			fn from(val: $name) -> Self {
				return val.0
			}
		}
	};

	// Each op lifts the underlier into the width-one packing, runs its arithmetic, and lowers back.
	(@arithmetic_via_packed $name:ident, $typ:ty) => {
		impl Mul<Self> for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				$crate::tracing::trace_multiplication!($name);
				type P = $crate::arch::PackedPrimitiveType<$typ, $name>;
				Self((P::from_underlier(self.0) * P::from_underlier(rhs.0)).to_underlier())
			}
		}

		impl $crate::arithmetic_traits::Square for $name {
			#[inline]
			fn square(self) -> Self {
				type P = $crate::arch::PackedPrimitiveType<$typ, $name>;
				Self(
					$crate::arithmetic_traits::Square::square(P::from_underlier(self.0)).to_underlier(),
				)
			}
		}

		impl $crate::arithmetic_traits::InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				type P = $crate::arch::PackedPrimitiveType<$typ, $name>;
				Self(
					$crate::arithmetic_traits::InvertOrZero::invert_or_zero(P::from_underlier(self.0))
						.to_underlier(),
				)
			}
		}

		impl $crate::arithmetic_traits::WideMul for $name {
			type Output = <$crate::arch::PackedPrimitiveType<$typ, $name> as $crate::arithmetic_traits::WideMul>::Output;

			#[inline]
			fn wide_mul(a: Self, b: Self) -> Self::Output {
				type P = $crate::arch::PackedPrimitiveType<$typ, $name>;
				<P as $crate::arithmetic_traits::WideMul>::wide_mul(
					P::from_underlier(a.0),
					P::from_underlier(b.0),
				)
			}

			#[inline]
			fn reduce(wide: Self::Output) -> Self {
				type P = $crate::arch::PackedPrimitiveType<$typ, $name>;
				Self(<P as $crate::arithmetic_traits::WideMul>::reduce(wide).to_underlier())
			}
		}
	};
}

pub(crate) use binary_field;

macro_rules! mul_by_binary_field_1b {
	($name:ident) => {
		impl Mul<BinaryField1b> for $name {
			type Output = Self;

			#[inline]
			#[allow(clippy::suspicious_arithmetic_impl)]
			fn mul(self, rhs: BinaryField1b) -> Self::Output {
				use $crate::underlier::{UnderlierType, WithUnderlier};

				$crate::tracing::trace_multiplication!(BinaryField128b, BinaryField1b);

				Self(self.0 & <$name as WithUnderlier>::Underlier::fill_with_bit(u8::from(rhs.0)))
			}
		}
	};
}

pub(crate) use mul_by_binary_field_1b;

macro_rules! impl_field_extension {
	($subfield_name:ident($subfield_typ:ty) < @$log_degree:expr => $name:ident($typ:ty)) => {
		impl TryFrom<$name> for $subfield_name {
			type Error = ();

			#[inline]
			fn try_from(elem: $name) -> Result<Self, Self::Error> {
				use $crate::underlier::{Divisible, NumCast, UnderlierType};

				// `elem` lies in the subfield iff every subfield-underlier limb above the
				// least-significant one is zero (equivalent to `elem >> N_BITS == 0`).
				let in_subfield = Divisible::<$subfield_typ>::ref_iter(&elem.0)
					.skip(1)
					.all(|limb| limb == <$subfield_typ as UnderlierType>::ZERO);
				if in_subfield {
					Ok($subfield_name(<$subfield_typ>::num_cast_from(elem.val())))
				} else {
					Err(())
				}
			}
		}

		impl From<$subfield_name> for $name {
			#[inline]
			fn from(elem: $subfield_name) -> Self {
				$name(<$typ>::from(elem.val()))
			}
		}

		impl Add<$subfield_name> for $name {
			type Output = Self;

			#[inline]
			fn add(self, rhs: $subfield_name) -> Self::Output {
				self + Self::from(rhs)
			}
		}

		impl Sub<$subfield_name> for $name {
			type Output = Self;

			#[inline]
			fn sub(self, rhs: $subfield_name) -> Self::Output {
				self - Self::from(rhs)
			}
		}

		impl AddAssign<$subfield_name> for $name {
			#[inline]
			fn add_assign(&mut self, rhs: $subfield_name) {
				*self = *self + rhs;
			}
		}

		impl SubAssign<$subfield_name> for $name {
			#[inline]
			fn sub_assign(&mut self, rhs: $subfield_name) {
				*self = *self - rhs;
			}
		}

		impl MulAssign<$subfield_name> for $name {
			#[inline]
			fn mul_assign(&mut self, rhs: $subfield_name) {
				*self = *self * rhs;
			}
		}

		impl Add<$name> for $subfield_name {
			type Output = $name;

			#[inline]
			fn add(self, rhs: $name) -> Self::Output {
				rhs + self
			}
		}

		impl Sub<$name> for $subfield_name {
			type Output = $name;

			#[allow(clippy::suspicious_arithmetic_impl)]
			#[inline]
			fn sub(self, rhs: $name) -> Self::Output {
				rhs + self
			}
		}

		impl Mul<$name> for $subfield_name {
			type Output = $name;

			#[inline]
			fn mul(self, rhs: $name) -> Self::Output {
				rhs * self
			}
		}

		impl ExtensionField<$subfield_name> for $name {
			const LOG_DEGREE: usize = $log_degree;

			#[inline]
			fn basis(i: usize) -> Self {
				use $crate::underlier::{Divisible, UnderlierType};

				assert!(
					i < 1 << $log_degree,
					"index {} out of range for degree {}",
					i,
					1 << $log_degree
				);
				// The `i`-th basis element sets subfield-underlier limb `i` to one, i.e. bit
				// `i * N_BITS` (equivalent to `ONE << (i * N_BITS)`).
				let mut underlier = <$typ as UnderlierType>::ZERO;
				Divisible::<$subfield_typ>::set(
					&mut underlier,
					i,
					<$subfield_typ as UnderlierType>::ONE,
				);
				Self(underlier)
			}

			#[inline]
			fn from_bases_sparse(
				base_elems: impl IntoIterator<Item = $subfield_name>,
				log_stride: usize,
			) -> Self {
				use $crate::underlier::{Divisible, UnderlierType};

				debug_assert!($name::N_BITS.is_power_of_two());
				let shift_step = ($subfield_name::N_BITS << log_stride) & ($name::N_BITS - 1);
				let mut underlier = <$typ as UnderlierType>::ZERO;
				let mut shift = 0;

				for elem in base_elems.into_iter() {
					assert!(shift < $name::N_BITS, "too many base elements for extension degree");
					// `shift` is a multiple of the subfield width, so it addresses limb
					// `shift / N_BITS`; OR the element in (matching the previous `|= .. << shift`).
					let limb = shift / $subfield_name::N_BITS;
					let acc = Divisible::<$subfield_typ>::get(&underlier, limb) | elem.val();
					Divisible::<$subfield_typ>::set(&mut underlier, limb, acc);
					shift += shift_step;
				}

				Self(underlier)
			}

			#[inline]
			fn iter_bases(&self) -> impl Iterator<Item = $subfield_name> {
				use binius_utils::iter::IterExtensions;
				use $crate::underlier::{Divisible, WithUnderlier};

				Divisible::<<$subfield_name as WithUnderlier>::Underlier>::ref_iter(&self.0)
					.map_skippable($subfield_name::from)
			}

			#[inline]
			unsafe fn get_base_unchecked(&self, i: usize) -> $subfield_name {
				use $crate::underlier::{Divisible, WithUnderlier};
				// Safety: the caller guarantees `i < Self::N` (over subfield elements).
				unsafe {
					$subfield_name::from_underlier(Divisible::<
						<$subfield_name as WithUnderlier>::Underlier,
					>::get_unchecked(&self.to_underlier(), i))
				}
			}

			#[inline]
			fn square_transpose(values: &mut [Self]) {
				crate::transpose::square_transforms_extension_field::<$subfield_name, Self>(values)
			}
		}
	};
}

pub(crate) use impl_field_extension;

binary_field!(pub BinaryField1b(U1), U1::new(0x1));

macro_rules! serialize_deserialize {
	($bin_type:ty) => {
		impl SerializeBytes for $bin_type {
			fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
				self.0.serialize(write_buf)
			}
		}

		impl DeserializeBytes for $bin_type {
			fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError> {
				Ok(Self(DeserializeBytes::deserialize(read_buf)?))
			}
		}
	};
}

serialize_deserialize!(BinaryField1b);

impl FixedSizeSerializeBytes for BinaryField1b {
	const BYTE_SIZE: usize = 1;
}

impl BinaryField1b {
	pub const fn new(value: U1) -> Self {
		Self(value)
	}

	/// Creates value without checking that it is within valid range (0 or 1)
	///
	/// # Safety
	/// Value should not exceed 1
	#[inline]
	pub unsafe fn new_unchecked(val: u8) -> Self {
		debug_assert!(val < 2, "val has to be less than 2, but it's {val}");

		Self::new(U1::new_unchecked(val))
	}
}

impl From<u8> for BinaryField1b {
	#[inline]
	fn from(val: u8) -> Self {
		Self::new(U1::new(val))
	}
}

impl From<BinaryField1b> for u8 {
	#[inline]
	fn from(value: BinaryField1b) -> Self {
		value.val().into()
	}
}

impl From<bool> for BinaryField1b {
	#[inline]
	fn from(value: bool) -> Self {
		Self::from(U1::new_unchecked(value.into()))
	}
}

#[cfg(test)]
pub(crate) mod tests {
	use binius_utils::{DeserializeBytes, SerializeBytes, bytes::BytesMut};
	use proptest::prelude::*;

	use super::BinaryField1b as BF1;
	use crate::{
		AESTowerField8b, BinaryField, BinaryField1b, BinaryField128bGhash, Field,
		arithmetic_traits::InvertOrZero,
	};

	#[test]
	fn test_gf2_add() {
		assert_eq!(BF1::from(0) + BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(0) + BF1::from(1), BF1::from(1));
		assert_eq!(BF1::from(1) + BF1::from(0), BF1::from(1));
		assert_eq!(BF1::from(1) + BF1::from(1), BF1::from(0));
	}

	#[test]
	fn test_gf2_sub() {
		assert_eq!(BF1::from(0) - BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(0) - BF1::from(1), BF1::from(1));
		assert_eq!(BF1::from(1) - BF1::from(0), BF1::from(1));
		assert_eq!(BF1::from(1) - BF1::from(1), BF1::from(0));
	}

	#[test]
	fn test_gf2_mul() {
		assert_eq!(BF1::from(0) * BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(0) * BF1::from(1), BF1::from(0));
		assert_eq!(BF1::from(1) * BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(1) * BF1::from(1), BF1::from(1));
	}

	pub(crate) fn is_binary_field_valid_generator<F: BinaryField>() -> bool {
		// Binary fields should contain a multiplicative subgroup of size 2^n - 1
		let mut order = if F::N_BITS == 128 {
			u128::MAX
		} else {
			(1 << F::N_BITS) - 1
		};

		// Naive factorization of group order - represented as a multiset of prime factors
		let mut factorization = Vec::new();

		let mut prime = 2;
		while prime * prime <= order {
			while order.is_multiple_of(prime) {
				order /= prime;
				factorization.push(prime);
			}

			prime += if prime > 2 { 2 } else { 1 };
		}

		if order > 1 {
			factorization.push(order);
		}

		// Iterate over all divisors (some may be tested several times if order is non-square-free)
		for mask in 0..(1 << factorization.len()) {
			let mut divisor = 1;

			for (bit_index, &prime) in factorization.iter().enumerate() {
				if (1 << bit_index) & mask != 0 {
					divisor *= prime;
				}
			}

			// Compute pow(generator, divisor) in log time
			divisor = divisor.reverse_bits();

			let mut pow_divisor = F::ONE;
			while divisor > 0 {
				pow_divisor *= pow_divisor;

				if divisor & 1 != 0 {
					pow_divisor *= F::MULTIPLICATIVE_GENERATOR;
				}

				divisor >>= 1;
			}

			// Generator invariant
			let is_root_of_unity = pow_divisor == F::ONE;
			let is_full_group = mask + 1 == 1 << factorization.len();

			if is_root_of_unity && !is_full_group || !is_root_of_unity && is_full_group {
				return false;
			}
		}

		true
	}

	#[test]
	fn test_multiplicative_generators() {
		assert!(is_binary_field_valid_generator::<BinaryField1b>());
		assert!(is_binary_field_valid_generator::<AESTowerField8b>());
		assert!(is_binary_field_valid_generator::<BinaryField128bGhash>());
	}

	#[test]
	fn test_field_degrees() {
		assert_eq!(BinaryField1b::N_BITS, 1);
		assert_eq!(AESTowerField8b::N_BITS, 8);
		assert_eq!(BinaryField128bGhash::N_BITS, 128);
	}

	#[test]
	fn test_field_formatting() {
		assert_eq!(format!("{}", BinaryField1b::from(1)), "0x1");
		assert_eq!(format!("{}", AESTowerField8b::from(3)), "0x03");
		assert_eq!(
			format!("{}", BinaryField128bGhash::new(5)),
			"0x00000000000000000000000000000005"
		);
	}

	#[test]
	fn test_inverse_on_zero() {
		assert!(BinaryField1b::ZERO.invert_or_zero().is_zero());
		assert!(AESTowerField8b::ZERO.invert_or_zero().is_zero());
		assert!(BinaryField128bGhash::ZERO.invert_or_zero().is_zero());
	}

	proptest! {
		#[test]
		fn test_inverse_8b(val in 1u8..) {
			let x = AESTowerField8b::new(val);
			// Safety: `val` is in `1..`, so `x` is non-zero.
			let x_inverse = unsafe { x.invert() };
			assert_eq!(x * x_inverse, AESTowerField8b::ONE);
		}

		#[test]
		fn test_inverse_128b(val in 1u128..) {
			let x = BinaryField128bGhash::from(val);
			// Safety: `val` is in `1..`, so `x` is non-zero.
			let x_inverse = unsafe { x.invert() };
			assert_eq!(x * x_inverse, BinaryField128bGhash::ONE);
		}
	}

	#[test]
	fn test_serialization() {
		let mut buffer = BytesMut::new();
		let b1 = BinaryField1b::from(0x1);
		let b8 = AESTowerField8b::new(0x12);
		let b128 = BinaryField128bGhash::new(0x147AD0369CF258BE8899AABBCCDDEEFF);

		b1.serialize(&mut buffer).unwrap();
		b8.serialize(&mut buffer).unwrap();
		b128.serialize(&mut buffer).unwrap();

		let mut read_buffer = buffer.freeze();

		assert_eq!(BinaryField1b::deserialize(&mut read_buffer).unwrap(), b1);
		assert_eq!(AESTowerField8b::deserialize(&mut read_buffer).unwrap(), b8);
		assert_eq!(BinaryField128bGhash::deserialize(&mut read_buffer).unwrap(), b128);
	}

	#[test]
	fn test_gf2_new_unchecked() {
		for i in 0..2 {
			assert_eq!(unsafe { BF1::new_unchecked(i) }, BF1::from(i));
		}
	}
}
