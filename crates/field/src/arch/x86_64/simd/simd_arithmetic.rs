// Copyright 2024-2025 Irreducible Inc.

use std::{any::TypeId, arch::x86_64::*};

use crate::{
	BinaryField, PackedField, TowerField,
	aes_field::AESTowerField8b,
	arch::{
		SimdStrategy,
		portable::{
			packed::PackedPrimitiveType, packed_arithmetic::PackedTowerField,
			reuse_multiply_arithmetic::Alpha,
		},
	},
	arithmetic_traits::{
		MulAlpha, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedPackedTransformationFactory,
		TaggedSquare,
	},
	linear_transformation::{FieldLinearTransformation, Transformation},
	packed::PackedBinaryField,
	underlier::{UnderlierType, UnderlierWithBitOps, WithUnderlier},
};

pub trait TowerSimdType: Sized + Copy + UnderlierWithBitOps {
	/// Blend odd and even elements
	fn blend_odd_even<Scalar: BinaryField>(a: Self, b: Self) -> Self;
	/// Set alpha to even elements
	fn set_alpha_even<Scalar: BinaryField>(self) -> Self;
	/// Apply `mask` to `a` (set zeros at positions where high bit of the `mask` is 0).
	fn apply_mask<Scalar: BinaryField>(mask: Self, a: Self) -> Self;

	/// Bit xor operation
	fn xor(a: Self, b: Self) -> Self;

	/// Shuffle 8-bit elements within 128-bit lanes
	fn shuffle_epi8(a: Self, b: Self) -> Self;

	/// Byte shifts within 128-bit lanes
	fn bslli_epi128<const IMM8: i32>(self) -> Self;
	fn bsrli_epi128<const IMM8: i32>(self) -> Self;

	/// Initialize value with a single element
	fn set1_epi128(val: __m128i) -> Self;
	fn set_epi_64(val: i64) -> Self;

	#[inline(always)]
	fn dup_shuffle<Scalar: BinaryField>() -> Self {
		let shuffle_mask_128 = unsafe {
			match Scalar::N_BITS.ilog2() {
				3 => _mm_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0),
				4 => _mm_set_epi8(13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0),
				5 => _mm_set_epi8(11, 10, 9, 8, 11, 10, 9, 8, 3, 2, 1, 0, 3, 2, 1, 0),
				6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0),
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(shuffle_mask_128)
	}

	#[inline(always)]
	fn flip_shuffle<Scalar: BinaryField>() -> Self {
		let flip_mask_128 = unsafe {
			match Scalar::N_BITS.ilog2() {
				3 => _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1),
				4 => _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2),
				5 => _mm_set_epi8(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4),
				6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8),
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(flip_mask_128)
	}

	/// Creates mask to propagate the highest bit form mask to other element bytes
	#[inline(always)]
	fn make_epi8_mask_shuffle<Scalar: BinaryField>() -> Self {
		let epi8_mask_128 = unsafe {
			match Scalar::N_BITS.ilog2() {
				4 => _mm_set_epi8(15, 15, 13, 13, 11, 11, 9, 9, 7, 7, 5, 5, 3, 3, 1, 1),
				5 => _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3),
				6 => _mm_set_epi8(15, 15, 15, 15, 15, 15, 15, 15, 7, 7, 7, 7, 7, 7, 7, 7),
				7 => _mm_set1_epi8(15),
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(epi8_mask_128)
	}

	#[inline(always)]
	fn alpha<Scalar: BinaryField>() -> Self {
		let alpha_128 = {
			match Scalar::N_BITS.ilog2() {
				3 => {
					// Compiler will optimize this if out for each instantiation
					let type_id = TypeId::of::<Scalar>();
					let value = if type_id == TypeId::of::<AESTowerField8b>() {
						0xd3u8 as i8
					} else {
						panic!("tower field not supported")
					};
					unsafe { _mm_set1_epi8(value) }
				}
				4 => unsafe { _mm_set1_epi16(0x0100) },
				5 => unsafe { _mm_set1_epi32(0x00010000) },
				6 => unsafe { _mm_set1_epi64x(0x0000000100000000) },
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(alpha_128)
	}

	#[inline(always)]
	fn even_mask<Scalar: BinaryField>() -> Self {
		let mask_128 = {
			match Scalar::N_BITS.ilog2() {
				3 => unsafe { _mm_set1_epi16(0x00FF) },
				4 => unsafe { _mm_set1_epi32(0x0000FFFF) },
				5 => unsafe { _mm_set1_epi64x(0x00000000FFFFFFFF) },
				6 => unsafe { _mm_set_epi64x(0, -1) },
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(mask_128)
	}
}

impl<U: UnderlierType + TowerSimdType, Scalar: TowerField> Alpha
	for PackedPrimitiveType<U, Scalar>
{
	#[inline(always)]
	fn alpha() -> Self {
		U::alpha::<Scalar>().into()
	}
}

#[inline(always)]
fn blend_odd_even<U, PT>(a: PT, b: PT) -> PT
where
	U: TowerSimdType,
	PT: PackedField<Scalar: TowerField> + WithUnderlier<Underlier = U>,
{
	PT::from_underlier(U::blend_odd_even::<PT::Scalar>(a.to_underlier(), b.to_underlier()))
}

#[inline(always)]
fn xor<U, PT>(a: PT, b: PT) -> PT
where
	U: TowerSimdType,
	PT: WithUnderlier<Underlier = U>,
{
	PT::from_underlier(U::xor(a.to_underlier(), b.to_underlier()))
}

#[inline(always)]
fn duplicate_odd<U, PT>(val: PT) -> PT
where
	U: TowerSimdType,
	PT: PackedField<Scalar: TowerField> + WithUnderlier<Underlier = U>,
{
	PT::from_underlier(U::shuffle_epi8(val.to_underlier(), U::dup_shuffle::<PT::Scalar>()))
}

#[inline(always)]
fn flip_even_odd<U, PT>(val: PT) -> PT
where
	U: TowerSimdType,
	PT: PackedField<Scalar: TowerField> + WithUnderlier<Underlier = U>,
{
	PT::from_underlier(U::shuffle_epi8(val.to_underlier(), U::flip_shuffle::<PT::Scalar>()))
}

impl<U, Scalar: TowerField> TaggedMul<SimdStrategy> for PackedPrimitiveType<U, Scalar>
where
	Self: PackedTowerField<Underlier = U>,
	U: TowerSimdType + UnderlierType,
{
	fn mul(self, rhs: Self) -> Self {
		// This fallback is needed to generically use SimdStrategy in benchmarks.
		if Scalar::TOWER_LEVEL <= 3 {
			return self * rhs;
		}

		let a = self.as_packed_subfield();
		let b = rhs.as_packed_subfield();

		// [a0_lo * b0_lo, a0_hi * b0_hi, a1_lo * b1_lo, a1_h1 * b1_hi, ...]
		let z0_even_z2_odd = a * b;

		// [a0_lo, b0_lo, a1_lo, b1_lo, ...]
		// [a0_hi, b0_hi, a1_hi, b1_hi, ...]
		let (lo, hi) = a.interleave(b, 0);
		// [a0_lo + a0_hi, b0_lo + b0_hi, a1_lo + a1_hi, b1lo + b1_hi, ...]
		let lo_plus_hi_a_even_b_odd = lo + hi;

		let alpha_even_z2_odd = <Self as PackedTowerField>::PackedDirectSubfield::from_underlier(
			z0_even_z2_odd
				.to_underlier()
				.set_alpha_even::<<Self as PackedTowerField>::DirectSubfield>(),
		);
		let (lhs, rhs) = lo_plus_hi_a_even_b_odd.interleave(alpha_even_z2_odd, 0);
		let z1_xor_z0z2_even_z2a_odd = lhs * rhs;

		let z1_xor_z0z2 = duplicate_odd(z1_xor_z0z2_even_z2a_odd);
		let zero_even_z1_xor_z2a_xor_z0z2_odd = xor(z1_xor_z0z2_even_z2a_odd, z1_xor_z0z2);

		let z2_even_z0_odd = flip_even_odd(z0_even_z2_odd);
		let z0z2 = xor(z0_even_z2_odd, z2_even_z0_odd);

		Self::from_packed_subfield(xor(zero_even_z1_xor_z2a_xor_z0z2_odd, z0z2))
	}
}

impl<U, Scalar: TowerField> TaggedMulAlpha<SimdStrategy> for PackedPrimitiveType<U, Scalar>
where
	Self: PackedTowerField<Underlier = U> + MulAlpha,
	<Self as PackedTowerField>::PackedDirectSubfield: MulAlpha,
	U: TowerSimdType + UnderlierType,
{
	#[inline]
	fn mul_alpha(self) -> Self {
		// This fallback is needed to generically use SimdStrategy in benchmarks.
		if Scalar::TOWER_LEVEL <= 3 {
			return MulAlpha::mul_alpha(self);
		}

		let a_0_a_1 = self.as_packed_subfield();
		let a_0_mul_alpha_a_1_mul_alpha = a_0_a_1.mul_alpha();

		let a_1_a_0 = flip_even_odd(self.as_packed_subfield());
		let a0_plus_a1_alpha = xor(a_0_mul_alpha_a_1_mul_alpha, a_1_a_0);

		Self::from_packed_subfield(blend_odd_even(a0_plus_a1_alpha, a_1_a_0))
	}
}

impl<U, Scalar: TowerField> TaggedSquare<SimdStrategy> for PackedPrimitiveType<U, Scalar>
where
	Self: PackedTowerField<Underlier = U>,
	<Self as PackedTowerField>::PackedDirectSubfield: MulAlpha,
	U: TowerSimdType + UnderlierType,
{
	fn square(self) -> Self {
		// This fallback is needed to generically use SimdStrategy in benchmarks.
		if Scalar::TOWER_LEVEL <= 3 {
			return PackedField::square(self);
		}

		let a_0_a_1 = self.as_packed_subfield();
		let a_0_sq_a_1_sq = PackedField::square(a_0_a_1);
		let a_1_sq_a_0_sq = flip_even_odd(a_0_sq_a_1_sq);
		let a_0_sq_plus_a_1_sq = a_0_sq_a_1_sq + a_1_sq_a_0_sq;
		let a_1_mul_alpha = a_0_sq_a_1_sq.mul_alpha();

		Self::from_packed_subfield(blend_odd_even(a_1_mul_alpha, a_0_sq_plus_a_1_sq))
	}
}

impl<U, Scalar: TowerField> TaggedInvertOrZero<SimdStrategy> for PackedPrimitiveType<U, Scalar>
where
	Self: PackedTowerField<Underlier = U>,
	<Self as PackedTowerField>::PackedDirectSubfield: MulAlpha,
	U: TowerSimdType + UnderlierType,
{
	fn invert_or_zero(self) -> Self {
		// This fallback is needed to generically use SimdStrategy in benchmarks.
		if Scalar::TOWER_LEVEL <= 3 {
			return PackedField::invert_or_zero(self);
		}

		let a_0_a_1 = self.as_packed_subfield();
		let a_1_a_0 = flip_even_odd(a_0_a_1);
		let a_1_mul_alpha = a_1_a_0.mul_alpha();
		let a_0_plus_a1_mul_alpha = xor(a_0_a_1, a_1_mul_alpha);
		let a_1_sq_a_0_sq = PackedField::square(a_1_a_0);
		let delta = xor(a_1_sq_a_0_sq, a_0_plus_a1_mul_alpha * a_0_a_1);
		let delta_inv = PackedField::invert_or_zero(delta);
		let delta_inv_delta_inv = duplicate_odd(delta_inv);
		let delta_multiplier = blend_odd_even(a_0_a_1, a_0_plus_a1_mul_alpha);

		Self::from_packed_subfield(delta_inv_delta_inv * delta_multiplier)
	}
}

/// SIMD packed field transformation.
/// The idea is similar to `PackedTransformation` but we use SIMD instructions
/// to multiply a component with zeros/ones by a basis vector.
pub struct SimdTransformation<OP> {
	bases: Vec<OP>,
	ones: OP,
}

#[allow(private_bounds)]
impl<OP> SimdTransformation<OP>
where
	OP: PackedBinaryField + WithUnderlier<Underlier: TowerSimdType + UnderlierWithBitOps>,
{
	pub fn new<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self {
		Self {
			bases: transformation
				.bases()
				.iter()
				.map(|base| OP::broadcast(*base))
				.collect(),
			// Set ones to the highest bit
			// This is the format that is used in SIMD masks
			ones: OP::one().mutate_underlier(|underlier| underlier << (OP::Scalar::N_BITS - 1)),
		}
	}
}

impl<U, IP, OP, IF, OF> Transformation<IP, OP> for SimdTransformation<OP>
where
	IP: PackedField<Scalar = IF> + WithUnderlier<Underlier = U>,
	OP: PackedField<Scalar = OF> + WithUnderlier<Underlier = U>,
	IF: BinaryField,
	OF: BinaryField,
	U: UnderlierWithBitOps + TowerSimdType,
{
	fn transform(&self, input: &IP) -> OP {
		let mut result = OP::zero();
		let ones = self.ones.to_underlier();
		let mut input = input.to_underlier();

		// Unlike `PackedTransformation`, we iterate from the highest bit to lowest one
		// keeping current component in the highest bit.
		for base in self.bases.iter().rev() {
			let bases_mask = input & ones;
			let component = U::apply_mask::<OP::Scalar>(bases_mask, base.to_underlier());
			result += OP::from_underlier(component);
			input = input << 1;
		}

		result
	}
}

impl<IP, OP> TaggedPackedTransformationFactory<SimdStrategy, OP> for IP
where
	IP: PackedBinaryField + WithUnderlier<Underlier: UnderlierWithBitOps>,
	OP: PackedBinaryField + WithUnderlier<Underlier = IP::Underlier>,
	IP::Underlier: TowerSimdType,
{
	type PackedTransformation<Data: AsRef<[<OP>::Scalar]> + Sync> = SimdTransformation<OP>;

	fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data> {
		SimdTransformation::new(transformation)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::test_utils::{
		define_invert_tests, define_multiply_tests, define_square_tests,
		define_transformation_tests,
	};

	define_multiply_tests!(TaggedMul<SimdStrategy>::mul, TaggedMul<SimdStrategy>);

	define_square_tests!(TaggedSquare<SimdStrategy>::square, TaggedSquare<SimdStrategy>);

	define_invert_tests!(
		TaggedInvertOrZero<SimdStrategy>::invert_or_zero,
		TaggedInvertOrZero<SimdStrategy>
	);

	#[allow(unused)]
	trait SelfPackedTransformationFactory:
		TaggedPackedTransformationFactory<SimdStrategy, Self>
	{
	}

	impl<T: TaggedPackedTransformationFactory<SimdStrategy, Self>> SelfPackedTransformationFactory
		for T
	{
	}

	define_transformation_tests!(SelfPackedTransformationFactory);
}
