// Copyright 2024-2025 Irreducible Inc.

use core::arch::x86_64::*;
use std::array;

use gfni_arithmetics::{GfniType, get_8x8_matrix};

use super::*;
use crate::{
	BinaryField, PackedField,
	arch::{GfniSpecializedStrategy512b, x86_64::m512::M512},
	arithmetic_traits::TaggedPackedTransformationFactory,
	linear_transformation::{FieldLinearTransformation, Transformation},
	underlier::WithUnderlier,
};

impl GfniType for M512 {
	#[inline(always)]
	fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self {
		unsafe { _mm512_gf2p8affine_epi64_epi8::<0>(x.0, a.0) }.into()
	}

	#[inline(always)]
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self {
		unsafe { _mm512_gf2p8mul_epi8(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self {
		unsafe { _mm512_gf2p8affineinv_epi64_epi8::<0>(x.0, a.0) }.into()
	}
}

/// Specialized GFNI transformation for AVX512-packed 128-bit binary fields.
/// Get advantage of the fact that we can multiply 8 8x8 matrices by some elements at once.
/// Theoretically we can have similar implementation for different scalar sizes, but it is not
/// implemented yet.
pub struct GfniTransformation512b {
	// Each element contains 8 8x8 matrices
	bases_8x8: [[__m512i; 2]; 16],
	// Permute mask for each column
	permute_masks: [__m512i; 16],
}

impl GfniTransformation512b {
	pub fn new<OP, Data>(transformation: FieldLinearTransformation<OP::Scalar, Data>) -> Self
	where
		OP: PackedField<Scalar: BinaryField<Underlier = u128>> + WithUnderlier<Underlier = M512>,
		Data: AsRef<[OP::Scalar]> + Sync,
	{
		let bases_8x8 = array::from_fn(|col| {
			array::from_fn(|row| unsafe {
				_mm512_set_epi64(
					get_8x8_matrix(&transformation, row + 14, col),
					get_8x8_matrix(&transformation, row + 12, col),
					get_8x8_matrix(&transformation, row + 10, col),
					get_8x8_matrix(&transformation, row + 8, col),
					get_8x8_matrix(&transformation, row + 6, col),
					get_8x8_matrix(&transformation, row + 4, col),
					get_8x8_matrix(&transformation, row + 2, col),
					get_8x8_matrix(&transformation, row, col),
				)
			})
		});

		let permute_masks = array::from_fn(|col| {
			let permute_bytes: [u8; 8] = array::from_fn(|i| (col + (i / 2) * 16) as u8);
			let permute_u64 = u64::from_le_bytes(permute_bytes);

			unsafe { _mm512_set1_epi64(permute_u64 as i64) }
		});

		Self {
			bases_8x8,
			permute_masks,
		}
	}
}

impl<IP, OP> Transformation<IP, OP> for GfniTransformation512b
where
	IP: PackedField<Scalar: WithUnderlier<Underlier = u128>> + WithUnderlier<Underlier = M512>,
	OP: PackedField<Scalar: WithUnderlier<Underlier = u128>> + WithUnderlier<Underlier = M512>,
{
	fn transform(&self, data: &IP) -> OP {
		let mut result = unsafe { _mm512_setzero_si512() };
		let data = data.to_underlier();

		let odd_mask = unsafe { _mm512_set1_epi16(0x00ff) };
		let even_mask = unsafe { _mm512_set1_epi16(0xff00u16 as i16) };
		for col in 0..16 {
			// Permute 8b elements from [b_0, ..., b_64] to
			// [b_col, b_col, b_(col + 16), b_(col + 16), b_(col + 32), b_(col + 32), b_(col + 48),
			// b_(col + 48), ...] This is a cross-lane operation and usually takes more cycles so
			// we are doing it once for the pair instead of twice.
			let permuted_data = unsafe { _mm512_permutexvar_epi8(self.permute_masks[col], data.0) };

			// Multiply 8 8x8 matrices by odd 8b elements
			let data = unsafe { _mm512_and_si512(permuted_data, odd_mask) };
			let product =
				unsafe { _mm512_gf2p8affine_epi64_epi8::<0>(data, self.bases_8x8[col][0]) };
			result = unsafe { _mm512_xor_si512(result, product) };

			// Multiply 8 8x8 matrices by even 8b elements
			let data = unsafe { _mm512_and_si512(permuted_data, even_mask) };
			let product =
				unsafe { _mm512_gf2p8affine_epi64_epi8::<0>(data, self.bases_8x8[col][1]) };
			result = unsafe { _mm512_xor_si512(result, product) };
		}

		// Permute 8b elements to the correct order
		let permute_mask = unsafe {
			_mm512_set_epi16(
				31, 27, 23, 19, 15, 11, 7, 3, 30, 26, 22, 18, 14, 10, 6, 2, 29, 25, 21, 17, 13, 9,
				5, 1, 28, 24, 20, 16, 12, 8, 4, 0,
			)
		};
		result = unsafe { _mm512_permutexvar_epi16(permute_mask, result) };

		OP::from_underlier(M512(result))
	}
}

impl<IP, OP> TaggedPackedTransformationFactory<GfniSpecializedStrategy512b, OP> for IP
where
	IP: PackedField<Scalar: BinaryField + WithUnderlier<Underlier = u128>>
		+ WithUnderlier<Underlier = M512>,
	OP: PackedField<Scalar: BinaryField + WithUnderlier<Underlier = u128>>
		+ WithUnderlier<Underlier = M512>,
{
	type PackedTransformation<Data: AsRef<[<OP>::Scalar]> + Sync> = GfniTransformation512b;

	fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data> {
		GfniTransformation512b::new::<OP, Data>(transformation)
	}
}
