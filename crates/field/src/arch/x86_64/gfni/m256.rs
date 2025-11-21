// Copyright 2024-2025 Irreducible Inc.

use core::arch::x86_64::*;
use std::array;

use gfni_arithmetics::{GfniType, get_8x8_matrix};
use seq_macro::seq;

use super::*;
use crate::{
	BinaryField, PackedField,
	arch::{GfniSpecializedStrategy256b, x86_64::m256::M256},
	arithmetic_traits::TaggedPackedTransformationFactory,
	linear_transformation::{FieldLinearTransformation, Transformation},
	underlier::WithUnderlier,
};

impl GfniType for M256 {
	#[inline(always)]
	fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self {
		unsafe { _mm256_gf2p8affine_epi64_epi8::<0>(x.0, a.0) }.into()
	}

	#[inline(always)]
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self {
		unsafe { _mm256_gf2p8mul_epi8(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self {
		unsafe { _mm256_gf2p8affineinv_epi64_epi8::<0>(x.0, a.0) }.into()
	}
}

/// Specialized GFNI transformation for AVX2-packed 128-bit binary fields.
/// Get advantage of the fact that we can multiply 4 8x8 matrices by some elements at once.
/// Theoretically we can have similar implementation for different scalar sizes, but it is not
/// implemented yet.
pub struct GfniTransformation256b {
	// Each element contains 4 8x8 matrices
	bases_8x8: [[__m256i; 4]; 16],
	// Shuffle mask from the permuted values to tra right position before matrix multiplication
	shuffles: [[__m256i; 4]; 16],
}

impl GfniTransformation256b {
	pub fn new<OP, Data>(transformation: FieldLinearTransformation<OP::Scalar, Data>) -> Self
	where
		OP: PackedField<Scalar: BinaryField<Underlier = u128>> + WithUnderlier<Underlier = M256>,
		Data: AsRef<[OP::Scalar]> + Sync,
	{
		let bases_8x8 = array::from_fn(|col| {
			array::from_fn(|row| unsafe {
				_mm256_set_epi64x(
					get_8x8_matrix(&transformation, row + 12, col),
					get_8x8_matrix(&transformation, row + 8, col),
					get_8x8_matrix(&transformation, row + 4, col),
					get_8x8_matrix(&transformation, row, col),
				)
			})
		});

		let shuffles = array::from_fn(|col| {
			array::from_fn(|row| {
				let mut shuffle_bytes = [255u8; 8];
				shuffle_bytes[row] = (col % 8) as u8;
				shuffle_bytes[row + 4] = (col % 8) as u8 + 8;

				let shuffle_u64 = u64::from_le_bytes(shuffle_bytes);

				unsafe { _mm256_set1_epi64x(shuffle_u64 as i64) }
			})
		});

		Self {
			bases_8x8,
			shuffles,
		}
	}
}

impl<IP, OP> Transformation<IP, OP> for GfniTransformation256b
where
	IP: PackedField<Scalar: WithUnderlier<Underlier = u128>> + WithUnderlier<Underlier = M256>,
	OP: PackedField<Scalar: WithUnderlier<Underlier = u128>> + WithUnderlier<Underlier = M256>,
{
	// This is a consequence of using `seq!` macro
	#[allow(clippy::erasing_op)]
	#[allow(clippy::identity_op)]
	fn transform(&self, data: &IP) -> OP {
		let mut result = unsafe { _mm256_setzero_si256() };
		let data = data.to_underlier();

		// We use `seq` macro to generate the loop because it produces more efficient code
		seq!(
			i in 0usize..2 {
				// Permute values [b_0, .. ,b_32 ] to
				// [b_0, b_1, .. b_7, b_16, ... , b_23, b_0, ... , b_7, b_16, ... , b_23] for the first octet
				// [b_8, .. b_15, b_24, ... , b_31, b_8, ... , b_15, b_24, ... , b_31] for the second octet
				let offset = (2* i) as i32;
				let permute_mask = unsafe { _mm256_set_epi32(5 + offset, 4 + offset, 1 + offset, offset, 5 + offset, 4 + offset, 1 + offset, offset) };
				let permuted_data = unsafe { _mm256_permutexvar_epi32(permute_mask, data.0) };

				for col in 8*i..8*(i + 1) {
					for row in 0..4 {
						let base = self.bases_8x8[col][row];
						let shuffle = self.shuffles[col][row];

						// Shuffle values to have
						// [0, ... 0, b_col, 0, ... 0, b_(col+16), 0, ... 0, b_col, 0, ... 0, b_(col+16), 0...]
						//             /\                  /\                  /\                  /\
						//            row                 row+4              row + 8              row + 12
						let data = unsafe { _mm256_shuffle_epi8(permuted_data, shuffle) };
						// Multiply by (A_col_row, A_(col_row+4), A_col_(row+12), A_col_(row+16))
						let product = unsafe { _mm256_gf2p8affine_epi64_epi8::<0>(data, base) };

						result = unsafe { _mm256_xor_si256(result, product) };
					}
				}
			}
		);

		// Reorder element to the right order
		let permute_mask = unsafe { _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0) };
		result = unsafe { _mm256_permutexvar_epi32(permute_mask, result) };

		OP::from_underlier(M256(result))
	}
}

impl<IP, OP> TaggedPackedTransformationFactory<GfniSpecializedStrategy256b, OP> for IP
where
	IP: PackedField<Scalar: BinaryField + WithUnderlier<Underlier = u128>>
		+ WithUnderlier<Underlier = M256>,
	OP: PackedField<Scalar: BinaryField + WithUnderlier<Underlier = u128>>
		+ WithUnderlier<Underlier = M256>,
{
	type PackedTransformation<Data: AsRef<[<OP>::Scalar]> + Sync> = GfniTransformation256b;

	fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data> {
		GfniTransformation256b::new::<OP, Data>(transformation)
	}
}
