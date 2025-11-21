// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_utils::checked_arithmetics::checked_int_div;

use crate::{
	BinaryField, PackedField, TowerField,
	arch::{
		GfniStrategy,
		portable::packed::PackedPrimitiveType,
		x86_64::{m128::m128_from_u128, simd::simd_arithmetic::TowerSimdType},
	},
	arithmetic_traits::{TaggedInvertOrZero, TaggedMul, TaggedPackedTransformationFactory},
	is_aes_tower,
	linear_transformation::{FieldLinearTransformation, Transformation},
	packed::PackedBinaryField,
	underlier::{Divisible, UnderlierType, WithUnderlier},
};

#[rustfmt::skip]
pub const IDENTITY_MAP: i64 = u64::from_le_bytes([
	0b10000000,
	0b01000000,
	0b00100000,
	0b00010000,
	0b00001000,
	0b00000100,
	0b00000010,
	0b00000001,
]) as i64;

pub(super) trait GfniType: Copy + TowerSimdType {
	fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self;
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self;
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self;
}

impl<U: GfniType + UnderlierType, Scalar: BinaryField> TaggedMul<GfniStrategy>
	for PackedPrimitiveType<U, Scalar>
{
	#[inline(always)]
	fn mul(self, rhs: Self) -> Self {
		U::gf2p8mul_epi8(self.0, rhs.0).into()
	}
}

impl<U: GfniType + UnderlierType, Scalar: TowerField> TaggedInvertOrZero<GfniStrategy>
	for PackedPrimitiveType<U, Scalar>
{
	#[inline(always)]
	fn invert_or_zero(self) -> Self {
		assert!(is_aes_tower::<Scalar>());
		assert!(Scalar::N_BITS == 8);

		let val_gfni = self.to_underlier();

		// Calculate inversion and linear transformation to the original field with a single
		// instruction
		let transform_after = U::set_epi_64(IDENTITY_MAP);
		let inv_gfni = U::gf2p8affineinv_epi64_epi8(val_gfni, transform_after);

		inv_gfni.into()
	}
}

/// Transformation that uses `gf2p8affine_epi64_epi8` transformation to apply linear transformation
/// to a 8-bit packed field. It appeared that this dedicated implementation is more efficient than
/// `GfniTransformationNxN<_, 1>`.
#[allow(private_bounds)]
pub struct GfniTransformation<OP>
where
	OP: WithUnderlier<Underlier: GfniType>,
{
	/// Value is filled with 64-bit linear transformation matrices
	bases_8x8: OP::Underlier,
}

/// Transpose i64 representing a 8x8 boolean matrix.
/// There may be a faster implementation for this but
/// it is used only during packed transformation creation, not at transformation itself.
fn transpose_8x8(mut matrix: i64) -> i64 {
	let mut result = 0;

	for i in 0..8 {
		for j in 0..8 {
			result |= (matrix & 1) << ((7 - j) * 8 + i);
			matrix >>= 1;
		}
	}

	result
}

/// Get 8x8 matrix block from the linear transformation.
/// `row` and `col` are indices of the 8x8 block in the transformation matrix.
pub(super) fn get_8x8_matrix<OF, Data>(
	transformation: &FieldLinearTransformation<OF, Data>,
	row: usize,
	col: usize,
) -> i64
where
	OF: BinaryField<Underlier: Divisible<u8>>,
	Data: AsRef<[OF]> + Sync,
{
	transpose_8x8(i64::from_le_bytes(array::from_fn(|k| {
		transformation.bases()[k + 8 * col]
			.to_underlier()
			.split_ref()[row]
	})))
}

#[allow(private_bounds)]
impl<OP> GfniTransformation<OP>
where
	OP: WithUnderlier<Underlier: GfniType>
		+ PackedBinaryField<Scalar: WithUnderlier<Underlier = u8>>,
{
	pub fn new<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self {
		debug_assert_eq!(OP::Scalar::N_BITS, 8);
		debug_assert_eq!(transformation.bases().len(), 8);
		let bases_8x8 =
			i64::from_le_bytes(array::from_fn(|i| transformation.bases()[i].to_underlier()));

		Self {
			bases_8x8: OP::Underlier::set_epi_64(transpose_8x8(bases_8x8)),
		}
	}
}

impl<IP, OP, U> Transformation<IP, OP> for GfniTransformation<OP>
where
	IP: PackedField<Scalar: WithUnderlier<Underlier = u8>> + WithUnderlier<Underlier = U>,
	OP: PackedField<Scalar: WithUnderlier<Underlier = u8>> + WithUnderlier<Underlier = U>,
	U: GfniType,
{
	fn transform(&self, data: &IP) -> OP {
		OP::from_underlier(U::gf2p8affine_epi64_epi8(data.to_underlier(), self.bases_8x8))
	}
}

impl<IP, OP, U> TaggedPackedTransformationFactory<GfniStrategy, OP> for IP
where
	IP: PackedField<Scalar: BinaryField + WithUnderlier<Underlier = u8>>
		+ WithUnderlier<Underlier = U>,
	OP: PackedField<Scalar: BinaryField + WithUnderlier<Underlier = u8>>
		+ WithUnderlier<Underlier = U>,
	U: GfniType,
{
	type PackedTransformation<Data: AsRef<[<OP>::Scalar]> + Sync> = GfniTransformation<OP>;

	fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data> {
		GfniTransformation::new(transformation)
	}
}

/// Linear transformation for packed scalars of size `BLOCKS*8`.
/// Splits elements itself and transformation matrix to 8-bit size blocks and uses
/// `gf2p8affine_epi64_epi8` to perform multiplications of those.
/// Transformation complexity is `BLOCKS^2/2` since two 8x8 matrices multiplications are done with a
/// single instruction.
///
/// All operations scale by lane count, so this transformation works for any register size.
/// `MATRICES` is actually a half of `BLOCKS`, since two 8x8 matrices are processed at once.
#[allow(private_bounds)]
pub struct GfniTransformationNxN<OP, const BLOCKS: usize, const MATRICES: usize>
where
	OP: WithUnderlier<Underlier: GfniType>,
{
	// Each element contains two matrices in a single 128-bit lane
	bases_8x8: [[OP::Underlier; MATRICES]; BLOCKS],
	// Each element contains a shuffle mask to to put 8-bit blocks in the right order.
	shuffles: [[OP::Underlier; MATRICES]; BLOCKS],
	// Final shuffle to put all 8-bit blocks in the right order in the result.
	final_shuffle: OP::Underlier,
}

#[allow(private_bounds)]
impl<OP, const BLOCKS: usize, const MATRICES: usize> GfniTransformationNxN<OP, BLOCKS, MATRICES>
where
	OP: WithUnderlier<Underlier: GfniType>
		+ PackedBinaryField<Scalar: WithUnderlier<Underlier: Divisible<u8>>>,
	[[OP::Underlier; MATRICES]; BLOCKS]: Default,
{
	pub fn new<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self {
		debug_assert_eq!(OP::Scalar::N_BITS, BLOCKS * 8);
		debug_assert_eq!(transformation.bases().len(), BLOCKS * 8);

		// Convert bases matrix into `BLOCKS`x`BLOCKS` matrix of 8x8 blocks.
		let bases_8x8 = array::from_fn(|col| {
			array::from_fn(|row| {
				let odd_matrix = get_8x8_matrix(&transformation, row, col) as u64 as u128;
				let even_matrix =
					get_8x8_matrix(&transformation, row + MATRICES, col) as u64 as u128;
				let matrices_u128 = (even_matrix << 64) | odd_matrix;
				let matrices_m128 = m128_from_u128!(matrices_u128);

				OP::Underlier::set1_epi128(matrices_m128)
			})
		});

		// precompute shuffles
		let shuffles = array::from_fn(|col| {
			array::from_fn(|row| {
				let mut half_u128_lane = [255u8; 8];
				for i in 0..checked_int_div(8, MATRICES) {
					half_u128_lane[row + i * MATRICES] = (col + i * BLOCKS) as u8;
				}

				let byte_indices = array::from_fn(|i| {
					// all shuffle indices are repeated with cycle 8.
					half_u128_lane[i % 8]
				});
				let mask_u128 = u128::from_le_bytes(byte_indices);
				let mask_m128 = m128_from_u128!(mask_u128);

				OP::Underlier::set1_epi128(mask_m128)
			})
		});
		let final_shuffle = {
			let mut shuffle = [0u8; 16];
			for i in 0..checked_int_div(16, BLOCKS) {
				for k in 0..MATRICES {
					shuffle[i * BLOCKS + k] = (i * MATRICES + k) as _;
					shuffle[i * BLOCKS + k + MATRICES] = (i * MATRICES + k + 8) as _;
				}
			}
			let mask_u128 = u128::from_le_bytes(shuffle);
			let mask_m128 = m128_from_u128!(mask_u128);

			OP::Underlier::set1_epi128(mask_m128)
		};

		Self {
			bases_8x8,
			shuffles,
			final_shuffle,
		}
	}
}

impl<U, IP, OP, const BLOCKS: usize, const MATRICES: usize> Transformation<IP, OP>
	for GfniTransformationNxN<OP, BLOCKS, MATRICES>
where
	IP: PackedField + WithUnderlier<Underlier = U>,
	OP: PackedField + WithUnderlier<Underlier = U>,
	U: GfniType + std::fmt::Debug,
{
	fn transform(&self, data: &IP) -> OP {
		let mut result = OP::Underlier::default();

		for col in 0..BLOCKS {
			for row in 0..MATRICES {
				// shuffle [b_0,... b_15] to have [0, .., b_col, 0, .. 0, ,b_col, 0, .. 0]
				//                                         /\                /\
				//                                         row            row + 8
				let shuffled = U::shuffle_epi8(data.to_underlier(), self.shuffles[col][row]);

				// Multiply `A_col_row` by `b_col` and `A_col_(row+MATRICES)` by `b_col`
				result =
					U::xor(result, U::gf2p8affine_epi64_epi8(shuffled, self.bases_8x8[col][row]));
			}
		}

		// put the 8-bit blocks to the right order
		result = U::shuffle_epi8(result, self.final_shuffle);

		OP::from_underlier(result)
	}
}

/// Implement packed transformation factory with GFNI instructions for scalars bigger than 8 bits
macro_rules! impl_transformation_with_gfni_nxn {
	($name:ty, $blocks:literal) => {
		impl<OP> $crate::linear_transformation::PackedTransformationFactory<OP> for $name
		where
			OP: $crate::packed::PackedBinaryField<
					Scalar: $crate::underlier::WithUnderlier<
						Underlier: $crate::underlier::Divisible<u8>,
					>,
				> + $crate::underlier::WithUnderlier<
					Underlier = <$name as $crate::underlier::WithUnderlier>::Underlier,
				>,
		{
			type PackedTransformation<
				Data: AsRef<[<OP as $crate::packed::PackedField>::Scalar]> + Sync,
			> = $crate::arch::x86_64::gfni::gfni_arithmetics::GfniTransformationNxN<
				OP,
				$blocks,
				{ $blocks / 2 },
			>;

			fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
				transformation: $crate::linear_transformation::FieldLinearTransformation<
					OP::Scalar,
					Data,
				>,
			) -> Self::PackedTransformation<Data> {
				$crate::arch::x86_64::gfni::gfni_arithmetics::GfniTransformationNxN::<
					OP,
					$blocks,
					{ $blocks / 2 },
				>::new(transformation)
			}
		}
	};
}

pub(crate) use impl_transformation_with_gfni_nxn;
