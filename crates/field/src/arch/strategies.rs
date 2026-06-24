// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use bytemuck::TransparentWrapper;

use crate::{
	BinaryField,
	arch::PackedPrimitiveType,
	arithmetic_traits::{Square, TaggedMul, TaggedSquare, WideMul},
	underlier::{Divisible, UnderlierType},
};

/// Packed strategy for arithmetic operations.
/// (Uses arithmetic operations with underlier and subfield to simultaneously calculate the result
/// for all packed values)
pub struct PackedStrategy;
/// Pairwise strategy. Apply the result of the operation to each packed element independently.
pub struct PairwiseStrategy;
/// Get result of operation from the table for each sub-element
pub struct PairwiseTableStrategy;
/// Applicable only for multiply by alpha and square operations.
/// Reuse multiplication operation for that.
pub struct ReuseMultiplyStrategy;

/// Use operations with GFNI instructions
pub struct GfniStrategy;
/// Use SIMD operations for packed arithmetic
pub struct SimdStrategy;
/// Specialized versions of the above to resolve conflicting implementations
pub struct GfniSpecializedStrategy256b;
pub struct GfniSpecializedStrategy512b;

/// Strategy for ScaledUnderlier operations that delegate to sub-underlier operations.
pub struct ScaledStrategy;

/// Strategy that squares by dividing the underlier into `SubU`-sized lanes, squaring each lane as a
/// `PackedPrimitiveType<SubU, F>`, and recombining. A generic fallback for packings that lack a
/// specialized full-width square.
pub struct DivideStrategy<SubU>(PhantomData<SubU>);

impl<U, SubU, F> TaggedSquare<DivideStrategy<SubU>> for PackedPrimitiveType<U, F>
where
	U: UnderlierType + Divisible<SubU>,
	SubU: UnderlierType,
	F: BinaryField,
	PackedPrimitiveType<SubU, F>: Square,
{
	#[inline]
	fn square(self) -> Self {
		let squared = Divisible::<SubU>::value_iter(self.to_underlier()).map(|lane| {
			PackedPrimitiveType::<SubU, F>::from_underlier(lane)
				.square()
				.to_underlier()
		});
		Self::from_underlier(Divisible::<SubU>::from_iter(squared))
	}
}

/// Widening-multiply wrapper that defers to per-`u8`-lane multiplication: it splits each underlier
/// into `u8` lanes, applies the 1-byte packing's [`WideMul`], and recombines. This is the `WideMul`
/// analogue of [`DivideStrategy`] — a generic fallback for packings whose underlier is
/// `Divisible<u8>`. It requires the 1-byte packing's wide product to already be the reduced element
/// (`Output = Self`), so `reduce` is the identity.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct ElementwiseWideMul<T>(T);

impl<U, F> WideMul for ElementwiseWideMul<PackedPrimitiveType<U, F>>
where
	U: UnderlierType + Divisible<u8>,
	F: BinaryField,
	PackedPrimitiveType<u8, F>: WideMul<Output = PackedPrimitiveType<u8, F>>,
{
	type Output = PackedPrimitiveType<U, F>;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		let a = Self::peel(a).to_underlier();
		let b = Self::peel(b).to_underlier();
		let product = Divisible::<u8>::value_iter(a)
			.zip(Divisible::<u8>::value_iter(b))
			.map(|(lhs, rhs)| {
				<PackedPrimitiveType<u8, F> as WideMul>::wide_mul(
					PackedPrimitiveType::from_underlier(lhs),
					PackedPrimitiveType::from_underlier(rhs),
				)
				.to_underlier()
			});
		PackedPrimitiveType::from_underlier(Divisible::<u8>::from_iter(product))
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		Self::wrap(wide)
	}
}

/// Strategy that defines multiplication as `reduce(wide_mul(a, b))`, deferring to the type's own
/// [`WideMul`] impl, making the widening multiply the single source of truth for both `Mul` and
/// `WideMul`. Used by every GHASH and AES packing.
pub struct MulFromWideMul;

impl<P: WideMul> TaggedMul<MulFromWideMul> for P {
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		P::reduce(P::wide_mul(self, rhs))
	}
}
