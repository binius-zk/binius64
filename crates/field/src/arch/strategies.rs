// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

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

/// Strategy that defines multiplication as `reduce(wide_mul(a, b))`, deferring to the type's own
/// [`WideMul`] impl. Used by every GHASH packing so the (CLMUL-accelerated or scaled) widening
/// multiply is the single source of truth for both `Mul` and `WideMul`.
pub struct GhashMulStrategy;

impl<P: WideMul> TaggedMul<GhashMulStrategy> for P {
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		P::reduce(P::wide_mul(self, rhs))
	}
}
