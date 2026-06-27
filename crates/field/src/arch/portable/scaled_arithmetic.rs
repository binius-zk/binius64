// Copyright 2025-2026 The Binius Developers

use std::array;

use bytemuck::{Pod, TransparentWrapper};

use super::packed::PackedPrimitiveType;
use crate::{
	BinaryField,
	arch::LaneWideProduct,
	arithmetic_traits::{InvertOrZero, Square, WideMul},
	underlier::{ScaledUnderlier, UnderlierType},
};

/// Wrapper for `ScaledUnderlier` multiplication that delegates to sub-underlier operations.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Scaled<T>(T);

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> std::ops::Mul
	for Scaled<PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>>
where
	PackedPrimitiveType<U, Scalar>: std::ops::Mul<Output = PackedPrimitiveType<U, Scalar>>,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		let (a, b) = (Self::peel(self), Self::peel(rhs));
		Self::wrap(PackedPrimitiveType::wrap(ScaledUnderlier(array::from_fn(|i| {
			let lhs_i = a.0.0[i];
			let rhs_i = b.0.0[i];
			PackedPrimitiveType::peel(
				PackedPrimitiveType::wrap(lhs_i) * PackedPrimitiveType::wrap(rhs_i),
			)
		}))))
	}
}

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> Square
	for Scaled<PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>>
where
	PackedPrimitiveType<U, Scalar>: Square,
{
	fn square(self) -> Self {
		let val = Self::peel(self);
		Self::wrap(PackedPrimitiveType::wrap(ScaledUnderlier(val.0.0.map(|sub_underlier| {
			PackedPrimitiveType::peel(Square::square(PackedPrimitiveType::wrap(sub_underlier)))
		}))))
	}
}

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> InvertOrZero
	for Scaled<PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>>
where
	PackedPrimitiveType<U, Scalar>: InvertOrZero,
{
	fn invert_or_zero(self) -> Self {
		let val = Self::peel(self);
		Self::wrap(PackedPrimitiveType::wrap(ScaledUnderlier(val.0.0.map(|sub_underlier| {
			PackedPrimitiveType::peel(InvertOrZero::invert_or_zero(PackedPrimitiveType::wrap(
				sub_underlier,
			)))
		}))))
	}
}

/// Widening multiply for a `ScaledUnderlier` packing: apply the sub-underlier packing's [`WideMul`]
/// to each of the `N` lanes independently, deferring reduction per lane via [`LaneWideProduct`].
/// The `Scaled` analogue of [`Divide`](crate::arch::Divide)'s `WideMul`, but addressing the inner
/// sub-underliers of `ScaledUnderlier` directly instead of splitting an underlier with `Divisible`.
impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> WideMul
	for Scaled<PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>>
where
	PackedPrimitiveType<U, Scalar>: WideMul,
	<PackedPrimitiveType<U, Scalar> as WideMul>::Output: Copy + Default,
{
	type Output = LaneWideProduct<<PackedPrimitiveType<U, Scalar> as WideMul>::Output, N>;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		let (a, b) = (Self::peel(a), Self::peel(b));
		LaneWideProduct(array::from_fn(|i| {
			<PackedPrimitiveType<U, Scalar> as WideMul>::wide_mul(
				PackedPrimitiveType::wrap(a.0.0[i]),
				PackedPrimitiveType::wrap(b.0.0[i]),
			)
		}))
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		Self::wrap(PackedPrimitiveType::wrap(ScaledUnderlier(array::from_fn(|i| {
			PackedPrimitiveType::peel(<PackedPrimitiveType<U, Scalar> as WideMul>::reduce(
				wide.0[i],
			))
		}))))
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::{aes_field::AESTowerField8b, arch::M128};

	// A two-lane `ScaledUnderlier` AES packing whose `M128` lanes carry their own `WideMul`.
	type Inner = PackedPrimitiveType<ScaledUnderlier<M128, 2>, AESTowerField8b>;
	type P = Scaled<Inner>;

	fn packing(lo: u128, hi: u128) -> P {
		P::wrap(Inner::from_underlier(ScaledUnderlier([M128::from_u128(lo), M128::from_u128(hi)])))
	}

	proptest! {
		// `reduce(wide_mul(a, b))` must agree with the `Scaled` multiply.
		#[test]
		fn wide_mul_reduce_matches_mul(
			a_lo in any::<u128>(), a_hi in any::<u128>(),
			b_lo in any::<u128>(), b_hi in any::<u128>(),
		) {
			let (a, b) = (packing(a_lo, a_hi), packing(b_lo, b_hi));
			let via_wide = P::peel(P::reduce(P::wide_mul(a, b)));
			let via_mul = P::peel(packing(a_lo, a_hi) * packing(b_lo, b_hi));
			prop_assert_eq!(via_wide, via_mul);
		}

		// Deferred per-lane accumulation: summing two wide products then reducing once must equal
		// the sum of the two reduced products.
		#[test]
		fn wide_mul_accumulates(
			a_lo in any::<u128>(), a_hi in any::<u128>(),
			b_lo in any::<u128>(), b_hi in any::<u128>(),
			c_lo in any::<u128>(), c_hi in any::<u128>(),
			d_lo in any::<u128>(), d_hi in any::<u128>(),
		) {
			let acc = P::wide_mul(packing(a_lo, a_hi), packing(b_lo, b_hi))
				+ P::wide_mul(packing(c_lo, c_hi), packing(d_lo, d_hi));
			let via_wide = P::peel(P::reduce(acc));
			let via_mul = P::peel(packing(a_lo, a_hi) * packing(b_lo, b_hi))
				+ P::peel(packing(c_lo, c_hi) * packing(d_lo, d_hi));
			prop_assert_eq!(via_wide, via_mul);
		}
	}
}
