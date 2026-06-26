// Copyright 2025-2026 The Binius Developers

use std::{
	array,
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::{Pod, TransparentWrapper};

use super::packed::PackedPrimitiveType;
use crate::{
	BinaryField,
	arch::M128,
	arithmetic_traits::{InvertOrZero, Square, WideMul},
	underlier::{Divisible, ScaledUnderlier, UnderlierType},
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

/// The unreduced product of a multi-lane packing: one independent width-1 widening product per
/// 128-bit lane. Lanes accumulate and reduce independently, mirroring the packing's structure. The
/// const parameter `N` is the lane count.
#[derive(Clone, Copy, Debug)]
pub struct ScaledWideProduct<O, const N: usize>([O; N]);

impl<O: Copy + Default, const N: usize> Default for ScaledWideProduct<O, N> {
	#[inline]
	fn default() -> Self {
		Self([O::default(); N])
	}
}

impl<O: Copy + Add<Output = O>, const N: usize> Add for ScaledWideProduct<O, N> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self(array::from_fn(|i| self.0[i] + rhs.0[i]))
	}
}

impl<O: Copy + Add<Output = O>, const N: usize> AddAssign for ScaledWideProduct<O, N> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<O: Copy + Sub<Output = O>, const N: usize> Sub for ScaledWideProduct<O, N> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self(array::from_fn(|i| self.0[i] - rhs.0[i]))
	}
}

impl<O: Copy + Sub<Output = O>, const N: usize> SubAssign for ScaledWideProduct<O, N> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<O: Copy + Default + Add<Output = O>, const N: usize> Sum for ScaledWideProduct<O, N> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

/// Widening-multiply wrapper for a multi-lane packing built from 128-bit lanes: it splits the
/// underlier into its `M128` lanes via [`Divisible`] and applies the width-1 packing's [`WideMul`]
/// to each. A single impl covers every multi-lane packing whose underlier is `Divisible<M128>` —
/// both the portable `ScaledUnderlier` packings and the x86_64 SIMD `M256`/`M512` ones — for any
/// field: GHASH lanes defer a 256-bit product and reduce at the end, while AES lanes carry the
/// already-reduced byte (`reduce` is then the identity). The const parameter `N` is the lane count
/// and must equal `<U as Divisible<M128>>::N` for the packed type's underlier.
///
/// Use the lane-count aliases [`Scaled2xWideMul`] and [`Scaled4xWideMul`] to avoid repeating the
/// const argument at each call site.
#[derive(TransparentWrapper)]
#[repr(transparent)]
pub struct ScaledWideMul<T, const N: usize>(pub T);

/// The width-1 packing for the active arch's 128-bit underlier — the per-lane field whose `WideMul`
/// drives each lane.
type Lane<Scalar> = PackedPrimitiveType<M128, Scalar>;

impl<U, Scalar, const N: usize> WideMul for ScaledWideMul<PackedPrimitiveType<U, Scalar>, N>
where
	U: UnderlierType + Divisible<M128>,
	Scalar: BinaryField,
	Lane<Scalar>: WideMul,
	<Lane<Scalar> as WideMul>::Output: Copy + Default,
{
	type Output = ScaledWideProduct<<Lane<Scalar> as WideMul>::Output, N>;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		debug_assert_eq!(N, <U as Divisible<M128>>::N, "N must equal Divisible<M128>::N");

		let a = Self::peel(a).to_underlier();
		let b = Self::peel(b).to_underlier();

		let mut out = [<Lane<Scalar> as WideMul>::Output::default(); N];
		for (slot, (lhs, rhs)) in out
			.iter_mut()
			.zip(<U as Divisible<M128>>::value_iter(a).zip(<U as Divisible<M128>>::value_iter(b)))
		{
			*slot = <Lane<Scalar>>::wide_mul(
				Lane::<Scalar>::from_underlier(lhs),
				Lane::<Scalar>::from_underlier(rhs),
			);
		}
		ScaledWideProduct(out)
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		let lanes = wide
			.0
			.into_iter()
			.map(|product| <Lane<Scalar> as WideMul>::reduce(product).to_underlier());
		Self::wrap(PackedPrimitiveType::from_underlier(<U as Divisible<M128>>::from_iter(lanes)))
	}
}

/// Convenience alias for the 2-lane (256-bit) scaled widening multiply.
pub type Scaled2xWideMul<T> = ScaledWideMul<T, 2>;

/// Convenience alias for the 4-lane (512-bit) scaled widening multiply.
pub type Scaled4xWideMul<T> = ScaledWideMul<T, 4>;
