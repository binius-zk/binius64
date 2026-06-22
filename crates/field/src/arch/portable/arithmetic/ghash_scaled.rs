// Copyright 2026 The Binius Developers

//! Scaled deferred-reduction GHASH widening multiply for multi-lane packings.
//!
//! A 256- or 512-bit GHASH packing is a vector of independent 128-bit GHASH elements. When no
//! wide-vector carryless multiply is available (e.g. no VPCLMULQDQ, or the portable build), the
//! widening multiply is built by applying the width-1 GHASH [`WideMul`] — itself CLMUL-accelerated
//! when possible, portable schoolbook otherwise — to each 128-bit lane. This keeps every GHASH
//! packing on a *deferring* `WideMul` (never the eager `TrivialWideMul`), so a sum of products
//! reduces once per lane at the end.
//!
//! [`ScaledGhashWideMul`] is the wrapper passed as the `wide_mul` strategy of the packing macro. A
//! single generic [`WideMul`] impl covers every multi-lane packing — both the portable
//! `ScaledUnderlier` packings and the x86_64 SIMD `M256`/`M512` ones — by splitting the underlier
//! into its 128-bit lanes via [`Divisible`]. The const parameter `N` is the lane count, which must
//! equal `<U as Divisible<M128>>::N` for the underlying packed type.

use std::{
	array,
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use crate::{
	BinaryField,
	arch::{M128, PackedPrimitiveType},
	arithmetic_traits::WideMul,
	underlier::{Divisible, UnderlierType},
};

/// The unreduced product of a multi-lane GHASH packing: one independent width-1 widening product
/// per 128-bit lane. Lanes accumulate and reduce independently, mirroring the packing's structure.
/// The const parameter `N` is the lane count.
#[derive(Clone, Copy, Debug)]
pub struct ScaledWideGhashProduct<O, const N: usize>([O; N]);

impl<O: Copy + Default, const N: usize> Default for ScaledWideGhashProduct<O, N> {
	#[inline]
	fn default() -> Self {
		Self([O::default(); N])
	}
}

impl<O: Copy + Add<Output = O>, const N: usize> Add for ScaledWideGhashProduct<O, N> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self(array::from_fn(|i| self.0[i] + rhs.0[i]))
	}
}

impl<O: Copy + Add<Output = O>, const N: usize> AddAssign for ScaledWideGhashProduct<O, N> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<O: Copy + Sub<Output = O>, const N: usize> Sub for ScaledWideGhashProduct<O, N> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self(array::from_fn(|i| self.0[i] - rhs.0[i]))
	}
}

impl<O: Copy + Sub<Output = O>, const N: usize> SubAssign for ScaledWideGhashProduct<O, N> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<O: Copy + Default + Add<Output = O>, const N: usize> Sum for ScaledWideGhashProduct<O, N> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

/// Widening-multiply wrapper for a multi-lane GHASH packing, applying the width-1 GHASH `WideMul`
/// to each 128-bit lane of the packing's underlier. The const parameter `N` is the number of
/// 128-bit lanes and must equal `<U as Divisible<M128>>::N` for the packed type's underlier.
///
/// Use the lane-count aliases [`Scaled2xGhashWideMul`] and [`Scaled4xGhashWideMul`] to avoid
/// repeating the const argument at each call site.
#[derive(TransparentWrapper)]
#[repr(transparent)]
pub struct ScaledGhashWideMul<T, const N: usize>(pub T);

/// The width-1 GHASH packing for the active arch's 128-bit underlier — the per-lane field whose
/// `WideMul` (CLMUL-accelerated or portable schoolbook) drives each lane.
type Lane<Scalar> = PackedPrimitiveType<M128, Scalar>;

impl<U, Scalar, const N: usize> WideMul for ScaledGhashWideMul<PackedPrimitiveType<U, Scalar>, N>
where
	U: UnderlierType + Divisible<M128>,
	Scalar: BinaryField,
	Lane<Scalar>: WideMul,
	<Lane<Scalar> as WideMul>::Output: Copy + Default,
{
	type Output = ScaledWideGhashProduct<<Lane<Scalar> as WideMul>::Output, N>;

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
		ScaledWideGhashProduct(out)
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

/// Convenience alias for the 2-lane (256-bit) scaled GHASH widening multiply.
pub type Scaled2xGhashWideMul<T> = ScaledGhashWideMul<T, 2>;

/// Convenience alias for the 4-lane (512-bit) scaled GHASH widening multiply.
pub type Scaled4xGhashWideMul<T> = ScaledGhashWideMul<T, 4>;
