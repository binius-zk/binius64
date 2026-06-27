// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	array,
	iter::Sum,
	marker::PhantomData,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use crate::{
	BinaryField,
	arch::PackedPrimitiveType,
	arithmetic_traits::{InvertOrZero, Square, WideMul},
	underlier::{Divisible, UnderlierType},
};

/// Pairwise strategy. Apply the result of the operation to each packed element independently.
pub struct PairwiseStrategy;

/// Strategy that splits the underlier into `SubU`-sized lanes, applies the sub-packing
/// `PackedPrimitiveType<SubU, F>`'s op to each lane, and recombines — a generic fallback for
/// packings that lack a specialized full-width [`Square`], [`InvertOrZero`], or [`WideMul`]. The
/// sub-underlier `SubU` is a `PhantomData` parameter so the packing type `T` stays last for the
/// macro's `Divide<SubU, $name, N>` form.
///
/// `N` is the lane count: callers always pass `N = <U as Divisible<SubU>>::N` (or the literal it
/// works out to). `Square`/`InvertOrZero` stream through [`Divisible`] and ignore `N`, but it is
/// still required so every `Divide` instantiation names its lane count explicitly. `WideMul` must
/// defer reduction, so it materializes one unreduced product per lane in an `N`-element
/// [`LaneWideProduct`] — and an associated const can't be an array length without
/// `generic_const_exprs`, which is why `N` is a const generic rather than read from `Divisible`.
#[repr(transparent)]
#[derive(TransparentWrapper)]
#[transparent(T)]
pub struct Divide<SubU, T, const N: usize>(T, PhantomData<SubU>);

impl<U, SubU, F, const N: usize> Square for Divide<SubU, PackedPrimitiveType<U, F>, N>
where
	U: UnderlierType + Divisible<SubU>,
	SubU: UnderlierType,
	F: BinaryField,
	PackedPrimitiveType<SubU, F>: Square,
{
	#[inline]
	fn square(self) -> Self {
		let val = Self::peel(self);
		let squared = Divisible::<SubU>::value_iter(val.to_underlier()).map(|lane| {
			PackedPrimitiveType::<SubU, F>::from_underlier(lane)
				.square()
				.to_underlier()
		});
		Self::wrap(PackedPrimitiveType::from_underlier(Divisible::<SubU>::from_iter(squared)))
	}
}

impl<U, SubU, F, const N: usize> InvertOrZero for Divide<SubU, PackedPrimitiveType<U, F>, N>
where
	U: UnderlierType + Divisible<SubU>,
	SubU: UnderlierType,
	F: BinaryField,
	PackedPrimitiveType<SubU, F>: InvertOrZero,
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		let val = Self::peel(self);
		let inverted = Divisible::<SubU>::value_iter(val.to_underlier()).map(|lane| {
			PackedPrimitiveType::<SubU, F>::from_underlier(lane)
				.invert_or_zero()
				.to_underlier()
		});
		Self::wrap(PackedPrimitiveType::from_underlier(Divisible::<SubU>::from_iter(inverted)))
	}
}

/// One independent deferred wide product per `SubU` lane of a [`Divide`] widening multiply. Lanes
/// accumulate (`Add`/`Sub`/`Sum`) and reduce independently, mirroring the packing structure, so a
/// sum of products is reduced only once per lane. `N` is the lane count.
#[derive(Clone, Copy, Debug)]
pub struct LaneWideProduct<O, const N: usize>([O; N]);

impl<O: Copy + Default, const N: usize> Default for LaneWideProduct<O, N> {
	#[inline]
	fn default() -> Self {
		Self([O::default(); N])
	}
}

impl<O: Copy + Add<Output = O>, const N: usize> Add for LaneWideProduct<O, N> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self(array::from_fn(|i| self.0[i] + rhs.0[i]))
	}
}

impl<O: Copy + Add<Output = O>, const N: usize> AddAssign for LaneWideProduct<O, N> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<O: Copy + Sub<Output = O>, const N: usize> Sub for LaneWideProduct<O, N> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self(array::from_fn(|i| self.0[i] - rhs.0[i]))
	}
}

impl<O: Copy + Sub<Output = O>, const N: usize> SubAssign for LaneWideProduct<O, N> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<O: Copy + Default + Add<Output = O>, const N: usize> Sum for LaneWideProduct<O, N> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

impl<U, SubU, F, const N: usize> WideMul for Divide<SubU, PackedPrimitiveType<U, F>, N>
where
	U: UnderlierType + Divisible<SubU>,
	SubU: UnderlierType,
	F: BinaryField,
	PackedPrimitiveType<SubU, F>: WideMul,
	<PackedPrimitiveType<SubU, F> as WideMul>::Output: Copy + Default,
{
	type Output = LaneWideProduct<<PackedPrimitiveType<SubU, F> as WideMul>::Output, N>;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		debug_assert_eq!(N, <U as Divisible<SubU>>::N, "N must equal Divisible<SubU>::N");

		let a = Self::peel(a).to_underlier();
		let b = Self::peel(b).to_underlier();

		let mut lanes = [<PackedPrimitiveType<SubU, F> as WideMul>::Output::default(); N];
		for (slot, (lhs, rhs)) in lanes
			.iter_mut()
			.zip(Divisible::<SubU>::value_iter(a).zip(Divisible::<SubU>::value_iter(b)))
		{
			*slot = <PackedPrimitiveType<SubU, F> as WideMul>::wide_mul(
				PackedPrimitiveType::from_underlier(lhs),
				PackedPrimitiveType::from_underlier(rhs),
			);
		}
		LaneWideProduct(lanes)
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		let lanes = wide.0.into_iter().map(|product| {
			<PackedPrimitiveType<SubU, F> as WideMul>::reduce(product).to_underlier()
		});
		Self::wrap(PackedPrimitiveType::from_underlier(Divisible::<SubU>::from_iter(lanes)))
	}
}

/// Wrapper that defines multiplication as `reduce(wide_mul(a, b))`, deferring to the type's own
/// [`WideMul`] impl, making the widening multiply the single source of truth for both `Mul` and
/// `WideMul`. Used by every GHASH and AES packing.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct MulFromWideMul<T>(T);

impl<P: WideMul> std::ops::Mul for MulFromWideMul<P> {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self {
		Self::wrap(P::reduce(P::wide_mul(Self::peel(self), Self::peel(rhs))))
	}
}
