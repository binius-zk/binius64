// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_utils::{checked_arithmetics::checked_int_div, iter::IterExtensions};

use super::{Divisible, NumCast, U1, U2, U4, UnderlierType, UnderlierWithBitOps};

/// The iteration strategy for the given underlier type 'U' that is treated as a packed collection
/// of 'T's.
pub trait IterationStrategy<T, U> {
	/// Iterate over the subvalues of the given reference.
	fn ref_iter(value: &U) -> impl Iterator<Item = T> + Send + Clone + '_;

	/// Iterate over the subvalues of the given value.
	fn value_iter(value: U) -> impl Iterator<Item = T> + Send + Clone;

	/// Iterate over the subvalues of the given slice.
	fn slice_iter(slice: &[U]) -> impl Iterator<Item = T> + Send + Clone + '_;
}

/// The iteration strategy for the given underlier type 'U' that is treated as a packed collection
/// of bits.
pub struct BitIterationStrategy;

impl<U> IterationStrategy<U1, U> for BitIterationStrategy
where
	U: Divisible<u8> + UnderlierWithBitOps,
	U1: NumCast<U>,
{
	#[inline]
	fn ref_iter(value: &U) -> impl Iterator<Item = U1> + Send + Clone + '_ {
		(0..U::BITS).map_skippable(move |i| unsafe { value.get_subvalue(i) })
	}

	#[inline]
	fn value_iter(value: U) -> impl Iterator<Item = U1> + Send + Clone {
		(0..U::BITS).map_skippable(move |i| unsafe { value.get_subvalue(i) })
	}

	#[inline]
	fn slice_iter(slice: &[U]) -> impl Iterator<Item = U1> + Send + Clone + '_ {
		U::split_slice(slice)
			.iter()
			.flat_map(|val| (0..8).map(move |i| U1::new(*val >> i)))
	}
}

/// Specialized iteration strategy for types that can be cast to an array of the elements of a
/// smaller type 'T'.
pub struct DivisibleStrategy;

impl<U: UnderlierType + Divisible<T>, T: UnderlierType> IterationStrategy<T, U>
	for DivisibleStrategy
{
	#[inline]
	fn ref_iter(value: &U) -> impl Iterator<Item = T> + Send + Clone + '_ {
		U::split_ref(value).iter().copied()
	}

	#[inline]
	fn value_iter(value: U) -> impl Iterator<Item = T> + Send + Clone {
		U::split_val(value).into_iter()
	}

	#[inline]
	fn slice_iter(slice: &[U]) -> impl Iterator<Item = T> + Send + Clone + '_ {
		U::split_slice(slice).iter().copied()
	}
}

/// Fallback iteration strategy for types that do not have a specialized strategy.
pub struct FallbackStrategy;

impl<T, U> IterationStrategy<T, U> for FallbackStrategy
where
	T: UnderlierType + NumCast<U>,
	U: UnderlierWithBitOps,
{
	#[inline]
	fn value_iter(value: U) -> impl Iterator<Item = T> + Send + Clone {
		(0..checked_int_div(U::BITS, T::BITS))
			.map_skippable(move |i| unsafe { value.get_subvalue(i) })
	}

	#[inline]
	fn ref_iter(value: &U) -> impl Iterator<Item = T> + Send + Clone + '_ {
		(0..checked_int_div(U::BITS, T::BITS))
			.map_skippable(move |i| unsafe { value.get_subvalue(i) })
	}

	#[inline]
	fn slice_iter(slice: &[U]) -> impl Iterator<Item = T> + Send + Clone + '_ {
		slice.iter().flat_map(Self::ref_iter)
	}
}

/// `IterationMethods<T, U>` is supposed to implement an optimal strategy for the pair of types `(T,
/// U)`.
#[derive(Default, Copy, Clone, Eq, PartialEq, Debug)]
pub struct IterationMethods<T, U>(PhantomData<(T, U)>);

macro_rules! impl_iteration {
	(@pairs $strategy:ident, $bigger:ty,) => {};
	(@pairs $strategy:ident, $bigger:ty, $smaller:ty, $($tail:ty,)*) => {
		impl $crate::underlier::IterationStrategy<$smaller, $bigger> for $crate::underlier::IterationMethods<$smaller, $bigger> {
			fn ref_iter(value: &$bigger) -> impl Iterator<Item = $smaller> + Send + Clone + '_ {
				$crate::underlier::$strategy::ref_iter(value)
			}

			fn value_iter(value: $bigger) -> impl Iterator<Item = $smaller> + Send + Clone {
				$crate::underlier::$strategy::value_iter(value)
			}

			fn slice_iter(slice: &[$bigger]) -> impl Iterator<Item = $smaller> + Send + Clone + '_ {
				$crate::underlier::$strategy::slice_iter(slice)
			}
		}

		impl_iteration!(@pairs $strategy, $bigger, $($tail,)*);
	};
	($bigger:ty, $(@strategy $strategy:ident, $($smaller:ty,)*)*) => {
		$(
			impl_iteration!(@pairs $strategy, $bigger, $($smaller,)*);
		)*
	};
}

pub(crate) use impl_iteration;

impl_iteration!(U1,
	@strategy FallbackStrategy, U1,
);
impl_iteration!(U2,
	@strategy FallbackStrategy, U1, U2,
);
impl_iteration!(U4,
	@strategy FallbackStrategy, U1, U2, U4,
);
impl_iteration!(u8,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8,
);
impl_iteration!(u16,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8, u16,
);
impl_iteration!(u32,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8, u16, u32,
);
impl_iteration!(u64,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8, u16, u32, u64,
);
impl_iteration!(u128,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8, u16, u32, u64, u128,
);
