// Copyright 2026 The Binius Developers

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

use binius_utils::{
	DeserializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
	serialization::{assert_enough_data_for, assert_enough_space_for},
};
use bytemuck::{Pod, Zeroable};
use derive_more::{From, Into};
use rand::{
	distr::{Distribution, StandardUniform},
	prelude::*,
};

use crate::{
	BinaryField,
	arch::portable::packed::PackedPrimitiveType,
	underlier::{
		Divisible, NumCast, SmallU, UnderlierType, impl_divisible_bitmask, impl_divisible_memcast,
		impl_divisible_self,
	},
};

/// 128-bit underlier for the portable build — a transparent wrapper over `u128`.
///
/// On x86_64/aarch64 `M128` is a SIMD register and on wasm32 (with `simd128`) a `v128`; here it is
/// a plain `u128` newtype. Wrapping rather than aliasing `u128` keeps `M128` a distinct type on
/// every target, so the `M128 <-> u128` conversions never collide with `u128`'s own reflexive
/// impls and the architecture-gated `BinaryField128bGhash` conversions need no cfg gate.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, From, Into)]
#[repr(transparent)]
pub struct M128(u128);

impl M128 {
	#[inline(always)]
	pub const fn from_u128(value: u128) -> Self {
		Self(value)
	}
}

impl From<u64> for M128 {
	#[inline(always)]
	fn from(value: u64) -> Self {
		Self(value as u128)
	}
}
impl From<u32> for M128 {
	#[inline(always)]
	fn from(value: u32) -> Self {
		Self(value as u128)
	}
}
impl From<u16> for M128 {
	#[inline(always)]
	fn from(value: u16) -> Self {
		Self(value as u128)
	}
}
impl From<u8> for M128 {
	#[inline(always)]
	fn from(value: u8) -> Self {
		Self(value as u128)
	}
}

impl<const N: usize> From<SmallU<N>> for M128 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		Self(value.val() as u128)
	}
}

impl<U: NumCast<u128>> NumCast<M128> for U {
	#[inline(always)]
	fn num_cast_from(val: M128) -> Self {
		Self::num_cast_from(val.0)
	}
}

impl SerializeBytes for M128 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u128_le(self.0);
		Ok(())
	}
}

impl DeserializeBytes for M128 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(Self(read_buf.get_u128_le()))
	}
}

unsafe impl Zeroable for M128 {}

unsafe impl Pod for M128 {}

impl_divisible_memcast!(M128, u128, u64, u32, u16, u8);
impl_divisible_bitmask!(M128, 1, 2, 4);
impl_divisible_self!(M128);

impl BitAnd for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitand(self, rhs: Self) -> Self::Output {
		Self(self.0 & rhs.0)
	}
}

impl BitAndAssign for M128 {
	#[inline(always)]
	fn bitand_assign(&mut self, rhs: Self) {
		self.0 &= rhs.0;
	}
}

impl BitOr for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(self.0 | rhs.0)
	}
}

impl BitOrAssign for M128 {
	#[inline(always)]
	fn bitor_assign(&mut self, rhs: Self) {
		self.0 |= rhs.0;
	}
}

impl BitXor for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl BitXorAssign for M128 {
	#[inline(always)]
	fn bitxor_assign(&mut self, rhs: Self) {
		self.0 ^= rhs.0;
	}
}

impl Not for M128 {
	type Output = Self;

	#[inline(always)]
	fn not(self) -> Self::Output {
		Self(!self.0)
	}
}

impl Shl<usize> for M128 {
	type Output = Self;

	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		Self(self.0 << rhs)
	}
}

impl Shr<usize> for M128 {
	type Output = Self;

	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		Self(self.0 >> rhs)
	}
}

impl Distribution<M128> for StandardUniform {
	#[inline]
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> M128 {
		M128(rng.random())
	}
}

impl std::fmt::Display for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{:032X}", self.0)
	}
}

impl std::fmt::Debug for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "M128({self})")
	}
}

impl std::fmt::LowerHex for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		std::fmt::LowerHex::fmt(&self.0, f)
	}
}

impl UnderlierType for M128 {
	const LOG_BITS: usize = 7;
	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const ONES: Self = Self(u128::MAX);

	#[inline(always)]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let (a, b) = self.0.interleave(other.0, log_block_len);
		(Self(a), Self(b))
	}
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	#[inline]
	fn from(value: u128) -> Self {
		Self::from(M128::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for u128 {
	#[inline]
	fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
		value.to_underlier().into()
	}
}
