// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::{
	DeserializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
	checked_arithmetics::checked_log_2,
};
use bytemuck::{Pod, TransparentWrapper, Zeroable};
use rand::{
	Rng,
	distr::{Distribution, StandardUniform},
};

use crate::{
	Field, PackedField,
	arithmetic_traits::MulAlpha,
	as_packed_field::PackScalar,
	linear_transformation::{
		FieldLinearTransformation, PackedTransformationFactory, Transformation,
	},
	packed::PackedBinaryField,
	underlier::{ScaledUnderlier, UnderlierType, WithUnderlier},
};

/// Packed field that just stores smaller packed field N times and performs all operations
/// one by one.
/// This makes sense for creating portable implementations for 256 and 512 packed sizes.
#[derive(PartialEq, Eq, Clone, Copy, Debug, bytemuck::TransparentWrapper)]
#[repr(transparent)]
pub struct ScaledPackedField<PT, const N: usize>(pub(super) [PT; N]);

impl<PT, const N: usize> ScaledPackedField<PT, N> {
	pub const WIDTH_IN_PT: usize = N;

	/// In general case PT != Self::Scalar, so this function has a different name from
	/// `PackedField::from_fn`
	pub fn from_direct_packed_fn(f: impl FnMut(usize) -> PT) -> Self {
		Self(std::array::from_fn(f))
	}

	/// We put implementation here to be able to use in the generic code.
	/// (`PackedField` is only implemented for certain types via macro).
	#[inline]
	pub(crate) unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self
	where
		PT: PackedField,
	{
		let log_n = checked_log_2(N);
		let values = if log_block_len >= PT::LOG_WIDTH {
			let offset = block_idx << (log_block_len - PT::LOG_WIDTH);
			let log_packed_block = log_block_len - PT::LOG_WIDTH;
			let log_smaller_block = PT::LOG_WIDTH.saturating_sub(log_n - log_packed_block);
			let smaller_block_index_mask = (1 << (PT::LOG_WIDTH - log_smaller_block)) - 1;
			array::from_fn(|i| unsafe {
				self.0
					.get_unchecked(offset + (i >> (log_n - log_packed_block)))
					.spread_unchecked(
						log_smaller_block,
						(i >> log_n.saturating_sub(log_block_len)) & smaller_block_index_mask,
					)
			})
		} else {
			let value_index = block_idx >> (PT::LOG_WIDTH - log_block_len);
			let log_inner_block_len = log_block_len.saturating_sub(log_n);
			let block_offset = block_idx & ((1 << (PT::LOG_WIDTH - log_block_len)) - 1);
			let block_offset = block_offset << (log_block_len - log_inner_block_len);

			array::from_fn(|i| unsafe {
				self.0.get_unchecked(value_index).spread_unchecked(
					log_inner_block_len,
					block_offset + (i >> (log_n + log_inner_block_len - log_block_len)),
				)
			})
		};

		Self(values)
	}
}

impl<PT, const N: usize> SerializeBytes for ScaledPackedField<PT, N>
where
	PT: SerializeBytes,
{
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		for elem in &self.0 {
			elem.serialize(&mut write_buf)?;
		}
		Ok(())
	}
}

impl<PT, const N: usize> DeserializeBytes for ScaledPackedField<PT, N>
where
	PT: DeserializeBytes,
{
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let mut result = Vec::with_capacity(N);
		for _ in 0..N {
			result.push(PT::deserialize(&mut read_buf)?);
		}

		match result.try_into() {
			Ok(arr) => Ok(Self(arr)),
			Err(_) => Err(SerializationError::InvalidConstruction {
				name: "ScaledPackedField",
			}),
		}
	}
}

impl<PT, const N: usize> Default for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
{
	fn default() -> Self {
		Self(Default::default())
	}
}

impl<U, PT, const N: usize> From<[U; N]> for ScaledPackedField<PT, N>
where
	PT: From<U>,
{
	fn from(value: [U; N]) -> Self {
		Self(value.map(Into::into))
	}
}

impl<U, PT, const N: usize> From<ScaledPackedField<PT, N>> for [U; N]
where
	U: From<PT>,
{
	fn from(value: ScaledPackedField<PT, N>) -> Self {
		value.0.map(Into::into)
	}
}

unsafe impl<PT: Zeroable, const N: usize> Zeroable for ScaledPackedField<PT, N> {}

unsafe impl<PT: Pod, const N: usize> Pod for ScaledPackedField<PT, N> {}

impl<PT: Copy + Add<Output = PT>, const N: usize> Add for ScaledPackedField<PT, N>
where
	Self: Default,
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self::from_direct_packed_fn(|i| self.0[i] + rhs.0[i])
	}
}

impl<PT: Copy + AddAssign, const N: usize> AddAssign for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn add_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] += rhs.0[i];
		}
	}
}

impl<PT: Copy + Sub<Output = PT>, const N: usize> Sub for ScaledPackedField<PT, N>
where
	Self: Default,
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		Self::from_direct_packed_fn(|i| self.0[i] - rhs.0[i])
	}
}

impl<PT: Copy + SubAssign, const N: usize> SubAssign for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn sub_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] -= rhs.0[i];
		}
	}
}

impl<PT: Copy + Mul<Output = PT>, const N: usize> Mul for ScaledPackedField<PT, N>
where
	Self: Default,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Self::from_direct_packed_fn(|i| self.0[i] * rhs.0[i])
	}
}

impl<PT: Copy + MulAssign, const N: usize> MulAssign for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn mul_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] *= rhs.0[i];
		}
	}
}

/// Currently we use this trait only in this file for compactness.
/// If it is useful in some other place it worth moving it to `arithmetic_traits.rs`
trait ArithmeticOps<Rhs>:
	Add<Rhs, Output = Self>
	+ AddAssign<Rhs>
	+ Sub<Rhs, Output = Self>
	+ SubAssign<Rhs>
	+ Mul<Rhs, Output = Self>
	+ MulAssign<Rhs>
{
}

impl<T, Rhs> ArithmeticOps<Rhs> for T where
	T: Add<Rhs, Output = Self>
		+ AddAssign<Rhs>
		+ Sub<Rhs, Output = Self>
		+ SubAssign<Rhs>
		+ Mul<Rhs, Output = Self>
		+ MulAssign<Rhs>
{
}

impl<PT: Add<Output = PT> + Copy, const N: usize> Sum for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |l, r| l + r)
	}
}

impl<PT: PackedField, const N: usize> Product for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
{
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		let one = Self([PT::one(); N]);
		iter.fold(one, |l, r| l * r)
	}
}

impl<PT: PackedField, const N: usize> PackedField for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
	Self: ArithmeticOps<PT::Scalar>,
{
	type Scalar = PT::Scalar;

	const LOG_WIDTH: usize = PT::LOG_WIDTH + checked_log_2(N);

	#[inline]
	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
		let outer_i = i / PT::WIDTH;
		let inner_i = i % PT::WIDTH;
		unsafe { self.0.get_unchecked(outer_i).get_unchecked(inner_i) }
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
		let outer_i = i / PT::WIDTH;
		let inner_i = i % PT::WIDTH;
		unsafe {
			self.0
				.get_unchecked_mut(outer_i)
				.set_unchecked(inner_i, scalar);
		}
	}

	#[inline]
	fn zero() -> Self {
		Self(array::from_fn(|_| PT::zero()))
	}

	#[inline]
	fn broadcast(scalar: Self::Scalar) -> Self {
		Self(array::from_fn(|_| PT::broadcast(scalar)))
	}

	#[inline]
	fn square(self) -> Self {
		Self(self.0.map(|v| v.square()))
	}

	#[inline]
	fn invert_or_zero(self) -> Self {
		Self(self.0.map(|v| v.invert_or_zero()))
	}

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let mut first = [Default::default(); N];
		let mut second = [Default::default(); N];

		if log_block_len >= PT::LOG_WIDTH {
			let block_in_pts = 1 << (log_block_len - PT::LOG_WIDTH);
			for i in (0..N).step_by(block_in_pts * 2) {
				first[i..i + block_in_pts].copy_from_slice(&self.0[i..i + block_in_pts]);
				first[i + block_in_pts..i + 2 * block_in_pts]
					.copy_from_slice(&other.0[i..i + block_in_pts]);

				second[i..i + block_in_pts]
					.copy_from_slice(&self.0[i + block_in_pts..i + 2 * block_in_pts]);
				second[i + block_in_pts..i + 2 * block_in_pts]
					.copy_from_slice(&other.0[i + block_in_pts..i + 2 * block_in_pts]);
			}
		} else {
			for i in 0..N {
				(first[i], second[i]) = self.0[i].interleave(other.0[i], log_block_len);
			}
		}

		(Self(first), Self(second))
	}

	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let mut first = [Default::default(); N];
		let mut second = [Default::default(); N];

		if log_block_len >= PT::LOG_WIDTH {
			let block_in_pts = 1 << (log_block_len - PT::LOG_WIDTH);
			for i in (0..N / 2).step_by(block_in_pts) {
				first[i..i + block_in_pts].copy_from_slice(&self.0[2 * i..2 * i + block_in_pts]);

				second[i..i + block_in_pts]
					.copy_from_slice(&self.0[2 * i + block_in_pts..2 * (i + block_in_pts)]);
			}

			for i in (0..N / 2).step_by(block_in_pts) {
				first[i + N / 2..i + N / 2 + block_in_pts]
					.copy_from_slice(&other.0[2 * i..2 * i + block_in_pts]);

				second[i + N / 2..i + N / 2 + block_in_pts]
					.copy_from_slice(&other.0[2 * i + block_in_pts..2 * (i + block_in_pts)]);
			}
		} else {
			for i in 0..N / 2 {
				(first[i], second[i]) = self.0[2 * i].unzip(self.0[2 * i + 1], log_block_len);
			}

			for i in 0..N / 2 {
				(first[i + N / 2], second[i + N / 2]) =
					other.0[2 * i].unzip(other.0[2 * i + 1], log_block_len);
			}
		}

		(Self(first), Self(second))
	}

	#[inline]
	unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self {
		unsafe { Self::spread_unchecked(self, log_block_len, block_idx) }
	}

	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		Self(array::from_fn(|i| PT::from_fn(|j| f(i * PT::WIDTH + j))))
	}

	#[inline]
	fn iter_slice(slice: &[Self]) -> impl Iterator<Item = Self::Scalar> + Send + Clone + '_ {
		// Safety: `Self` has the same layout as `[PT; N]` because it is a transparent wrapper.
		let cast_slice =
			unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const [PT; N], slice.len()) };

		PT::iter_slice(cast_slice.as_flattened())
	}
}

impl<PT: PackedField, const N: usize> Distribution<ScaledPackedField<PT, N>> for StandardUniform
where
	[PT; N]: Default,
{
	fn sample<R: Rng + ?Sized>(&self, mut rng: &mut R) -> ScaledPackedField<PT, N> {
		ScaledPackedField(array::from_fn(|_| PT::random(&mut rng)))
	}
}

impl<PT: PackedField + MulAlpha, const N: usize> MulAlpha for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
{
	#[inline]
	fn mul_alpha(self) -> Self {
		Self(self.0.map(|v| v.mul_alpha()))
	}
}

/// Per-element transformation as a scaled packed field.
pub struct ScaledTransformation<I> {
	inner: I,
}

impl<I> ScaledTransformation<I> {
	const fn new(inner: I) -> Self {
		Self { inner }
	}
}

impl<OP, IP, const N: usize, I> Transformation<ScaledPackedField<IP, N>, ScaledPackedField<OP, N>>
	for ScaledTransformation<I>
where
	I: Transformation<IP, OP>,
{
	fn transform(&self, data: &ScaledPackedField<IP, N>) -> ScaledPackedField<OP, N> {
		ScaledPackedField::from_direct_packed_fn(|i| self.inner.transform(&data.0[i]))
	}
}

impl<OP, IP, const N: usize> PackedTransformationFactory<ScaledPackedField<OP, N>>
	for ScaledPackedField<IP, N>
where
	Self: PackedBinaryField,
	ScaledPackedField<OP, N>: PackedBinaryField<Scalar = OP::Scalar>,
	OP: PackedBinaryField,
	IP: PackedTransformationFactory<OP>,
{
	type PackedTransformation<Data: AsRef<[OP::Scalar]> + Sync> =
		ScaledTransformation<IP::PackedTransformation<Data>>;

	fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<
			<ScaledPackedField<OP, N> as PackedField>::Scalar,
			Data,
		>,
	) -> Self::PackedTransformation<Data> {
		ScaledTransformation::new(IP::make_packed_transformation(transformation))
	}
}

/// The only thing that prevents us from having pure generic implementation of `ScaledPackedField`
/// is that we can't have generic operations both with `Self` and `PT::Scalar`
/// (it leads to `conflicting implementations of trait` error).
/// That's why we implement one of those in a macro.
macro_rules! packed_scaled_field {
	($name:ident = [$inner:ty;$size:literal]) => {
		pub type $name = $crate::arch::portable::packed_scaled::ScaledPackedField<$inner, $size>;

		impl std::ops::Add<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			type Output = Self;

			#[inline]
			fn add(mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) -> Self {
				let broadcast = <$inner as $crate::packed::PackedField>::broadcast(rhs);
				for v in self.0.iter_mut() {
					*v += broadcast;
				}

				self
			}
		}

		impl std::ops::AddAssign<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			#[inline]
			fn add_assign(&mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) {
				let broadcast = <$inner as $crate::packed::PackedField>::broadcast(rhs);
				for v in self.0.iter_mut() {
					*v += broadcast;
				}
			}
		}

		impl std::ops::Sub<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			type Output = Self;

			#[inline]
			fn sub(mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) -> Self {
				let broadcast = <$inner as $crate::packed::PackedField>::broadcast(rhs);
				for v in self.0.iter_mut() {
					*v -= broadcast;
				}

				self
			}
		}

		impl std::ops::SubAssign<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			#[inline]
			fn sub_assign(&mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) {
				let broadcast = <$inner as $crate::packed::PackedField>::broadcast(rhs);
				for v in self.0.iter_mut() {
					*v -= broadcast;
				}
			}
		}

		impl std::ops::Mul<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			type Output = Self;

			#[inline]
			fn mul(mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) -> Self {
				let broadcast = <$inner as $crate::packed::PackedField>::broadcast(rhs);
				for v in self.0.iter_mut() {
					*v *= broadcast;
				}

				self
			}
		}

		impl std::ops::MulAssign<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			#[inline]
			fn mul_assign(&mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) {
				let broadcast = <$inner as $crate::packed::PackedField>::broadcast(rhs);
				for v in self.0.iter_mut() {
					*v *= broadcast;
				}
			}
		}
	};
}

pub(crate) use packed_scaled_field;

unsafe impl<PT, const N: usize> WithUnderlier for ScaledPackedField<PT, N>
where
	PT: WithUnderlier<Underlier: Pod>,
{
	type Underlier = ScaledUnderlier<PT::Underlier, N>;

	fn to_underlier(self) -> Self::Underlier {
		TransparentWrapper::peel(self)
	}

	fn to_underlier_ref(&self) -> &Self::Underlier {
		TransparentWrapper::peel_ref(self)
	}

	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		TransparentWrapper::peel_mut(self)
	}

	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		TransparentWrapper::peel_slice(val)
	}

	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		TransparentWrapper::peel_slice_mut(val)
	}

	fn from_underlier(val: Self::Underlier) -> Self {
		TransparentWrapper::wrap(val)
	}

	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		TransparentWrapper::wrap_ref(val)
	}

	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		TransparentWrapper::wrap_mut(val)
	}

	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		TransparentWrapper::wrap_slice(val)
	}

	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		TransparentWrapper::wrap_slice_mut(val)
	}
}

impl<U, F, const N: usize> PackScalar<F> for ScaledUnderlier<U, N>
where
	U: PackScalar<F> + UnderlierType + Pod,
	F: Field,
	ScaledPackedField<U::Packed, N>: PackedField<Scalar = F> + WithUnderlier<Underlier = Self>,
{
	type Packed = ScaledPackedField<U::Packed, N>;
}

unsafe impl<PT, U, const N: usize> TransparentWrapper<ScaledUnderlier<U, N>>
	for ScaledPackedField<PT, N>
where
	PT: WithUnderlier<Underlier = U>,
{
}
