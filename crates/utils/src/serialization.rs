// Copyright 2024-2025 Irreducible Inc.

use bytemuck::Pod;
use bytes::{Buf, BufMut};
use hybrid_array::{Array, ArraySize};
use thiserror::Error;

/// Serialize data to a byte buffer.
pub trait SerializeBytes {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError>;

	/// Serializes every element of `slice` back-to-back, with no length prefix.
	///
	/// The default loops, calling [`serialize`](Self::serialize) once per element.
	/// That default is always correct.
	/// Overriding this method is an optimization, never a correctness requirement.
	///
	/// Two kinds of type override it.
	/// An integer whose serialized bytes are its in-memory bytes calls [`serialize_slice_le`].
	/// A transparent wrapper over such an integer forwards to its underlier's implementation.
	/// The wrapper forwards here exactly as its [`serialize`](Self::serialize) forwards.
	fn serialize_slice(slice: &[Self], write_buf: impl BufMut) -> Result<(), SerializationError>
	where
		Self: Sized,
	{
		let mut write_buf = write_buf;
		slice
			.iter()
			.try_for_each(|item| item.serialize(&mut write_buf))
	}
}

/// Deserialize data from a byte buffer.
pub trait DeserializeBytes: Sized {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>;
}

/// A type whose byte-serialized form has a fixed, compile-time-known size.
///
/// Implementors must guarantee that [`SerializeBytes::serialize`] writes exactly
/// [`BYTE_SIZE`](Self::BYTE_SIZE) bytes and that [`DeserializeBytes::deserialize`] reads exactly
/// [`BYTE_SIZE`](Self::BYTE_SIZE) bytes.
pub trait FixedSizeSerializeBytes: SerializeBytes + DeserializeBytes {
	/// The exact number of bytes written by `serialize` and read by `deserialize`.
	const BYTE_SIZE: usize;
}

macro_rules! impl_fixed_size_serialize_bytes {
	($ty:ty, $size:expr) => {
		impl FixedSizeSerializeBytes for $ty {
			const BYTE_SIZE: usize = $size;
		}
	};
}

impl_fixed_size_serialize_bytes!(u8, 1);
impl_fixed_size_serialize_bytes!(u16, 2);
impl_fixed_size_serialize_bytes!(u32, 4);
impl_fixed_size_serialize_bytes!(u64, 8);
impl_fixed_size_serialize_bytes!(u128, 16);
// `usize` is serialized as a `u32`.
impl_fixed_size_serialize_bytes!(usize, 4);
impl_fixed_size_serialize_bytes!(bool, 1);

/// Serializes `slice` as one bulk copy of its raw memory, on a little-endian target.
///
/// `T`'s serialized form must be its in-memory bytes, in little-endian order.
/// Every fixed-width integer written with a `put_*_le` satisfies that on a little-endian target.
/// There the whole call is a single `put_slice`, in place of `slice.len()` word-sized writes.
///
/// On a big-endian target the two byte orders disagree, so this falls back to the per-element loop.
/// The bytes written are identical either way; only the number of writes differs.
pub fn serialize_slice_le<T: SerializeBytes + Pod>(
	slice: &[T],
	mut write_buf: impl BufMut,
) -> Result<(), SerializationError> {
	#[cfg(target_endian = "little")]
	{
		let bytes: &[u8] = bytemuck::cast_slice(slice);
		assert_enough_space_for(&write_buf, bytes.len())?;
		write_buf.put_slice(bytes);
		Ok(())
	}
	#[cfg(target_endian = "big")]
	{
		slice
			.iter()
			.try_for_each(|item| item.serialize(&mut write_buf))
	}
}

#[derive(Error, Debug, Clone)]
pub enum SerializationError {
	#[error("Write buffer is full")]
	WriteBufferFull,
	#[error("Not enough data in read buffer to deserialize")]
	NotEnoughBytes,
	#[error("Unknown enum variant index {name}::{index}")]
	UnknownEnumVariant { name: &'static str, index: u8 },
	#[error("FromUtf8Error: {0}")]
	FromUtf8Error(#[from] std::string::FromUtf8Error),
	#[error("Invalid construction of {name}")]
	InvalidConstruction { name: &'static str },
	#[error("usize {size} is too large to serialize (max is {max})", max = u32::MAX)]
	UsizeTooLarge { size: usize },
}

impl<T: SerializeBytes + ?Sized> SerializeBytes for &T {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		(**self).serialize(write_buf)
	}
}

impl SerializeBytes for usize {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		let value: u32 = (*self)
			.try_into()
			.map_err(|_| SerializationError::UsizeTooLarge { size: *self })?;
		SerializeBytes::serialize(&value, &mut write_buf)
	}
}

impl DeserializeBytes for usize {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let value: u32 = DeserializeBytes::deserialize(&mut read_buf)?;
		Ok(value as Self)
	}
}

impl SerializeBytes for u128 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u128_le(*self);
		Ok(())
	}

	fn serialize_slice(slice: &[Self], write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_slice_le(slice, write_buf)
	}
}

impl DeserializeBytes for u128 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u128_le())
	}
}

impl SerializeBytes for u64 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u64_le(*self);
		Ok(())
	}

	fn serialize_slice(slice: &[Self], write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_slice_le(slice, write_buf)
	}
}

impl DeserializeBytes for u64 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u64_le())
	}
}

impl SerializeBytes for u32 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u32_le(*self);
		Ok(())
	}

	fn serialize_slice(slice: &[Self], write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_slice_le(slice, write_buf)
	}
}

impl DeserializeBytes for u32 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u32_le())
	}
}

impl SerializeBytes for u16 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u16_le(*self);
		Ok(())
	}

	fn serialize_slice(slice: &[Self], write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_slice_le(slice, write_buf)
	}
}

impl DeserializeBytes for u16 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u16_le())
	}
}

impl SerializeBytes for u8 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u8(*self);
		Ok(())
	}

	fn serialize_slice(slice: &[Self], write_buf: impl BufMut) -> Result<(), SerializationError> {
		serialize_slice_le(slice, write_buf)
	}
}

impl DeserializeBytes for u8 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u8())
	}
}

impl SerializeBytes for bool {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		u8::serialize(&(*self as u8), write_buf)
	}
}

impl DeserializeBytes for bool {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(u8::deserialize(read_buf)? != 0)
	}
}

impl<T> SerializeBytes for std::marker::PhantomData<T> {
	fn serialize(&self, _write_buf: impl BufMut) -> Result<(), SerializationError> {
		Ok(())
	}
}

impl<T> DeserializeBytes for std::marker::PhantomData<T> {
	fn deserialize(_read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self)
	}
}

impl SerializeBytes for &str {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		let bytes = self.as_bytes();
		SerializeBytes::serialize(&bytes.len(), &mut write_buf)?;
		assert_enough_space_for(&write_buf, bytes.len())?;
		write_buf.put_slice(bytes);
		Ok(())
	}
}

impl SerializeBytes for String {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		SerializeBytes::serialize(&self.as_str(), &mut write_buf)
	}
}

impl DeserializeBytes for String {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let len = DeserializeBytes::deserialize(&mut read_buf)?;
		assert_enough_data_for(&read_buf, len)?;
		Ok(Self::from_utf8(read_buf.copy_to_bytes(len).to_vec())?)
	}
}

impl<T: SerializeBytes> SerializeBytes for [T] {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		SerializeBytes::serialize(&self.len(), &mut write_buf)?;
		T::serialize_slice(self, write_buf)
	}
}

impl<T: SerializeBytes> SerializeBytes for Vec<T> {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		SerializeBytes::serialize(self.as_slice(), &mut write_buf)
	}
}

impl<T: DeserializeBytes> DeserializeBytes for Vec<T> {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let len: usize = DeserializeBytes::deserialize(&mut read_buf)?;
		(0..len)
			.map(|_| DeserializeBytes::deserialize(&mut read_buf))
			.collect()
	}
}

impl<T: SerializeBytes> SerializeBytes for Option<T> {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		match self {
			Some(value) => {
				SerializeBytes::serialize(&true, &mut write_buf)?;
				SerializeBytes::serialize(value, &mut write_buf)?;
			}
			None => {
				SerializeBytes::serialize(&false, write_buf)?;
			}
		}
		Ok(())
	}
}

impl<T: DeserializeBytes> DeserializeBytes for Option<T> {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(if bool::deserialize(&mut read_buf)? {
			Some(T::deserialize(&mut read_buf)?)
		} else {
			None
		})
	}
}

impl<U: SerializeBytes, V: SerializeBytes> SerializeBytes for (U, V) {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		U::serialize(&self.0, &mut write_buf)?;
		V::serialize(&self.1, write_buf)
	}
}

impl<U: DeserializeBytes, V: DeserializeBytes> DeserializeBytes for (U, V) {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok((U::deserialize(&mut read_buf)?, V::deserialize(read_buf)?))
	}
}

impl<T: SerializeBytes, const N: usize> SerializeBytes for [T; N] {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		for val in self {
			val.serialize(&mut write_buf)?;
		}
		Ok(())
	}
}

impl<T: DeserializeBytes, const N: usize> DeserializeBytes for [T; N] {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		array_util::try_from_fn(|_| T::deserialize(&mut read_buf))
	}
}

impl<U: ArraySize> SerializeBytes for Array<u8, U> {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, U::USIZE)?;
		write_buf.put_slice(self);
		Ok(())
	}
}

impl<U: ArraySize> DeserializeBytes for Array<u8, U> {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		assert_enough_data_for(&read_buf, U::USIZE)?;
		let mut ret = Self::default();
		read_buf.copy_to_slice(&mut ret);
		Ok(ret)
	}
}

#[inline]
pub fn assert_enough_space_for(
	write_buf: &impl BufMut,
	size: usize,
) -> Result<(), SerializationError> {
	if write_buf.remaining_mut() < size {
		return Err(SerializationError::WriteBufferFull);
	}
	Ok(())
}

#[inline]
pub fn assert_enough_data_for(read_buf: &impl Buf, size: usize) -> Result<(), SerializationError> {
	if read_buf.remaining() < size {
		return Err(SerializationError::NotEnoughBytes);
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use std::fmt::Debug;

	use hybrid_array::sizes::U32;
	use proptest::{collection::vec, prelude::any, proptest};
	use rand::prelude::*;

	use super::*;

	/// Asserts that `T`'s bulk `serialize_slice` agrees with the per-element loop it replaces.
	///
	/// The loop is the reference: it is the trait default, and consumers trust the two to agree.
	/// Round-tripping through `deserialize` then checks the bytes are readable, not merely equal.
	fn check_serialize_slice_matches_loop<T>(values: &[T])
	where
		T: SerializeBytes + DeserializeBytes + Copy + PartialEq + Debug,
	{
		let mut expected = Vec::new();
		for value in values {
			value.serialize(&mut expected).unwrap();
		}

		let mut actual = Vec::new();
		T::serialize_slice(values, &mut actual).unwrap();

		assert_eq!(actual, expected);

		// `serialize_slice` writes no length prefix.
		// Read back exactly `values.len()` elements from a cursor over the same buffer.
		let mut cursor = actual.as_slice();
		let deserialized: Vec<T> = (0..values.len())
			.map(|_| T::deserialize(&mut cursor).unwrap())
			.collect();
		assert_eq!(deserialized, values);
		assert!(cursor.is_empty());
	}

	proptest! {
		/// Every fixed-width integer underlier, at lengths from empty through a few cache lines.
		#[test]
		fn test_integer_serialize_slice_matches_loop(
			bytes in vec(any::<u8>(), 0..300),
			shorts in vec(any::<u16>(), 0..300),
			words in vec(any::<u32>(), 0..300),
			longs in vec(any::<u64>(), 0..300),
			wide in vec(any::<u128>(), 0..300),
		) {
			check_serialize_slice_matches_loop(&bytes);
			check_serialize_slice_matches_loop(&shorts);
			check_serialize_slice_matches_loop(&words);
			check_serialize_slice_matches_loop(&longs);
			check_serialize_slice_matches_loop(&wide);
		}
	}

	#[test]
	fn test_serialize_slice_rejects_a_buffer_one_byte_short() {
		let values = [1u64, 2, 3, 4];

		// The bulk copy checks the whole slice's byte length, not one element's.
		// One byte short of four words must still be refused.
		let mut buf = [0u8; 4 * size_of::<u64>() - 1];
		let err = u64::serialize_slice(&values, buf.as_mut_slice()).unwrap_err();

		assert!(matches!(err, SerializationError::WriteBufferFull));
	}

	#[test]
	fn test_generic_array_serialize_deserialize() {
		let mut rng = StdRng::seed_from_u64(0);

		let mut data = Array::<u8, U32>::default();
		rng.fill_bytes(&mut data);

		let mut buf = Vec::new();
		data.serialize(&mut buf).unwrap();

		let data_deserialized = Array::<u8, U32>::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(data_deserialized, data);
	}
}
