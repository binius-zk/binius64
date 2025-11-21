// Copyright 2023-2025 Irreducible Inc.

use binius_utils::random_access_sequence::RandomAccessSequence;
use bytemuck::{Pod, zeroed_vec};

use crate::{
	AESTowerField8b, BinaryField128bGhash, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
	PackedBinaryGhash4x128b, PackedField,
	arch::{
		packed_8::*, packed_16::*, packed_32::*, packed_64::*, packed_128::*, packed_256::*,
		packed_512::*, packed_aes_8::*, packed_aes_16::*, packed_aes_32::*, packed_aes_64::*,
		packed_aes_128::*, packed_aes_256::*, packed_aes_512::*,
	},
};

/// A marker trait that the slice of packed values can be iterated as a sequence of bytes.
/// The order of the iteration by BinaryField1b subfield elements and bits within iterated bytes
/// must be the same.
///
/// # Safety
/// The implementor must ensure that the cast of the slice of packed values to the slice of bytes
/// is safe and preserves the order of the 1-bit elements.
#[allow(unused)]
unsafe trait SequentialBytes: Pod {}

unsafe impl SequentialBytes for PackedBinaryField8x1b {}
unsafe impl SequentialBytes for PackedBinaryField16x1b {}
unsafe impl SequentialBytes for PackedBinaryField32x1b {}
unsafe impl SequentialBytes for PackedBinaryField64x1b {}
unsafe impl SequentialBytes for PackedBinaryField128x1b {}
unsafe impl SequentialBytes for PackedBinaryField256x1b {}
unsafe impl SequentialBytes for PackedBinaryField512x1b {}

unsafe impl SequentialBytes for AESTowerField8b {}

unsafe impl SequentialBytes for PackedAESBinaryField1x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField2x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField4x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField8x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField16x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField32x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField64x8b {}

unsafe impl SequentialBytes for BinaryField128bGhash {}

unsafe impl SequentialBytes for PackedBinaryGhash1x128b {}
unsafe impl SequentialBytes for PackedBinaryGhash2x128b {}
unsafe impl SequentialBytes for PackedBinaryGhash4x128b {}

/// Returns true if T implements `SequentialBytes` trait.
/// Use a hack that exploits that array copying is optimized for the `Copy` types.
/// Unfortunately there is no more proper way to perform this check this in Rust at runtime.
#[inline(always)]
#[allow(clippy::redundant_clone)] // this is intentional in this method
pub fn is_sequential_bytes<T>() -> bool {
	struct X<U>(bool, std::marker::PhantomData<U>);

	impl<U> Clone for X<U> {
		fn clone(&self) -> Self {
			Self(false, std::marker::PhantomData)
		}
	}

	impl<U: SequentialBytes> Copy for X<U> {}

	let value = [X::<T>(true, std::marker::PhantomData)];
	let cloned = value.clone();

	cloned[0].0
}

/// Returns if we can iterate over bytes, each representing 8 1-bit values.
#[inline(always)]
pub fn can_iterate_bytes<P: PackedField>() -> bool {
	// Packed fields with sequential byte order
	is_sequential_bytes::<P>()
}

/// Callback for byte iteration.
/// We can't return different types from the `iterate_bytes` and Fn traits don't support associated
/// types that's why we use a callback with a generic function.
pub trait ByteIteratorCallback {
	fn call(&mut self, iter: impl Iterator<Item = u8>);
}

/// Iterate over bytes of a slice of the packed values.
/// The method panics if the packed field doesn't support byte iteration, so use `can_iterate_bytes`
/// to check it.
#[inline(always)]
pub fn iterate_bytes<P: PackedField>(data: &[P], callback: &mut impl ByteIteratorCallback) {
	if is_sequential_bytes::<P>() {
		// Safety: `P` implements `SequentialBytes` trait, so the following cast is safe
		// and preserves the order.
		let bytes = unsafe {
			std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
		};
		callback.call(bytes.iter().copied());
	} else {
		unreachable!("packed field doesn't support byte iteration")
	}
}

/// Create a lookup table for partial sums of 8 consequent elements with coefficients corresponding
/// to bits in a byte. The lookup table has the following structure:
/// [
///     partial_sum_chunk_0_7_byte_0, partial_sum_chunk_0_7_byte_1, ...,
/// partial_sum_chunk_0_7_byte_255,     partial_sum_chunk_8_15_byte_0,
/// partial_sum_chunk_8_15_byte_1, ..., partial_sum_chunk_8_15_byte_255,    ...
/// ]
pub fn create_partial_sums_lookup_tables<P: PackedField>(
	values: impl RandomAccessSequence<P>,
) -> Vec<P> {
	let len = values.len();
	assert!(len.is_multiple_of(8));

	let mut result = zeroed_vec(len * 32);

	for (chunk_idx, chunk_start) in (0..len).step_by(8).enumerate() {
		let sums = &mut result[chunk_idx * 256..(chunk_idx + 1) * 256];

		for j in 0..8 {
			let value = values.get(chunk_start + j);
			let mask = 1 << j;
			for i in (mask..256).step_by(mask * 2) {
				for k in 0..mask {
					sums[i + k] += value;
				}
			}
		}
	}

	result
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{PackedBinaryField1x1b, PackedBinaryField2x1b, PackedBinaryField4x1b};

	#[test]
	fn test_sequential_bits() {
		assert!(is_sequential_bytes::<PackedBinaryField8x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField32x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField64x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField128x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField256x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField512x1b>());

		assert!(is_sequential_bytes::<AESTowerField8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField1x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField2x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField4x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField8x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField16x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField32x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField64x8b>());

		assert!(is_sequential_bytes::<BinaryField128bGhash>());
		assert!(is_sequential_bytes::<PackedBinaryGhash1x128b>());
		assert!(is_sequential_bytes::<PackedBinaryGhash2x128b>());
		assert!(is_sequential_bytes::<PackedBinaryGhash4x128b>());

		assert!(!is_sequential_bytes::<PackedBinaryField1x1b>());
		assert!(!is_sequential_bytes::<PackedBinaryField2x1b>());
		assert!(!is_sequential_bytes::<PackedBinaryField4x1b>());
	}

	#[test]
	fn test_partial_sums_basic() {
		let v1 = AESTowerField8b::from(1);
		let v2 = AESTowerField8b::from(2);
		let v3 = AESTowerField8b::from(3);
		let v4 = AESTowerField8b::from(4);
		let v5 = AESTowerField8b::from(5);
		let v6 = AESTowerField8b::from(6);
		let v7 = AESTowerField8b::from(7);
		let v8 = AESTowerField8b::from(8);

		let values = vec![v1, v2, v3, v4, v5, v6, v7, v8];

		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		// Check specific precomputed sums
		assert_eq!(lookup_table[0b0000_0000], AESTowerField8b::from(0));
		assert_eq!(lookup_table[0b0000_0001], v1);
		assert_eq!(lookup_table[0b0000_0011], v1 + v2);
		assert_eq!(lookup_table[0b0000_0111], v1 + v2 + v3);
		assert_eq!(lookup_table[0b0000_1111], v1 + v2 + v3 + v4);
		assert_eq!(lookup_table[0b0001_1111], v1 + v2 + v3 + v4 + v5);
		assert_eq!(lookup_table[0b0011_1111], v1 + v2 + v3 + v4 + v5 + v6);
		assert_eq!(lookup_table[0b0111_1111], v1 + v2 + v3 + v4 + v5 + v6 + v7);
		assert_eq!(lookup_table[0b1111_1111], v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8);
	}

	#[test]
	fn test_partial_sums_all_zeros() {
		let values = vec![AESTowerField8b::from(0); 8];
		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		for &l in lookup_table.iter().take(256) {
			assert_eq!(l, AESTowerField8b::from(0));
		}
	}

	#[test]
	fn test_partial_sums_single_element() {
		let mut values = vec![AESTowerField8b::from(0); 8];
		// Set only the fourth element (index 3)
		values[3] = AESTowerField8b::from(10);

		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		// Only cases where the 4th bit is set should have non-zero sums
		assert_eq!(lookup_table[0b0000_0000], AESTowerField8b::from(0));
		assert_eq!(lookup_table[0b0000_1000], AESTowerField8b::from(10));
		assert_eq!(lookup_table[0b0000_1100], AESTowerField8b::from(10));
		assert_eq!(lookup_table[0b0001_1000], AESTowerField8b::from(10));
		assert_eq!(lookup_table[0b1111_1111], AESTowerField8b::from(10));
	}

	#[test]
	fn test_partial_sums_alternating_values() {
		let v1 = AESTowerField8b::from(10);
		let v2 = AESTowerField8b::from(20);
		let v3 = AESTowerField8b::from(30);
		let v4 = AESTowerField8b::from(40);

		let zero = AESTowerField8b::from(0);

		let values = vec![v1, zero, v2, zero, v3, zero, v4, zero];

		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		// Expect only the even indexed elements to contribute to the sum
		assert_eq!(lookup_table[0b0000_0000], zero);
		assert_eq!(lookup_table[0b0000_0001], v1);
		assert_eq!(lookup_table[0b0000_0101], v1 + v2);
		assert_eq!(lookup_table[0b0000_1111], v1 + v2);
		assert_eq!(lookup_table[0b1111_1111], v1 + v2 + v3 + v4);
	}
}
