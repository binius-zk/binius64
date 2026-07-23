// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Traits for packed field elements which support SIMD implementations.
//!
//! Interfaces are derived from [`plonky2`](https://github.com/mir-protocol/plonky2).

use std::{
	fmt::Debug,
	iter,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::iter::IterExtensions;
use bytemuck::Zeroable;

use super::{Random, arithmetic_traits::Square};
use crate::{BinaryField, Divisible, Maskable, WideMul, field::FieldOps};

/// A packed field represents a vector of underlying field elements.
///
/// Arithmetic operations on packed field elements can be accelerated with SIMD CPU instructions.
/// The vector width is a constant, `WIDTH`. This trait requires that the width must be a power of
/// two.
pub trait PackedField:
	Default
	+ Debug
	+ Clone
	+ Copy
	+ Eq
	+ Sized
	+ FieldOps
	+ Add<Self::Scalar, Output = Self>
	+ Sub<Self::Scalar, Output = Self>
	+ Mul<Self::Scalar, Output = Self>
	+ AddAssign<Self::Scalar>
	+ SubAssign<Self::Scalar>
	+ MulAssign<Self::Scalar>
	+ Send
	+ Sync
	+ Zeroable
	+ Random
	+ WideMul<Output: Debug + Send + Sync + 'static>
	+ 'static
	// A packed field divides into its `WIDTH` scalars. Scalar element access (`get`/`set` and
	// their `_unchecked` variants), broadcast, and the scalar iterators are all provided by this
	// supertrait.
	+ Divisible<Self::Scalar>
	// A packed field supports branchless per-lane masking over its scalars.
	+ Maskable<Self::Scalar>
{
	/// Base-2 logarithm of the number of field elements packed into one packed element.
	///
	/// This is the number of scalars the packed field divides into, i.e. its `Divisible` log-count.
	const LOG_WIDTH: usize = <Self as Divisible<Self::Scalar>>::LOG_N;

	/// The number of field elements packed into one packed element.
	///
	/// WIDTH is guaranteed to equal 2^LOG_WIDTH.
	const WIDTH: usize = 1 << Self::LOG_WIDTH;

	#[inline]
	fn into_iter(self) -> impl Iterator<Item = Self::Scalar> + Send + Clone {
		(0..Self::WIDTH).map_skippable(move |i|
			// Safety: `i` is always less than `WIDTH`
			unsafe { self.get_unchecked(i) })
	}

	#[inline]
	fn iter(&self) -> impl Iterator<Item = Self::Scalar> + Send + Clone + '_ {
		(0..Self::WIDTH).map_skippable(move |i|
			// Safety: `i` is always less than `WIDTH`
			unsafe { self.get_unchecked(i) })
	}

	#[inline]
	fn iter_slice(slice: &[Self]) -> impl Iterator<Item = Self::Scalar> + Send + Clone + '_ {
		slice.iter().flat_map(Self::iter)
	}

	/// Initialize zero position with `scalar`, set other elements to zero.
	#[inline(always)]
	fn set_single(scalar: Self::Scalar) -> Self {
		let mut result = Self::default();
		result.set(0, scalar);
		result
	}

	/// Construct a packed field element from a function that returns scalar values by index.
	fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self;

	/// Construct a packed field element from a sequence of scalars.
	///
	/// If the number of values in the sequence is less than the packing width, the remaining
	/// elements are set to zero. If greater than the packing width, the excess elements are
	/// ignored.
	#[inline]
	fn from_scalars(values: impl IntoIterator<Item = Self::Scalar>) -> Self {
		let mut result = Self::default();
		for (i, val) in values.into_iter().take(Self::WIDTH).enumerate() {
			result.set(i, val);
		}
		result
	}

	/// Returns the value to the power `exp`.
	fn pow(self, exp: u64) -> Self {
		let mut res = Self::one();
		for i in (0..64).rev() {
			res = Square::square(res);
			if ((exp >> i) & 1) == 1 {
				res.mul_assign(self)
			}
		}
		res
	}

	/// Interleaves blocks of this packed vector with another packed vector.
	///
	/// The operation can be seen as stacking the two vectors, dividing them into 2x2 matrices of
	/// blocks, where each block is 2^`log_block_width` elements, and transposing the matrices.
	///
	/// Consider this example, where `LOG_WIDTH` is 3 and `log_block_len` is 1:
	///     A = [a0, a1, a2, a3, a4, a5, a6, a7]
	///     B = [b0, b1, b2, b3, b4, b5, b6, b7]
	///
	/// The interleaved result is
	///     A' = [a0, a1, b0, b1, a4, a5, b4, b5]
	///     B' = [a2, a3, b2, b3, a6, a7, b6, b7]
	///
	/// ## Preconditions
	/// * `log_block_len` must be strictly less than `LOG_WIDTH`.
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self);

	/// Unzips interleaved blocks of this packed vector with another packed vector.
	///
	/// Consider this example, where `LOG_WIDTH` is 3 and `log_block_len` is 1:
	///    A = [a0, a1, b0, b1, a2, a3, b2, b3]
	///    B = [a4, a5, b4, b5, a6, a7, b6, b7]
	///
	/// The transposed result is
	///    A' = [a0, a1, a2, a3, a4, a5, a6, a7]
	///    B' = [b0, b1, b2, b3, b4, b5, b6, b7]
	///
	/// ## Preconditions
	/// * `log_block_len` must be strictly less than `LOG_WIDTH`.
	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self);

	/// Spread takes a block of elements within a packed field and repeats them to the full packing
	/// width.
	///
	/// Spread can be seen as an extension of the functionality of [`Divisible::broadcast`].
	///
	/// ## Examples
	///
	/// ```
	/// use binius_field::{BinaryField1b, PackedField, PackedBinaryField8x1b};
	///
	/// let input =
	///     PackedBinaryField8x1b::from_scalars([0, 1, 0, 1, 0, 1, 0, 1].map(BinaryField1b::from));
	/// assert_eq!(
	///     input.spread(0, 1),
	///     PackedBinaryField8x1b::from_scalars([1, 1, 1, 1, 1, 1, 1, 1].map(BinaryField1b::from))
	/// );
	/// assert_eq!(
	///     input.spread(1, 0),
	///     PackedBinaryField8x1b::from_scalars([0, 0, 0, 0, 1, 1, 1, 1].map(BinaryField1b::from))
	/// );
	/// assert_eq!(
	///     input.spread(2, 0),
	///     PackedBinaryField8x1b::from_scalars([0, 0, 1, 1, 0, 0, 1, 1].map(BinaryField1b::from))
	/// );
	/// assert_eq!(input.spread(3, 0), input);
	/// ```
	///
	/// ## Preconditions
	///
	/// * `log_block_len` must be less than or equal to `LOG_WIDTH`.
	/// * `block_idx` must be less than `2^(Self::LOG_WIDTH - log_block_len)`.
	#[inline]
	fn spread(self, log_block_len: usize, block_idx: usize) -> Self {
		assert!(log_block_len <= Self::LOG_WIDTH);
		assert!(block_idx < 1 << (Self::LOG_WIDTH - log_block_len));

		// Safety: is guaranteed by the preconditions.
		unsafe { self.spread_unchecked(log_block_len, block_idx) }
	}

	/// Unsafe version of [`Self::spread`].
	///
	/// # Safety
	/// The caller must ensure that `log_block_len` is less than or equal to `LOG_WIDTH` and
	/// `block_idx` is less than `2^(Self::LOG_WIDTH - log_block_len)`.
	#[inline]
	unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self {
		let block_len = 1 << log_block_len;
		let repeat = 1 << (Self::LOG_WIDTH - log_block_len);

		Self::from_scalars(
			self.iter()
				.skip(block_idx * block_len)
				.take(block_len)
				.flat_map(|elem| iter::repeat_n(elem, repeat)),
		)
	}
}

#[inline(always)]
pub fn get_packed_slice<P: PackedField>(packed: &[P], i: usize) -> P::Scalar {
	assert!(i >> P::LOG_WIDTH < packed.len(), "index out of bounds");

	unsafe { get_packed_slice_unchecked(packed, i) }
}

/// Returns the scalar at the given index without bounds checking.
/// # Safety
/// The caller must ensure that `i` is less than `P::WIDTH * packed.len()`.
#[inline(always)]
pub unsafe fn get_packed_slice_unchecked<P: PackedField>(packed: &[P], i: usize) -> P::Scalar {
	// TODO: Consider putting a get_in_slice method on Divisible

	// Safety:
	// - `i / P::WIDTH` is within the bounds of `packed` if `i` is less than `P::WIDTH *
	//   packed.len()`
	// - `i % P::WIDTH` is always less than `P::WIDTH
	unsafe {
		packed
			.get_unchecked(i >> P::LOG_WIDTH)
			.get_unchecked(i % P::WIDTH)
	}
}

/// Sets the scalar at the given index without bounds checking.
/// # Safety
/// The caller must ensure that `i` is less than `P::WIDTH * packed.len()`.
#[inline]
pub unsafe fn set_packed_slice_unchecked<P: PackedField>(
	packed: &mut [P],
	i: usize,
	scalar: P::Scalar,
) {
	// TODO: Consider putting a set_in_slice method on Divisible

	// Safety: if `i` is less than `P::WIDTH * packed.len()`, then
	// - `i / P::WIDTH` is within the bounds of `packed`
	// - `i % P::WIDTH` is always less than `P::WIDTH
	unsafe {
		packed
			.get_unchecked_mut(i >> P::LOG_WIDTH)
			.set_unchecked(i % P::WIDTH, scalar)
	}
}

/// A helper trait to make the generic bounds shorter
pub trait PackedBinaryField: PackedField<Scalar: BinaryField> {}

impl<PT> PackedBinaryField for PT where PT: PackedField<Scalar: BinaryField> {}

#[cfg(test)]
mod tests {
	use rand::prelude::*;

	use crate::{
		AESTowerField8b, BinaryField1b, BinaryField128bGhash, PackedAESBinaryField1x8b,
		PackedAESBinaryField16x8b, PackedAESBinaryField32x8b, PackedAESBinaryField64x8b,
		PackedBinaryField1x1b, PackedBinaryField2x1b, PackedBinaryField4x1b, PackedBinaryField8x1b,
		PackedBinaryField16x1b, PackedBinaryField32x1b, PackedBinaryField64x1b,
		PackedBinaryField128x1b, PackedBinaryField256x1b, PackedBinaryField512x1b,
		PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
		SlicedGhashSq1x256b, SlicedGhashSq2x256b, SlicedGhashSq4x256b,
	};

	trait PackedFieldTest {
		fn run<P: PackedField>(&self);
	}

	/// Run the test for all the packed fields defined in this crate.
	fn run_for_all_packed_fields(test: &impl PackedFieldTest) {
		// B1
		test.run::<BinaryField1b>();
		test.run::<PackedBinaryField1x1b>();
		test.run::<PackedBinaryField2x1b>();
		test.run::<PackedBinaryField4x1b>();
		test.run::<PackedBinaryField8x1b>();
		test.run::<PackedBinaryField16x1b>();
		test.run::<PackedBinaryField32x1b>();
		test.run::<PackedBinaryField64x1b>();
		test.run::<PackedBinaryField128x1b>();
		test.run::<PackedBinaryField256x1b>();
		test.run::<PackedBinaryField512x1b>();

		// AES
		test.run::<AESTowerField8b>();
		test.run::<PackedAESBinaryField1x8b>();
		test.run::<PackedAESBinaryField16x8b>();
		test.run::<PackedAESBinaryField32x8b>();
		test.run::<PackedAESBinaryField64x8b>();

		// GHASH
		test.run::<BinaryField128bGhash>();
		test.run::<PackedBinaryGhash1x128b>();
		test.run::<PackedBinaryGhash2x128b>();
		test.run::<PackedBinaryGhash4x128b>();

		// GHASH² in a sliced layout
		test.run::<SlicedGhashSq1x256b>();
		test.run::<SlicedGhashSq2x256b>();
		test.run::<SlicedGhashSq4x256b>();
	}

	fn check_value_iteration<P: PackedField>(mut rng: impl Rng) {
		let packed = P::random(&mut rng);
		let mut iter = packed.iter();
		for i in 0..P::WIDTH {
			assert_eq!(packed.get(i), iter.next().unwrap());
		}
		assert!(iter.next().is_none());
	}

	fn check_ref_iteration<P: PackedField>(mut rng: impl Rng) {
		let packed = P::random(&mut rng);
		let mut iter = packed.into_iter();
		for i in 0..P::WIDTH {
			assert_eq!(packed.get(i), iter.next().unwrap());
		}
		assert!(iter.next().is_none());
	}

	struct PackedFieldIterationTest;

	impl PackedFieldTest for PackedFieldIterationTest {
		fn run<P: PackedField>(&self) {
			let mut rng = StdRng::seed_from_u64(0);

			check_value_iteration::<P>(&mut rng);
			check_ref_iteration::<P>(&mut rng);
		}
	}

	#[test]
	fn test_iteration() {
		run_for_all_packed_fields(&PackedFieldIterationTest);
	}
}
