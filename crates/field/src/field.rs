// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::Display,
	hash::Hash,
	iter::{self, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes, SerializeBytes};

use crate::{
	PackedField,
	arithmetic_traits::{InvertOrZero, Square},
};

/// An element of a finite field.
///
/// A finite field (also called a Galois field) has order `p^k` where `p` is the
/// [`CHARACTERISTIC`](Self::CHARACTERISTIC) and `k` is the
/// [`ORDER_EXPONENT`](Self::ORDER_EXPONENT).
pub trait Field:
	PackedField<Scalar = Self>
	+ Display
	+ Hash
	+ SerializeBytes
	+ DeserializeBytes
	+ FixedSizeSerializeBytes
{
	/// The zero element of the field, the additive identity.
	const ZERO: Self;

	/// The one element of the field, the multiplicative identity.
	const ONE: Self;

	/// The characteristic `p` of the field. The field order is `p^k` where `k` is
	/// [`ORDER_EXPONENT`](Self::ORDER_EXPONENT).
	const CHARACTERISTIC: usize;

	/// The exponent `k` such that the field order equals `CHARACTERISTIC^k`.
	const ORDER_EXPONENT: usize;

	/// Fixed generator of the multiplicative group.
	const MULTIPLICATIVE_GENERATOR: Self;

	/// Returns true iff this element is zero.
	fn is_zero(&self) -> bool {
		*self == Self::ZERO
	}

	/// Doubles this element.
	#[must_use]
	fn double(&self) -> Self;

	/// Exponentiates `self` by `exp`, where `exp` is a little-endian order integer
	/// exponent.
	fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
		let mut res = Self::ONE;
		for e in exp.as_ref().iter().rev() {
			for i in (0..64).rev() {
				res = res.square();

				if ((*e >> i) & 1) == 1 {
					res.mul_assign(self);
				}
			}
		}

		res
	}
}

/// Operations for types that represent vectors of field elements.
///
/// This trait abstracts over:
/// - [`Field`] types (single field elements, which are trivially vectors of length 1)
/// - [`PackedField`] types (SIMD-accelerated vectors of field elements)
/// - Symbolic field types (for constraint system representations)
///
/// Mathematically, instances of this trait represent vectors of field elements where
/// arithmetic operations like addition, subtraction, multiplication, squaring, and
/// inversion are defined element-wise. For a packed field with width N, multiplying
/// two values performs N independent field multiplications in parallel.
///
/// # Required Methods
///
/// - [`zero()`](Self::zero) - Returns the additive identity (all elements are zero)
/// - [`one()`](Self::one) - Returns the multiplicative identity (all elements are one)
pub trait FieldOps:
	Clone
	+ Neg<Output = Self>
	+ Add<Output = Self>
	+ Sub<Output = Self>
	+ Mul<Output = Self>
	+ Sum
	+ Product
	+ for<'a> Add<&'a Self, Output = Self>
	+ for<'a> Sub<&'a Self, Output = Self>
	+ for<'a> Mul<&'a Self, Output = Self>
	+ for<'a> Sum<&'a Self>
	+ for<'a> Product<&'a Self>
	+ AddAssign
	+ SubAssign
	+ MulAssign
	+ for<'a> AddAssign<&'a Self>
	+ for<'a> SubAssign<&'a Self>
	+ for<'a> MulAssign<&'a Self>
	+ Square
	+ InvertOrZero
{
	type Scalar: Field;

	/// Returns the zero element (additive identity).
	fn zero() -> Self;

	/// Returns the one element (multiplicative identity).
	fn one() -> Self;

	/// Transpose the subfield elements in a slice of field elements.
	///
	/// ## Arguments
	///
	/// * `elems` - a slice of $n$ elements, where $n$ is the degee of the extension of
	///   `Self::Scalar` over `FSub`. They are overwritten with the result elements.
	///
	/// ## Preconditions
	///
	/// * `elems.len()` must equal `Self::Scalar::DEGREE`
	fn square_transpose<FSub: Field>(elems: &mut [Self])
	where
		Self::Scalar: ExtensionField<FSub>;
}

impl<F: Field> FieldOps for F {
	type Scalar = F;

	fn zero() -> Self {
		Self::ZERO
	}

	fn one() -> Self {
		Self::ONE
	}

	fn square_transpose<FSub: Field>(elems: &mut [Self])
	where
		F: ExtensionField<FSub>,
	{
		<F as ExtensionField<FSub>>::square_transpose(elems);
	}
}

pub trait ExtensionField<F: Field>:
	Field
	+ From<F>
	+ TryInto<F>
	+ Add<F, Output = Self>
	+ Sub<F, Output = Self>
	+ Mul<F, Output = Self>
	+ AddAssign<F>
	+ SubAssign<F>
	+ MulAssign<F>
{
	/// Base-2 logarithm of the extension degree.
	const LOG_DEGREE: usize;

	/// Extension degree.
	///
	/// `DEGREE` is guaranteed to equal `2^LOG_DEGREE`.
	const DEGREE: usize = 1 << Self::LOG_DEGREE;

	/// For `0 <= i < DEGREE`, returns `i`-th basis field element.
	///
	/// # Preconditions
	///
	/// * `i` must be in the range [0, `Self::DEGREE`).
	fn basis(i: usize) -> Self;

	/// Create an extension field element from a slice of base field elements in order
	/// consistent with `basis(i)` return values.
	/// Potentially faster than taking an inner product with a vector of basis elements.
	///
	/// # Preconditions
	///
	/// * `base_elems` must have at most `DEGREE` elements.
	#[inline]
	fn from_bases(base_elems: impl IntoIterator<Item = F>) -> Self {
		Self::from_bases_sparse(base_elems, 0)
	}

	/// A specialized version of `from_bases` which assumes that only base field
	/// elements with indices dividing `2^log_stride` can be nonzero.
	///
	/// `base_elems` should have length at most `ceil(DEGREE / 2^LOG_STRIDE)`. Note that
	/// [`ExtensionField::from_bases`] is a special case of `from_bases_sparse` with `log_stride =
	/// 0`.
	///
	/// # Preconditions
	///
	/// * `log_stride` must be at most `LOG_DEGREE`.
	/// * `base_elems` must have at most `ceil(DEGREE / 2^log_stride)` elements.
	fn from_bases_sparse(base_elems: impl IntoIterator<Item = F>, log_stride: usize) -> Self;

	/// Iterator over base field elements.
	fn iter_bases(&self) -> impl Iterator<Item = F>;

	/// Returns the i-th base field element.
	#[inline]
	fn get_base(&self, i: usize) -> F {
		assert!(i < Self::DEGREE, "index out of bounds");
		unsafe { self.get_base_unchecked(i) }
	}

	/// Returns the i-th base field element without bounds checking.
	///
	/// # Safety
	/// `i` must be less than `DEGREE`.
	unsafe fn get_base_unchecked(&self, i: usize) -> F;

	/// Transpose square block of subfield elements within `values` in place.
	///
	/// # Preconditions
	///
	/// * `values.len()` must equal `DEGREE`.
	fn square_transpose(values: &mut [Self]);
}

impl<F: Field> ExtensionField<F> for F {
	const LOG_DEGREE: usize = 0;

	#[inline(always)]
	fn basis(i: usize) -> Self {
		assert!(i == 0, "index {i} out of range for degree 1");
		Self::ONE
	}

	#[inline(always)]
	fn from_bases_sparse(base_elems: impl IntoIterator<Item = F>, log_stride: usize) -> Self {
		assert!(log_stride == 0, "log_stride must be 0 for degree-1 extension");
		let mut base_elems = base_elems.into_iter();
		base_elems.next().unwrap_or(Self::ZERO)
	}

	#[inline(always)]
	fn iter_bases(&self) -> impl Iterator<Item = F> {
		iter::once(*self)
	}

	#[inline(always)]
	unsafe fn get_base_unchecked(&self, i: usize) -> F {
		debug_assert_eq!(i, 0);
		*self
	}

	#[inline]
	fn square_transpose(values: &mut [Self]) {
		assert!(values.len() == 1, "values.len() must be 1 for degree-1 extension");
	}
}
