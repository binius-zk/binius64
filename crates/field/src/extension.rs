// Copyright 2023-2025 Irreducible Inc.

use std::{
	iter,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::{Field, error::Error};

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
	/// ## Pre-conditions
	///
	/// * `i` must be in the range [0, `Self::DEGREE`).
	#[inline]
	fn basis(i: usize) -> Self {
		Self::basis_checked(i).expect("pre-condition: 0 <= i < DEGREE")
	}

	/// For `0 <= i < DEGREE`, returns `i`-th basis field element.
	///
	/// This is the same operation as [`Self::basis`], but it returns an error result instead of
	/// panicking if the index is out of range.
	///
	/// ## Throws
	///
	/// * `Error::ExtensionDegreeMismatch` unless `i` is in the range [0, `Self::DEGREE`).
	fn basis_checked(i: usize) -> Result<Self, Error>;

	/// Create an extension field element from a slice of base field elements in order
	/// consistent with `basis(i)` return values.
	/// Potentially faster than taking an inner product with a vector of basis elements.
	#[inline]
	fn from_bases(base_elems: impl IntoIterator<Item = F>) -> Result<Self, Error> {
		Self::from_bases_sparse(base_elems, 0)
	}

	/// A specialized version of `from_bases` which assumes that only base field
	/// elements with indices dividing `2^log_stride` can be nonzero.
	///
	/// `base_elems` should have length at most `ceil(DEGREE / 2^LOG_STRIDE)`. Note that
	/// [`ExtensionField::from_bases`] is a special case of `from_bases_sparse` with `log_stride =
	/// 0`.
	fn from_bases_sparse(
		base_elems: impl IntoIterator<Item = F>,
		log_stride: usize,
	) -> Result<Self, Error>;

	/// Iterator over base field elements.
	fn iter_bases(&self) -> impl Iterator<Item = F>;

	/// Convert into an iterator over base field elements.
	fn into_iter_bases(self) -> impl Iterator<Item = F>;

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
	fn square_transpose(values: &mut [Self]) -> Result<(), Error>;
}

impl<F: Field> ExtensionField<F> for F {
	const LOG_DEGREE: usize = 0;

	#[inline(always)]
	fn basis_checked(i: usize) -> Result<Self, Error> {
		if i != 0 {
			return Err(Error::ExtensionDegreeMismatch);
		}
		Ok(Self::ONE)
	}

	#[inline(always)]
	fn from_bases_sparse(
		base_elems: impl IntoIterator<Item = F>,
		log_stride: usize,
	) -> Result<Self, Error> {
		let mut base_elems = base_elems.into_iter();
		if log_stride != 0 {
			return Err(Error::ExtensionDegreeMismatch);
		}

		match base_elems.next() {
			Some(elem) => Ok(elem),
			None => Ok(Self::ZERO),
		}
	}

	#[inline(always)]
	fn iter_bases(&self) -> impl Iterator<Item = F> {
		iter::once(*self)
	}

	#[inline(always)]
	fn into_iter_bases(self) -> impl Iterator<Item = F> {
		iter::once(self)
	}

	#[inline(always)]
	unsafe fn get_base_unchecked(&self, i: usize) -> F {
		debug_assert_eq!(i, 0);
		*self
	}

	#[inline]
	fn square_transpose(values: &mut [Self]) -> Result<(), Error> {
		if values.len() != 1 {
			return Err(Error::MismatchedLengths);
		}

		Ok(())
	}
}
