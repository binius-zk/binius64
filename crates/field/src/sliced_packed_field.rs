// Copyright 2026 The Binius Developers

//! A packed extension field in a *sliced* (struct-of-arrays) memory layout.
//!
//! An extension field element `x = c_0·β_0 + … + c_{N-1}·β_{N-1}` over a subfield `FSub` is a
//! vector of `N = DEGREE` subfield coordinates in the basis `{β_j}`. [`SlicedPackedField`] packs
//! `WIDTH` such extension scalars by storing each *coordinate* of every element in its own packed
//! subfield register:
//!
//! ```text
//! coords[0] = [ c_0(x_0), c_0(x_1), …, c_0(x_{WIDTH-1}) ]   // β_0 coordinate of every lane
//! coords[1] = [ c_1(x_0), c_1(x_1), …, c_1(x_{WIDTH-1}) ]   // β_1 coordinate of every lane
//! …
//! ```
//!
//! The coordinates of a single extension element are *not* adjacent in memory — hence "sliced".
//! This is the layout that lets a batch multiply run as a handful of packed subfield multiplies
//! over the whole batch (Karatsuba over the extension), instead of a schoolbook product per lane.
//!
//! # What is generic and what is not
//!
//! Everything that does not depend on the extension's multiplication rule is provided here,
//! generically, for any `PSub: PackedField` and any scalar `F: ExtensionField<PSub::Scalar>`:
//! scalar access, broadcast, iteration, addition, masking, interleave/unzip/spread, and
//! `square_transpose`. The layout makes these uniform: a lane permutation (interleave, spread) or a
//! bitwise op (add, mask) applies to each coordinate register identically, and scalar access reads
//! or writes the `N` coordinate registers at one lane through the [`ExtensionField`] basis.
//!
//! The field arithmetic is written per concrete extension: a type supplies a custom [`WideMul`]
//! (the widening multiply, with a deferred reduction), plus [`Square`] and [`InvertOrZero`]. `Mul`
//! is then blanket-implemented as `reduce(wide_mul(a, b))`, mirroring how the scalar fields are
//! defined. A concrete extension whose coordinate `PSub` is a [`PackedPrimitiveType`] can reach
//! into its underlier for optimizations a generic packed field cannot express. See
//! `packed_ghash_sq` for the GHASH² instantiation.
//!
//! # The `F` type parameter
//!
//! The scalar `F` is carried as a phantom parameter rather than derived from `(PSub, N)`. A degree
//! and a subfield do not name a unique extension, and Rust requires a type parameter used only as
//! `type Scalar = F` to appear in the self type. This mirrors [`PackedPrimitiveType<U, Scalar>`],
//! which likewise carries its scalar. The invariant `N == <F as ExtensionField<PSub::Scalar>>::
//! DEGREE` is upheld by the concrete type aliases.
//!
//! [`PackedPrimitiveType`]: crate::arch::PackedPrimitiveType
//! [`PackedPrimitiveType<U, Scalar>`]: crate::arch::PackedPrimitiveType

use std::{
	array,
	fmt::Debug,
	iter::{Product, Sum},
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bytemuck::Zeroable;
use rand::distr::{Distribution, StandardUniform};

use crate::{
	Divisible, ExtensionField, Field, Maskable, PackedField, WideMul,
	arithmetic_traits::{InvertOrZero, Square},
	field::FieldOps,
};

/// A packed extension field stored as `N` packed subfield coordinate registers.
///
/// `F` is the extension scalar, `PSub` the packed subfield holding one coordinate of every lane,
/// and `N = <F as ExtensionField<PSub::Scalar>>::DEGREE` the extension degree. See the module
/// documentation for the layout and for which operations are generic here versus supplied per
/// concrete extension.
///
/// Concrete packings are named through type aliases; see `packed_ghash_sq` for GHASH².
///
/// ```
/// use binius_field::{Divisible, Field, GhashSq256b, PackedField, SlicedGhashSq2x256b};
///
/// let scalars = [GhashSq256b::ONE, GhashSq256b::MULTIPLICATIVE_GENERATOR];
/// let a = SlicedGhashSq2x256b::from_scalars(scalars);
/// let squared = a * a;
/// for i in 0..SlicedGhashSq2x256b::WIDTH {
///     assert_eq!(squared.get(i), scalars[i] * scalars[i]);
/// }
/// ```
#[repr(transparent)]
pub struct SlicedPackedField<F, PSub, const N: usize>([PSub; N], PhantomData<F>);

// `Clone`/`Copy`/`Eq` are implemented by hand rather than derived: the `PhantomData<F>` field is
// always `Copy`, so these should bound only on `PSub` and not drag an `F: Copy` obligation into
// every generic impl.
impl<F, PSub: Copy, const N: usize> Clone for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<F, PSub: Copy, const N: usize> Copy for SlicedPackedField<F, PSub, N> {}

impl<F, PSub: PartialEq, const N: usize> PartialEq for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn eq(&self, other: &Self) -> bool {
		self.0 == other.0
	}
}

impl<F, PSub: Eq, const N: usize> Eq for SlicedPackedField<F, PSub, N> {}

impl<F, PSub: PackedField, const N: usize> SlicedPackedField<F, PSub, N> {
	/// Wraps `N` coordinate registers, where `coords[j]` holds the `β_j` coordinate of every lane.
	#[inline]
	pub const fn from_coords(coords: [PSub; N]) -> Self {
		Self(coords, PhantomData)
	}

	/// Unwraps the `N` coordinate registers.
	#[inline]
	pub const fn to_coords(self) -> [PSub; N] {
		self.0
	}
}

impl<F, PSub: PackedField, const N: usize> Default for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn default() -> Self {
		Self::from_coords([PSub::default(); N])
	}
}

impl<F, PSub: PackedField, const N: usize> Debug for SlicedPackedField<F, PSub, N> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "SlicedPacked<{N}>({:?})", self.0)
	}
}

// SAFETY: the struct is `[PSub; N]` plus a zero-sized `PhantomData`; an all-zero bit pattern is a
// valid `[PSub; N]` whenever `PSub: Zeroable`.
unsafe impl<F, PSub: Zeroable, const N: usize> Zeroable for SlicedPackedField<F, PSub, N> {}

// --- Additive group: coordinate-wise (a binary field is characteristic two, so neg is identity).

impl<F, PSub: PackedField, const N: usize> Neg for SlicedPackedField<F, PSub, N> {
	type Output = Self;

	#[inline]
	fn neg(self) -> Self {
		self
	}
}

impl<F, PSub: PackedField, const N: usize> Add for SlicedPackedField<F, PSub, N> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self::from_coords(array::from_fn(|j| self.0[j] + rhs.0[j]))
	}
}

impl<F, PSub: PackedField, const N: usize> Sub for SlicedPackedField<F, PSub, N> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self::from_coords(array::from_fn(|j| self.0[j] - rhs.0[j]))
	}
}

impl<F, PSub: PackedField, const N: usize> Add<&Self> for SlicedPackedField<F, PSub, N> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: &Self) -> Self {
		self + *rhs
	}
}

impl<F, PSub: PackedField, const N: usize> Sub<&Self> for SlicedPackedField<F, PSub, N> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: &Self) -> Self {
		self - *rhs
	}
}

impl<F, PSub: PackedField, const N: usize> AddAssign for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<F, PSub: PackedField, const N: usize> SubAssign for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<F, PSub: PackedField, const N: usize> AddAssign<&Self> for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn add_assign(&mut self, rhs: &Self) {
		*self = *self + *rhs;
	}
}

impl<F, PSub: PackedField, const N: usize> SubAssign<&Self> for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn sub_assign(&mut self, rhs: &Self) {
		*self = *self - *rhs;
	}
}

impl<F, PSub: PackedField, const N: usize> Sum for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

impl<'a, F, PSub: PackedField, const N: usize> Sum<&'a Self> for SlicedPackedField<F, PSub, N> {
	#[inline]
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + *x)
	}
}

// --- Multiply: blanket-implemented in terms of the per-extension `WideMul`.

impl<F, PSub: PackedField, const N: usize> Mul for SlicedPackedField<F, PSub, N>
where
	Self: WideMul,
{
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self {
		Self::reduce(Self::wide_mul(self, rhs))
	}
}

impl<F, PSub: PackedField, const N: usize> Mul<&Self> for SlicedPackedField<F, PSub, N>
where
	Self: Mul<Output = Self>,
{
	type Output = Self;

	#[inline]
	fn mul(self, rhs: &Self) -> Self {
		self * *rhs
	}
}

impl<F, PSub: PackedField, const N: usize> MulAssign for SlicedPackedField<F, PSub, N>
where
	Self: Mul<Output = Self>,
{
	#[inline]
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl<F, PSub: PackedField, const N: usize> MulAssign<&Self> for SlicedPackedField<F, PSub, N>
where
	Self: Mul<Output = Self>,
{
	#[inline]
	fn mul_assign(&mut self, rhs: &Self) {
		*self = *self * *rhs;
	}
}

impl<F, PSub: PackedField, const N: usize> Product for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	Self: Mul<Output = Self>,
{
	#[inline]
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(<Self as Divisible<F>>::broadcast(F::ONE), |acc, x| acc * x)
	}
}

impl<'a, F, PSub: PackedField, const N: usize> Product<&'a Self> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	Self: Mul<Output = Self>,
{
	#[inline]
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(<Self as Divisible<F>>::broadcast(F::ONE), |acc, x| acc * *x)
	}
}

// --- Scalar (extension-element) operations broadcast the scalar across every lane.

impl<F, PSub, const N: usize> Add<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
{
	type Output = Self;

	#[inline]
	fn add(self, rhs: F) -> Self {
		self + <Self as Divisible<F>>::broadcast(rhs)
	}
}

impl<F, PSub, const N: usize> Sub<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
{
	type Output = Self;

	#[inline]
	fn sub(self, rhs: F) -> Self {
		self - <Self as Divisible<F>>::broadcast(rhs)
	}
}

impl<F, PSub, const N: usize> Mul<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
	Self: Mul<Output = Self>,
{
	type Output = Self;

	#[inline]
	fn mul(self, rhs: F) -> Self {
		self * <Self as Divisible<F>>::broadcast(rhs)
	}
}

impl<F, PSub, const N: usize> AddAssign<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
{
	#[inline]
	fn add_assign(&mut self, rhs: F) {
		*self = *self + rhs;
	}
}

impl<F, PSub, const N: usize> SubAssign<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
{
	#[inline]
	fn sub_assign(&mut self, rhs: F) {
		*self = *self - rhs;
	}
}

impl<F, PSub, const N: usize> MulAssign<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
	Self: Mul<Output = Self>,
{
	#[inline]
	fn mul_assign(&mut self, rhs: F) {
		*self = *self * <Self as Divisible<F>>::broadcast(rhs);
	}
}

// --- Scalar access: a lane's extension scalar is read from / written to the `N` coordinates.

impl<F, PSub, const N: usize> Divisible<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
{
	// One extension scalar per subfield lane: the packing width is `PSub::WIDTH`.
	const LOG_N: usize = PSub::LOG_WIDTH;

	#[inline]
	fn value_iter(value: Self) -> impl ExactSizeIterator<Item = F> + Send + Clone {
		(0..Self::N).map(move |i| unsafe { value.get_unchecked(i) })
	}

	#[inline]
	fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = F> + Send + Clone + '_ {
		let value = *value;
		(0..Self::N).map(move |i| unsafe { value.get_unchecked(i) })
	}

	#[inline]
	fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = F> + Send + Clone + '_ {
		(0..slice.len() * Self::N).map(move |global| {
			let (elem, lane) = (global / Self::N, global % Self::N);
			// SAFETY: `lane < Self::N` by construction.
			unsafe { slice[elem].get_unchecked(lane) }
		})
	}

	#[inline]
	unsafe fn get_unchecked(&self, index: usize) -> F {
		// SAFETY: `index < Self::N == PSub::WIDTH` by the caller's contract, so each coordinate
		// access is in bounds.
		F::from_bases((0..N).map(|j| unsafe { self.0[j].get_unchecked(index) }))
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, index: usize, val: F) {
		for (j, coord) in self.0.iter_mut().enumerate() {
			// SAFETY: `index < Self::N == PSub::WIDTH`; `j < N == DEGREE` so `get_base(j)` is in
			// range.
			unsafe {
				coord.set_unchecked(index, val.get_base_unchecked(j));
			}
		}
	}

	#[inline]
	fn broadcast(val: F) -> Self {
		Self::from_coords(array::from_fn(|j| PSub::broadcast(val.get_base(j))))
	}

	#[inline]
	fn from_iter(mut iter: impl Iterator<Item = F>) -> Self {
		let mut result = Self::default();
		for i in 0..Self::N {
			match iter.next() {
				Some(val) => result.set(i, val),
				None => break,
			}
		}
		result
	}
}

// --- Lane masking applies the same per-lane mask to every coordinate.

impl<F, PSub, const N: usize> Maskable<F> for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
{
	type Mask = PSub::Mask;

	#[inline]
	fn make_mask(selectors: impl Iterator<Item = bool>) -> Self::Mask {
		PSub::make_mask(selectors)
	}

	#[inline]
	fn select(&self, mask: &Self::Mask) -> Self {
		Self::from_coords(array::from_fn(|j| self.0[j].select(mask)))
	}
}

impl<F, PSub, const N: usize> FieldOps for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
	Self: Square + InvertOrZero + Mul<Output = Self>,
{
	type Scalar = F;

	#[inline]
	fn zero() -> Self {
		Self::default()
	}

	#[inline]
	fn one() -> Self {
		<Self as Divisible<F>>::broadcast(F::ONE)
	}

	fn square_transpose<FSub: Field>(elems: &mut [Self])
	where
		F: ExtensionField<FSub>,
	{
		let degree = <F as ExtensionField<FSub>>::DEGREE;
		assert_eq!(elems.len(), degree);

		// Transpose the `degree × degree` matrix of `FSub` coordinates independently in each lane.
		// Reading a whole column before writing keeps the in-place update free of read-after-write
		// hazards within the lane.
		for lane in 0..PSub::WIDTH {
			let column = (0..degree).map(|j| elems[j].get(lane)).collect::<Vec<F>>();
			for (i, elem) in elems.iter_mut().enumerate() {
				let transposed = <F as ExtensionField<FSub>>::from_bases(
					(0..degree).map(|j| <F as ExtensionField<FSub>>::get_base(&column[j], i)),
				);
				elem.set(lane, transposed);
			}
		}
	}
}

impl<F, PSub, const N: usize> PackedField for SlicedPackedField<F, PSub, N>
where
	F: ExtensionField<PSub::Scalar>,
	PSub: PackedField,
	Self:
		Square + InvertOrZero + Mul<Output = Self> + WideMul<Output: Debug + Send + Sync + 'static>,
{
	// LOG_WIDTH defaults to `<Self as Divisible<F>>::LOG_N == PSub::LOG_WIDTH`; scalar access is
	// provided by the `Divisible<F>` impl above.

	#[inline]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_WIDTH);
		// The lane permutation is data-independent, so interleaving every coordinate register with
		// the same block length keeps each lane's coordinates together.
		let pairs: [(PSub, PSub); N] =
			array::from_fn(|j| self.0[j].interleave(other.0[j], log_block_len));
		(Self::from_coords(pairs.map(|(c, _)| c)), Self::from_coords(pairs.map(|(_, d)| d)))
	}

	#[inline]
	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_WIDTH);
		let pairs: [(PSub, PSub); N] =
			array::from_fn(|j| self.0[j].unzip(other.0[j], log_block_len));
		(Self::from_coords(pairs.map(|(c, _)| c)), Self::from_coords(pairs.map(|(_, d)| d)))
	}

	#[inline]
	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		let mut result = Self::default();
		for i in 0..Self::WIDTH {
			result.set(i, f(i));
		}
		result
	}

	#[inline]
	unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self {
		// Spread repeats a block of lanes; the same lane pattern applies to each coordinate.
		Self::from_coords(array::from_fn(|j| unsafe {
			self.0[j].spread_unchecked(log_block_len, block_idx)
		}))
	}
}

impl<F, PSub: PackedField, const N: usize> Distribution<SlicedPackedField<F, PSub, N>>
	for StandardUniform
{
	#[inline]
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> SlicedPackedField<F, PSub, N> {
		SlicedPackedField::from_coords(array::from_fn(|_| PSub::random(&mut *rng)))
	}
}
