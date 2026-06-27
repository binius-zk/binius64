// Copyright 2026 The Binius Developers

//! Packed [`GhashSq256b`] in a sliced (struct-of-arrays) memory layout.
//!
//! A [`GhashSq256b`] element is `a + b*Y` over GHASH, with `Y^2 = Y + X^-1`.
//! The sliced layout stores the two coordinates of `WIDTH` elements as `[P; 2]`:
//! - `coords[0]` packs the `a` coordinate (coefficient of `1`) of every lane.
//! - `coords[1]` packs the `b` coordinate (coefficient of `Y`) of every lane.
//!
//! The two coordinates of one element are not adjacent in memory, hence "sliced".
//! This is what lets the multiply batch its GHASH products across all lanes at once.

use std::{
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bytemuck::Zeroable;
use rand::Rng;

use crate::{
	BinaryField128bGhash, Divisible, ExtensionField, FieldOps, GhashSq256b, Maskable, PackedField,
	Random, WideMul,
	arithmetic_traits::{InvertOrZero, Square},
};

/// The inverse of the GHASH generator `x`, as a GHASH field element.
///
/// Multiplying by this constant is the `mul_inv_x` map that folds `Y^2 = Y + X^-1` into the `{1,
/// Y}` basis. Its value is `mul_inv_x(1) = 0x43 + x^127` (bits 0, 1, 6, and 127).
const GHASH_INV_X: BinaryField128bGhash =
	BinaryField128bGhash::new(0x80000000000000000000000000000043);

/// Multiplies every GHASH lane of `p` by `X^-1`, the inverse of the GHASH generator.
#[inline]
fn ghash_mul_inv_x<P: PackedField<Scalar = BinaryField128bGhash>>(p: P) -> P {
	// `p * X^-1` per lane; the scalar multiply broadcasts the constant across lanes.
	p * GHASH_INV_X
}

/// A packed vector of [`GhashSq256b`] elements stored in sliced (struct-of-arrays) layout.
///
/// The packing width equals that of the inner GHASH packing `P`.
/// Each lane `i` holds the element `coords[0].get(i) + coords[1].get(i) * Y`.
///
/// Multiplication is the sliced Karatsuba algorithm over GHASH:
/// for `x = x_0 + x_1*Y` and `y = y_0 + y_1*Y` with `Y^2 = Y + X^-1`,
/// - `z_0 = x_0*y_0 + (x_1*y_1)*X^-1`
/// - `z_1 = (x_0+x_1)*(y_0+y_1) + x_0*y_0`
///
/// Each of the three GHASH products runs once across all lanes, so the cost amortizes over `WIDTH`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PackedSlicedGhashSq<P: PackedField<Scalar = BinaryField128bGhash>>([P; 2]);

unsafe impl<P: PackedField<Scalar = BinaryField128bGhash>> Zeroable for PackedSlicedGhashSq<P> {}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Neg for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn neg(self) -> Self {
		// Characteristic two: negation is the identity.
		self
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Add for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		// Addition is coordinatewise XOR of the two sliced coordinates.
		Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Sub for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		// Characteristic two: subtraction is coordinatewise XOR, the same as addition.
		Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Mul for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self {
		let [x0, x1] = self.0;
		let [y0, y1] = rhs.0;

		// Karatsuba over GHASH: three packed GHASH products across all lanes.
		//
		//     t0 = x_0 * y_0
		//     t2 = x_1 * y_1
		//     t1 = (x_0 + x_1) * (y_0 + y_1)
		let t0 = x0 * y0;
		let t2 = x1 * y1;
		let t1 = (x0 + x1) * (y0 + y1);

		// Fold `Y^2 = Y + X^-1` into the basis: z_0 = t0 + t2*X^-1, z_1 = t1 + t0.
		let z0 = t0 + ghash_mul_inv_x(t2);
		let z1 = t1 + t0;
		Self([z0, z1])
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> AddAssign for PackedSlicedGhashSq<P> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> SubAssign for PackedSlicedGhashSq<P> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> MulAssign for PackedSlicedGhashSq<P> {
	#[inline]
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Add<&Self> for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: &Self) -> Self {
		self + *rhs
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Sub<&Self> for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: &Self) -> Self {
		self - *rhs
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Mul<&Self> for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: &Self) -> Self {
		self * *rhs
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> AddAssign<&Self> for PackedSlicedGhashSq<P> {
	#[inline]
	fn add_assign(&mut self, rhs: &Self) {
		*self = *self + *rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> SubAssign<&Self> for PackedSlicedGhashSq<P> {
	#[inline]
	fn sub_assign(&mut self, rhs: &Self) {
		*self = *self - *rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> MulAssign<&Self> for PackedSlicedGhashSq<P> {
	#[inline]
	fn mul_assign(&mut self, rhs: &Self) {
		*self = *self * *rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Sum for PackedSlicedGhashSq<P> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

impl<'a, P: PackedField<Scalar = BinaryField128bGhash>> Sum<&'a Self> for PackedSlicedGhashSq<P> {
	#[inline]
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.copied().sum()
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Product for PackedSlicedGhashSq<P> {
	#[inline]
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		// The identity is `1 + 0*Y`: a-coordinate all ones, b-coordinate all zeros.
		iter.fold(Self([P::one(), P::zero()]), |acc, x| acc * x)
	}
}

impl<'a, P: PackedField<Scalar = BinaryField128bGhash>> Product<&'a Self>
	for PackedSlicedGhashSq<P>
{
	#[inline]
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.copied().product()
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Add<GhashSq256b> for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: GhashSq256b) -> Self {
		self + Self::broadcast(rhs)
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Sub<GhashSq256b> for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: GhashSq256b) -> Self {
		self - Self::broadcast(rhs)
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Mul<GhashSq256b> for PackedSlicedGhashSq<P> {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: GhashSq256b) -> Self {
		self * Self::broadcast(rhs)
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> AddAssign<GhashSq256b>
	for PackedSlicedGhashSq<P>
{
	#[inline]
	fn add_assign(&mut self, rhs: GhashSq256b) {
		*self = *self + rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> SubAssign<GhashSq256b>
	for PackedSlicedGhashSq<P>
{
	#[inline]
	fn sub_assign(&mut self, rhs: GhashSq256b) {
		*self = *self - rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> MulAssign<GhashSq256b>
	for PackedSlicedGhashSq<P>
{
	#[inline]
	fn mul_assign(&mut self, rhs: GhashSq256b) {
		*self = *self * rhs;
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Square for PackedSlicedGhashSq<P> {
	#[inline]
	fn square(self) -> Self {
		let [x0, x1] = self.0;

		// Characteristic two kills the cross term: (x_0 + x_1*Y)^2 = (x_0^2 + x_1^2*X^-1) +
		// x_1^2*Y.
		let t0 = Square::square(x0);
		let t2 = Square::square(x1);
		Self([t0 + ghash_mul_inv_x(t2), t2])
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> InvertOrZero for PackedSlicedGhashSq<P> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		let [a, b] = self.0;

		// The conjugate of `a + b*Y` under `Y -> Y + 1` is `(a + b) + b*Y`.
		// Its norm `N = a^2 + a*b + b^2*X^-1` lies in GHASH.
		// The inverse is the conjugate times `N^-1`.
		let norm = Square::square(a) + a * b + ghash_mul_inv_x(Square::square(b));
		let norm_inv = norm.invert_or_zero();

		// A zero element gives norm 0, so packed `invert_or_zero` returns 0 lanes, hence 0 here.
		Self([(a + b) * norm_inv, b * norm_inv])
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> FieldOps for PackedSlicedGhashSq<P> {
	type Scalar = GhashSq256b;

	#[inline]
	fn zero() -> Self {
		Self([P::zero(), P::zero()])
	}

	#[inline]
	fn one() -> Self {
		// `1 + 0*Y`: a-coordinate all ones, b-coordinate all zeros.
		Self([P::one(), P::zero()])
	}

	fn square_transpose<FSub: crate::Field>(elems: &mut [Self])
	where
		GhashSq256b: ExtensionField<FSub>,
	{
		let degree = <GhashSq256b as ExtensionField<FSub>>::DEGREE;
		assert_eq!(elems.len(), degree);

		// The transpose runs independently per lane.
		// Each lane's `degree` scalars are gathered, transposed by the scalar routine, then written
		// back. The scratch buffer holds one column at a time.
		let mut column = vec![GhashSq256b::default(); degree];
		for lane in 0..P::WIDTH {
			for (slot, elem) in column.iter_mut().zip(elems.iter()) {
				*slot = elem.get(lane);
			}
			<GhashSq256b as ExtensionField<FSub>>::square_transpose(&mut column);
			for (elem, value) in elems.iter_mut().zip(column.iter()) {
				elem.set(lane, *value);
			}
		}
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Divisible<GhashSq256b>
	for PackedSlicedGhashSq<P>
{
	// One GhashSq lane per GHASH lane of the inner packing.
	const LOG_N: usize = P::LOG_WIDTH;

	#[inline]
	fn value_iter(value: Self) -> impl ExactSizeIterator<Item = GhashSq256b> + Send + Clone {
		let [a, b] = value.0;
		// Zip the two coordinate streams lane by lane into `a + b*Y`.
		P::value_iter(a)
			.zip(P::value_iter(b))
			.map(|(a, b)| GhashSq256b::from_bases([a, b]))
	}

	#[inline]
	fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = GhashSq256b> + Send + Clone + '_ {
		P::ref_iter(&value.0[0])
			.zip(P::ref_iter(&value.0[1]))
			.map(|(a, b)| GhashSq256b::from_bases([a, b]))
	}

	#[inline]
	fn slice_iter(
		slice: &[Self],
	) -> impl ExactSizeIterator<Item = GhashSq256b> + Send + Clone + '_ {
		// Lanes run element by element, each element contributing its `P::WIDTH` lanes in order.
		let total = slice.len() * P::WIDTH;
		(0..total).map(move |i| {
			let elem = &slice[i / P::WIDTH];
			let lane = i % P::WIDTH;
			GhashSq256b::from_bases([elem.0[0].get(lane), elem.0[1].get(lane)])
		})
	}

	#[inline]
	unsafe fn get_unchecked(&self, index: usize) -> GhashSq256b {
		// Safety: `index < Self::N = P::WIDTH` by the caller's contract.
		let a = unsafe { self.0[0].get_unchecked(index) };
		let b = unsafe { self.0[1].get_unchecked(index) };
		GhashSq256b::from_bases([a, b])
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, index: usize, val: GhashSq256b) {
		// Split the element into its `(a, b)` coordinates and write each into its coordinate slice.
		let a = val.get_base(0);
		let b = val.get_base(1);
		// Safety: `index < Self::N = P::WIDTH` by the caller's contract.
		unsafe {
			self.0[0].set_unchecked(index, a);
			self.0[1].set_unchecked(index, b);
		}
	}

	#[inline]
	fn broadcast(val: GhashSq256b) -> Self {
		// Broadcast each coordinate of the element across all lanes of its coordinate slice.
		Self([P::broadcast(val.get_base(0)), P::broadcast(val.get_base(1))])
	}

	#[inline]
	fn from_iter(iter: impl Iterator<Item = GhashSq256b>) -> Self {
		// Fill lanes from the iterator, leaving any unfilled lanes zero.
		let mut result = Self::default();
		for (i, val) in iter.take(P::WIDTH).enumerate() {
			result.set(i, val);
		}
		result
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Maskable<GhashSq256b>
	for PackedSlicedGhashSq<P>
{
	// A GhashSq lane and the GHASH lane it is built from share an index, so the inner mask applies.
	type Mask = <P as Maskable<BinaryField128bGhash>>::Mask;

	#[inline]
	fn make_mask(selectors: impl Iterator<Item = bool>) -> Self::Mask {
		P::make_mask(selectors)
	}

	#[inline]
	fn select(&self, mask: &Self::Mask) -> Self {
		// Masking a GhashSq lane masks both of its coordinates with the same per-lane mask.
		Self([self.0[0].select(mask), self.0[1].select(mask)])
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> WideMul for PackedSlicedGhashSq<P> {
	// The reduction is already folded into the multiply, so the wide product is the element itself.
	type Output = Self;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self {
		a * b
	}

	#[inline]
	fn reduce(wide: Self) -> Self {
		wide
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> Random for PackedSlicedGhashSq<P> {
	#[inline]
	fn random(mut rng: impl Rng) -> Self {
		Self([P::random(&mut rng), P::random(&mut rng)])
	}
}

impl<P: PackedField<Scalar = BinaryField128bGhash>> PackedField for PackedSlicedGhashSq<P> {
	#[inline]
	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		let mut result = Self::default();
		for i in 0..P::WIDTH {
			result.set(i, f(i));
		}
		result
	}

	#[inline]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		// Lanes line up with the inner packing, so interleave each coordinate slice identically.
		let (a0, a1) = self.0[0].interleave(other.0[0], log_block_len);
		let (b0, b1) = self.0[1].interleave(other.0[1], log_block_len);
		(Self([a0, b0]), Self([a1, b1]))
	}

	#[inline]
	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let (a0, a1) = self.0[0].unzip(other.0[0], log_block_len);
		let (b0, b1) = self.0[1].unzip(other.0[1], log_block_len);
		(Self([a0, b0]), Self([a1, b1]))
	}

	#[inline]
	unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self {
		// Safety: the preconditions on `log_block_len`/`block_idx` are forwarded unchanged to `P`.
		unsafe {
			Self([
				self.0[0].spread_unchecked(log_block_len, block_idx),
				self.0[1].spread_unchecked(log_block_len, block_idx),
			])
		}
	}
}

/// Sliced packing of a single [`GhashSq256b`] element (`WIDTH = 1`).
pub type PackedGhashSq1x256b = PackedSlicedGhashSq<crate::PackedBinaryGhash1x128b>;

/// Sliced packing of two [`GhashSq256b`] elements (`WIDTH = 2`).
pub type PackedGhashSq2x256b = PackedSlicedGhashSq<crate::PackedBinaryGhash2x128b>;

/// Sliced packing of four [`GhashSq256b`] elements (`WIDTH = 4`).
pub type PackedGhashSq4x256b = PackedSlicedGhashSq<crate::PackedBinaryGhash4x128b>;

#[cfg(test)]
mod tests {
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	// Builds a GhashSq256b from its `(a, b)` GHASH coordinates.
	fn ghash_sq(a: u128, b: u128) -> GhashSq256b {
		GhashSq256b::from_bases([BinaryField128bGhash::new(a), BinaryField128bGhash::new(b)])
	}

	// Strategy producing a vector of `width` random GhashSq256b scalars.
	fn arb_scalars(width: usize) -> impl Strategy<Value = Vec<GhashSq256b>> {
		prop::collection::vec(any::<[u128; 2]>().prop_map(|[a, b]| ghash_sq(a, b)), width)
	}

	// Pins the packed operation to the scalar reference, lane by lane, for one packing P.
	//
	// Fixture: pack `width` random scalars per operand, run the packed op, then check every lane
	// equals the scalar GhashSq256b op on the corresponding inputs.
	fn check_against_scalar<P: PackedField<Scalar = BinaryField128bGhash>>() {
		let width = PackedSlicedGhashSq::<P>::WIDTH;
		let mut runner = proptest::test_runner::TestRunner::deterministic();

		runner
			.run(&(arb_scalars(width), arb_scalars(width)), |(xs, ys)| {
				// Pack the scalar inputs into the sliced layout.
				let px = PackedSlicedGhashSq::<P>::from_scalars(xs.iter().copied());
				let py = PackedSlicedGhashSq::<P>::from_scalars(ys.iter().copied());

				// Multiply: each lane must equal the scalar product of its inputs.
				let prod = px * py;
				for i in 0..width {
					prop_assert_eq!(prod.get(i), xs[i] * ys[i]);
				}

				// Square: each lane must equal the scalar square of its input.
				let sq = Square::square(px);
				for i in 0..width {
					prop_assert_eq!(sq.get(i), Square::square(xs[i]));
				}

				// Invert-or-zero: each lane inverts independently, with 0 mapping to 0.
				let inv = px.invert_or_zero();
				for i in 0..width {
					prop_assert_eq!(inv.get(i), xs[i].invert_or_zero());
				}

				Ok(())
			})
			.unwrap();
	}

	#[test]
	fn mul_square_invert_match_scalar() {
		// Exercise every concrete width against the scalar GhashSq256b reference.
		check_against_scalar::<crate::PackedBinaryGhash1x128b>();
		check_against_scalar::<crate::PackedBinaryGhash2x128b>();
		check_against_scalar::<crate::PackedBinaryGhash4x128b>();
	}

	// Packs scalars, reads them back, and checks the round trip preserves every lane.
	fn check_get_set<P: PackedField<Scalar = BinaryField128bGhash>>(mut rng: impl rand::Rng) {
		let width = PackedSlicedGhashSq::<P>::WIDTH;

		// from_scalars then get reproduces each input scalar.
		let scalars: Vec<GhashSq256b> = (0..width).map(|_| GhashSq256b::random(&mut rng)).collect();
		let packed = PackedSlicedGhashSq::<P>::from_scalars(scalars.iter().copied());
		for i in 0..width {
			assert_eq!(packed.get(i), scalars[i]);
		}

		// set overwrites a single lane, leaving the others intact.
		let mut packed = packed;
		let replacement = GhashSq256b::random(&mut rng);
		packed.set(0, replacement);
		assert_eq!(packed.get(0), replacement);
		for i in 1..width {
			assert_eq!(packed.get(i), scalars[i]);
		}

		// broadcast fills every lane with the same scalar.
		let b = PackedSlicedGhashSq::<P>::broadcast(replacement);
		for i in 0..width {
			assert_eq!(b.get(i), replacement);
		}
	}

	#[test]
	fn get_set_broadcast_round_trip() {
		let mut rng = StdRng::seed_from_u64(0);
		check_get_set::<crate::PackedBinaryGhash1x128b>(&mut rng);
		check_get_set::<crate::PackedBinaryGhash2x128b>(&mut rng);
		check_get_set::<crate::PackedBinaryGhash4x128b>(&mut rng);
	}

	#[test]
	fn arithmetic_identities() {
		let mut rng = StdRng::seed_from_u64(1);

		// Width-2 packing: identity, distributivity, and characteristic-two facts.
		let a = PackedGhashSq2x256b::random(&mut rng);
		let b = PackedGhashSq2x256b::random(&mut rng);
		let c = PackedGhashSq2x256b::random(&mut rng);

		// Multiplicative identity.
		assert_eq!(a * <PackedGhashSq2x256b as FieldOps>::one(), a);

		// Distributivity of multiplication over addition.
		assert_eq!(a * (b + c), a * b + a * c);

		// Characteristic two: addition is its own inverse, negation is the identity.
		assert_eq!(a + a, PackedGhashSq2x256b::default());
		assert_eq!(-a, a);

		// Squaring agrees with self-multiplication.
		assert_eq!(Square::square(a), a * a);

		// A nonzero element times its inverse is one.
		let inv = a.invert_or_zero();
		assert_eq!(a * inv, <PackedGhashSq2x256b as FieldOps>::one());
	}

	#[test]
	fn interleave_unzip_are_inverse() {
		let mut rng = StdRng::seed_from_u64(2);

		// Width-2 packing has a single valid block length: log_block_len = 0.
		let a = PackedGhashSq2x256b::random(&mut rng);
		let b = PackedGhashSq2x256b::random(&mut rng);

		// Interleave then unzip at the same block length restores the inputs.
		let (i0, i1) = a.interleave(b, 0);
		let (u0, u1) = i0.unzip(i1, 0);
		assert_eq!(u0, a);
		assert_eq!(u1, b);
	}

	// Independent oracle: transpose the `degree x degree` matrix whose row `i` is the GHASH-basis
	// expansion of `column[i]`, built only from the separately-tested basis accessors.
	fn naive_transpose_column(column: &[GhashSq256b]) -> Vec<GhashSq256b> {
		let degree = column.len();
		// Flatten to the row-major coordinate grid: coords[i*degree + j] is base `j` of element
		// `i`.
		let coords: Vec<BinaryField128bGhash> = column
			.iter()
			.flat_map(ExtensionField::<BinaryField128bGhash>::iter_bases)
			.collect();
		// Read the grid column-major to produce the transposed elements.
		(0..degree)
			.map(|i| GhashSq256b::from_bases((0..degree).map(|j| coords[j * degree + i])))
			.collect()
	}

	// Checks that the packed transpose matches the per-lane naive transpose for one packing P.
	fn check_square_transpose<P: PackedField<Scalar = BinaryField128bGhash>>(
		mut rng: impl rand::Rng,
	) {
		let width = PackedSlicedGhashSq::<P>::WIDTH;
		// GhashSq256b is a degree-2 extension of GHASH, so the transposed block is 2 x 2.
		let degree = <GhashSq256b as ExtensionField<BinaryField128bGhash>>::DEGREE;

		// Build `degree` packed elements of random scalars.
		let mut packed: Vec<PackedSlicedGhashSq<P>> = (0..degree)
			.map(|_| PackedSlicedGhashSq::<P>::random(&mut rng))
			.collect();

		// Expected: transpose each lane's column independently with the naive oracle.
		let mut expected = packed.clone();
		for lane in 0..width {
			let column: Vec<GhashSq256b> = expected.iter().map(|e| e.get(lane)).collect();
			let transposed = naive_transpose_column(&column);
			for (elem, value) in expected.iter_mut().zip(transposed) {
				elem.set(lane, value);
			}
		}

		// The packed transpose must agree lane for lane.
		<PackedSlicedGhashSq<P> as FieldOps>::square_transpose::<BinaryField128bGhash>(&mut packed);
		assert_eq!(packed, expected);
	}

	#[test]
	fn square_transpose_matches_naive() {
		let mut rng = StdRng::seed_from_u64(3);
		check_square_transpose::<crate::PackedBinaryGhash1x128b>(&mut rng);
		check_square_transpose::<crate::PackedBinaryGhash2x128b>(&mut rng);
		check_square_transpose::<crate::PackedBinaryGhash4x128b>(&mut rng);
	}
}
