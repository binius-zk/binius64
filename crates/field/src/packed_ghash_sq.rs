// Copyright 2026 The Binius Developers

//! Packed [`GhashSq256b`] as a [`PackedPrimitiveType`] view over a bit-vector underlier.
//!
//! A [`GhashSq256b`] element is `a + b*Y` over GHASH, with `Y^2 = Y + X^-1`.
//!
//! Lane `i` of a packing occupies the `i`-th 256-bit window:
//! - the low 128 bits hold the `a` coordinate,
//! - the high 128 bits hold the `b` coordinate.
//!
//! Only the field arithmetic is defined here.
//! Every other operation is inherited from the generic [`PackedPrimitiveType`] implementation.
//!
//! The multiply batches its three GHASH products across all lanes:
//! - read the underlier as 128-bit coordinate lanes `[a_0, b_0, a_1, b_1, ...]`,
//! - regroup the even lanes into an all-`a` slice and the odd lanes into an all-`b` slice,
//! - run the three Karatsuba GHASH products over those slices,
//! - interleave the result coordinates back into the packed layout.

use crate::{
	BinaryField128bGhash, GhashSq256b, PackedField, WideMul,
	arch::{M128, M256, M512, PackedPrimitiveType},
	arithmetic_traits::{InvertOrZero, Square},
	underlier::{Divisible, ScaledUnderlier, UnderlierType},
};

/// The inverse of the GHASH generator `x`, as a GHASH field element.
///
/// Multiplying by this constant folds `Y^2 = Y + X^-1` back into the `{1, Y}` basis.
/// Its value `0x43 + x^127` sets bits 0, 1, 6 and 127.
const GHASH_INV_X: BinaryField128bGhash =
	BinaryField128bGhash::new(0x80000000000000000000000000000043);

/// The 1024-bit underlier backing a width-4 [`GhashSq256b`] packing.
///
/// No native 1024-bit register exists, so it is always two 512-bit halves.
type M1024 = ScaledUnderlier<M512, 2>;

/// Multiplies every GHASH lane of `p` by `X^-1`, the inverse of the GHASH generator.
#[inline]
fn mul_inv_x<P: PackedField<Scalar = BinaryField128bGhash>>(p: P) -> P {
	// The scalar multiply broadcasts the constant across every lane.
	p * GHASH_INV_X
}

/// Splits a packed [`GhashSq256b`] underlier into its all-`a` and all-`b` GHASH slices.
///
/// The underlier reads as 128-bit coordinate lanes `[a_0, b_0, a_1, b_1, ...]`.
/// The even lanes become the all-`a` slice, the odd lanes the all-`b` slice.
/// Each slice is repacked as a half-width GHASH packing.
#[inline]
fn split<OU, HU>(
	g: PackedPrimitiveType<OU, GhashSq256b>,
) -> (
	PackedPrimitiveType<HU, BinaryField128bGhash>,
	PackedPrimitiveType<HU, BinaryField128bGhash>,
)
where
	OU: UnderlierType + Divisible<M128>,
	HU: UnderlierType + Divisible<M128>,
	PackedPrimitiveType<HU, BinaryField128bGhash>: PackedField<Scalar = BinaryField128bGhash>,
{
	let underlier = g.to_underlier();
	let a = <HU as Divisible<M128>>::from_iter(Divisible::<M128>::value_iter(underlier).step_by(2));
	let b = <HU as Divisible<M128>>::from_iter(
		Divisible::<M128>::value_iter(underlier).skip(1).step_by(2),
	);
	(PackedPrimitiveType::from_underlier(a), PackedPrimitiveType::from_underlier(b))
}

/// Rejoins all-`a` and all-`b` GHASH slices into a packed [`GhashSq256b`] underlier.
///
/// Inverse of [`split`]: interleaves the coordinate lanes back into `[a_0, b_0, a_1, b_1, ...]`.
#[inline]
fn join<OU, HU>(
	a: PackedPrimitiveType<HU, BinaryField128bGhash>,
	b: PackedPrimitiveType<HU, BinaryField128bGhash>,
) -> PackedPrimitiveType<OU, GhashSq256b>
where
	OU: UnderlierType + Divisible<M128>,
	HU: UnderlierType + Divisible<M128>,
	PackedPrimitiveType<HU, BinaryField128bGhash>: PackedField<Scalar = BinaryField128bGhash>,
{
	let a_lanes = Divisible::<M128>::value_iter(a.to_underlier());
	let b_lanes = Divisible::<M128>::value_iter(b.to_underlier());
	let interleaved = a_lanes.zip(b_lanes).flat_map(|(a, b)| [a, b]);
	let underlier = <OU as Divisible<M128>>::from_iter(interleaved);
	PackedPrimitiveType::from_underlier(underlier)
}

/// Multiplies two packed [`GhashSq256b`] values via Karatsuba over GHASH.
///
/// For `x = x_0 + x_1*Y` and `y = y_0 + y_1*Y`, with `Y^2 = Y + X^-1`:
///
/// ```text
/// z_0 = x_0*y_0 + (x_1*y_1)*X^-1
/// z_1 = (x_0 + x_1)*(y_0 + y_1) + x_0*y_0
/// ```
///
/// Each of the three GHASH products runs once across all lanes, so the cost amortizes over `WIDTH`.
#[inline]
fn mul_impl<OU, HU>(
	x: PackedPrimitiveType<OU, GhashSq256b>,
	y: PackedPrimitiveType<OU, GhashSq256b>,
) -> PackedPrimitiveType<OU, GhashSq256b>
where
	OU: UnderlierType + Divisible<M128>,
	HU: UnderlierType + Divisible<M128>,
	PackedPrimitiveType<HU, BinaryField128bGhash>: PackedField<Scalar = BinaryField128bGhash>,
{
	let (x0, x1) = split(x);
	let (y0, y1) = split(y);

	// Three packed GHASH products:
	//     t0 = x_0 * y_0
	//     t2 = x_1 * y_1
	//     t1 = (x_0 + x_1) * (y_0 + y_1)   carries the cross term x_0*y_1 + x_1*y_0
	let t0 = x0 * y0;
	let t2 = x1 * y1;
	let t1 = (x0 + x1) * (y0 + y1);

	// Fold `Y^2 = Y + X^-1` into the basis.
	// The two `t2` (cross term and `Y^2`) cancel in characteristic two, so z_1 = t1 + t0.
	let z0 = t0 + mul_inv_x(t2);
	let z1 = t1 + t0;
	join(z0, z1)
}

/// Squares a packed [`GhashSq256b`] value.
///
/// In characteristic two the cross term vanishes:
/// `(x_0 + x_1*Y)^2 = (x_0^2 + x_1^2*X^-1) + x_1^2*Y`.
#[inline]
fn square_impl<OU, HU>(
	x: PackedPrimitiveType<OU, GhashSq256b>,
) -> PackedPrimitiveType<OU, GhashSq256b>
where
	OU: UnderlierType + Divisible<M128>,
	HU: UnderlierType + Divisible<M128>,
	PackedPrimitiveType<HU, BinaryField128bGhash>: PackedField<Scalar = BinaryField128bGhash>,
{
	let (x0, x1) = split(x);
	let t0 = Square::square(x0);
	let t2 = Square::square(x1);
	join(t0 + mul_inv_x(t2), t2)
}

/// Inverts a packed [`GhashSq256b`] value, mapping zero lanes to zero.
///
/// The conjugate of `a + b*Y` under `Y -> Y + 1` is `(a + b) + b*Y`.
/// Its norm `N = a^2 + a*b + b^2*X^-1` lies in GHASH.
/// The inverse is the conjugate scaled by `N^-1`.
/// A zero element has norm zero, so the GHASH inverse returns zero lanes, hence zero here.
#[inline]
fn invert_impl<OU, HU>(
	x: PackedPrimitiveType<OU, GhashSq256b>,
) -> PackedPrimitiveType<OU, GhashSq256b>
where
	OU: UnderlierType + Divisible<M128>,
	HU: UnderlierType + Divisible<M128>,
	PackedPrimitiveType<HU, BinaryField128bGhash>: PackedField<Scalar = BinaryField128bGhash>,
{
	let (a, b) = split(x);
	let norm = Square::square(a) + a * b + mul_inv_x(Square::square(b));
	let norm_inv = norm.invert_or_zero();
	join((a + b) * norm_inv, b * norm_inv)
}

/// Defines a packed [`GhashSq256b`] type over outer underlier `$ou`, whose halves are `$hu`.
///
/// The packing is a plain [`PackedPrimitiveType`].
/// Every operation except field arithmetic is inherited from the generic implementations.
/// Field arithmetic delegates to the half-width GHASH packing, which carries the platform's SIMD.
/// The widening multiply is trivial because the reduction is already folded into the multiply.
macro_rules! define_packed_ghash_sq {
	($name:ident, $ou:ty, $hu:ty) => {
		pub type $name = PackedPrimitiveType<$ou, GhashSq256b>;

		impl std::ops::Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				$crate::tracing::trace_multiplication!($name);
				mul_impl::<$ou, $hu>(self, rhs)
			}
		}

		impl Square for $name {
			#[inline]
			fn square(self) -> Self {
				square_impl::<$ou, $hu>(self)
			}
		}

		impl InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				invert_impl::<$ou, $hu>(self)
			}
		}

		impl WideMul for $name {
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
	};
}

define_packed_ghash_sq!(PackedGhashSq1x256b, M256, M128);
define_packed_ghash_sq!(PackedGhashSq2x256b, M512, M256);
define_packed_ghash_sq!(PackedGhashSq4x256b, M1024, M512);

#[cfg(test)]
mod tests {
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{ExtensionField, Random, field::FieldOps};

	// Builds a GhashSq256b from its `(a, b)` GHASH coordinates.
	fn ghash_sq(a: u128, b: u128) -> GhashSq256b {
		GhashSq256b::from_bases([BinaryField128bGhash::new(a), BinaryField128bGhash::new(b)])
	}

	// Strategy producing `width` random GhashSq256b scalars.
	fn arb_scalars(width: usize) -> impl Strategy<Value = Vec<GhashSq256b>> {
		prop::collection::vec(any::<[u128; 2]>().prop_map(|[a, b]| ghash_sq(a, b)), width)
	}

	// Pins the packed arithmetic to the scalar GhashSq256b reference, lane by lane.
	//
	// Pack `WIDTH` random scalars per operand.
	// Run each packed op, then check every lane equals the scalar op on the same inputs.
	fn check_against_scalar<P: PackedField<Scalar = GhashSq256b>>() {
		let width = P::WIDTH;
		let mut runner = proptest::test_runner::TestRunner::deterministic();

		runner
			.run(&(arb_scalars(width), arb_scalars(width)), |(xs, ys)| {
				let px = P::from_scalars(xs.iter().copied());
				let py = P::from_scalars(ys.iter().copied());

				// Multiply: each lane equals the scalar product of its inputs.
				let prod = px * py;
				for i in 0..width {
					prop_assert_eq!(prod.get(i), xs[i] * ys[i]);
				}

				// Square: each lane equals the scalar square of its input.
				let sq = Square::square(px);
				for i in 0..width {
					prop_assert_eq!(sq.get(i), Square::square(xs[i]));
				}

				// Invert-or-zero: each lane inverts independently, with zero mapping to zero.
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
		check_against_scalar::<PackedGhashSq1x256b>();
		check_against_scalar::<PackedGhashSq2x256b>();
		check_against_scalar::<PackedGhashSq4x256b>();
	}

	#[test]
	fn arithmetic_identities() {
		let mut rng = StdRng::seed_from_u64(0);

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
}
