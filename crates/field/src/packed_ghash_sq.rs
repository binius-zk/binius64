// Copyright 2026 The Binius Developers

//! Packed [`GhashSq256b`] in two layouts: sliced (struct-of-arrays) and interleaved
//! (array-of-structs).
//!
//! [`GhashSq256b`] is the degree-two extension `a + b·Y` of the GHASH field, with `Y² = X·Y + X`.
//! Both layouts reduce a batch multiply to three packed GHASH multiplies (Karatsuba over `Y`)
//! rather than a schoolbook product per element, and both defer the GHASH reductions and the
//! multiply-by-`X` — all `GF(2)`-linear — so an inner product reduces once at the end.
//!
//! - The **sliced** packings ([`SlicedGhashSq256b`]) store the `a` and `b` coordinates of every
//!   lane in two separate packed GHASH registers via [`SlicedPackedField`]; `Mul` and the rest of
//!   the [`PackedField`] surface come generically from [`SlicedPackedField`].
//! - The **interleaved** packings ([`PackedGhashSq1x256b`] / [`PackedGhashSq2x256b`], i.e.
//!   `PackedPrimitiveType<M256/M512, GhashSq256b>`) store each scalar as one contiguous 256-bit
//!   value — the same layout as the scalar field. The width-one packing carries the field
//!   arithmetic (and the scalar [`GhashSq256b`] derives its own from it), while the width-two
//!   packing divides into two width-one lanes.
//!
//! In both, the coordinate register is a [`PackedPrimitiveType`], so the multiply-by-`X` in the
//! reduction is a per-lane bit shift ([`ghash_mul_x`]) rather than a full field multiply.
//!
//! The width-one packing's widening multiply is architecture-specific — see [`GhashSqWideMul1x`],
//! which selects between batching the Karatsuba diagonal into one 256-bit carry-less multiply and
//! keeping the three products separate to defer the multiply-by-`X` past a reduction.

use std::{
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use crate::{
	BinaryField128bGhash, Divisible, GhashSq256b, PackedBinaryGhash2x128b, PackedField, WideMul,
	arch::{
		Divide, GhashSqWideMul1x, M128, M256, M512, MulFromWideMul, PackedPrimitiveType,
		portable::packed_macros::{portable_macros::*, *},
	},
	arithmetic_traits::{InvertOrZero, Square},
	cast_base, cast_ext,
	sliced_packed_field::SlicedPackedField,
	underlier::{UnderlierType, WithUnderlier},
};

/// The packed GHASH coordinate register backing a `SlicedGhashSq256b<U>`.
type Ghash<U> = PackedPrimitiveType<U, BinaryField128bGhash>;

/// A GHASH² packing whose two GHASH coordinates pack into `PackedPrimitiveType<U, Ghash128b>`.
pub type SlicedGhashSq256b<U> = SlicedPackedField<GhashSq256b, Ghash<U>, 2>;
/// Packed `GhashSq256b` holding one extension scalar (the degenerate width-one packing).
pub type SlicedGhashSq1x256b = SlicedGhashSq256b<M128>;
/// Packed `GhashSq256b` holding two extension scalars.
pub type SlicedGhashSq2x256b = SlicedGhashSq256b<M256>;
/// Packed `GhashSq256b` holding four extension scalars.
pub type SlicedGhashSq4x256b = SlicedGhashSq256b<M512>;

/// The unreduced widening product of the coordinate GHASH multiply.
type GhashWide<U> = <Ghash<U> as WideMul>::Output;

/// Multiplies every 128-bit GHASH lane of an underlier by `X`.
///
/// `X` scaling is `GF(2)`-linear — a per-lane bit shift with a fixed compensation, not a field
/// multiply — so this is far cheaper than a CLMUL. It reuses the scalar
/// [`BinaryField128bGhash::mul_x`] on each 128-bit lane, which every supported underlier divides
/// into.
#[inline]
fn ghash_mul_x<U: Divisible<M128>>(u: U) -> U {
	U::from_iter(Divisible::<M128>::value_iter(u).map(|lane| {
		BinaryField128bGhash::from_underlier(lane)
			.mul_x()
			.to_underlier()
	}))
}

/// Multiplies every GHASH lane of a packed coordinate by `X`.
#[inline]
fn mul_x<U: UnderlierType + Divisible<M128>>(coord: Ghash<U>) -> Ghash<U> {
	Ghash::<U>::from_underlier(ghash_mul_x(coord.to_underlier()))
}

/// The unreduced product of two GHASH² elements, as three separate GHASH products.
///
/// Holds the three Karatsuba GHASH widening products, deferring both the GHASH reductions and the
/// multiply-by-`X`. Since those are all `GF(2)`-linear, an inner product over GHASH² accumulates
/// these by XOR and reduces once at the end. Used both by the sliced packings and — where there is
/// no wider carry-less multiply to batch the diagonal into — by the interleaved width-one packing.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlicedGhashSqWide<W> {
	/// Unreduced `a·e`, the low diagonal Karatsuba product.
	pub(crate) t0: W,
	/// Unreduced `b·f`, the high diagonal Karatsuba product.
	pub(crate) t2: W,
	/// Unreduced `(a+b)·(e+f)`, the Karatsuba cross product.
	pub(crate) t1: W,
}

impl<W: Add<Output = W>> Add for SlicedGhashSqWide<W> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self {
			t0: self.t0 + rhs.t0,
			t2: self.t2 + rhs.t2,
			t1: self.t1 + rhs.t1,
		}
	}
}

impl<W: Sub<Output = W>> Sub for SlicedGhashSqWide<W> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self {
			t0: self.t0 - rhs.t0,
			t2: self.t2 - rhs.t2,
			t1: self.t1 - rhs.t1,
		}
	}
}

impl<W: AddAssign> AddAssign for SlicedGhashSqWide<W> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		self.t0 += rhs.t0;
		self.t2 += rhs.t2;
		self.t1 += rhs.t1;
	}
}

impl<W: SubAssign> SubAssign for SlicedGhashSqWide<W> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		self.t0 -= rhs.t0;
		self.t2 -= rhs.t2;
		self.t1 -= rhs.t1;
	}
}

impl<W: Default + Add<Output = W>> Sum for SlicedGhashSqWide<W> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

impl<U> WideMul for SlicedGhashSq256b<U>
where
	U: UnderlierType + Divisible<M128>,
	Ghash<U>: PackedField<Scalar = BinaryField128bGhash> + WideMul,
{
	type Output = SlicedGhashSqWide<GhashWide<U>>;

	/// Karatsuba over `Y`: defers the three GHASH products `a·e`, `b·f`, `(a+b)·(e+f)`.
	#[inline]
	fn wide_mul(lhs: Self, rhs: Self) -> Self::Output {
		let [a, b] = lhs.to_coords();
		let [e, f] = rhs.to_coords();

		SlicedGhashSqWide {
			t0: <Ghash<U> as WideMul>::wide_mul(a, e),
			t2: <Ghash<U> as WideMul>::wide_mul(b, f),
			t1: <Ghash<U> as WideMul>::wide_mul(a + b, e + f),
		}
	}

	/// Reduces the three products and folds `Y² = X·Y + X`. With the Karatsuba cross term recovered
	/// as `t₁ + t₀ + t₂`: `z₀ = t₀ + X·t₂`, `z₁ = (t₁ + t₀ + t₂) + X·t₂ = z₀ + t₁ + t₂`.
	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		let t0 = <Ghash<U> as WideMul>::reduce(wide.t0);
		let t2 = <Ghash<U> as WideMul>::reduce(wide.t2);
		let t1 = <Ghash<U> as WideMul>::reduce(wide.t1);

		let z0 = t0 + mul_x(t2);
		Self::from_coords([z0, z0 + t1 + t2])
	}
}

impl<U> Square for SlicedGhashSq256b<U>
where
	U: UnderlierType + Divisible<M128>,
	Ghash<U>: PackedField<Scalar = BinaryField128bGhash>,
{
	/// `(a + b·Y)² = (a² + X·b²) + (X·b²)·Y` — the cross term vanishes in characteristic two, and
	/// `Y² = X·Y + X`.
	#[inline]
	fn square(self) -> Self {
		let [a, b] = self.to_coords();

		let t0 = Square::square(a);
		let t2 = Square::square(b);

		let x_t2 = mul_x(t2);
		Self::from_coords([t0 + x_t2, x_t2])
	}
}

impl<U> InvertOrZero for SlicedGhashSq256b<U>
where
	U: UnderlierType + Divisible<M128>,
	Ghash<U>: PackedField<Scalar = BinaryField128bGhash>,
{
	/// Inverts through the norm of the degree-two extension. The conjugate of `u = a + b·Y` sends
	/// `Y` to the other root of `Y² + X·Y + X` (the roots sum to `X` and multiply to `X`), giving
	/// `ū = (a + X·b) + b·Y`. Its norm `N = u·ū = a² + X·b·(a + b)` lies in GHASH, and
	/// `u⁻¹ = ū·N⁻¹`. A zero lane has norm zero, so `invert_or_zero` returns zero there.
	#[inline]
	fn invert_or_zero(self) -> Self {
		let [a, b] = self.to_coords();

		let norm = Square::square(a) + mul_x(a * b + Square::square(b));
		let norm_inv = norm.invert_or_zero();

		Self::from_coords([(a + mul_x(b)) * norm_inv, b * norm_inv])
	}
}

// ---------------------------------------------------------------------------
// Interleaved (array-of-structs) packings: `PackedPrimitiveType<M256/M512, GhashSq256b>`.
//
// Unlike the sliced packings above, these store each GHASH² scalar as one contiguous 256-bit value
// (low 128 bits = coefficient of `1`, high 128 bits = coefficient of `Y`) — the same layout as the
// scalar `GhashSq256b`. The width-one M256 packing carries the field arithmetic (the scalar field
// derives its own `Mul`/`Square`/`InvertOrZero`/`WideMul` from it via `binary_field!`); the
// width-two M512 packing divides into two independent M256 lanes.
//
// The width-one `WideMul` itself is architecture-specific and lives with the other per-target
// strategies under `arch`, reached here through the `GhashSqWideMul1x` alias.
// ---------------------------------------------------------------------------

/// The GHASH coordinates `[a, b]` of a width-one GHASH² element `a + b·Y`.
///
/// The coordinates already sit in the two 128-bit lanes of the 256-bit value, so this is a free
/// reinterpretation followed by two lane reads.
#[inline]
pub(crate) fn ghash_sq_coords(elem: PackedGhashSq1x256b) -> [BinaryField128bGhash; 2] {
	let coords = cast_base::<BinaryField128bGhash, _>(elem);
	[coords.get(0), coords.get(1)]
}

/// Assembles a width-one GHASH² element `a + b·Y` from its GHASH coordinates `[a, b]`.
#[inline]
pub(crate) fn ghash_sq_from_coords(coords: [BinaryField128bGhash; 2]) -> PackedGhashSq1x256b {
	cast_ext::<BinaryField128bGhash, _>(PackedBinaryGhash2x128b::from_scalars(coords))
}

/// [`Square`] strategy for [`PackedGhashSq1x256b`].
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashSqSquare<T>(T);

impl Square for GhashSqSquare<PackedGhashSq1x256b> {
	/// `(a + b·Y)² = (a² + X·b²) + (X·b²)·Y` — the cross term vanishes in characteristic two.
	#[inline]
	fn square(self) -> Self {
		let sq = Square::square(cast_base::<BinaryField128bGhash, _>(Self::peel(self)));

		let x_t2 = sq.get(1).mul_x();
		Self::wrap(ghash_sq_from_coords([sq.get(0) + x_t2, x_t2]))
	}
}

/// [`InvertOrZero`] strategy for [`PackedGhashSq1x256b`].
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashSqInvert<T>(T);

impl InvertOrZero for GhashSqInvert<PackedGhashSq1x256b> {
	/// Inverts through the norm: conjugate `ū = (a + X·b) + b·Y` (the roots of `Y² + X·Y + X` sum
	/// to `X` and multiply to `X`), norm `N = a² + X·b·(a + b)`, and `u⁻¹ = ū·N⁻¹`.
	#[inline]
	fn invert_or_zero(self) -> Self {
		let [a, b] = ghash_sq_coords(Self::peel(self));

		let norm = Square::square(a) + (a * b + Square::square(b)).mul_x();
		let norm_inv = norm.invert_or_zero();

		Self::wrap(ghash_sq_from_coords([(a + b.mul_x()) * norm_inv, b * norm_inv]))
	}
}

/// [`Divide`] strategy specializing the width-two M512 packing into two width-one M256 lanes.
type GhashSqDivide2x<T> = Divide<M256, T, 2>;

define_packed_binary_field!(
	PackedGhashSq1x256b,
	GhashSq256b,
	M256,
	(MulFromWideMul),
	(GhashSqSquare),
	(GhashSqInvert),
	(GhashSqWideMul1x)
);

define_packed_binary_field!(
	PackedGhashSq2x256b,
	GhashSq256b,
	M512,
	(MulFromWideMul),
	(GhashSqDivide2x),
	(GhashSqDivide2x),
	(GhashSqDivide2x)
);

#[cfg(test)]
mod tests {
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		Divisible, Field, PackedField, Random,
		arithmetic_traits::{InvertOrZero, Square},
		field::FieldOps,
	};

	// Every packing of `GhashSq256b` must agree lane-by-lane with the scalar reference field, which
	// is tested independently in `ghash_sq`. Each check is run for all three widths.

	fn check_arithmetic<P: PackedField<Scalar = GhashSq256b>>(mut rng: impl Rng) {
		let a = P::random(&mut rng);
		let b = P::random(&mut rng);

		let sum = a + b;
		let diff = a - b;
		let prod = a * b;
		let sq = Square::square(a);
		let inv = InvertOrZero::invert_or_zero(a);

		for i in 0..P::WIDTH {
			let (x, y) = (a.get(i), b.get(i));
			assert_eq!(sum.get(i), x + y);
			assert_eq!(diff.get(i), x - y);
			assert_eq!(prod.get(i), x * y);
			assert_eq!(sq.get(i), Square::square(x));
			assert_eq!(inv.get(i), x.invert_or_zero());
			// `invert_or_zero` is a genuine inverse away from zero.
			if x != GhashSq256b::ZERO {
				assert_eq!(x * inv.get(i), GhashSq256b::ONE);
			}
		}
	}

	fn check_wide_mul<P>(mut rng: impl Rng)
	where
		P: PackedField<Scalar = GhashSq256b> + WideMul,
	{
		// The deferred widening form must match the eager product, and accumulating before a single
		// reduction must match summing the reductions (both the multiply-by-`X` and the GHASH
		// reduction are `GF(2)`-linear).
		let (a1, b1) = (P::random(&mut rng), P::random(&mut rng));
		let (a2, b2) = (P::random(&mut rng), P::random(&mut rng));

		assert_eq!(P::reduce(P::wide_mul(a1, b1)), a1 * b1);
		let deferred = P::reduce(P::wide_mul(a1, b1) + P::wide_mul(a2, b2));
		assert_eq!(deferred, a1 * b1 + a2 * b2);
	}

	fn check_scalar_ops<P: PackedField<Scalar = GhashSq256b>>(mut rng: impl Rng) {
		let a = P::random(&mut rng);
		let s = GhashSq256b::random(&mut rng);

		let broadcast = <P as Divisible<GhashSq256b>>::broadcast(s);
		let scaled = a * s;
		for i in 0..P::WIDTH {
			assert_eq!(broadcast.get(i), s);
			assert_eq!(scaled.get(i), a.get(i) * s);
		}

		// `one` is the multiplicative identity in every lane.
		assert_eq!(a * <P as FieldOps>::one(), a);
	}

	fn check_get_set_iter<P: PackedField<Scalar = GhashSq256b>>(mut rng: impl Rng) {
		let mut a = P::random(&mut rng);
		for i in 0..P::WIDTH {
			let v = GhashSq256b::random(&mut rng);
			a.set(i, v);
			assert_eq!(a.get(i), v);
		}

		// `from_scalars(iter())` round-trips.
		let scalars: Vec<_> = a.iter().collect();
		assert_eq!(P::from_scalars(scalars.iter().copied()), a);
	}

	/// Reference [`PackedField::interleave`] over the scalar sequence, per the documented 2×2
	/// block transpose: output `x ∈ {0, 1}` takes, at block position `t`, block `2·⌊t/2⌋ + x` from
	/// the first operand when `t` is even and from the second when `t` is odd.
	fn ref_interleave<S: Copy>(a: &[S], b: &[S], lbl: usize) -> (Vec<S>, Vec<S>) {
		let s = 1usize << lbl;
		let nb = a.len() / s;
		let build = |x: usize| -> Vec<S> {
			let mut out = Vec::with_capacity(a.len());
			for t in 0..nb {
				let (src, blk) = if t % 2 == 0 {
					(a, t + x)
				} else {
					(b, t - 1 + x)
				};
				out.extend_from_slice(&src[blk * s..blk * s + s]);
			}
			out
		};
		(build(0), build(1))
	}

	/// Reference [`PackedField::unzip`] over the scalar sequence: concatenate the `nb` blocks of
	/// the first operand then the `nb` of the second, and split the resulting `2·nb` blocks into
	/// the even-indexed (first output) and odd-indexed (second output).
	fn ref_unzip<S: Copy>(a: &[S], b: &[S], lbl: usize) -> (Vec<S>, Vec<S>) {
		let s = 1usize << lbl;
		let nb = a.len() / s;
		let block = |i: usize| -> &[S] {
			if i < nb {
				&a[i * s..i * s + s]
			} else {
				&b[(i - nb) * s..(i - nb) * s + s]
			}
		};
		let (mut out_a, mut out_b) = (Vec::with_capacity(a.len()), Vec::with_capacity(a.len()));
		for i in 0..2 * nb {
			if i % 2 == 0 {
				out_a.extend_from_slice(block(i));
			} else {
				out_b.extend_from_slice(block(i));
			}
		}
		(out_a, out_b)
	}

	fn check_interleave_unzip<P: PackedField<Scalar = GhashSq256b>>(mut rng: impl Rng) {
		let a = P::random(&mut rng);
		let b = P::random(&mut rng);
		let (sa, sb): (Vec<_>, Vec<_>) = (a.iter().collect(), b.iter().collect());

		for log_block_len in 0..P::LOG_WIDTH {
			let (c, d) = a.interleave(b, log_block_len);
			let (ec, ed) = ref_interleave(&sa, &sb, log_block_len);
			assert_eq!(c, P::from_scalars(ec));
			assert_eq!(d, P::from_scalars(ed));

			let (u, v) = a.unzip(b, log_block_len);
			let (eu, ev) = ref_unzip(&sa, &sb, log_block_len);
			assert_eq!(u, P::from_scalars(eu));
			assert_eq!(v, P::from_scalars(ev));
		}
	}

	macro_rules! width_tests {
		($mod:ident, $ty:ty) => {
			mod $mod {
				use super::*;

				#[test]
				fn arithmetic() {
					for seed in 0..64 {
						check_arithmetic::<$ty>(StdRng::seed_from_u64(seed));
					}
				}

				#[test]
				fn wide_mul() {
					for seed in 0..64 {
						check_wide_mul::<$ty>(StdRng::seed_from_u64(seed));
					}
				}

				#[test]
				fn scalar_ops() {
					for seed in 0..64 {
						check_scalar_ops::<$ty>(StdRng::seed_from_u64(seed));
					}
				}

				#[test]
				fn get_set_iter() {
					for seed in 0..64 {
						check_get_set_iter::<$ty>(StdRng::seed_from_u64(seed));
					}
				}

				#[test]
				fn interleave_unzip() {
					for seed in 0..64 {
						check_interleave_unzip::<$ty>(StdRng::seed_from_u64(seed));
					}
				}
			}
		};
	}

	width_tests!(width1, SlicedGhashSq1x256b);
	width_tests!(width2, SlicedGhashSq2x256b);
	width_tests!(width4, SlicedGhashSq4x256b);

	// The interleaved `PackedPrimitiveType` packings must agree lane-by-lane with the same scalar
	// reference field as the sliced packings above.
	width_tests!(packed_width1, PackedGhashSq1x256b);
	width_tests!(packed_width2, PackedGhashSq2x256b);
}
