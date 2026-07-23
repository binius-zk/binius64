// Copyright 2026 The Binius Developers

//! Portable GHASH² widening multiply: three independent GHASH multiplies over the coordinates.
//!
//! This is the strategy for targets whose widest carry-less multiply is 128 bits, where batching
//! the Karatsuba diagonal into a two-lane packed GHASH multiply buys nothing — that packed multiply
//! decomposes back into two independent 128-bit multiplies, each with its own reduction. Keeping
//! the three Karatsuba products separate instead lets the multiply-by-`X` of the irreducible
//! polynomial be applied to an *unreduced* product ([`MulXWide`]), which folds it into a reduction
//! that has to happen anyway: two GHASH reductions per GHASH² product rather than three.

use bytemuck::TransparentWrapper;

use crate::{
	BinaryField128bGhash, PackedGhashSq1x256b, SlicedGhashSqWide, WideMul,
	arithmetic_traits::MulXWide,
	packed_ghash_sq::{ghash_sq_coords, ghash_sq_from_coords},
};

/// The unreduced product of a single GHASH coordinate multiply.
type GhashWide = <BinaryField128bGhash as WideMul>::Output;

/// [`WideMul`] strategy for [`PackedGhashSq1x256b`] keeping the three Karatsuba products of the
/// coordinate multiply separate, so that the multiply-by-`X` can be deferred into a reduction.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashSqSlicedWideMul<T>(T);

impl WideMul for GhashSqSlicedWideMul<PackedGhashSq1x256b> {
	type Output = SlicedGhashSqWide<GhashWide>;

	/// Karatsuba over `Y`: defers the three GHASH products `a·e`, `b·f`, `(a+b)·(e+f)`.
	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		let [a0, a1] = ghash_sq_coords(Self::peel(a));
		let [b0, b1] = ghash_sq_coords(Self::peel(b));

		SlicedGhashSqWide {
			t0: BinaryField128bGhash::wide_mul(a0, b0),
			t2: BinaryField128bGhash::wide_mul(a1, b1),
			t1: BinaryField128bGhash::wide_mul(a0 + a1, b0 + b1),
		}
	}

	/// Folds `Y² = X·Y + X`, recovering the Karatsuba cross term as `t₁ + t₀ + t₂`:
	/// `z₀ = t₀ + X·t₂` and `z₁ = (t₁ + t₀ + t₂) + X·t₂ = z₀ + t₁ + t₂`.
	///
	/// Scaling `t₂` by `X` while it is still unreduced turns `z₀` into a single reduction of an
	/// accumulated wide product, so the two coordinates cost two reductions in total.
	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		let z0 = BinaryField128bGhash::reduce(wide.t0 + wide.t2.mul_x_wide());
		let z1 = z0 + BinaryField128bGhash::reduce(wide.t1 + wide.t2);

		Self::wrap(ghash_sq_from_coords([z0, z1]))
	}
}
