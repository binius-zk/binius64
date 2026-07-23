// Copyright 2026 The Binius Developers

//! GHASH² widening multiply for targets with a 256-bit carry-less multiply.
//!
//! The two diagonal Karatsuba products `[a·e, b·f]` are batched into a single
//! [`PackedBinaryGhash2x128b`] widening multiply — one VPCLMUL over both 128-bit lanes — leaving
//! only the Karatsuba cross product to a scalar widening multiply.

use std::{
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use crate::{
	BinaryField128bGhash, Divisible, PackedBinaryGhash2x128b, PackedGhashSq1x256b, WideMul,
	cast_base, packed_ghash_sq::ghash_sq_from_coords,
};

/// The two unreduced GHASH products `[a·e, b·f]` batched into one packed widening multiply.
type DiagWide = <PackedBinaryGhash2x128b as WideMul>::Output;
/// The single unreduced GHASH cross product `(a+b)·(e+f)` from a scalar widening multiply.
type CrossWide = <BinaryField128bGhash as WideMul>::Output;

/// The unreduced product of two [`PackedGhashSq1x256b`] elements.
///
/// Holds the three GHASH widening products of the Karatsuba decomposition over `Y`, deferring both
/// the GHASH reductions and the multiply-by-`X` — all `GF(2)`-linear — so an inner product over
/// GHASH² accumulates these by XOR and reduces only once at the end.
#[derive(Clone, Copy, Default, Debug)]
pub struct WideGhashSqProduct {
	/// Unreduced `[a·e, b·f]`, the diagonal Karatsuba products in their two packed GHASH lanes.
	diag: DiagWide,
	/// Unreduced `(a+b)·(e+f)`, the Karatsuba cross product.
	cross: CrossWide,
}

impl Add for WideGhashSqProduct {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self {
			diag: self.diag + rhs.diag,
			cross: self.cross + rhs.cross,
		}
	}
}

impl AddAssign for WideGhashSqProduct {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		self.diag += rhs.diag;
		self.cross += rhs.cross;
	}
}

impl Sub for WideGhashSqProduct {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self {
			diag: self.diag - rhs.diag,
			cross: self.cross - rhs.cross,
		}
	}
}

impl SubAssign for WideGhashSqProduct {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		self.diag -= rhs.diag;
		self.cross -= rhs.cross;
	}
}

impl Sum for WideGhashSqProduct {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

/// [`WideMul`] strategy for [`PackedGhashSq1x256b`] batching the Karatsuba diagonal into a single
/// two-lane packed GHASH widening multiply.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashSqHybridWideMul<T>(T);

impl WideMul for GhashSqHybridWideMul<PackedGhashSq1x256b> {
	type Output = WideGhashSqProduct;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		// The GHASH² coordinates already sit in the two 128-bit lanes of the 256-bit value, so
		// viewing an operand as a packed GHASH pair is a free reinterpretation.
		let a = cast_base::<BinaryField128bGhash, _>(Self::peel(a));
		let b = cast_base::<BinaryField128bGhash, _>(Self::peel(b));

		WideGhashSqProduct {
			// Diagonal `[a·e, b·f]` as one two-lane packed widening multiply.
			diag: PackedBinaryGhash2x128b::wide_mul(a, b),
			// Karatsuba cross product `(a+b)·(e+f)` as a scalar widening multiply.
			cross: BinaryField128bGhash::wide_mul(a.get(0) + a.get(1), b.get(0) + b.get(1)),
		}
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		// Reduce the batched diagonal back to `t_0 = a·e`, `t_2 = b·f`, and the cross to `t_1`.
		let diag = PackedBinaryGhash2x128b::reduce(wide.diag);
		let t0 = diag.get(0);
		let t2 = diag.get(1);
		let t1 = BinaryField128bGhash::reduce(wide.cross);

		// Fold `Y² = X·Y + X`, recovering the cross term as `t_1 + t_0 + t_2`:
		// `z_0 = t_0 + X·t_2`, `z_1 = (t_1 + t_0 + t_2) + X·t_2 = z_0 + t_1 + t_2`.
		let z0 = t0 + t2.mul_x();
		Self::wrap(ghash_sq_from_coords([z0, z0 + t1 + t2]))
	}
}
