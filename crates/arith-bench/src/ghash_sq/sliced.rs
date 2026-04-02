// Copyright 2026 The Binius Developers
//! Sliced multiplication for GHASH² elements.
//!
//! In the sliced representation, a pair of GHASH² elements `(a₀ + b₀Y, a₁ + b₁Y)` is stored as
//! `[U; 2]` where `U` is a packed underlier containing two GHASH elements. The first element
//! `U` packs the lower limbs `(a₀, a₁)` and the second packs the upper limbs `(b₀, b₁)`. The
//! limbs for each extension element are not adjacent in memory — hence "sliced".
//!
//! This layout enables SIMD-parallel multiplication of multiple GHASH² elements, since the
//! underlying GHASH operations operate on all packed lanes simultaneously.

use crate::Underlier;

/// Multiply packed GHASH² elements in sliced representation.
///
/// Given `x = [x0, x1]` and `y = [y0, y1]` representing packed GHASH² elements where `x0`/`y0`
/// hold the lower limbs and `x1`/`y1` hold the upper limbs, computes the product in the extension
/// field defined by Y² + Y + X⁻¹.
///
/// Uses Karatsuba multiplication with three base-field multiplications and one multiply-by-X⁻¹.
#[inline]
pub fn mul_sliced<U: Underlier>(
	x: [U; 2],
	y: [U; 2],
	ghash_mul: impl Fn(U, U) -> U,
	ghash_mul_inv_x: impl Fn(U) -> U,
) -> [U; 2] {
	let [x0, x1] = x;
	let [y0, y1] = y;

	let t0 = ghash_mul(x0, y0);
	let t2 = ghash_mul(x1, y1);
	let t1 = ghash_mul(U::xor(x0, x1), U::xor(y0, y1));

	let z0 = U::xor(t0, ghash_mul_inv_x(t2));
	let z1 = U::xor(t1, t0);
	[z0, z1]
}
