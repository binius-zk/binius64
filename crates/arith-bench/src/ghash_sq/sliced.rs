// Copyright 2026 The Binius Developers
//! Sliced multiplication and squaring for GHASH² elements.
//!
//! In the sliced representation, a pair of GHASH² elements `(a₀ + b₀Y, a₁ + b₁Y)` is stored as
//! `[U; 2]` where `U` is a packed underlier containing two GHASH elements. The first element
//! `U` packs the lower limbs `(a₀, a₁)` and the second packs the upper limbs `(b₀, b₁)`. The
//! limbs for each extension element are not adjacent in memory — hence "sliced".
//!
//! This layout enables SIMD-parallel multiplication of multiple GHASH² elements, since the
//! underlying GHASH operations operate on all packed lanes simultaneously.
//!
//! The extension is defined by the irreducible polynomial `Y² + X·Y + X` over the GHASH field, so
//! `Y² = X·Y + X`. Reducing this way multiplies by `X` (a left shift) rather than by `X⁻¹`, and the
//! multiply-by-`X` is applied to the *unreduced* widening product, which saves one base-field
//! reduction per multiply (two instead of three).

use crate::Underlier;

/// Widening (unreduced) multiply of packed GHASH² elements in sliced representation.
///
/// Returns the three raw base-field products `[t0, t1, t2] = [x0·y0, (x0+x1)·(y0+y1), x1·y1]` — the
/// Karatsuba terms for `Y² + X·Y + X` — still *unreduced*. No combination or multiply-by-`X` is
/// done here: those are F2-linear and deferred to [`reduce_sliced`], so an inner product
/// XOR-accumulates the three products (`W` is an [`Underlier`] via the blanket array impl) and pays
/// the reduction and the `X` exactly once at the end rather than per term.
#[inline]
pub fn mul_wide_sliced<U: Underlier, W>(
	x: [U; 2],
	y: [U; 2],
	ghash_wide_mul: impl Fn(U, U) -> W,
) -> [W; 3] {
	let [x0, x1] = x;
	let [y0, y1] = y;

	let t0 = ghash_wide_mul(x0, y0);
	let t1 = ghash_wide_mul(U::xor(x0, x1), U::xor(y0, y1));
	let t2 = ghash_wide_mul(x1, y1);

	[t0, t1, t2]
}

/// Reduce the three raw products `[t0, t1, t2]` from [`mul_wide_sliced`] into the two GHASH²
/// coefficients.
///
/// Over `Y² + X·Y + X` (so `Y² = X·Y + X`), writing `x = x0 + x1·Y` and `y = y0 + y1·Y`:
///
/// ```text
/// z0 = x0·y0 + X·(x1·y1)             = t0 + X·t2
/// z1 = x0·y1 + x1·y0 + X·(x1·y1)     = (t1 + t0 + t2) + X·t2
/// ```
///
/// with Karatsuba recovering the cross term `x0·y1 + x1·y0` as `t1 + t0 + t2`. The multiply-by-`X`
/// (a plain shift of the unreduced `t2`) and the XOR combinations live here, applied to the raw
/// products, so only the two output coefficients are reduced — two base-field reductions instead of
/// three. An inner product therefore performs each reduction (and the `X`) only once, on the
/// accumulated products.
#[inline]
pub fn reduce_sliced<U: Underlier, W: Underlier>(
	[t0, t1, t2]: [W; 3],
	ghash_reduce: impl Fn(W) -> U,
	ghash_mul_x_wide: impl Fn(W) -> W,
) -> [U; 2] {
	let x_t2 = ghash_mul_x_wide(t2);
	let z0 = ghash_reduce(W::xor(t0, x_t2));
	let z1 = U::xor(z0, ghash_reduce(W::xor(t1, t2)));
	[z0, z1]
}

/// Multiply packed GHASH² elements in sliced representation.
///
/// Given `x = [x0, x1]` and `y = [y0, y1]` representing packed GHASH² elements where `x0`/`y0`
/// hold the lower limbs and `x1`/`y1` hold the upper limbs, computes the product in the extension
/// field defined by `Y² + X·Y + X`.
///
/// Composes [`mul_wide_sliced`] with [`reduce_sliced`].
#[inline]
pub fn mul_sliced<U: Underlier, W: Underlier>(
	x: [U; 2],
	y: [U; 2],
	ghash_wide_mul: impl Fn(U, U) -> W,
	ghash_reduce: impl Fn(W) -> U,
	ghash_mul_x_wide: impl Fn(W) -> W,
) -> [U; 2] {
	reduce_sliced(mul_wide_sliced(x, y, ghash_wide_mul), ghash_reduce, ghash_mul_x_wide)
}

/// Square a packed GHASH² element in sliced representation.
///
/// Given `x = [x0, x1]` representing a packed GHASH² element where `x0` holds the lower limb and
/// `x1` holds the upper limb, computes the square in the extension field defined by `Y² + X·Y + X`.
///
/// Uses the identity `(a + bY)² = a² + b²·Y² = (a² + X·b²) + (X·b²)·Y`, requiring two base-field
/// squarings and one multiply-by-`X`.
#[inline]
pub fn square_sliced<U: Underlier>(
	x: [U; 2],
	ghash_square: impl Fn(U) -> U,
	ghash_mul_x: impl Fn(U) -> U,
) -> [U; 2] {
	let [x0, x1] = x;

	let a2 = ghash_square(x0);
	let x_b2 = ghash_mul_x(ghash_square(x1));

	let z0 = U::xor(a2, x_b2);
	let z1 = x_b2;
	[z0, z1]
}
