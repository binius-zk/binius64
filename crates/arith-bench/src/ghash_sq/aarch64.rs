// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication and squaring using aarch64 `PMULL` instructions.
//!
//! Elements are held as `poly64x2_t` and the unreduced widening products as the three 128-bit
//! limbs `[poly64x2_t; 3]` of [`crate::ghash::aarch64`], so the arithmetic stays in NEON registers.

use core::arch::aarch64::poly64x2_t;

use crate::ghash::aarch64 as ghash;

/// Multiply packed GHASH² elements in sliced representation using `PMULL` arithmetic.
#[inline]
pub fn mul_sliced(x: [poly64x2_t; 2], y: [poly64x2_t; 2]) -> [poly64x2_t; 2] {
	super::sliced::mul_sliced(x, y, ghash::mul_wide_schoolbook, ghash::reduce, ghash::mul_x_wide)
}

/// Widening (unreduced) multiply of packed GHASH² elements in sliced representation using `PMULL`
/// arithmetic. Returns the three raw GHASH products (`[poly64x2_t; 3]` each); see
/// [`super::sliced::mul_wide_sliced`] and reduce with [`reduce_sliced`].
#[inline]
pub fn mul_wide_sliced(x: [poly64x2_t; 2], y: [poly64x2_t; 2]) -> [[poly64x2_t; 3]; 3] {
	super::sliced::mul_wide_sliced(x, y, ghash::mul_wide_schoolbook)
}

/// Reduce the three raw products from [`mul_wide_sliced`] into a GHASH² element using `PMULL`
/// arithmetic; see [`super::sliced::reduce_sliced`].
#[inline]
pub fn reduce_sliced(t: [[poly64x2_t; 3]; 3]) -> [poly64x2_t; 2] {
	super::sliced::reduce_sliced(t, ghash::reduce, ghash::mul_x_wide)
}

/// Square packed GHASH² elements in sliced representation using `PMULL` arithmetic.
#[inline]
pub fn square_sliced(x: [poly64x2_t; 2]) -> [poly64x2_t; 2] {
	super::sliced::square_sliced(x, ghash::square, ghash::mul_x)
}

#[cfg(test)]
mod tests {
	use core::arch::aarch64::{vreinterpretq_p64_p128, vreinterpretq_p128_p64};

	use proptest::prelude::*;

	use super::*;
	use crate::{Underlier, ghash_sq::soft64};

	fn to_poly(x: [u128; 2]) -> [poly64x2_t; 2] {
		x.map(|v| unsafe { vreinterpretq_p64_p128(v) })
	}

	fn from_poly(x: [poly64x2_t; 2]) -> [u128; 2] {
		x.map(|v| unsafe { vreinterpretq_p128_p64(v) })
	}

	proptest! {
		// The PMULL sliced multiply agrees with the soft64 reference.
		#[test]
		fn test_mul_sliced_matches_soft64(a in any::<[u128; 2]>(), b in any::<[u128; 2]>()) {
			prop_assert_eq!(from_poly(mul_sliced(to_poly(a), to_poly(b))), soft64::mul_sliced(a, b));
		}

		// The PMULL sliced square agrees with the soft64 reference.
		#[test]
		fn test_square_sliced_matches_soft64(a in any::<[u128; 2]>()) {
			prop_assert_eq!(from_poly(square_sliced(to_poly(a))), soft64::square_sliced(a));
		}

		// Deferred reduction: accumulating the three raw products by XOR and calling reduce_sliced
		// once equals summing the reduced products.
		#[test]
		fn test_wide_sliced_deferred_reduction(
			a1 in any::<[u128; 2]>(), b1 in any::<[u128; 2]>(),
			a2 in any::<[u128; 2]>(), b2 in any::<[u128; 2]>(),
		) {
			let acc = <[[poly64x2_t; 3]; 3]>::xor(
				mul_wide_sliced(to_poly(a1), to_poly(b1)),
				mul_wide_sliced(to_poly(a2), to_poly(b2)),
			);
			let sum = <[u128; 2]>::xor(soft64::mul_sliced(a1, b1), soft64::mul_sliced(a2, b2));
			prop_assert_eq!(from_poly(reduce_sliced(acc)), sum);
		}
	}
}
