// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication and squaring using the soft64 GHASH implementation.

use crate::ghash;

/// Multiply packed GHASH² elements in sliced representation using soft64 arithmetic.
#[inline]
pub fn mul_sliced(x: [u128; 2], y: [u128; 2]) -> [u128; 2] {
	super::sliced::mul_sliced(
		x,
		y,
		ghash::soft64::mul_wide,
		ghash::soft64::reduce,
		ghash::soft64::mul_x_wide,
	)
}

/// Widening (unreduced) multiply of packed GHASH² elements in sliced representation using soft64
/// arithmetic. Returns the three raw GHASH products (`[u64; 4]` each); see
/// [`super::sliced::mul_wide_sliced`] and reduce with [`reduce_sliced`].
#[inline]
pub fn mul_wide_sliced(x: [u128; 2], y: [u128; 2]) -> [[u64; 4]; 3] {
	super::sliced::mul_wide_sliced(x, y, ghash::soft64::mul_wide)
}

/// Reduce the three raw products from [`mul_wide_sliced`] into a GHASH² element using soft64
/// arithmetic; see [`super::sliced::reduce_sliced`].
#[inline]
pub fn reduce_sliced(t: [[u64; 4]; 3]) -> [u128; 2] {
	super::sliced::reduce_sliced(t, ghash::soft64::reduce, ghash::soft64::mul_x_wide)
}

/// Square packed GHASH² elements in sliced representation using soft64 arithmetic.
#[inline]
pub fn square_sliced(x: [u128; 2]) -> [u128; 2] {
	super::sliced::square_sliced(x, ghash::soft64::square, ghash::soft64::mul_x)
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::{
		Underlier,
		ghash::ONE,
		test_utils::multiplication_tests::{
			test_mul_associative, test_mul_commutative, test_mul_distributive,
			test_square_equals_mul,
		},
	};

	/// The multiplicative identity in GHASH²: 1 + 0*Y.
	const IDENTITY: [u128; 2] = [ONE, 0];

	proptest! {
		#[test]
		fn test_ghash_sq_soft64_mul_commutative(
			a in any::<[u128; 2]>(),
			b in any::<[u128; 2]>(),
		) {
			test_mul_commutative(a, b, mul_sliced, "GHASH²");
		}

		#[test]
		fn test_ghash_sq_soft64_mul_associative(
			a in any::<[u128; 2]>(),
			b in any::<[u128; 2]>(),
			c in any::<[u128; 2]>(),
		) {
			test_mul_associative(a, b, c, mul_sliced, "GHASH²");
		}

		#[test]
		fn test_ghash_sq_soft64_mul_distributive(
			a in any::<[u128; 2]>(),
			b in any::<[u128; 2]>(),
			c in any::<[u128; 2]>(),
		) {
			test_mul_distributive(a, b, c, mul_sliced, "GHASH²");
		}

		#[test]
		fn test_ghash_sq_soft64_mul_identity(
			a in any::<[u128; 2]>(),
		) {
			let result = mul_sliced(a, IDENTITY);
			assert!(
				<[u128; 2]>::is_equal(result, a),
				"The provided identity is not the multiplicative identity in GHASH²"
			);
		}

		#[test]
		fn test_ghash_sq_soft64_square_equals_mul(
			a in any::<[u128; 2]>(),
		) {
			test_square_equals_mul(a, mul_sliced, square_sliced, "GHASH²");
		}

		// Deferred reduction: accumulating the three raw products by XOR and calling reduce_sliced
		// once equals summing the reduced products — the F2-linear property the inner-product
		// benchmark relies on (the multiply-by-X is likewise deferred into reduce_sliced).
		#[test]
		fn test_ghash_sq_soft64_wide_deferred_reduction(
			a1 in any::<[u128; 2]>(), b1 in any::<[u128; 2]>(),
			a2 in any::<[u128; 2]>(), b2 in any::<[u128; 2]>(),
		) {
			let acc = <[[u64; 4]; 3]>::xor(mul_wide_sliced(a1, b1), mul_wide_sliced(a2, b2));
			let sum = <[u128; 2]>::xor(mul_sliced(a1, b1), mul_sliced(a2, b2));
			prop_assert_eq!(reduce_sliced(acc), sum);
		}
	}
}
