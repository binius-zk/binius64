// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication using the soft64 GHASH implementation.

use crate::ghash;

/// Multiply packed GHASH² elements in sliced representation using soft64 arithmetic.
#[inline]
pub fn mul_sliced(x: [u128; 2], y: [u128; 2]) -> [u128; 2] {
	super::sliced::mul_sliced(x, y, ghash::soft64::mul, ghash::soft64::mul_inv_x)
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
	}
}
