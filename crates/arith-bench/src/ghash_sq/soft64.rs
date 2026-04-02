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
	use crate::ghash::ONE;

	/// The multiplicative identity in GHASH²: 1 + 0*Y.
	const IDENTITY: [u128; 2] = [ONE, 0];

	proptest! {
		#[test]
		fn test_ghash_sq_soft64_mul_commutative(
			a0 in any::<u128>(),
			a1 in any::<u128>(),
			b0 in any::<u128>(),
			b1 in any::<u128>(),
		) {
			let a = [a0, a1];
			let b = [b0, b1];
			let ab = mul_sliced(a, b);
			let ba = mul_sliced(b, a); // spellchecker:disable-line
			prop_assert_eq!(ab, ba, "GHASH² soft64 multiplication is not commutative"); // spellchecker:disable-line
		}

		#[test]
		fn test_ghash_sq_soft64_mul_associative(
			a0 in any::<u128>(),
			a1 in any::<u128>(),
			b0 in any::<u128>(),
			b1 in any::<u128>(),
			c0 in any::<u128>(),
			c1 in any::<u128>(),
		) {
			let a = [a0, a1];
			let b = [b0, b1];
			let c = [c0, c1];
			let ab_c = mul_sliced(mul_sliced(a, b), c);
			let a_bc = mul_sliced(a, mul_sliced(b, c));
			prop_assert_eq!(ab_c, a_bc, "GHASH² soft64 multiplication is not associative");
		}

		#[test]
		fn test_ghash_sq_soft64_mul_distributive(
			a0 in any::<u128>(),
			a1 in any::<u128>(),
			b0 in any::<u128>(),
			b1 in any::<u128>(),
			c0 in any::<u128>(),
			c1 in any::<u128>(),
		) {
			let a = [a0, a1];
			let b = [b0, b1];
			let c = [c0, c1];
			let b_plus_c = [b0 ^ c0, b1 ^ c1];
			let a_times_b_plus_c = mul_sliced(a, b_plus_c);
			let ab_plus_ac = {
				let ab = mul_sliced(a, b);
				let ac = mul_sliced(a, c);
				[ab[0] ^ ac[0], ab[1] ^ ac[1]]
			};
			prop_assert_eq!(a_times_b_plus_c, ab_plus_ac,
				"GHASH² soft64 multiplication does not satisfy the distributive law");
		}

		#[test]
		fn test_ghash_sq_soft64_mul_identity(
			a0 in any::<u128>(),
			a1 in any::<u128>(),
		) {
			let a = [a0, a1];
			let result = mul_sliced(a, IDENTITY);
			prop_assert_eq!(result, a,
				"The provided identity is not the multiplicative identity in GHASH²");
		}
	}
}
