// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	BinaryField128bGhash,
	arch::{
		GhashInvert1x, GhashInvert2x, GhashInvert4x, GhashSquare1x, GhashSquare2x, GhashSquare4x,
		GhashWideMul1x, GhashWideMul2x, GhashWideMul4x, M128, M256, M512, MulFromWideMul,
		portable::packed_macros::{portable_macros::*, *},
	},
};

define_packed_binary_field!(
	PackedBinaryGhash1x128b,
	BinaryField128bGhash,
	M128,
	(MulFromWideMul),
	(GhashSquare1x),
	(GhashInvert1x),
	(GhashWideMul1x)
);

define_packed_binary_field!(
	PackedBinaryGhash2x128b,
	BinaryField128bGhash,
	M256,
	(MulFromWideMul),
	(GhashSquare2x),
	(GhashInvert2x),
	(GhashWideMul2x)
);

define_packed_binary_field!(
	PackedBinaryGhash4x128b,
	BinaryField128bGhash,
	M512,
	(MulFromWideMul),
	(GhashSquare4x),
	(GhashInvert4x),
	(GhashWideMul4x)
);

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, proptest};

	use super::*;
	use crate::{
		BinaryField128bGhash, PackedField, packed_binary_field::test_utils::packed_field_tests,
		underlier::WithUnderlier,
	};

	fn check_get_set<const WIDTH: usize, PT>(a: [u128; WIDTH], b: [u128; WIDTH])
	where
		PT: PackedField<Scalar = BinaryField128bGhash>
			+ WithUnderlier<Underlier: From<[u128; WIDTH]>>,
	{
		let mut val = PT::from_underlier(a.into());
		for i in 0..WIDTH {
			assert_eq!(val.get(i), BinaryField128bGhash::from(a[i]));
			val.set(i, BinaryField128bGhash::from(b[i]));
			assert_eq!(val.get(i), BinaryField128bGhash::from(b[i]));
		}
	}

	proptest! {
		#[test]
		fn test_get_set_256(a in any::<[u128; 2]>(), b in any::<[u128; 2]>()) {
			check_get_set::<2, PackedBinaryGhash2x128b>(a, b);
		}

		#[test]
		fn test_get_set_512(a in any::<[u128; 4]>(), b in any::<[u128; 4]>()) {
			check_get_set::<4, PackedBinaryGhash4x128b>(a, b);
		}
	}

	packed_field_tests!(ghash_1x128b, PackedBinaryGhash1x128b);
	packed_field_tests!(ghash_2x128b, PackedBinaryGhash2x128b);
	packed_field_tests!(ghash_4x128b, PackedBinaryGhash4x128b);

	#[test]
	fn test_wide_mul_zero_inputs() {
		use super::PackedBinaryGhash1x128b as P;
		use crate::{WideMul, field::FieldOps};

		let zero = P::default();
		let one = P::one();

		assert_eq!(P::reduce(P::wide_mul(zero, zero)), zero);
		assert_eq!(P::reduce(P::wide_mul(zero, one)), zero);
		assert_eq!(P::reduce(P::wide_mul(one, zero)), zero);
		assert_eq!(P::reduce(P::wide_mul(one, one)), one);

		let wide_zero = <P as WideMul>::Output::default();
		assert_eq!(P::reduce(wide_zero), zero);
	}

	#[test]
	fn test_wide_mul_single_accumulation() {
		use rand::{SeedableRng, rngs::StdRng};

		use super::PackedBinaryGhash1x128b as P;
		use crate::{Random, WideMul};

		let mut rng = StdRng::seed_from_u64(77);
		let a = P::random(&mut rng);
		let b = P::random(&mut rng);

		let wide = P::wide_mul(a, b);
		let sum = wide + <P as WideMul>::Output::default();
		assert_eq!(P::reduce(sum), a * b);
	}
}
