// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	Ghash128b,
	arch::{
		GhashInvert1x, GhashInvert2x, GhashInvert4x, GhashPreparedMul1x, GhashPreparedMul2x,
		GhashPreparedMul4x, GhashSquare1x, GhashSquare2x, GhashSquare4x, GhashWideMul1x,
		GhashWideMul2x, GhashWideMul4x, M128, M256, M512, portable::packed_macros::*,
	},
};

define_packed_binary_field!(
	PackedBinaryGhash1x128b,
	Ghash128b,
	M128,
	(GhashSquare1x),
	(GhashInvert1x),
	(GhashWideMul1x),
	(GhashPreparedMul1x)
);

define_packed_binary_field!(
	PackedBinaryGhash2x128b,
	Ghash128b,
	M256,
	(GhashSquare2x),
	(GhashInvert2x),
	(GhashWideMul2x),
	(GhashPreparedMul2x)
);

define_packed_binary_field!(
	PackedBinaryGhash4x128b,
	Ghash128b,
	M512,
	(GhashSquare4x),
	(GhashInvert4x),
	(GhashWideMul4x),
	(GhashPreparedMul4x)
);

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, proptest};

	use super::*;
	use crate::{
		Ghash128b, PackedField, PreparedMul, packed_fields::test_utils::packed_field_tests,
		underlier::UnderlierView,
	};

	/// The bit patterns a random proptest is unlikely to reach.
	///
	/// `2` is the field element `X`, `0x87` the modulus' low part, and the two top-bit patterns are
	/// the operands whose product reaches the highest degree.
	const BOUNDARY_SCALARS: [u128; 7] = [0, 1, 2, 0x87, u128::MAX, 1 << 127, (1 << 127) | 1];

	/// Every pairing of the boundary patterns multiplies the same prepared as unprepared.
	///
	/// A rotation of the pattern list fills the lanes with distinct patterns, so the pairs cover a
	/// lane-dependent bug as well: a packing wider than one lane sees a different pattern in each.
	fn check_mul_prepared_boundaries<P: PackedField<Scalar = Ghash128b>>() {
		let rotation = |first: usize| {
			P::from_scalars((0..P::WIDTH).map(|lane| {
				Ghash128b::from(BOUNDARY_SCALARS[(first + lane) % BOUNDARY_SCALARS.len()])
			}))
		};

		for i in 0..BOUNDARY_SCALARS.len() {
			for j in 0..BOUNDARY_SCALARS.len() {
				let (x, y) = (rotation(i), rotation(j));

				assert_eq!(x.mul_prepared(&y.prepare()), x * y, "rotations {i} and {j}");
				assert_eq!(y.mul_prepared(&x.prepare()), x * y, "rotations {j} and {i}");
			}
		}
	}

	fn check_get_set<const WIDTH: usize, PT>(a: [u128; WIDTH], b: [u128; WIDTH])
	where
		PT: PackedField<Scalar = Ghash128b> + UnderlierView<Underlier: From<[u128; WIDTH]>>,
	{
		let mut val = PT::from_underlier(a.into());
		for i in 0..WIDTH {
			assert_eq!(val.get(i), Ghash128b::from(a[i]));
			val.set(i, Ghash128b::from(b[i]));
			assert_eq!(val.get(i), Ghash128b::from(b[i]));
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
	fn mul_prepared_on_boundary_patterns() {
		check_mul_prepared_boundaries::<PackedBinaryGhash1x128b>();
		check_mul_prepared_boundaries::<PackedBinaryGhash2x128b>();
		check_mul_prepared_boundaries::<PackedBinaryGhash4x128b>();
	}

	/// Preparing a broadcast scalar must agree with preparing the scalar itself.
	///
	/// The scalar field derives its arithmetic from the width-one packing, so this pins the two
	/// entry points to the same field element — lane by lane, at every packing width.
	#[test]
	fn prepared_broadcast_agrees_with_the_scalar_path() {
		use rand::{SeedableRng, rngs::StdRng};

		use crate::Random;

		fn check<P: PackedField<Scalar = Ghash128b>>(rng: &mut impl rand::Rng) {
			for _ in 0..64 {
				let value = P::random(&mut *rng);
				let multiplier = Ghash128b::random(&mut *rng);

				let prepared = P::broadcast(multiplier).prepare();
				let scalar_prepared = multiplier.prepare();

				let product = value.mul_prepared(&prepared);
				assert_eq!(product, value * P::broadcast(multiplier));
				for i in 0..P::WIDTH {
					assert_eq!(product.get(i), value.get(i).mul_prepared(&scalar_prepared));
				}
			}
		}

		let mut rng = StdRng::seed_from_u64(1234);
		check::<PackedBinaryGhash1x128b>(&mut rng);
		check::<PackedBinaryGhash2x128b>(&mut rng);
		check::<PackedBinaryGhash4x128b>(&mut rng);
	}

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
