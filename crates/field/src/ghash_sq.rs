// Copyright 2026 The Binius Developers

//! Binary field implementation of GF(2^256) as a degree-two extension of the GHASH field.
//!
//! Elements are pairs `(a, b)` representing `a + b·Y`, where `a` and `b` are elements of
//! [`BinaryField128bGhash`]. The extension is defined by the irreducible polynomial
//! `Y² + X·Y + X` over GHASH, so that `Y² = X·Y + X`.
//!
//! The field is backed by [`M256`], with the low 128 bits holding the coefficient of `1` (`a`) and
//! the high 128 bits holding the coefficient of `Y` (`b`). This is the same layout as
//! [`PackedBinaryGhash2x128b`] (two GHASH lanes in an `M256`) and matches the `{1, Y}` basis used
//! by the `ExtensionField<BinaryField128bGhash>` implementation.
//!
//! Reducing with `Y² = X·Y + X` multiplies by `X` (a left shift) rather than by `X⁻¹`, and the
//! multiply-by-`X` folds into the reduction. Multiplication batches the two GHASH products that
//! share the AVX2 256-bit CLMUL into a single [`PackedBinaryGhash2x128b`] multiply (the
//! `mul_m256i_hybrid` algorithm).

use std::{
	fmt::{Debug, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{
	DeserializeBytes, FixedSizeSerializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
};
use bytemuck::{Pod, Zeroable};

use super::{
	binary_field::{BinaryField, BinaryField1b, binary_field, impl_field_extension},
	extension::ExtensionField,
};
use crate::{
	BinaryField128bGhash, Divisible, Field, PackedBinaryGhash2x128b, PackedField,
	arch::{M128, M256, m256_from_u128s},
	mul_by_binary_field_1b,
	underlier::U1,
};

// The multiplicative generator is `Y` (low 128 bits = 0, high 128 bits = 1).
// `tests::test_multiplicative_generator` verifies it generates GF(2^256)* against the known
// factorization of 2^256 - 1.
//
// The field's `Mul`/`Square`/`InvertOrZero`/`WideMul` are the width-one packing's arithmetic,
// derived by the `binary_field!` macro from [`PackedGhashSq1x256b`] (`PackedPrimitiveType<M256,
// GhashSq256b>`), whose implementation lives in `packed_ghash_sq`.
binary_field!(pub GhashSq256b(M256), m256_from_u128s(0, 1));

unsafe impl Pod for GhashSq256b {}

impl GhashSq256b {
	/// Splits the element into its `(a, b)` coefficients over GHASH, where `self = a + b·Y`.
	#[inline]
	fn to_coeffs(self) -> [BinaryField128bGhash; 2] {
		// `GhashSq256b` and `PackedBinaryGhash2x128b` share the `M256` underlier and lane layout
		// (low lane = coefficient of `1`, high lane = coefficient of `Y`), so this reinterprets.
		let packed = PackedBinaryGhash2x128b::from_underlier(self.0);
		[packed.get(0), packed.get(1)]
	}

	/// Builds an element from its `(a, b)` coefficients over GHASH, so that `self = a + b·Y`.
	#[inline]
	fn from_coeffs(coeffs: [BinaryField128bGhash; 2]) -> Self {
		Self(PackedBinaryGhash2x128b::from_scalars(coeffs).to_underlier())
	}
}

// Degree-two extension over GHASH: the low 128 bits are the coefficient of `1`, the high 128 bits
// the coefficient of `Y`. `square_transpose` uses the packed fast path via
// `PackedBinaryGhash2x128b`.
impl_field_extension!(BinaryField128bGhash(M128) < @1 => GhashSq256b(M256));

// Extension over GF(2): the 256 underlier bits are the coordinates in the `BinaryField1b` basis.
impl_field_extension!(BinaryField1b(U1) < @8 => GhashSq256b(M256));

mul_by_binary_field_1b!(GhashSq256b);

// Scalar multiplication by a GHASH subfield element scales both extension coordinates.
impl Mul<BinaryField128bGhash> for GhashSq256b {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: BinaryField128bGhash) -> Self::Output {
		let [a, b] = self.to_coeffs();
		Self::from_coeffs([a * rhs, b * rhs])
	}
}

impl SerializeBytes for GhashSq256b {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for GhashSq256b {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self(DeserializeBytes::deserialize(read_buf)?))
	}
}

impl FixedSizeSerializeBytes for GhashSq256b {
	const BYTE_SIZE: usize = 32;
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::arithmetic_traits::{InvertOrZero, Square, WideMul};

	/// `X`, the generator of the GHASH field in the standard polynomial basis.
	const GHASH_X: u128 = 0x02;

	/// Prime factorization of `2²⁵⁶ - 1` (the multiplicative group order). These are the
	/// Fermat-number factors `F₀..F₇`: `2²⁵⁶ - 1 = ∏_{k=0}^{7} (2^{2^k} + 1)`.
	const ORDER_PRIME_FACTORS: [u128; 11] = [
		3,
		5,
		17,
		257,
		65537,
		641,
		6700417,
		274177,
		67280421310721,
		59649589127497217,
		5704689200685129054721,
	];

	fn ghash_sq(a: u128, b: u128) -> GhashSq256b {
		GhashSq256b::from_coeffs([BinaryField128bGhash::new(a), BinaryField128bGhash::new(b)])
	}

	fn arb_elem() -> impl Strategy<Value = GhashSq256b> {
		any::<[u128; 2]>().prop_map(|[a, b]| ghash_sq(a, b))
	}

	/// Independent reference for [`ExtensionField::square_transpose`]: transposes the
	/// `DEGREE × DEGREE` matrix whose row `i` is the `F`-basis expansion of `values[i]`. Built only
	/// from the (separately tested) `iter_bases`/`from_bases` accessors, to check the packed
	/// `square_transpose` fast path.
	fn naive_square_transpose<F: BinaryField>(values: &[GhashSq256b]) -> Vec<GhashSq256b>
	where
		GhashSq256b: ExtensionField<F>,
	{
		let degree = <GhashSq256b as ExtensionField<F>>::DEGREE;
		assert_eq!(values.len(), degree);
		let coords: Vec<F> = values
			.iter()
			.flat_map(|v| ExtensionField::<F>::iter_bases(v))
			.collect();
		(0..degree)
			.map(|i| {
				<GhashSq256b as ExtensionField<F>>::from_bases(
					(0..degree).map(|j| coords[j * degree + i]),
				)
			})
			.collect()
	}

	/// Computes `(2²⁵⁶ - 1) / p` as little-endian `u64` limbs via bit-by-bit long division. The
	/// remainder stays below `p` (≤ 73 bits), so it fits in a `u128`.
	fn order_cofactor(p: u128) -> [u64; 4] {
		let mut quotient = [0u64; 4];
		let mut rem: u128 = 0;
		// The dividend `2²⁵⁶ - 1` is 256 set bits, processed most-significant first.
		for bit in (0..256).rev() {
			rem = (rem << 1) | 1;
			let q_bit = if rem >= p {
				rem -= p;
				1u64
			} else {
				0
			};
			quotient[bit / 64] |= q_bit << (bit % 64);
		}
		quotient
	}

	/// A nonzero element generates the full multiplicative group iff `g^((2²⁵⁶-1)/p) ≠ 1` for
	/// every prime `p` dividing the group order.
	fn is_generator(g: GhashSq256b) -> bool {
		ORDER_PRIME_FACTORS
			.iter()
			.all(|&p| Field::pow(&g, order_cofactor(p)) != GhashSq256b::ONE)
	}

	#[test]
	fn test_quadratic_relation() {
		// The extension is defined by `Y² + X·Y + X = 0`, i.e. `Y² = X·Y + X`, whose coordinates in
		// the `{1, Y}` basis are `(X, X)`.
		let y = ghash_sq(0, 1);
		assert_eq!(y * y, ghash_sq(GHASH_X, GHASH_X));
	}

	#[test]
	fn test_subfield_embedding() {
		// Products of GHASH-subfield elements agree with GHASH multiplication.
		let a = BinaryField128bGhash::new(0x0123456789abcdef0123456789abcdef);
		let b = BinaryField128bGhash::new(0xfedcba9876543210fedcba9876543210);
		assert_eq!(GhashSq256b::from(a) * GhashSq256b::from(b), GhashSq256b::from(a * b),);
	}

	#[test]
	fn test_multiplicative_generator() {
		assert!(
			is_generator(GhashSq256b::MULTIPLICATIVE_GENERATOR),
			"baked MULTIPLICATIVE_GENERATOR is not a generator of GF(2^256)*",
		);
	}

	#[test]
	#[ignore = "search utility: prints a valid generator literal to bake into the field"]
	fn find_generator() {
		for b in 1u128..256 {
			for a in 0u128..256 {
				let candidate = ghash_sq(a, b);
				if is_generator(candidate) {
					let [low, high] = candidate.to_coeffs();
					panic!(
						"found generator: a={a:#x}, b={b:#x} -> m256_from_u128s({:#034x}, {:#034x})",
						u128::from(low.val()),
						u128::from(high.val()),
					);
				}
			}
		}
		panic!("no generator found in search range");
	}

	proptest! {
		#[test]
		fn test_mul_commutative(a in arb_elem(), b in arb_elem()) {
			prop_assert_eq!(a * b, b * a);
		}

		#[test]
		fn test_mul_associative(a in arb_elem(), b in arb_elem(), c in arb_elem()) {
			prop_assert_eq!((a * b) * c, a * (b * c));
		}

		#[test]
		fn test_mul_distributive(a in arb_elem(), b in arb_elem(), c in arb_elem()) {
			prop_assert_eq!(a * (b + c), a * b + a * c);
		}

		#[test]
		fn test_mul_identity(a in arb_elem()) {
			prop_assert_eq!(a * GhashSq256b::ONE, a);
		}

		#[test]
		fn test_square_equals_mul(a in arb_elem()) {
			prop_assert_eq!(Square::square(a), a * a);
		}

		#[test]
		fn test_wide_mul_deferred_reduction(
			pairs in prop::collection::vec((arb_elem(), arb_elem()), 1..16),
		) {
			// Inner product over GHASH^2: the deferred form must equal the eager form.
			//
			//     eager:    sum_i reduce(wide_mul(a_i, b_i))  =  sum_i a_i * b_i
			//     deferred: reduce( sum_i wide_mul(a_i, b_i) )
			//
			// This holds because the GHASH reduction and the multiply-by-`X` are both GF(2)-linear.
			// So a sum of products costs one reduction, not one per term.
			let eager: GhashSq256b = pairs.iter().map(|&(a, b)| a * b).sum();
			let deferred = GhashSq256b::reduce(
				pairs.iter().map(|&(a, b)| GhashSq256b::wide_mul(a, b)).sum(),
			);
			prop_assert_eq!(deferred, eager);
		}

		#[test]
		fn test_invert(a in arb_elem()) {
			let inv = a.invert_or_zero();
			if a == GhashSq256b::ZERO {
				prop_assert_eq!(inv, GhashSq256b::ZERO);
			} else {
				prop_assert_eq!(a * inv, GhashSq256b::ONE);
			}
		}

		#[test]
		fn test_serialization_roundtrip(a in arb_elem()) {
			let mut buf = Vec::new();
			a.serialize(&mut buf).unwrap();
			prop_assert_eq!(buf.len(), GhashSq256b::BYTE_SIZE);
			let b = GhashSq256b::deserialize(buf.as_slice()).unwrap();
			prop_assert_eq!(a, b);
		}

		#[test]
		fn test_ghash_extension_bases_roundtrip(a in arb_elem()) {
			let bases: Vec<BinaryField128bGhash> =
				ExtensionField::<BinaryField128bGhash>::iter_bases(&a).collect();
			prop_assert_eq!(bases.len(), 2);
			prop_assert_eq!(
				<GhashSq256b as ExtensionField<BinaryField128bGhash>>::from_bases(bases),
				a,
			);
		}

		#[test]
		fn test_b1b_extension_bases_roundtrip(a in arb_elem()) {
			let bases: Vec<BinaryField1b> =
				ExtensionField::<BinaryField1b>::iter_bases(&a).collect();
			prop_assert_eq!(bases.len(), 256);
			prop_assert_eq!(
				<GhashSq256b as ExtensionField<BinaryField1b>>::from_bases(bases),
				a,
			);
		}

		#[test]
		fn test_square_transpose_ghash(a in arb_elem(), b in arb_elem()) {
			let mut values = [a, b];
			let expected = naive_square_transpose::<BinaryField128bGhash>(&values);
			<GhashSq256b as ExtensionField<BinaryField128bGhash>>::square_transpose(&mut values);
			prop_assert_eq!(values.as_slice(), expected.as_slice());
		}

		#[test]
		fn test_square_transpose_b1b(values in prop::collection::vec(arb_elem(), 256)) {
			let mut values = values;
			let expected = naive_square_transpose::<BinaryField1b>(&values);
			<GhashSq256b as ExtensionField<BinaryField1b>>::square_transpose(&mut values);
			prop_assert_eq!(values, expected);
		}
	}
}
