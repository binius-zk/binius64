// Copyright 2026 The Binius Developers

//! Itoh-Tsujii inversion for the GHASH field `GF(2^128)`.
//!
//! For a non-zero `x`, the inverse is `x^(2^128 - 2) = (x^(2^127 - 1))^2`. The exponent `2^127 - 1`
//! is built up with an addition chain on the powers `beta_k := x^(2^k - 1)`, using the identity
//!
//! ```text
//! beta_{a+b} = (beta_a)^(2^b) * beta_b.
//! ```
//!
//! Squaring `beta_a` repeatedly `b` times (the `x -> x^(2^b)` power map) is an `F_2`-linear
//! transformation. We precompute each power map as a [`BytewiseLookupTransformation`] (the [Method
//! of Four Russians]), wrapped into a `GhashB128 -> GhashB128` transform, and hold them in a
//! process-wide [`LazyLock`] so the tables are computed once and shared read-only across all
//! threads.
//!
//! [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>

use std::{array, iter, ops::Mul, sync::LazyLock};

use bytemuck::TransparentWrapper;

use crate::{
	BinaryField1b, Divisible, ExtensionField,
	arch::M128,
	arithmetic_traits::{InvertOrZero, Square},
	ghash::BinaryField128bGhash as GhashB128,
	linear_transformation::{
		BytewiseLookupTransformation, BytewiseLookupTransformationFactory,
		InputWrappingTransformationFactory, LinearTransformationFactory,
		OutputWrappingTransformationFactory, Transformation, WrappingTransformation,
	},
};

/// Number of bits in a GHASH element.
const FIELD_BITS: usize = 128;

/// A precomputed `x -> x^(2^n)` power map as a byte-lookup transform on `GhashB128`.
///
/// The underlying [`BytewiseLookupTransformation`] operates on the underlier `M128`; the input and
/// output wrappers lift it to a `GhashB128 -> GhashB128` transform.
type GhashPowerMap =
	WrappingTransformation<BytewiseLookupTransformation<M128, M128>, GhashB128, GhashB128>;

/// The power maps needed by the Itoh-Tsujii addition chain for the GHASH field.
///
/// Each field holds the transform for one power map `x -> x^(2^n)`, for the values of `n` that
/// appear in the chain (`pow_2_7` is reused for both the `7 -> 14` and `56 -> 63` steps).
struct GhashPowerMapTables {
	pow_2_3: GhashPowerMap,
	pow_2_7: GhashPowerMap,
	pow_2_14: GhashPowerMap,
	pow_2_28: GhashPowerMap,
	pow_2_63: GhashPowerMap,
}

impl GhashPowerMapTables {
	fn new() -> Self {
		Self {
			pow_2_3: compute_power_map_transform(3),
			pow_2_7: compute_power_map_transform(7),
			pow_2_14: compute_power_map_transform(14),
			pow_2_28: compute_power_map_transform(28),
			pow_2_63: compute_power_map_transform(63),
		}
	}
}

static GHASH_POWER_MAP_TABLES: LazyLock<GhashPowerMapTables> =
	LazyLock::new(GhashPowerMapTables::new);

/// Build the byte-lookup transform for the power map `x -> x^(2^n)` over `GhashB128`.
///
/// The power map is the `F_2`-linear transformation whose matrix has one column per input bit
/// (`compute_power_map_matrix`). [`BytewiseLookupTransformation`] turns that column set into
/// byte-indexed lookup tables; the input/output wrappers make it accept and return `GhashB128`.
fn compute_power_map_transform(n: usize) -> GhashPowerMap {
	let matrix = compute_power_map_matrix(n);
	OutputWrappingTransformationFactory::<_, GhashB128, GhashB128>::new(
		InputWrappingTransformationFactory::<_, GhashB128, M128>::new(
			BytewiseLookupTransformationFactory,
		),
	)
	.create(&matrix)
}

/// Compute the matrix of the `F_2`-linear power map `x -> x^(2^n)`.
///
/// Column `i` is the image of the `i`-th basis element, i.e. `basis(i)^(2^n)`, obtained by squaring
/// `n` times.
fn compute_power_map_matrix(n: usize) -> [GhashB128; FIELD_BITS] {
	array::from_fn(|i| {
		let basis = <GhashB128 as ExtensionField<BinaryField1b>>::basis(i);
		iter::successors(Some(basis), |basis_pow_2_i| Some(basis_pow_2_i.square()))
			.nth(n)
			.expect("closure always returns Some")
	})
}

/// Invert each GHASH element (scalar or packed) via the Itoh-Tsujii algorithm.
///
/// Zero elements map to zero, matching `InvertOrZero` semantics.
///
/// The bound is phrased in terms of the field operations (`Square`, `Mul`) plus
/// `Divisible<GhashB128>` rather than `P: PackedField`. `PackedField`'s blanket impl lists
/// `InvertOrZero` in its where-clause, so requiring it here would form a trait-resolution cycle
/// when this function backs the `InvertOrZero` impls. `Divisible<GhashB128>` carries no such
/// obligation, keeps the function statically GHASH-typed, and is satisfied both by the GHASH packed
/// fields and (reflexively) by the scalar `BinaryField128bGhash`, so the scalar inverts directly
/// without routing through a packed type.
pub fn invert_b128<P>(x: P) -> P
where
	P: Copy + Square + Mul<Output = P> + Divisible<GhashB128>,
{
	let tables = &*GHASH_POWER_MAP_TABLES;

	// Addition chain for 127: 1, 2, 3, 6, 7, 14, 28, 56, 63, 126, 127.
	let beta_1 = x;
	let beta_2 = beta_1.square() * beta_1;
	let beta_3 = beta_2.square() * beta_1;
	let beta_6 = pow_2_n(beta_3, &tables.pow_2_3) * beta_3;
	let beta_7 = beta_6.square() * beta_1;
	let beta_14 = pow_2_n(beta_7, &tables.pow_2_7) * beta_7;
	let beta_28 = pow_2_n(beta_14, &tables.pow_2_14) * beta_14;
	let beta_56 = pow_2_n(beta_28, &tables.pow_2_28) * beta_28;
	let beta_63 = pow_2_n(beta_56, &tables.pow_2_7) * beta_7;
	let beta_126 = pow_2_n(beta_63, &tables.pow_2_63) * beta_63;
	let beta_127 = beta_126.square() * beta_1;
	// x^(-1) = (x^(2^127 - 1))^2.
	beta_127.square()
}

/// Apply the power map `x -> x^(2^n)` to every GHASH scalar of `x`.
fn pow_2_n<P>(x: P, power_map: &GhashPowerMap) -> P
where
	P: Divisible<GhashB128>,
{
	Divisible::<GhashB128>::from_iter(
		Divisible::<GhashB128>::value_iter(x).map(|scalar| power_map.transform(&scalar)),
	)
}

/// `InvertOrZero` strategy wrapper backed by the [Itoh-Tsujii](invert_b128) inversion.
///
/// This is the single inversion strategy for the GHASH field across every architecture — there is
/// no carryless-multiply inverse, so the same addition-chain algorithm applies whether the square
/// and multiply underneath are CLMUL/PMULL-accelerated or software. Each arch type-aliases its
/// `GhashInvert1x` to this wrapper.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashItohTsujii<T>(T);

impl<P> InvertOrZero for GhashItohTsujii<P>
where
	P: Copy + Square + Mul<Output = P> + Divisible<GhashB128>,
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		Self::wrap(invert_b128(Self::peel(self)))
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::{Field, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedField};

	#[test]
	fn test_compute_power_map_matrix_is_squaring() {
		// The 2^1 power map is just squaring; column i must equal basis(i)^2.
		let matrix = compute_power_map_matrix(1);
		for i in 0..FIELD_BITS {
			let basis = <GhashB128 as ExtensionField<BinaryField1b>>::basis(i);
			assert_eq!(matrix[i], basis.square());
		}
	}

	#[test]
	fn test_power_map_transform_matches_repeated_squaring() {
		let power_map = compute_power_map_transform(7);
		for &raw in &[0u128, 1, 2, 0x87, 0x21ac73a21d46a21badd6747bcdfc5d4d] {
			let x = GhashB128::from(raw);
			let mut expected = x;
			for _ in 0..7 {
				expected = expected.square();
			}
			assert_eq!(power_map.transform(&x), expected);
		}
	}

	#[test]
	fn test_invert_b128_known_values() {
		let one = PackedBinaryGhash1x128b::broadcast(GhashB128::ONE);
		assert_eq!(invert_b128(one), one);

		let zero = PackedBinaryGhash1x128b::broadcast(GhashB128::ZERO);
		assert_eq!(invert_b128(zero), zero);
	}

	// `invert_b128` now backs `InvertOrZero` itself, so the multiplicative-inverse property (with
	// `0 -> 0`) is the independent oracle: given a separately-tested `mul`, `x * x^-1 == 1` fully
	// characterizes invert-or-zero.
	proptest! {
		#[test]
		fn test_invert_b128_is_multiplicative_inverse_scalar(raw in any::<u128>()) {
			let x = GhashB128::from(raw);
			let inv = invert_b128(x);
			if x == GhashB128::ZERO {
				prop_assert_eq!(inv, GhashB128::ZERO);
			} else {
				prop_assert_eq!(x * inv, GhashB128::ONE);
			}
		}

		#[test]
		fn test_invert_b128_is_multiplicative_inverse_1x(raw in any::<u128>()) {
			let scalar = GhashB128::from(raw);
			let x = PackedBinaryGhash1x128b::broadcast(scalar);
			let inv = invert_b128(x);
			if scalar == GhashB128::ZERO {
				prop_assert_eq!(inv, x);
			} else {
				prop_assert_eq!(x * inv, PackedBinaryGhash1x128b::broadcast(GhashB128::ONE));
			}
		}

		#[test]
		fn test_invert_b128_is_multiplicative_inverse_2x(a in any::<u128>(), b in any::<u128>()) {
			let x = PackedBinaryGhash2x128b::from_scalars([a, b].map(GhashB128::from));
			let inv = invert_b128(x);
			let ones = PackedBinaryGhash2x128b::from_scalars(
				[a, b].map(|raw| {
					if GhashB128::from(raw) == GhashB128::ZERO {
						GhashB128::ZERO
					} else {
						GhashB128::ONE
					}
				}),
			);
			prop_assert_eq!(x * inv, ones);
		}
	}
}
