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
//! transformation. We precompute each power map as a byte-indexed lookup table (the [Method of Four
//! Russians]) and hold them in a process-wide [`LazyLock`], so the tables are computed once and
//! shared read-only across all threads.
//!
//! [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>

use std::{array, iter, ops::Mul, sync::LazyLock};

use crate::{
	BinaryField1b, Divisible, ExtensionField, Field, arithmetic_traits::Square,
	ghash::BinaryField128bGhash as GhashB128, util::expand_subset_sums_array,
};

/// Number of bits in a GHASH element.
const FIELD_BITS: usize = 128;
/// Number of bytes in a GHASH element.
const FIELD_BYTES: usize = FIELD_BITS / 8;

/// A single byte-indexed lookup table for one `x -> x^(2^n)` power map.
type PowerMapTable = [[GhashB128; 256]; FIELD_BYTES];

/// The power-map lookup tables needed by the Itoh-Tsujii addition chain for the GHASH field.
///
/// Each field holds the table for one power map `x -> x^(2^n)`, for the values of `n` that appear
/// in the chain (`pow_2_7` is reused for both the `7 -> 14` and `56 -> 63` steps).
struct GhashPowerMapTables {
	pow_2_3: PowerMapTable,
	pow_2_7: PowerMapTable,
	pow_2_14: PowerMapTable,
	pow_2_28: PowerMapTable,
	pow_2_63: PowerMapTable,
}

impl GhashPowerMapTables {
	fn new() -> Self {
		Self {
			pow_2_3: compute_power_map_byte_lookup_tables(3),
			pow_2_7: compute_power_map_byte_lookup_tables(7),
			pow_2_14: compute_power_map_byte_lookup_tables(14),
			pow_2_28: compute_power_map_byte_lookup_tables(28),
			pow_2_63: compute_power_map_byte_lookup_tables(63),
		}
	}
}

static GHASH_POWER_MAP_TABLES: LazyLock<GhashPowerMapTables> =
	LazyLock::new(GhashPowerMapTables::new);

/// Compute a byte-wise lookup table of the power map `x -> x^(2^n)` as an `F_2`-linear
/// transformation.
///
/// The transformation matrix has one column per input bit (`compute_power_map_matrix`). The columns
/// are split into [`FIELD_BYTES`] chunks of 8 bits, and for each chunk we precompute the linear
/// combination of its columns for every possible byte value. Applying the transform to an input
/// then reduces to one table lookup per input byte, XOR-ing the results together (see
/// [`apply_power_map`]).
fn compute_power_map_byte_lookup_tables(n: usize) -> PowerMapTable {
	let matrix = compute_power_map_matrix(n);
	array::from_fn(|byte_idx| {
		let chunk: [GhashB128; 8] = array::from_fn(|bit| matrix[byte_idx * 8 + bit]);
		expand_subset_sums_array(chunk)
	})
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

/// Apply the power map `x -> x^(2^n)` to every GHASH scalar of `x`, using the precomputed byte
/// lookup table.
fn pow_2_n<P>(x: P, table: &PowerMapTable) -> P
where
	P: Divisible<GhashB128>,
{
	Divisible::<GhashB128>::from_iter(
		Divisible::<GhashB128>::value_iter(x).map(|scalar| apply_power_map(scalar, table)),
	)
}

/// Apply a byte-wise lookup table power map to a single GHASH scalar.
///
/// The element is split into its little-endian bytes; byte `j` selects from `table[j]` (which
/// covers input bits `8*j .. 8*j + 8`), and the looked-up linear combinations are XOR-ed together.
fn apply_power_map(x: GhashB128, table: &PowerMapTable) -> GhashB128 {
	let bytes = u128::from(x).to_le_bytes();
	iter::zip(table, bytes)
		.map(|(table, byte)| table[byte as usize])
		.fold(GhashB128::ZERO, |acc, term| acc + term)
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::{
		PackedField,
		arch::{
			packed_ghash_128::PackedBinaryGhash1x128b, packed_ghash_256::PackedBinaryGhash2x128b,
		},
	};

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
	fn test_apply_power_map_matches_repeated_squaring() {
		let tables = compute_power_map_byte_lookup_tables(7);
		for &raw in &[0u128, 1, 2, 0x87, 0x21ac73a21d46a21badd6747bcdfc5d4d] {
			let x = GhashB128::from(raw);
			let mut expected = x;
			for _ in 0..7 {
				expected = expected.square();
			}
			assert_eq!(apply_power_map(x, &tables), expected);
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
