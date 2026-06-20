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
//! Russians]) and store it in thread-local storage so the tables are computed once per thread.
//!
//! [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>

use std::{array, iter, thread::LocalKey};

use crate::{
	BinaryField1b, ExtensionField, Field, PackedField, arithmetic_traits::Square,
	ghash::BinaryField128bGhash as GhashB128, util::expand_subset_sums_array,
};

/// Number of bits in a GHASH element.
const FIELD_BITS: usize = 128;
/// Number of bytes in a GHASH element.
const FIELD_BYTES: usize = FIELD_BITS / 8;

thread_local! {
	static POW_2_3_TABLES: [[GhashB128; 256]; FIELD_BYTES] = compute_power_map_byte_lookup_tables(3);
	static POW_2_7_TABLES: [[GhashB128; 256]; FIELD_BYTES] = compute_power_map_byte_lookup_tables(7);
	static POW_2_14_TABLES: [[GhashB128; 256]; FIELD_BYTES] =
		compute_power_map_byte_lookup_tables(14);
	static POW_2_28_TABLES: [[GhashB128; 256]; FIELD_BYTES] =
		compute_power_map_byte_lookup_tables(28);
	static POW_2_63_TABLES: [[GhashB128; 256]; FIELD_BYTES] =
		compute_power_map_byte_lookup_tables(63);
}

/// Compute a byte-wise lookup table of the power map `x -> x^(2^n)` as an `F_2`-linear
/// transformation.
///
/// The transformation matrix has one column per input bit (`compute_power_map_matrix`). The columns
/// are split into [`FIELD_BYTES`] chunks of 8 bits, and for each chunk we precompute the linear
/// combination of its columns for every possible byte value. Applying the transform to an input
/// then reduces to one table lookup per input byte, XOR-ing the results together (see
/// [`apply_power_map`]).
fn compute_power_map_byte_lookup_tables(n: usize) -> [[GhashB128; 256]; FIELD_BYTES] {
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

/// Invert a packed vector of GHASH elements via the Itoh-Tsujii algorithm.
///
/// Zero elements map to zero, matching `InvertOrZero` semantics.
pub fn invert_b128<P>(x: P) -> P
where
	P: PackedField<Scalar = GhashB128>,
{
	// Addition chain for 127: 1, 2, 3, 6, 7, 14, 28, 56, 63, 126, 127.
	let beta_1 = x;
	let beta_2 = beta_1.square() * beta_1;
	let beta_3 = beta_2.square() * beta_1;
	let beta_6 = pow_2_n(beta_3, &POW_2_3_TABLES) * beta_3;
	let beta_7 = beta_6.square() * beta_1;
	let beta_14 = pow_2_n(beta_7, &POW_2_7_TABLES) * beta_7;
	let beta_28 = pow_2_n(beta_14, &POW_2_14_TABLES) * beta_14;
	let beta_56 = pow_2_n(beta_28, &POW_2_28_TABLES) * beta_28;
	let beta_63 = pow_2_n(beta_56, &POW_2_7_TABLES) * beta_7;
	let beta_126 = pow_2_n(beta_63, &POW_2_63_TABLES) * beta_63;
	let beta_127 = beta_126.square() * beta_1;
	// x^(-1) = (x^(2^127 - 1))^2.
	beta_127.square()
}

/// Apply the power map `x -> x^(2^n)` to every scalar of a packed vector, using the precomputed
/// byte lookup tables held in the given thread-local.
fn pow_2_n<P>(x: P, tables: &'static LocalKey<[[GhashB128; 256]; FIELD_BYTES]>) -> P
where
	P: PackedField<Scalar = GhashB128>,
{
	tables.with(|tables| P::from_scalars(x.into_iter().map(|x| apply_power_map(x, tables))))
}

/// Apply a byte-wise lookup table power map to a single GHASH scalar.
///
/// The element is split into its little-endian bytes; byte `j` selects from `tables[j]` (which
/// covers input bits `8*j .. 8*j + 8`), and the looked-up linear combinations are XOR-ed together.
fn apply_power_map(x: GhashB128, tables: &[[GhashB128; 256]; FIELD_BYTES]) -> GhashB128 {
	let bytes = u128::from(x).to_le_bytes();
	iter::zip(tables, bytes)
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
		arithmetic_traits::InvertOrZero,
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

	proptest! {
		#[test]
		fn test_invert_b128_matches_invert_or_zero_1x(raw in any::<u128>()) {
			let x = PackedBinaryGhash1x128b::broadcast(GhashB128::from(raw));
			prop_assert_eq!(invert_b128(x), x.invert_or_zero());
		}

		#[test]
		fn test_invert_b128_is_multiplicative_inverse(raw in any::<u128>()) {
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
		fn test_invert_b128_matches_invert_or_zero_2x(a in any::<u128>(), b in any::<u128>()) {
			let x = PackedBinaryGhash2x128b::from_scalars(
				[a, b].map(GhashB128::from),
			);
			prop_assert_eq!(invert_b128(x), x.invert_or_zero());
		}
	}
}
