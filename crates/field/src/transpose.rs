// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_utils::checked_arithmetics::checked_log_2;

use super::packed::PackedField;
use crate::{BinaryField, ExtensionField, PackedSubfield, UnderlierView, packed_extension};

/// Transpose square blocks of elements within packed field elements in place.
///
/// The input elements are interpreted as a rectangular matrix with height `n = 2^n` in row-major
/// order. This matrix is interpreted as a vector of square matrices of field elements, and each
/// square matrix is transposed in-place.
///
/// # Arguments
///
/// * `log_n`: The base-2 logarithm of the dimension of the n x n square matrix. Must be less than
///   or equal to the base-2 logarithm of the packing width.
/// * `elems`: The packed field elements, length is a power-of-two multiple of `1 << log_n`.
///
/// # Preconditions
///
/// * `log_n` must be at most `P::LOG_WIDTH`.
/// * `elems.len()` must be a power of two and at least `2^log_n`.
///
/// A caller whose dimensions are compile-time constants should use the fixed-size form below.
/// That form unrolls the butterfly and keeps the array in registers.
pub fn transpose_square_blocks<P: PackedField>(log_n: usize, elems: &mut [P]) {
	assert!(P::LOG_WIDTH >= log_n, "dimension n of square blocks must divide packing width");

	let size = elems.len();
	assert!(size.is_power_of_two(), "elems length must be a power of two, got {size}");
	let log_size = checked_log_2(size);
	assert!(
		log_size >= log_n,
		"elems must have length at least 2^log_n = {}, got {size}",
		1 << log_n
	);

	let log_w = log_size - log_n;

	// See Hacker's Delight, Section 7-3.
	// https://dl.acm.org/doi/10.5555/2462741
	for i in 0..log_n {
		for j in 0..1 << (log_n - i - 1) {
			for k in 0..1 << (log_w + i) {
				let idx0 = (j << (log_w + i + 1)) | k;
				let idx1 = idx0 | (1 << (log_w + i));

				let v0 = elems[idx0];
				let v1 = elems[idx1];
				let (v0, v1) = v0.interleave(v1, i);
				elems[idx0] = v0;
				elems[idx1] = v1;
			}
		}
	}
}

/// Transposes square blocks of scalars across a fixed-size array of packed elements, in place.
///
/// # Overview
///
/// The runtime-sized form in this module computes the same permutation.
/// This form is for a caller whose block dimension and array length are both constants.
///
/// Constant sizes let the compiler unroll the butterfly.
/// The whole array then stays in registers, which is what a caller in a hot loop wants.
///
/// # Algorithm
///
/// A butterfly network over `LOG_N` rounds, as in Hacker's Delight, Section 7-3.
/// Round `i` interleaves element pairs `2^(log_w + i)` apart at block granularity `2^i`.
///
/// # Preconditions
///
/// All three are checked at compile time, so a violating instantiation fails to build:
///
/// * The array length must be a power of two.
/// * The block dimension must not exceed the base-2 log of the array length.
/// * The block dimension must not exceed the base-2 log of the packed width.
pub fn transpose_square_blocks_array<P: PackedField, const LOG_N: usize, const S: usize>(
	elems: &mut [P; S],
) {
	const {
		assert!(LOG_N <= P::LOG_WIDTH, "LOG_N must not exceed the packed width");
		assert!(LOG_N <= checked_log_2(S), "LOG_N must not exceed the array length");
	}

	let log_size = checked_log_2(S);

	// Elements per block that stays contiguous through the butterfly.
	let log_w = log_size - LOG_N;

	for i in 0..LOG_N {
		for j in 0..1 << (LOG_N - i - 1) {
			for k in 0..1 << (log_w + i) {
				// Partner elements for this round, one stride apart.
				let idx0 = (j << (log_w + i + 1)) | k;
				let idx1 = idx0 | (1 << (log_w + i));

				// Interleaving at block granularity 2^i swaps the axes one bit at a time.
				let (v0, v1) = elems[idx0].interleave(elems[idx1], i);
				elems[idx0] = v0;
				elems[idx1] = v1;
			}
		}
	}
}

pub fn square_transforms_extension_field<F, FE>(values: &mut [FE])
where
	F: BinaryField,
	FE: PackedField<Scalar: ExtensionField<F>> + UnderlierView,
	PackedSubfield<FE, F>: PackedField<Scalar = F>,
{
	transpose_square_blocks(
		FE::Scalar::LOG_DEGREE,
		packed_extension::cast_bases_mut::<F, FE>(values),
	);
}

#[cfg(test)]
mod tests {
	use std::array;

	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{PackedBinaryField64x1b, PackedBinaryField128x1b, PackedField, Random};

	#[test]
	fn test_transpose_square_blocks_128x1b() {
		let mut elems = [
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
		];
		transpose_square_blocks(3, &mut elems);

		let expected = [
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
		];
		assert_eq!(elems, expected);
	}

	#[test]
	fn test_transpose_square_blocks_128x1b_multi_row() {
		let mut elems = [
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
		];
		transpose_square_blocks(1, &mut elems);

		let expected = [
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
		];
		assert_eq!(elems, expected);
	}

	// The fixed-size form exists only to unroll the loop, so it must compute exactly the
	// permutation the runtime form computes. Pinning them equal is what justifies keeping both.
	//
	//     same input -> runtime form   -> A
	//                -> fixed-size form -> B
	//     A == B for every block dimension the array length admits
	fn check_forms_agree<P, const LOG_N: usize, const S: usize>(seed: u64)
	where
		P: PackedField + Random,
	{
		let mut rng = StdRng::seed_from_u64(seed);

		// Random lanes over many trials cover every one of the S * WIDTH scalar positions.
		for _ in 0..100 {
			let input: [P; S] = array::from_fn(|_| P::random(&mut rng));

			let mut runtime = input;
			transpose_square_blocks(LOG_N, &mut runtime);

			let mut fixed = input;
			transpose_square_blocks_array::<P, LOG_N, S>(&mut fixed);

			assert_eq!(fixed, runtime, "forms disagree at LOG_N = {LOG_N}, S = {S}");
		}
	}

	#[test]
	fn fixed_size_form_agrees_with_runtime_form() {
		// Cover both row widths the callers run at, and every block dimension each admits.
		//
		//     64 lanes  -> LOG_N up to 6, array length 8 admits up to 3
		//     128 lanes -> LOG_N up to 7, array length 8 admits up to 3
		check_forms_agree::<PackedBinaryField64x1b, 0, 8>(0);
		check_forms_agree::<PackedBinaryField64x1b, 1, 8>(1);
		check_forms_agree::<PackedBinaryField64x1b, 3, 8>(2);
		check_forms_agree::<PackedBinaryField128x1b, 3, 8>(3);

		// A block dimension equal to the array length exercises the widest butterfly.
		check_forms_agree::<PackedBinaryField64x1b, 4, 16>(4);
		check_forms_agree::<PackedBinaryField128x1b, 5, 32>(5);
	}

	#[test]
	fn transpose_exchanges_element_axis_with_low_scalar_bits() {
		let mut rng = StdRng::seed_from_u64(0);

		// The permutation itself, stated directly rather than through either implementation.
		// Splitting a scalar position into a high part and its low three bits:
		//
		//     input : element r, position 8i + j  =  value at (r, 8i + j)
		//     output: element j, position 8i + t  =  value at (t, 8i + j)
		//
		// So the element index and the low three bits of the position trade places.
		for _ in 0..100 {
			let input: [PackedBinaryField128x1b; 8] =
				array::from_fn(|_| PackedBinaryField128x1b::random(&mut rng));
			let mut output = input;
			transpose_square_blocks_array::<_, 3, 8>(&mut output);

			// Read both sides as scalars, so the assertion is about positions and not underliers.
			let scalars = |elems: &[PackedBinaryField128x1b; 8]| {
				elems
					.iter()
					.map(|e| e.iter().collect::<Vec<_>>())
					.collect::<Vec<_>>()
			};
			let before = scalars(&input);
			let after = scalars(&output);

			// High part of the position, which the permutation leaves alone.
			for i in 0..PackedBinaryField128x1b::WIDTH / 8 {
				// Element of the output, which is the low three bits of the input position.
				for j in 0..8 {
					// Element of the input, which becomes the low three bits of the output.
					for t in 0..8 {
						assert_eq!(
							after[j][i * 8 + t],
							before[t][i * 8 + j],
							"i={i}, j={j}, t={t}"
						);
					}
				}
			}
		}
	}

	proptest! {
		#[test]
		fn transpose_is_an_involution(values in prop::collection::vec(any::<u128>(), 8)) {
			// Exchanging two axes twice restores the original layout.
			// This holds for the fixed-size form on any input, so it is a property, not a case.
			let input: [PackedBinaryField128x1b; 8] =
				array::from_fn(|i| PackedBinaryField128x1b::from(values[i]));

			let mut roundtrip = input;
			transpose_square_blocks_array::<_, 3, 8>(&mut roundtrip);
			transpose_square_blocks_array::<_, 3, 8>(&mut roundtrip);

			prop_assert_eq!(roundtrip, input);
		}
	}
}
