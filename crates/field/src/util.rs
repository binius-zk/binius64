// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_utils::{checked_arithmetics::checked_int_div, rayon::prelude::*};

use crate::{ExtensionField, Field, PackedField, packed::get_packed_slice_unchecked};

/// Computes the inner product of two vectors without checking that the lengths are equal
#[inline]
pub fn inner_product_unchecked<F, FE>(
	a: impl IntoIterator<Item = FE>,
	b: impl IntoIterator<Item = F>,
) -> FE
where
	F: Field,
	FE: ExtensionField<F>,
{
	iter::zip(a, b).map(|(a_i, b_i)| a_i * b_i).sum()
}

/// Calculate inner product for potentially big slices of xs and ys.
/// The number of elements in xs has to be less or equal to the number of elements in ys.
pub fn inner_product_par<FX, PX, PY>(xs: &[PX], ys: &[PY]) -> FX
where
	PX: PackedField<Scalar = FX>,
	PY: PackedField,
	FX: ExtensionField<PY::Scalar>,
{
	assert!(
		PX::WIDTH * xs.len() <= PY::WIDTH * ys.len(),
		"Y elements has to be at least as wide as X elements"
	);

	// If number of elements in xs is less than number of elements in ys this will be because due to
	// packing so we can use single-threaded version of the function.
	if PX::WIDTH * xs.len() < PY::WIDTH * ys.len() {
		return inner_product_unchecked(PackedField::iter_slice(xs), PackedField::iter_slice(ys));
	}

	let calc_product_by_ys = |xs: &[PX], ys: &[PY]| {
		let mut result = FX::ZERO;

		for (j, y) in ys.iter().enumerate() {
			for (k, y) in y.iter().enumerate() {
				result += unsafe { get_packed_slice_unchecked(xs, j * PY::WIDTH + k) } * y
			}
		}

		result
	};

	// These magic numbers were chosen experimentally to have a reasonable performance
	// for the calls with small number of elements.
	// For different field sizes, the numbers may need to be adjusted.
	const CHUNK_SIZE: usize = 64;
	if ys.len() < 16 * CHUNK_SIZE {
		calc_product_by_ys(xs, ys)
	} else {
		// According to benchmark results iterating by chunks here is more efficient than using
		// `par_iter` with `min_length` directly.
		ys.par_chunks(CHUNK_SIZE)
			.enumerate()
			.map(|(i, ys)| {
				let offset = i * checked_int_div(CHUNK_SIZE * PY::WIDTH, PX::WIDTH);
				calc_product_by_ys(&xs[offset..], ys)
			})
			.sum()
	}
}

/// Iterate the powers of a given value, beginning with 1 (the 0'th power).
pub fn powers<F: Field>(val: F) -> impl Iterator<Item = F> {
	iter::successors(Some(F::ONE), move |&power| Some(power * val))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{BinaryField128bGhash, PackedBinaryGhash4x128b};

	type P = PackedBinaryGhash4x128b;
	type F = BinaryField128bGhash;

	#[test]
	fn test_inner_product_par_equal_length() {
		// xs and ys have the same number of packed elements
		let xs1 = F::new(1);
		let xs2 = F::new(2);
		let xs = vec![P::set_single(xs1), P::set_single(xs2)];
		let ys1 = F::new(3);
		let ys2 = F::new(4);
		let ys = vec![P::set_single(ys1), P::set_single(ys2)];

		let result = inner_product_par::<F, P, P>(&xs, &ys);
		let expected = xs1 * ys1 + xs2 * ys2;

		assert_eq!(result, expected);
	}

	#[test]
	fn test_inner_product_par_unequal_length() {
		// ys is larger than xs due to packing differences
		let xs1 = F::new(1);
		let xs = vec![P::set_single(xs1)];
		let ys1 = F::new(2);
		let ys2 = F::new(3);
		let ys = vec![P::set_single(ys1), P::set_single(ys2)];

		let result = inner_product_par::<F, P, P>(&xs, &ys);
		let expected = xs1 * ys1;

		assert_eq!(result, expected);
	}

	#[test]
	fn test_inner_product_par_large_input_single_threaded() {
		// Large input but not enough to trigger parallel execution
		let size = 256;
		let xs: Vec<P> = (0..size)
			.map(|i| P::set_single(F::new(i as u128)))
			.collect();
		let ys: Vec<P> = (0..size)
			.map(|i| P::set_single(F::new((i + 1) as u128)))
			.collect();

		let result = inner_product_par::<F, P, P>(&xs, &ys);

		let expected = (0..size)
			.map(|i| F::new(i as u128) * F::new((i + 1) as u128))
			.sum::<F>();

		assert_eq!(result, expected);
	}

	#[test]
	fn test_inner_product_par_large_input_par() {
		// Large input to test parallel execution
		let size = 2000;
		let xs: Vec<P> = (0..size)
			.map(|i| P::set_single(F::new(i as u128)))
			.collect();
		let ys: Vec<P> = (0..size)
			.map(|i| P::set_single(F::new((i + 1) as u128)))
			.collect();

		let result = inner_product_par::<F, P, P>(&xs, &ys);

		let expected = (0..size)
			.map(|i| F::new(i as u128) * F::new((i + 1) as u128))
			.sum::<F>();

		assert_eq!(result, expected);
	}

	#[test]
	fn test_inner_product_par_empty() {
		// Case: Empty input should return 0
		let xs: Vec<P> = vec![];
		let ys: Vec<P> = vec![];

		let result = inner_product_par::<F, P, P>(&xs, &ys);
		let expected = F::ZERO;

		assert_eq!(result, expected);
	}
}
