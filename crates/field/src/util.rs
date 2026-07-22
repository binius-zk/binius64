// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use crate::{Field, PackedField, UnderlierType, field::FieldOps};

/// An arithmetic function over field elements, generic in the field it evaluates in.
///
/// A closure `FnOnce(&[F]) -> F` is monomorphic: it runs in one fixed field.
/// Carrying the genericity on the method instead lets one value run in many fields:
/// - natively, in the verifier's own base field `F`;
/// - over any larger field `E` whose scalar is `F`.
pub trait FieldFn<F: Field> {
	/// Evaluates the function on `inputs` in the field `E`, returning one element.
	///
	/// The scalar of `E` is the base field `F`.
	/// The `From<F>` bound lets the function embed base-field constants into `E`.
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, inputs: &[E]) -> E;

	/// Evaluates the function on `inputs` natively in the base field `F`.
	///
	/// The default is `self.call::<F>(inputs)`; implementors may override with a base-field
	/// specialized fast path (e.g. deferred `WideMul` reduction) that the generic
	/// [`call`](Self::call) — which cannot assume `E: WideMul` — can't express. Callers evaluating
	/// in `F` should prefer this.
	fn call_native(&self, inputs: &[F]) -> F {
		self.call::<F>(inputs)
	}
}

/// Iterate the powers of a given value, beginning with 1 (the 0'th power).
pub fn powers<F: FieldOps>(val: F) -> impl Iterator<Item = F> {
	iter::successors(Some(F::one()), move |power| Some(power.clone() * val.clone()))
}

/// Expands an array of field elements into all possible subset sums.
///
/// For an input array `[a, b, c]`, this computes all possible sums of subsets:
/// `[0, a, b, a+b, c, a+c, b+c, a+b+c]`
///
/// This is used to create lookup tables for the Method of Four Russians optimization,
/// where we precompute all possible combinations of a small set of values to avoid
/// doing the additions at runtime.
///
/// ## Type Parameters
///
/// * `F` - The field element type
/// * `N` - Size of the input array
/// * `N_EXP2` - Size of the output array, must be 2^N
///
/// ## Arguments
///
/// * `elems` - Input array of N field elements
///
/// ## Returns
///
/// An array of size N_EXP2 containing all possible subset sums of the input elements
///
/// ## Preconditions
///
/// * N_EXP2 must equal 2^N
///
/// ## Example
///
/// ```ignore
/// let input = [F::ONE, F::from(2)];
/// let sums = expand_subset_sums_array(input);
/// // sums = [F::ZERO, F::ONE, F::from(2), F::from(3)]
/// ```
pub fn expand_subset_sums_array<P: PackedField, const N: usize, const N_EXP2: usize>(
	elems: [P; N],
) -> [P; N_EXP2] {
	assert_eq!(N_EXP2, 1 << N);

	let mut expanded = [P::zero(); N_EXP2];
	for (i, elem_i) in elems.into_iter().enumerate() {
		let span = &mut expanded[..1 << (i + 1)];
		let (lo_half, hi_half) = span.split_at_mut(1 << i);
		for (lo_half_i, hi_half_i) in iter::zip(lo_half, hi_half) {
			*hi_half_i = *lo_half_i + elem_i;
		}
	}
	expanded
}

/// Expands `elems` into all `2^N` subset XOR combinations, indexed by subset bitmask.
///
/// Entry `mask` holds the XOR of `elems[i]` over every bit `i` set in `mask`. This is the
/// bitwise-XOR analogue of [`expand_subset_sums_array`] over raw underliers, used to build Method
/// of Four Russians lookup tables.
///
/// ## Preconditions
///
/// * `N_EXP2` must equal `2^N`
pub fn expand_subset_xors<U: UnderlierType, const N: usize, const N_EXP2: usize>(
	elems: [U; N],
) -> [U; N_EXP2] {
	assert_eq!(N_EXP2, 1 << N);

	let mut expanded = [U::ZERO; N_EXP2];
	for (i, elem_i) in elems.into_iter().enumerate() {
		let span = &mut expanded[..1 << (i + 1)];
		let (lo_half, hi_half) = span.split_at_mut(1 << i);
		for (lo_half_i, hi_half_i) in iter::zip(lo_half, hi_half) {
			*hi_half_i = *lo_half_i ^ elem_i;
		}
	}
	expanded
}

#[cfg(test)]
mod tests {
	use std::array;

	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{BinaryField128bGhash, Random};

	#[test]
	fn test_powers_against_pow() {
		let generator = BinaryField128bGhash::MULTIPLICATIVE_GENERATOR;
		let power_values: Vec<_> = powers(generator).take(10).collect();

		for i in 0..10 {
			assert_eq!(power_values[i], generator.pow(i as u64));
		}
	}

	type F = BinaryField128bGhash;

	/// Expands `N` random elements and asserts that entry `index` of the resulting `2^N`-sized
	/// lookup table equals the subset sum selected by the set bits of `index`.
	fn check_subset_sums<const N: usize, const N_EXP2: usize>(seed: u64, index: usize) {
		let mut rng = StdRng::seed_from_u64(seed);
		let elems: [F; N] = array::from_fn(|_| F::random(&mut rng));

		let result = expand_subset_sums_array::<_, N, N_EXP2>(elems);
		assert_eq!(result.len(), N_EXP2);

		// Compute expected sum based on the binary representation of index.
		let index = index % N_EXP2;
		let mut expected = F::ZERO;
		for (bit_pos, &elem) in elems.iter().enumerate() {
			if (index >> bit_pos) & 1 == 1 {
				expected += elem;
			}
		}

		assert_eq!(
			result[index], expected,
			"index {index} should hold the subset sum for its binary representation"
		);
	}

	proptest! {
		#[test]
		fn test_expand_subset_sums_array_correctness(
			n in 0usize..=8,  // Input length (small to avoid exponential blowup)
			index in 0usize..256,  // Index to check
		) {
			// Dispatch to the const-generic helper: `expand_subset_sums_array` needs the output
			// length `2^n` at compile time.
			match n {
				0 => check_subset_sums::<0, 1>(n as u64, index),
				1 => check_subset_sums::<1, 2>(n as u64, index),
				2 => check_subset_sums::<2, 4>(n as u64, index),
				3 => check_subset_sums::<3, 8>(n as u64, index),
				4 => check_subset_sums::<4, 16>(n as u64, index),
				5 => check_subset_sums::<5, 32>(n as u64, index),
				6 => check_subset_sums::<6, 64>(n as u64, index),
				7 => check_subset_sums::<7, 128>(n as u64, index),
				8 => check_subset_sums::<8, 256>(n as u64, index),
				_ => unreachable!("n is constrained to 0..=8"),
			}
		}
	}
}
