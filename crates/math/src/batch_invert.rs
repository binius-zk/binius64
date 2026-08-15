// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

//! Batch multiplicative inversion via Montgomery's trick.

use std::iter;

use binius_field::{Field, PackedField};

/// Reusable batch inversion context that owns its scratch buffers.
///
/// Reusing one instance across many same-size calls avoids reallocating on every call.
pub struct BatchInversion<P: PackedField> {
	/// Number of packed elements this instance is sized for.
	n: usize,
	/// Scratch space used by the pairwise-tree recursion.
	scratchpad: Vec<P>,
	/// Flat scalar index of every zero found by the most recent zero-tolerant call.
	zero_indices: Vec<usize>,
}

impl<P: PackedField> BatchInversion<P> {
	/// Creates a new batch inversion context sized for `n` packed elements.
	///
	/// # Arguments
	/// - `n`: the number of packed elements every future call must be invoked with.
	///
	/// # Panics
	/// Panics if `n` is 0.
	pub fn new(n: usize) -> Self {
		// No elements to invert, and nothing to allocate, when n is 0.
		assert!(n > 0, "n must be greater than 0");

		Self {
			n,
			scratchpad: vec![P::zero(); min_scratchpad_size(n)],
			zero_indices: Vec::new(),
		}
	}

	/// Inverts every element of the slice in place.
	///
	/// # Arguments
	/// - `elements`: the slice to invert in place.
	///
	/// # Safety
	/// Every scalar element must be non-zero.
	/// Behavior is undefined if any scalar is zero.
	///
	/// # Panics
	/// Panics if the slice length does not equal the `n` given at construction.
	pub fn invert_nonzero(&mut self, elements: &mut [P]) {
		assert_eq!(
			elements.len(),
			self.n,
			"elements.len() must equal n (expected {}, got {})",
			self.n,
			elements.len()
		);

		self.batch_invert_nonzero(elements);
	}

	/// Inverts every element of the slice in place, leaving zero elements as zero.
	///
	/// # Arguments
	/// - `elements`: the slice to invert in place.
	///
	/// # Panics
	/// Panics if the slice length does not equal the `n` given at construction.
	pub fn invert_or_zero(&mut self, elements: &mut [P]) {
		assert_eq!(
			elements.len(),
			self.n,
			"elements.len() must equal n (expected {}, got {})",
			self.n,
			elements.len()
		);

		// Zero has no inverse, so swap every zero scalar for a one, recording where.
		self.zero_indices.clear();
		for (packed_idx, packed) in elements.iter_mut().enumerate() {
			for lane in 0..P::WIDTH {
				if packed.get(lane) == P::Scalar::ZERO {
					packed.set(lane, P::Scalar::ONE);
					self.zero_indices.push(packed_idx * P::WIDTH + lane);
				}
			}
		}

		// Every scalar is non-zero now, so batch-invert directly.
		self.invert_nonzero(elements);

		// Restore the zeros — inverting one just gives one back.
		for &scalar_idx in &self.zero_indices {
			elements[scalar_idx / P::WIDTH].set(scalar_idx % P::WIDTH, P::Scalar::ZERO);
		}
	}

	/// Runs the pairwise-tree inversion using this context's own scratch buffer.
	fn batch_invert_nonzero(&mut self, elements: &mut [P]) {
		batch_invert_nonzero_with_scratchpad(elements, &mut self.scratchpad);
	}
}

/// Size of the scratchpad needed by the pairwise-tree recursion.
///
/// The recursion halves the element count at every level until it reaches 1.
/// It needs one scratch slot per level: `ceil(n/2) + ceil(n/4) + ... + 1`.
///
/// # Arguments
/// - `n`: the number of elements the recursion starts from.
///
/// # Returns
/// The total number of scratch slots needed across every level below the top.
///
/// # Panics
/// Panics if `n` is 0.
fn min_scratchpad_size(mut n: usize) -> usize {
	assert!(n > 0);

	let mut size = 0;
	// Sum each level's element count until only one element is left.
	while n > 1 {
		n = n.div_ceil(2);
		size += n;
	}
	size
}

/// Inverts every element of the slice in place, organized as a balanced binary tree.
///
/// # Arguments
/// - `elements`: the slice to invert in place.
/// - `scratchpad`: scratch space for the recursion, with one slot per element at every level below
///   the top.
///
/// # Safety
/// Every element must be non-zero.
/// Behavior is undefined if any scalar is zero.
///
/// # Algorithm
/// Each level pairs element `i` with element `half + i`, multiplying to halve the count.
/// Recursing down reaches a single combined product, which gets inverted directly.
/// Unwinding multiplies that inverse back against the saved products.
/// This recovers every individual inverse.
///
/// Elements paired at the same level never depend on each other.
/// So the CPU can pipeline their multiplications instead of stalling on one chain.
///
/// Walking through 4 elements `a, b, c, d`:
/// ```text
/// elements:             [ a,   b,   c,   d  ]
/// pairwise products:    [ a*c,     b*d      ]   (pairs i with half + i)
/// recurse to 1 element: invert (a*c)*(b*d) once
/// unwind one level:     [ (a*c)^-1, (b*d)^-1 ]
/// unwind one level:     [ a^-1, b^-1, c^-1, d^-1 ]
/// ```
fn batch_invert_nonzero_with_scratchpad<P: PackedField>(elements: &mut [P], scratchpad: &mut [P]) {
	debug_assert!(!elements.is_empty());

	if elements.len() == 1 {
		// Safety: inputs are non-zero, so their product is non-zero in every lane.
		// A packed type inverts every lane on its own — no manual unpacking needed.
		elements[0] = unsafe { elements[0].invert() };
		return;
	}

	// The next level's products go in the front of the scratch buffer.
	// The rest stays free for deeper levels of the recursion.
	let next_layer_len = elements.len().div_ceil(2);
	let (next_layer, remaining) = scratchpad.split_at_mut(next_layer_len);

	// Down: combine pairs into the next, half-as-long level.
	product_layer(elements, next_layer);
	// Recurse until a single combined product is left, then invert it directly.
	batch_invert_nonzero_with_scratchpad(next_layer, remaining);
	// Up: turn the single inverse for this level back into one inverse per element.
	unproduct_layer(next_layer, elements);
}

/// Computes element-wise products of the top and bottom halves of a slice.
///
/// Pairs `input[i]` with `input[half + i]`.
/// The middle element is copied through unpaired when the input length is odd.
///
/// # Arguments
/// - `input`: the elements to pair up and multiply.
/// - `output`: destination for the products, with length `input.len().div_ceil(2)`.
///
/// # Panics
/// Panics in debug builds if `output.len() != input.len().div_ceil(2)`.
#[inline]
fn product_layer<P: PackedField>(input: &[P], output: &mut [P]) {
	debug_assert_eq!(output.len(), input.len().div_ceil(2));

	// The bottom half has exactly output.len() elements.
	// The top half is whatever remains, one shorter when the length is odd.
	let (lo, hi) = input.split_at(output.len());
	let mut out_lo_iter = iter::zip(output, lo);

	// Odd length: the last bottom-half element has no partner — copy it through.
	if hi.len() < out_lo_iter.len() {
		let Some((out_i, lo_i)) = out_lo_iter.next_back() else {
			// Always called with 2 or more elements, so this iterator is never empty.
			unreachable!("out_lo_iter.len() must be greater than zero");
		};
		*out_i = *lo_i;
	}
	// Every remaining pair has both halves: multiply them together.
	for ((out_i, &lo_i), &hi_i) in iter::zip(out_lo_iter, hi) {
		*out_i = lo_i * hi_i;
	}
}

/// Unwinds a pairwise product pass to recover individual inverses.
///
/// Given inverted pair-products and the original paired values, recovers:
/// - `output[i] = input[i] * output[half + i]` (inverse of the bottom-half element)
/// - `output[half + i] = input[i] * output[i]` (inverse of the top-half element)
///
/// # Arguments
/// - `input`: the inverted product for each pair, from the level above.
/// - `output`: the original paired elements on entry, overwritten with their inverses.
///
/// # Panics
/// Panics in debug builds if `input.len() != output.len().div_ceil(2)`.
#[inline]
fn unproduct_layer<P: PackedField>(input: &[P], output: &mut [P]) {
	debug_assert_eq!(input.len(), output.len().div_ceil(2));

	// Mirrors the split from the product pass.
	// The bottom half pairs one-to-one with `input`.
	// The top half is whatever remains.
	let (lo, hi) = output.split_at_mut(input.len());
	let mut lo_in_iter = iter::zip(lo, input);

	// Odd length: the last element was unpaired, so its own product is its inverse.
	if hi.len() < lo_in_iter.len() {
		let Some((lo_i, in_i)) = lo_in_iter.next_back() else {
			// Always called with 1 or more pairs, so this iterator is never empty.
			unreachable!("out_lo_iter.len() must be greater than zero");
		};
		*lo_i = *in_i;
	}
	// Each pair recovers both halves, using their shared inverse and saved values.
	for ((lo_i, &in_i), hi_i) in iter::zip(lo_in_iter, hi) {
		let lo_tmp = *lo_i;
		let hi_tmp = *hi_i;
		*lo_i = in_i * hi_tmp;
		*hi_i = in_i * lo_tmp;
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash as Ghash, Random, arithmetic_traits::InvertOrZero};
	use proptest::prelude::*;
	use rand::{Rng, SeedableRng, rngs::StdRng, seq::IteratorRandom};

	use super::*;

	/// Shared helper to test batch inversion with a given inverter.
	fn invert_with_inverter(
		inverter: &mut BatchInversion<Ghash>,
		n: usize,
		n_zeros: usize,
		rng: &mut impl Rng,
	) {
		assert!(n_zeros <= n, "n_zeros must be <= n");

		// Pick n_zeros distinct positions out of n to force to zero.
		// Every other position stays random and non-zero.
		let zero_indices: Vec<usize> = (0..n).sample(rng, n_zeros);

		// Build the input slice from those positions.
		let mut state = Vec::with_capacity(n);
		for i in 0..n {
			if zero_indices.contains(&i) {
				state.push(Ghash::ZERO);
			} else {
				state.push(Ghash::random(&mut *rng));
			}
		}

		// Reference result: invert every element independently, one call per element.
		let expected: Vec<Ghash> = state
			.iter()
			.map(|x| InvertOrZero::invert_or_zero(*x))
			.collect();

		// Result under test: invert the whole batch through the zero-tolerant entry point.
		inverter.invert_or_zero(&mut state);

		// The batched result must match the per-element reference exactly, zeros included.
		assert_eq!(state, expected);
	}

	fn test_batch_inversion_for_size(n: usize, n_zeros: usize, rng: &mut impl Rng) {
		// Fresh context sized for exactly n elements.
		let mut inverter = BatchInversion::<Ghash>::new(n);
		invert_with_inverter(&mut inverter, n, n_zeros, rng);
	}

	fn test_batch_inversion_nonzero_for_size(n: usize, rng: &mut impl Rng) {
		// Every element is random and non-zero.
		// So the non-zero-only entry point is safe to use directly.
		let mut state = Vec::with_capacity(n);
		for _ in 0..n {
			state.push(Ghash::random(&mut *rng));
		}

		// Reference result: invert every element independently, one call per element.
		let expected: Vec<Ghash> = state
			.iter()
			.map(|x| InvertOrZero::invert_or_zero(*x))
			.collect();

		let mut inverter = BatchInversion::<Ghash>::new(n);
		inverter.invert_nonzero(&mut state);

		// The batched result must match the per-element reference exactly.
		assert_eq!(state, expected);
	}

	proptest! {
		#[test]
		fn test_batch_inversion(n in 1usize..=16, n_zeros in 0usize..=16) {
			// n_zeros counts positions to zero out of n, so it can never exceed n.
			// Discard the proptest cases where the generator picked past that.
			prop_assume!(n_zeros <= n);
			let mut rng = StdRng::seed_from_u64(0);
			test_batch_inversion_for_size(n, n_zeros, &mut rng);
		}

		#[test]
		fn test_batch_inversion_nonzero(n in 1usize..=16) {
			let mut rng = StdRng::seed_from_u64(0);
			test_batch_inversion_nonzero_for_size(n, &mut rng);
		}
	}

	#[test]
	fn test_batch_inversion_reuse() {
		let mut rng = StdRng::seed_from_u64(0);
		// One context, reused across every zero count from 0 to 8 below.
		// This checks that the zero mask from one call never leaks into the next.
		let mut inverter = BatchInversion::<Ghash>::new(8);

		for n_zeros in 0..=8 {
			invert_with_inverter(&mut inverter, 8, n_zeros, &mut rng);
		}
	}

	#[test]
	fn test_batch_inversion_packed() {
		use crate::test_utils::Packed128b;

		let mut rng = StdRng::seed_from_u64(0);
		const N: usize = 4;

		// Packed128b packs 4 scalar lanes into one packed element.
		// So 4 packed elements cover 16 scalars in total.
		// Place zeros at 2 of those 16 positions: word 1's lane 0, and word 2's lane 2.
		//
		//     word:  0            1            2            3
		//     lane:  [_, _, _, _] [0, _, _, _] [_, _, 0, _] [_, _, _, _]
		let mut state: Vec<Packed128b> = (0..N)
			.map(|i| {
				Packed128b::from_fn(|lane| {
					if (i == 1 && lane == 0) || (i == 2 && lane == 2) {
						Ghash::ZERO
					} else {
						Ghash::random(&mut rng)
					}
				})
			})
			.collect();

		// Reference result: invert every scalar lane independently.
		let expected: Vec<Packed128b> = state
			.iter()
			.map(|packed| Packed128b::from_scalars(packed.iter().map(InvertOrZero::invert_or_zero)))
			.collect();

		// Result under test: invert the whole batch of packed elements at once.
		let mut inverter = BatchInversion::<Packed128b>::new(N);
		inverter.invert_or_zero(&mut state);

		assert_eq!(state, expected);
	}
}
