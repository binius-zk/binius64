// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField128bGhash as Ghash, Field};
use binius_utils::random_access_sequence::RandomAccessSequence;

/// Efficiently inverts multiple field elements simultaneously using Montgomery's batch inversion
/// trick.
///
/// This function implements a binary tree approach to batch field inversion, which is significantly
/// more efficient than inverting each element individually when dealing with many elements.
///
/// # Algorithm Overview
/// 1. **Product Tree Construction**: Build a binary tree bottom-up where leaves are input elements
///    and each internal node stores the product of its children using linear addressing
/// 2. **Root Inversion**: Perform single expensive field inversion on the root (product of all
///    elements)
/// 3. **Inverse Propagation**: Propagate the root inverse down through tree levels to compute
///    individual element inverses
///
/// # Performance Benefits
/// - **Single inversion**: Only one expensive field inversion instead of N inversions
/// - **Linear addressing**: Simple addition-based indexing for better cache locality
/// - **Zero handling**: Graceful handling of zero elements without division-by-zero errors
///
/// # Parameters
/// - `elements`: Array of field elements to invert (modified in-place)
/// - `scratchpad`: Working memory buffer, must be at least `2*N-1` elements
///
/// # Requirements
/// - `N` must be a power of 2 and ≥ 2
/// - `scratchpad.len() >= 2*N-1` for intermediate computations
#[inline]
pub fn batch_invert<const N: usize>(elements: &mut [Ghash], scratchpad: &mut [Ghash]) {
	assert!(N.is_power_of_two() && N >= 2, "N must be a power of 2 and >= 2");
	assert_eq!(elements.len(), N);
	assert!(scratchpad.len() >= 2 * N - 1, "scratchpad too small");

	let zero = Ghash::ZERO;
	let one = Ghash::ONE;
	let levels = N.ilog2() as usize;

	// Phase 1: Setup - Copy input elements, replacing zeros with ones
	// This prevents division-by-zero while preserving zero semantics in final output
	for i in 0..N {
		scratchpad[i] = if elements[i] == zero {
			one // Temporary replacement - restored to zero in final phase
		} else {
			elements[i]
		};
	}

	// Phase 2: Build product tree bottom-up using linear addressing
	// Each level combines pairs from the previous level into products
	let mut dest_offset = N; // Current write position in scratchpad

	// Build intermediate tree levels (N/2, N/4, N/8, ... down to 2 elements)
	for level in 1..levels {
		let level_size = N >> level; // Number of products at this level
		let src_offset = dest_offset - (level_size * 2); // Read from previous level

		// Combine adjacent pairs: scratchpad[2*i] * scratchpad[2*i+1]
		for i in 0..level_size {
			scratchpad[dest_offset + i] =
				scratchpad[src_offset + 2 * i] * scratchpad[src_offset + 2 * i + 1];
		}
		dest_offset += level_size; // Move to next level's storage
	}

	// Final level: multiply the last two products to get root
	let src_offset = dest_offset - 2;
	scratchpad[dest_offset] = scratchpad[src_offset] * scratchpad[src_offset + 1];

	// Phase 3: Invert root product (Montgomery's key insight: single inversion)
	scratchpad[dest_offset] = scratchpad[dest_offset]
		.invert()
		.expect("factors are non-zero, so product is non-zero");

	// Phase 4: Propagate inverse down tree levels (reverse order)
	// Each level computes inverses from the level above
	for level in 1..levels {
		let level_size = 1 << level; // Size doubles each level going down
		let src_offset = dest_offset; // Read from current position
		dest_offset -= level_size; // Move down to previous level

		// For each pair, compute inverses using: inv(a*b) * b = inv(a), inv(a*b) * a = inv(b)
		for i in 0..level_size >> 1 {
			let left_product = scratchpad[dest_offset + 2 * i]; // Original product a
			scratchpad[dest_offset + 2 * i] =
				scratchpad[dest_offset + 2 * i + 1] * scratchpad[src_offset + i]; // inv(a) = b * inv(a*b)
			scratchpad[dest_offset + 2 * i + 1] = left_product * scratchpad[src_offset + i]; // inv(b) = a * inv(a*b)
		}
	}

	// Phase 5: Extract final inverses and restore zero semantics
	// The last layer of products could be done in the loop immediately above,
	// but for speed we avoid an extra copy by merging it copying from the
	// scratchpad.
	for i in 0..N / 2 {
		let j = 2 * i;
		// Restore original zeros (marked in Phase 1)
		elements[j] = if elements[j] == zero {
			zero
		} else {
			scratchpad[j + 1] * scratchpad[dest_offset + i]
		};
		elements[j + 1] = if elements[j + 1] == zero {
			zero
		} else {
			scratchpad[j] * scratchpad[dest_offset + i]
		};
	}
}

/// Reusable batch inversion context that owns its scratch buffers.
///
/// This struct manages the memory needed for batch field inversions, allowing
/// efficient reuse across multiple inversion operations of the same size.
///
/// # Example
/// ```
/// use binius_field::{BinaryField128bGhash, Field};
/// use binius_math::BatchInversion;
///
/// let mut inverter = BatchInversion::<BinaryField128bGhash>::new(8);
/// let mut elements = [BinaryField128bGhash::ONE; 8];
/// inverter.invert(&mut elements);
/// ```
pub struct BatchInversion<F: Field> {
	n: usize,
	scratchpad: Vec<F>,
	is_zero: Vec<bool>,
}

impl<F: Field> BatchInversion<F> {
	/// Creates a new batch inversion context for slices of size `n`.
	///
	/// Allocates the necessary scratch space:
	/// - `scratchpad`: Storage for intermediate products during recursion
	/// - `is_zero`: Tracking vector for zero elements
	///
	/// # Parameters
	/// - `n`: The size of slices this instance will handle
	///
	/// # Panics
	/// Panics if `n` is 0.
	pub fn new(n: usize) -> Self {
		assert!(n > 0, "n must be greater than 0");

		let scratchpad_size = min_scratchpad_size(n);
		Self {
			n,
			scratchpad: vec![F::ZERO; scratchpad_size],
			is_zero: vec![false; n],
		}
	}

	/// Inverts non-zero elements in-place.
	///
	/// # Parameters
	/// - `elements`: Mutable slice to invert in-place
	///
	/// # Preconditions
	/// All elements must be non-zero. Behavior is undefined if any element is zero.
	///
	/// # Panics
	/// Panics if `elements.len() != n` (the size specified at construction).
	pub fn invert(&mut self, elements: &mut [F]) {
		assert_eq!(
			elements.len(),
			self.n,
			"elements.len() must equal n (expected {}, got {})",
			self.n,
			elements.len()
		);

		batch_invert_nonzero(elements, &mut self.scratchpad);
	}

	/// Inverts elements in-place, handling zeros gracefully.
	///
	/// Zero elements remain zero after inversion, while non-zero elements
	/// are replaced with their multiplicative inverses.
	///
	/// # Parameters
	/// - `elements`: Mutable slice to invert in-place
	///
	/// # Panics
	/// Panics if `elements.len() != n` (the size specified at construction).
	pub fn invert_or_zero(&mut self, elements: &mut [F]) {
		assert_eq!(
			elements.len(),
			self.n,
			"elements.len() must equal n (expected {}, got {})",
			self.n,
			elements.len()
		);

		// Mark zeros and replace with ones
		for (element_i, is_zero_i) in iter::zip(&mut *elements, &mut self.is_zero) {
			if *element_i == F::ZERO {
				*element_i = F::ONE;
				*is_zero_i = true;
			} else {
				*is_zero_i = false;
			}
		}

		// Perform inversion on non-zero elements
		self.invert(elements);

		// Restore zeros
		for (element_i, is_zero_i) in iter::zip(elements, &self.is_zero) {
			if *is_zero_i {
				*element_i = F::ZERO;
			}
		}
	}
}

fn min_scratchpad_size(mut n: usize) -> usize {
	assert!(n > 0);

	let mut size = 0;
	while n > 1 {
		n = n.div_ceil(2);
		size += n;
	}
	size
}

fn batch_invert_nonzero<F: Field>(elements: &mut [F], scratchpad: &mut [F]) {
	debug_assert!(!elements.is_empty());

	if elements.len() == 1 {
		let element = elements.first_mut().expect("len == 1");
		let inv = element
			.invert()
			.expect("precondition: elements contains no zeros");
		*element = inv;
		return;
	}

	let next_layer_len = elements.len().div_ceil(2);
	let (next_layer, remaining) = scratchpad.split_at_mut(next_layer_len);
	product_layer(elements, next_layer);
	batch_invert_nonzero(next_layer, remaining);
	unproduct_layer(next_layer, elements);
}

#[inline]
fn product_layer<F: Field>(input: &[F], output: &mut [F]) {
	debug_assert_eq!(output.len(), input.len().div_ceil(2));

	let (in_pairs, in_remaining) = input.as_chunks::<2>();
	let (out_head, out_remaining) = output.split_at_mut(in_pairs.len());
	for (out_i, [in_lhs, in_rhs]) in iter::zip(out_head, in_pairs) {
		*out_i = *in_lhs * *in_rhs;
	}
	if !out_remaining.is_empty() {
		out_remaining[0] = in_remaining[0];
	}
}

#[inline]
fn unproduct_layer<F: Field>(input: &[F], output: &mut [F]) {
	debug_assert_eq!(input.len(), output.len().div_ceil(2));

	let (out_pairs, out_remaining) = output.as_chunks_mut::<2>();
	let (in_head, in_remaining) = input.split_at(out_pairs.len());
	for (in_i, [out_lhs, out_rhs]) in iter::zip(in_head, out_pairs) {
		let out_lhs_tmp = *out_lhs;
		let out_rhs_tmp = *out_rhs;
		*out_lhs = *in_i * out_rhs_tmp;
		*out_rhs = *in_i * out_lhs_tmp;
	}
	if !out_remaining.is_empty() {
		out_remaining[0] = in_remaining[0];
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Random, arithmetic_traits::InvertOrZero};
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

		// Sample indices for zeros without replacement
		let zero_indices: Vec<usize> = (0..n).choose_multiple(rng, n_zeros);

		// Create state vector with zeros at sampled indices
		let mut state = Vec::with_capacity(n);
		for i in 0..n {
			if zero_indices.contains(&i) {
				state.push(Ghash::ZERO);
			} else {
				state.push(Ghash::random(&mut *rng));
			}
		}

		let expected: Vec<Ghash> = state.iter().map(|x| x.invert_or_zero()).collect();

		inverter.invert_or_zero(&mut state);

		assert_eq!(state, expected);
	}

	fn test_batch_inversion_for_size(n: usize, n_zeros: usize, rng: &mut impl Rng) {
		let mut inverter = BatchInversion::<Ghash>::new(n);
		invert_with_inverter(&mut inverter, n, n_zeros, rng);
	}

	fn test_batch_inversion_nonzero_for_size(n: usize, rng: &mut impl Rng) {
		let mut state = Vec::with_capacity(n);
		for _ in 0..n {
			state.push(Ghash::random(&mut *rng));
		}

		let expected: Vec<Ghash> = state.iter().map(|x| x.invert_or_zero()).collect();

		let mut inverter = BatchInversion::<Ghash>::new(n);
		inverter.invert(&mut state);

		assert_eq!(state, expected);
	}

	proptest! {
		#[test]
		fn test_batch_inversion(n in 1usize..=16, n_zeros in 0usize..=16) {
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
		let mut inverter = BatchInversion::<Ghash>::new(8);

		// Test reusing the same inverter multiple times
		for n_zeros in 0..=8 {
			invert_with_inverter(&mut inverter, 8, n_zeros, &mut rng);
		}
	}
}
