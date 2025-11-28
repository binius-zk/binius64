// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{Field, PackedField};

/// Reusable batch inversion context that owns its scratch buffers.
///
/// This struct manages the memory needed for batch field inversions, allowing
/// efficient reuse across multiple inversion operations of the same size.
///
/// # Example
/// ```
/// use binius_field::{BinaryField128bGhash, Field};
/// use binius_math::batch_invert::BatchInversion;
///
/// let mut inverter = BatchInversion::<BinaryField128bGhash>::new(8);
/// let mut elements = [BinaryField128bGhash::ONE; 8];
/// inverter.invert_or_zero(&mut elements);
/// ```
pub struct BatchInversion<P: PackedField> {
	n: usize,
	scratchpad: Vec<P>,
	is_zero: Vec<bool>,
	/// Nested inverter for handling the base case when WIDTH > 1.
	/// When we recurse down to a single packed element, we need to
	/// batch-invert its WIDTH scalar elements.
	scalar_inverter: Option<Box<BatchInversion<P::Scalar>>>,
}

impl<P: PackedField> BatchInversion<P> {
	/// Creates a new batch inversion context for slices of size `n`.
	///
	/// Allocates the necessary scratch space:
	/// - `scratchpad`: Storage for intermediate products during recursion
	/// - `is_zero`: Tracking vector for zero elements (one per scalar)
	/// - `scalar_inverter`: Nested inverter for base case when WIDTH > 1
	///
	/// # Parameters
	/// - `n`: The number of packed elements this instance will handle
	///
	/// # Panics
	/// Panics if `n` is 0.
	pub fn new(n: usize) -> Self {
		assert!(n > 0, "n must be greater than 0");

		let scratchpad_size = min_scratchpad_size(n);
		let scalar_inverter = if P::WIDTH > 1 {
			Some(Box::new(BatchInversion::<P::Scalar>::new(P::WIDTH)))
		} else {
			None
		};
		Self {
			n,
			scratchpad: vec![P::zero(); scratchpad_size],
			is_zero: vec![false; n * P::WIDTH],
			scalar_inverter,
		}
	}

	/// Inverts non-zero elements in-place.
	///
	/// # Parameters
	/// - `elements`: Mutable slice to invert in-place
	///
	/// # Preconditions
	/// All scalar elements must be non-zero. Behavior is undefined if any scalar is zero.
	///
	/// # Panics
	/// Panics if `elements.len() != n` (the size specified at construction).
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

	/// Inverts elements in-place, handling zeros gracefully.
	///
	/// Zero scalar elements remain zero after inversion, while non-zero scalars
	/// are replaced with their multiplicative inverses.
	///
	/// # Parameters
	/// - `elements`: Mutable slice to invert in-place
	///
	/// # Panics
	/// Panics if `elements.len() != n` (the size specified at construction).
	pub fn invert_or_zero(&mut self, elements: &mut [P]) {
		assert_eq!(
			elements.len(),
			self.n,
			"elements.len() must equal n (expected {}, got {})",
			self.n,
			elements.len()
		);

		// Mark zeros at scalar level and replace with ones
		for (packed_idx, packed) in elements.iter_mut().enumerate() {
			for lane in 0..P::WIDTH {
				let scalar_idx = packed_idx * P::WIDTH + lane;
				let scalar = packed.get(lane);
				if scalar == P::Scalar::ZERO {
					packed.set(lane, P::Scalar::ONE);
					self.is_zero[scalar_idx] = true;
				} else {
					self.is_zero[scalar_idx] = false;
				}
			}
		}

		// Perform inversion on non-zero elements
		self.invert_nonzero(elements);

		// Restore zeros at scalar level
		for (packed_idx, packed) in elements.iter_mut().enumerate() {
			for lane in 0..P::WIDTH {
				let scalar_idx = packed_idx * P::WIDTH + lane;
				if self.is_zero[scalar_idx] {
					packed.set(lane, P::Scalar::ZERO);
				}
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

impl<P: PackedField> BatchInversion<P> {
	fn batch_invert_nonzero(&mut self, elements: &mut [P]) {
		self.batch_invert_nonzero_with_scratchpad(elements, &mut self.scratchpad.clone());
	}

	fn batch_invert_nonzero_with_scratchpad(&mut self, elements: &mut [P], scratchpad: &mut [P]) {
		debug_assert!(!elements.is_empty());

		if elements.len() == 1 {
			let packed = &mut elements[0];
			if P::WIDTH == 1 {
				// Direct scalar inversion
				let scalar = packed.get(0);
				let inv = scalar
					.invert()
					.expect("precondition: elements contains no zeros");
				packed.set(0, inv);
			} else {
				// Unpack, batch invert scalars, repack
				let mut scalars = packed.into_iter().collect::<Vec<_>>();
				self.scalar_inverter
					.as_mut()
					.expect("scalar_inverter must be Some when WIDTH > 1")
					.invert_nonzero(&mut scalars);
				*packed = P::from_scalars(scalars);
			}
			return;
		}

		let next_layer_len = elements.len().div_ceil(2);
		let (next_layer, remaining) = scratchpad.split_at_mut(next_layer_len);
		product_layer(elements, next_layer);
		self.batch_invert_nonzero_with_scratchpad(next_layer, remaining);
		unproduct_layer(next_layer, elements);
	}
}

#[inline]
fn product_layer<P: PackedField>(input: &[P], output: &mut [P]) {
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
fn unproduct_layer<P: PackedField>(input: &[P], output: &mut [P]) {
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

		let expected: Vec<Ghash> = state
			.iter()
			.map(|x| InvertOrZero::invert_or_zero(*x))
			.collect();

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

		let expected: Vec<Ghash> = state
			.iter()
			.map(|x| InvertOrZero::invert_or_zero(*x))
			.collect();

		let mut inverter = BatchInversion::<Ghash>::new(n);
		inverter.invert_nonzero(&mut state);

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

	/// Test batch inversion with a packed field (WIDTH > 1)
	#[test]
	fn test_batch_inversion_packed() {
		use crate::test_utils::Packed128b;

		let mut rng = StdRng::seed_from_u64(0);
		const N: usize = 4;

		// Create packed elements with some zeros at various positions
		let mut state: Vec<Packed128b> = (0..N)
			.map(|i| {
				Packed128b::from_fn(|lane| {
					// Put zeros at specific positions
					if (i == 1 && lane == 0) || (i == 2 && lane == 2) {
						Ghash::ZERO
					} else {
						Ghash::random(&mut rng)
					}
				})
			})
			.collect();

		// Compute expected by inverting each scalar
		let expected: Vec<Packed128b> = state
			.iter()
			.map(|packed| Packed128b::from_scalars(packed.iter().map(InvertOrZero::invert_or_zero)))
			.collect();

		let mut inverter = BatchInversion::<Packed128b>::new(N);
		inverter.invert_or_zero(&mut state);

		assert_eq!(state, expected);
	}
}
