// Copyright 2026 The Binius Developers
// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 Authors

//! Compression functions folding several digests into one, one digest at a time or in batch.
//!
//! The one-at-a-time interface is taken from
//! [p3_symmetric](https://github.com/Plonky3/Plonky3/blob/main/symmetric/src/compression.rs) in
//! [Plonky3].
//!
//! Plonky3 is dual-licensed under MIT OR Apache 2.0. We use it under Apache 2.0.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

use std::{array, mem::MaybeUninit};

use binius_utils::rayon::prelude::*;

/// An `N`-to-1 compression function used to build the inner nodes of a hash tree.
///
/// It folds `N` values into a single value of the same type.
/// Applied level by level, it turns the children of an inner node into that node's value.
pub trait CompressionFunction<T, const N: usize>: Clone {
	/// Maps the `N` inputs down to a single output of the same type.
	///
	/// In a hash tree this folds the `N` child node values into their parent node value.
	fn compress(&self, input: [T; N]) -> T;
}

/// The batch form of an `N`-to-1 compression function.
///
/// A hash tree level compresses many sibling groups at once, all independent of each other.
/// Implementations turn that independence into speed by filling SIMD lanes or threads.
pub trait ParallelPseudoCompression<T, const N: usize> {
	/// The one-group-at-a-time compression function this batches over.
	type Compression: CompressionFunction<T, N>;

	/// Returns the one-group-at-a-time function the batch form agrees with.
	fn compression(&self) -> &Self::Compression;

	/// Compresses each consecutive group of `N` inputs into one output.
	///
	/// ```text
	///     out[0] = compress(inputs[0..N])
	///     out[1] = compress(inputs[N..2*N])
	///     ...
	/// ```
	///
	/// Every output slot is written before this returns.
	/// Callers may therefore treat the whole output as initialized.
	///
	/// # Arguments
	///
	/// * `inputs` - the groups to compress, laid out back to back.
	/// * `out` - one slot per group, in the same order.
	///
	/// # Panics
	///
	/// Panics unless the input length is exactly `N` times the output length.
	fn parallel_compress(&self, inputs: &[T], out: &mut [MaybeUninit<T>]);
}

/// Lifts any `N`-to-1 compression function to the batch form by spreading groups over threads.
///
/// This is the fallback for hash functions with no vectorized batch implementation of their own.
/// Each group still goes through the scalar function, so only the outer loop is parallel.
#[derive(Debug, Clone, Default)]
pub struct ParallelCompressionAdaptor<C> {
	compression: C,
}

impl<C> ParallelCompressionAdaptor<C> {
	/// Wraps a compression function so it can stand in wherever the batch form is required.
	pub const fn new(compression: C) -> Self {
		Self { compression }
	}
}

impl<T, C, const ARITY: usize> ParallelPseudoCompression<T, ARITY> for ParallelCompressionAdaptor<C>
where
	T: Clone + Send + Sync,
	C: CompressionFunction<T, ARITY> + Sync,
{
	type Compression = C;

	fn compression(&self) -> &Self::Compression {
		&self.compression
	}

	fn parallel_compress(&self, inputs: &[T], out: &mut [MaybeUninit<T>]) {
		// Invariant: every output slot consumes exactly one group of ARITY inputs.
		assert_eq!(inputs.len(), ARITY * out.len(), "Input length must be N * output length");

		inputs
			// Group i of the input lines up with slot i of the output.
			.par_chunks_exact(ARITY)
			.zip(out.par_iter_mut())
			.for_each(|(chunk, slot)| {
				// The scalar function takes a fixed-size array, so materialize one per group.
				let group: [T; ARITY] = array::from_fn(|j| chunk[j].clone());
				// Groups share no state, so each thread writes only the slot it owns.
				slot.write(self.compression.compress(group));
			});
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;

	/// A stand-in three-to-one compression, chosen so the expected output is obvious by hand.
	#[derive(Clone, Debug)]
	struct XorCompression;

	impl CompressionFunction<u64, 3> for XorCompression {
		fn compress(&self, input: [u64; 3]) -> u64 {
			input[0] ^ input[1] ^ input[2]
		}
	}

	/// Folds the groups through the adaptor.
	/// Pins every output to the one-at-a-time fold of that same group.
	fn check_adaptor(groups: &[[u64; 3]]) {
		// Flatten to the back-to-back layout the batch form reads.
		let inputs: Vec<u64> = groups.iter().flatten().copied().collect();
		let mut out = vec![MaybeUninit::<u64>::uninit(); groups.len()];

		ParallelCompressionAdaptor::new(XorCompression).parallel_compress(&inputs, &mut out);

		// Invariant: batching changes the order of the work, never its result.
		for (slot, group) in out.into_iter().zip(groups) {
			assert_eq!(unsafe { slot.assume_init() }, XorCompression.compress(*group));
		}
	}

	#[test]
	fn test_parallel_compress_boundaries() {
		// Extreme words exercise an all-zero and an all-ones input in every slot of a group.
		check_adaptor(&[
			[0, 0, 0],
			[u64::MAX; 3],
			[0, u64::MAX, 0],
			[u64::MAX, 0, u64::MAX],
		]);

		// A group of three equal words xors to that word, and a repeated word cancels itself out.
		check_adaptor(&[[7, 7, 7], [7, 7, 0]]);

		// Nothing to compress is a valid batch: no slot is written and no panic fires.
		check_adaptor(&[]);
	}

	proptest! {
		#[test]
		fn parallel_compress_matches_scalar(
			groups in prop::collection::vec(prop::array::uniform3(any::<u64>()), 0..40usize),
		) {
			// Batch counts here straddle rayon's splitting threshold in both directions.
			check_adaptor(&groups);
		}
	}

	#[test]
	#[should_panic(expected = "Input length must be N * output length")]
	fn test_mismatched_input_length() {
		// Four inputs cannot fill two groups of three, so the length check rejects the call.
		let inputs = vec![1u64, 2, 3, 4];
		let mut out = [MaybeUninit::<u64>::uninit(); 2];

		ParallelCompressionAdaptor::new(XorCompression).parallel_compress(&inputs, &mut out);
	}
}
