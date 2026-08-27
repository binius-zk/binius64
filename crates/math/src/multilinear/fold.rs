// Copyright 2026 The Binius Developers

//! Fixing the highest variable of a multilinear to a value.
//!
//! Fixing one variable of an `n`-variate multilinear leaves an `(n-1)`-variate one:
//!
//! ```text
//! g(X_0, ..., X_{n-2}) = f(X_0, ..., X_{n-2}, r)
//! ```
//!
//! Coefficients are stored with the highest variable selecting which half of the buffer a
//! coefficient falls in.
//! So fixing the highest variable pairs the two halves and leaves the result in the first one.

use std::ops::{Deref, DerefMut};

use binius_compute::{Allocator, BufferData, VecLike};
use binius_field::{Field, PackedField};
use binius_utils::{
	random_access_sequence::RandomAccessSequence,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};

use crate::{FieldBuffer, FieldVec, line::extrapolate_line};

/// Fixes the highest variable of a multilinear to a value, in place.
///
/// ```text
/// g(X_0, ..., X_{n-2}) = f(X_0, ..., X_{n-2}, r)
/// ```
///
/// The result occupies the first half of the buffer.
/// The buffer then reports one variable fewer.
///
/// ## Preconditions
///
/// * the buffer must have at least one variable
pub fn fold_highest_var_inplace<P: PackedField, Data: BufferData<P>>(
	values: &mut FieldBuffer<P, Data>,
	scalar: P::Scalar,
) {
	// Each scalar of the result costs one multiplication.
	// Broadcasting the challenge once lets every packed word reuse the same multiplier.
	let broadcast_scalar = P::broadcast(scalar);
	{
		// The two halves are the multilinear specialized to 0 and to 1 on the highest variable.
		let mut split = values.split_half_mut();
		let (mut lo, mut hi) = split.halves();
		// Interpolate the line through each pair at the challenge, overwriting the low half.
		(lo.as_mut(), hi.as_mut())
			.into_par_iter()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(lo_i, hi_i)| {
				*lo_i = extrapolate_line(*lo_i, *hi_i, broadcast_scalar);
			});
	}

	// The result occupies a prefix, so the truncation drops the scalars past it.
	values.truncate(values.log_len() - 1);
}

/// Fixes the highest variable of a multilinear to a value, writing into memory from an allocator.
///
/// ```text
/// g(X_0, ..., X_{n-2}) = f(X_0, ..., X_{n-2}, r)
/// ```
///
/// Each output coefficient interpolates the line through one pair of input coefficients.
/// The input is left untouched.
///
/// Use this when the input is borrowed or must be preserved.
/// Otherwise prefer the form that overwrites the input.
///
/// ## Preconditions
///
/// * the buffer must have at least one variable
pub fn fold_highest_var<A: Allocator, P: PackedField, Data: Deref<Target = [P]>>(
	alloc: &A,
	values: &FieldBuffer<P, Data>,
	scalar: P::Scalar,
) -> FieldVec<P, A> {
	assert!(values.log_len() > 0, "precondition: buffer must have at least one variable");

	// The two halves are the multilinear specialized to 0 and to 1 on the highest variable.
	let broadcast_scalar = P::broadcast(scalar);
	let (lo, hi) = values.split_half();

	// Interpolate the line through each pair at the challenge directly into a fresh buffer
	// drawn from the allocator, writing the uninitialized spare capacity in parallel rather
	// than zero-filling first.
	let len = lo.as_ref().len();
	let mut data = alloc.alloc::<P>(len);
	let spare = &mut data.spare_capacity_mut()[..len];
	(spare, lo.as_ref(), hi.as_ref())
		.into_par_iter()
		.with_min_task(WorkPerItem::FieldMuls)
		.for_each(|(out, &lo_i, &hi_i)| {
			out.write(extrapolate_line(lo_i, hi_i, broadcast_scalar));
		});
	// SAFETY: the parallel loop initialized all `len` slots.
	unsafe { data.set_len(len) };
	FieldBuffer::new(values.log_len() - 1, data)
}

/// Overwrites a buffer with the high fold of a bit sequence by a tensor.
///
/// The bits are the coefficients of a multilinear whose values are all zero or one.
/// Each output vertex fixes that polynomial's low-indexed variables to that vertex.
/// What remains is then paired with the tensor.
///
/// This runs on one thread.
///
/// ## Preconditions
///
/// * the bit count must be a power of two
/// * the bit count must equal the output length times the tensor length
pub fn binary_fold_high<P, DataOut, DataIn>(
	values: &mut FieldBuffer<P, DataOut>,
	tensor: &FieldBuffer<P, DataIn>,
	bits: &(impl RandomAccessSequence<bool> + Sync),
) where
	P: PackedField,
	DataOut: DerefMut<Target = [P]>,
	DataIn: Deref<Target = [P]>,
{
	assert!(bits.len().is_power_of_two(), "precondition: bits length must be a power of two");

	let values_log_len = values.log_len();
	// Below one packed word the buffer still occupies a whole word, so only the live lanes count.
	let width = P::WIDTH.min(values.len());

	assert_eq!(
		1 << (values_log_len + tensor.log_len()),
		bits.len(),
		"precondition: bits length must equal values length times tensor length"
	);

	values
		.iter_packed_mut()
		.enumerate()
		.for_each(|(i, packed)| {
			*packed = P::from_scalars((0..width).map(|j| {
				// The output vertex this lane holds, as an index into the bits' low variables.
				let scalar_index = i << P::LOG_WIDTH | j;
				let mut acc = P::Scalar::ZERO;

				// Sum the tensor entries whose bit is set, over the bits' high variables.
				// Multiplication by a bit is a selection, so no field multiplication is needed.
				for (k, tensor_packed) in tensor.iter_packed().enumerate() {
					for (l, tensor_scalar) in tensor_packed.iter().take(tensor.len()).enumerate() {
						let tensor_scalar_index = k << P::LOG_WIDTH | l;
						if bits.get(tensor_scalar_index << values_log_len | scalar_index) {
							acc += tensor_scalar;
						}
					}
				}

				acc
			}));
		});
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_compute::GlobalAllocator;
	use binius_utils::rayon::task_size::min_len_for_work;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::{
		multilinear::eq::eq_ind_partial_eval,
		test_utils::{B128, Packed128b, random_field_buffer, random_scalars},
	};

	type P = Packed128b;
	type F = B128;

	// The packing width is four scalars, so this range straddles it in both directions.
	const MAX_VARS: usize = 8;

	#[test]
	fn fold_splits_above_the_task_threshold() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a fold splits only once each half holds two minimum tasks.
		// Below that it runs inline, leaving the parallel path unexercised.
		//
		//     words in one half = 2^(n_vars - 1) / scalars per word
		//     smallest n_vars with words in one half >= 2 * minimum
		//
		// Every other fold test is smaller, so this one covers the split.
		let min_len = min_len_for_work(WorkPerItem::FieldMuls);
		let n_vars = (2 * min_len * P::WIDTH).next_power_of_two().ilog2() as usize + 1;
		let half = 1 << (n_vars - 1);
		let original = random_field_buffer::<P>(&mut rng, n_vars);
		let challenge = random_scalars::<F>(&mut rng, 1)[0];

		let mut folded = original.clone();
		fold_highest_var_inplace(&mut folded, challenge);
		assert_eq!(folded.log_len(), n_vars - 1);

		// Scalar reference: each output interpolates one (lo, hi) pair at the challenge.
		for i in 0..half {
			let expected = extrapolate_line(original.get(i), original.get(i | half), challenge);
			assert_eq!(folded.get(i), expected, "mismatch at index {i}");
		}
	}

	#[test]
	fn folding_a_slice_backed_buffer_matches_folding_an_owned_one() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a fold shrinks its buffer, which each backing store does its own way.
		// A vector drops its tail, a mutable slice re-slices itself.
		// Both must leave the same coefficients behind.
		//
		// Fixture state: one 5-variable buffer, folded twice by the same challenge.
		//
		//     owned store      [ c_0 ... c_31 ]  -> [ c'_0 ... c'_15 ]
		//     slice store      [ c_0 ... c_31 ]  -> [ c'_0 ... c'_15 ]
		let original = random_field_buffer::<P>(&mut rng, 5);
		let scalar = random_scalars::<F>(&mut rng, 1)[0];

		let mut expected = original.clone();
		fold_highest_var_inplace(&mut expected, scalar);

		let mut owned = original;
		let mut slice = owned.as_mut_view();
		fold_highest_var_inplace(&mut slice, scalar);

		assert_eq!(slice.log_len(), 4);
		assert_eq!(slice, expected.as_mut_view());
	}

	proptest! {
		#[test]
		fn the_two_folds_agree(
			n_vars in 1..=MAX_VARS,
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let original = random_field_buffer::<P>(&mut rng, n_vars);
			let scalar = random_scalars::<F>(&mut rng, 1)[0];

			// Out of place leaves the input alone and returns a fresh half-size buffer.
			let out_of_place = fold_highest_var(&GlobalAllocator, &original, scalar);

			// In place overwrites the input's first half and shrinks it.
			let mut in_place = original;
			fold_highest_var_inplace(&mut in_place, scalar);

			prop_assert_eq!(out_of_place.log_len(), n_vars - 1);
			prop_assert_eq!(out_of_place, in_place);
		}

		#[test]
		fn the_binary_fold_matches_folding_the_widened_bits(
			dest_vars in 0..=6usize,
			tensor_vars in 0..=4usize,
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, tensor_vars);
			let tensor = eq_ind_partial_eval::<P>(&point);

			// The bit count is the product of the two lengths, as the precondition demands.
			let bits = repeat_with(|| rng.random())
				.take(1 << (dest_vars + tensor_vars))
				.collect::<Vec<bool>>();

			let mut folded = FieldBuffer::<P>::zeros(dest_vars);
			binary_fold_high(&mut folded, &tensor, &bits.as_slice());

			// Reference: widen the bits to field elements and fold the tensor's variables off.
			let scalars = bits
				.iter()
				.map(|&bit| if bit { F::ONE } else { F::ZERO })
				.collect::<Vec<F>>();
			let mut reference = FieldBuffer::<P>::from_values(&scalars);
			for &coord in point.iter().rev() {
				fold_highest_var_inplace(&mut reference, coord);
			}

			prop_assert_eq!(folded, reference);
		}
	}
}
