// Copyright 2024-2025 Irreducible Inc.

use std::ops::{Deref, DerefMut};

use binius_field::{Field, PackedField};
use binius_utils::{random_access_sequence::RandomAccessSequence, rayon::prelude::*};

use crate::{FieldBuffer, line::extrapolate_line_packed};

/// Computes the partial evaluation of a multilinear on its highest variable, inplace.
///
/// Each scalar of the result requires one multiplication to compute. Multilinear evaluations
/// occupy a prefix of the field buffer; scalars after the truncated length are zeroed out.
///
/// ## Preconditions
///
/// * `values.log_len() >= 1` (buffer must have at least 2 elements)
pub fn fold_highest_var_inplace<P: PackedField, Data: DerefMut<Target = [P]>>(
	values: &mut FieldBuffer<P, Data>,
	scalar: P::Scalar,
) {
	let broadcast_scalar = P::broadcast(scalar);
	{
		let mut split = values.split_half_mut();
		let (mut lo, mut hi) = split.halves();
		(lo.as_mut(), hi.as_mut())
			.into_par_iter()
			.for_each(|(lo_i, hi_i)| {
				*lo_i = extrapolate_line_packed(*lo_i, *hi_i, broadcast_scalar)
			});
	}

	values.truncate(values.log_len() - 1);
}

/// Computes the partial evaluation of a multilinear on its highest variable, out of place.
///
/// This is the out-of-place counterpart of [`fold_highest_var_inplace`].
/// It reads `values` and returns a fresh half-size buffer, leaving the input untouched.
/// Use it when the input is borrowed or must be preserved; otherwise prefer the in-place version.
///
/// ## Preconditions
///
/// * `values.log_len() >= 1` (buffer must have at least 2 elements)
pub fn fold_highest_var<P: PackedField, Data: Deref<Target = [P]>>(
	values: &FieldBuffer<P, Data>,
	scalar: P::Scalar,
) -> FieldBuffer<P> {
	assert!(values.log_len() > 0, "precondition: buffer must have at least one variable");

	// The two halves are the multilinear specialized to 0 and to 1 on the highest variable.
	let broadcast_scalar = P::broadcast(scalar);
	let (lo, hi) = values.split_half_ref();

	// Interpolate the line through each (lo, hi) pair at the challenge into a fresh buffer.
	let folded = (lo.as_ref(), hi.as_ref())
		.into_par_iter()
		.map(|(&lo_i, &hi_i)| extrapolate_line_packed(lo_i, hi_i, broadcast_scalar))
		.collect();
	FieldBuffer::new(values.log_len() - 1, folded)
}

/// Computes the fold high of a binary multilinear with a fold tensor.
///
/// Binary multilinear is represented transparently by a boolean sequence.
/// Fold high meaning: for every hypercube vertex of the result, we specialize lower
/// indexed variables of the binary multilinear to the vertex coordinates and take an
/// inner product of the remaining multilinear and the tensor.
///
/// This method is single threaded.
///
/// ## Preconditions
///
/// * `bits.len()` must be a power of two
/// * `bits.len()` must equal `values.len() * tensor.len()`
pub fn binary_fold_high<P, DataOut, DataIn>(
	values: &mut FieldBuffer<P, DataOut>,
	tensor: &FieldBuffer<P, DataIn>,
	bits: impl RandomAccessSequence<bool> + Sync,
) where
	P: PackedField,
	DataOut: DerefMut<Target = [P]>,
	DataIn: Deref<Target = [P]> + Sync,
{
	assert!(bits.len().is_power_of_two(), "precondition: bits length must be a power of two");

	let values_log_len = values.log_len();
	let width = P::WIDTH.min(values.len());

	assert_eq!(
		1 << (values_log_len + tensor.log_len()),
		bits.len(),
		"precondition: bits length must equal values length times tensor length"
	);

	values
		.as_mut()
		.iter_mut()
		.enumerate()
		.for_each(|(i, packed)| {
			*packed = P::from_scalars((0..width).map(|j| {
				let scalar_index = i << P::LOG_WIDTH | j;
				let mut acc = P::Scalar::ZERO;

				for (k, tensor_packed) in tensor.as_ref().iter().enumerate() {
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

	use rand::prelude::*;

	use super::*;
	use crate::{
		multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
		test_utils::{B128, Packed128b, random_field_buffer, random_scalars},
	};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn test_fold_highest_var_inplace() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 10;

		let point = random_scalars::<F>(&mut rng, n_vars);
		let mut multilinear = random_field_buffer::<P>(&mut rng, n_vars);

		let eval = evaluate(&multilinear, &point);

		for &scalar in point.iter().rev() {
			fold_highest_var_inplace(&mut multilinear, scalar);
		}

		assert_eq!(multilinear.get(0), eval);
	}

	#[test]
	fn test_fold_highest_var_matches_inplace() {
		let mut rng = StdRng::seed_from_u64(0);

		for n_vars in 1..=10 {
			let multilinear = random_field_buffer::<P>(&mut rng, n_vars);
			let challenge = random_scalars::<F>(&mut rng, 1)[0];

			// Out-of-place: leaves the input untouched, returns a fresh buffer.
			let out_of_place = fold_highest_var(&multilinear, challenge);

			// In-place reference: fold a copy and compare.
			let mut in_place = multilinear;
			fold_highest_var_inplace(&mut in_place, challenge);

			assert_eq!(
				out_of_place, in_place,
				"out-of-place fold must equal in-place (n_vars={n_vars})"
			);
		}
	}

	fn test_binary_fold_high_conforms_to_regular_fold_high_helper(
		n_vars: usize,
		tensor_n_vars: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);

		let point = random_scalars::<F>(&mut rng, tensor_n_vars);

		let tensor = eq_ind_partial_eval::<P>(&point);

		let bits = repeat_with(|| rng.random())
			.take(1 << n_vars)
			.collect::<Vec<bool>>();

		let bits_scalars = bits
			.iter()
			.map(|&b| if b { F::ONE } else { F::ZERO })
			.collect::<Vec<F>>();

		let mut bits_buffer = FieldBuffer::<P>::from_values(&bits_scalars);

		let mut binary_fold_result = FieldBuffer::<P>::zeros(n_vars - tensor_n_vars);
		binary_fold_high(&mut binary_fold_result, &tensor, bits.as_slice());

		for &scalar in point.iter().rev() {
			fold_highest_var_inplace(&mut bits_buffer, scalar);
		}

		assert_eq!(bits_buffer, binary_fold_result);
	}

	#[test]
	fn test_binary_fold_high_conforms_to_regular_fold_high() {
		for (n_vars, tensor_n_vars) in [(2, 0), (2, 1), (4, 4), (10, 3)] {
			test_binary_fold_high_conforms_to_regular_fold_high_helper(n_vars, tensor_n_vars)
		}
	}
}
