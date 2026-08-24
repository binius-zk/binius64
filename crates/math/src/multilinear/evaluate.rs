// Copyright 2025 Irreducible Inc.

use std::ops::DerefMut;

use binius_field::field::FieldOps;

/// Evaluates a multilinear polynomial at a given point in-place using scalar operations.
///
/// This is a simple variant of multilinear evaluation that works directly on slices of scalars
/// with only a `FieldOps` bound. For each coordinate (highest to lowest), it folds the upper
/// half into the lower half: `evals[j] += r * (evals[j + half] - evals[j])`.
///
/// The final result is stored in `evals[0]` after all folds.
///
/// # Arguments
/// * `evals` - The 2^n evaluations over the boolean hypercube, modified in-place
/// * `point` - The n coordinates at which to evaluate the polynomial
///
/// # Panics
///
/// Panics if `evals.len() != 1 << point.len()`.
pub fn evaluate_inplace_scalars<F: FieldOps>(
	mut evals: impl DerefMut<Target = [F]>,
	point: &[F],
) -> F {
	assert_eq!(evals.len(), 1 << point.len(), "precondition: evals length must be 2^point.len()");

	for (log_half_len, point_i) in point.iter().enumerate().rev() {
		let half_len = 1 << log_half_len;
		for j in 0..half_len {
			let delta = evals[j + half_len].clone() - evals[j].clone();
			evals[j] += point_i.clone() * delta;
		}
	}
	evals[0].clone()
}

#[cfg(test)]
mod tests {
	use rand::prelude::*;

	use super::*;
	use crate::{
		multilinear::MultilinearMut,
		test_utils::{B128, Packed128b, random_field_buffer, random_scalars},
	};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn test_evaluate_inplace_scalars_consistency() {
		let mut rng = StdRng::seed_from_u64(0);

		for log_n in [0, P::LOG_WIDTH - 1, P::LOG_WIDTH, 10] {
			let buffer = random_field_buffer::<P>(&mut rng, log_n);
			let point = random_scalars::<F>(&mut rng, log_n);

			let result_inplace = buffer.clone().evaluate_inplace(&point);

			let scalar_evals = buffer.iter_scalars().collect::<Vec<_>>();
			let result_scalar = evaluate_inplace_scalars(scalar_evals, &point);

			assert_eq!(result_inplace, result_scalar, "mismatch at log_n={log_n}");
		}
	}
}
