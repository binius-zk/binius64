// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Evaluating a multilinear polynomial at a point off the hypercube.

use std::ops::{Deref, DerefMut};

use binius_compute::BufferData;
use binius_field::{Field, PackedField, field::FieldOps};
use binius_utils::rayon::prelude::*;

use crate::{
	FieldBuffer,
	inner_product::inner_product_buffers,
	multilinear::{fold::fold_highest_var_inplace, hypercube::Hypercube},
};

/// Evaluates a multilinear polynomial at a point, leaving the coefficients in place.
///
/// The point holds one coordinate per variable.
/// The result is a single field element.
/// Memory used is on the order of the square root of the coefficient count.
///
/// ## Preconditions
///
/// * the point must hold one coordinate per variable of the polynomial
pub fn evaluate<F, P, Data>(evals: &FieldBuffer<P, Data>, point: &[F]) -> F
where
	F: Field,
	P: PackedField<Scalar = F>,
	Data: Deref<Target = [P]>,
{
	assert_eq!(
		point.len(),
		evals.log_len(),
		"precondition: point length must equal evals log length"
	);

	// The point splits in half, and the first half gets at least one packed word's worth.
	// Expanding only that half costs memory on the order of the square root of the whole.
	let first_half_len = (point.len() / 2).max(P::LOG_WIDTH).min(point.len());
	let (first_coords, remaining_coords) = point.split_at(first_half_len);
	let eq_tensor = Hypercube::One.expand(first_coords).build::<P>();

	// With nothing left over the expansion covers every variable, so one pairing finishes.
	if remaining_coords.is_empty() {
		return inner_product_buffers(evals, &eq_tensor);
	}

	// Otherwise each chunk pairs with the expansion, and the resulting scalars are the
	// residual multilinear over the coordinates not yet used.
	let scalars = evals
		.par_chunks(first_half_len)
		.map(|chunk| inner_product_buffers(&chunk, &eq_tensor))
		.collect::<Vec<_>>();

	evaluate_inplace(FieldBuffer::<P>::from_values(&scalars), remaining_coords)
}

/// Evaluates a multilinear polynomial at a point, consuming the coefficients.
///
/// One variable is fixed at a time, in place, so nothing beyond the buffer is allocated.
/// Each fold halves the buffer, and the last one leaves the single result.
///
/// ## Preconditions
///
/// * the point must hold one coordinate per variable of the polynomial
pub fn evaluate_inplace<F, P, Data>(mut evals: FieldBuffer<P, Data>, coords: &[F]) -> F
where
	F: Field,
	P: PackedField<Scalar = F>,
	Data: BufferData<P>,
{
	assert_eq!(
		coords.len(),
		evals.log_len(),
		"precondition: coords length must equal evals log length"
	);

	// Fixing the highest variable first keeps the survivors in a prefix, so an `n`-variate
	// polynomial costs `2^n - 1` multiplications and no memory beyond the buffer.
	for &coord in coords.iter().rev() {
		fold_highest_var_inplace(&mut evals, coord);
	}

	assert_eq!(evals.len(), 1);
	evals.get(0)
}

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
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::{
		inner_product::inner_product_par,
		test_utils::{
			B128, Packed128b, index_to_hypercube_point, random_field_buffer, random_scalars,
		},
	};

	type P = Packed128b;
	type F = B128;

	// The packing width is four scalars, so this range straddles it in both directions.
	const MAX_VARS: usize = 8;

	#[test]
	fn test_evaluate_inplace_scalars_consistency() {
		let mut rng = StdRng::seed_from_u64(0);

		for log_n in [0, P::LOG_WIDTH - 1, P::LOG_WIDTH, 10] {
			let buffer = random_field_buffer::<P>(&mut rng, log_n);
			let point = random_scalars::<F>(&mut rng, log_n);

			let result_inplace = evaluate_inplace(buffer.clone(), &point);

			let scalar_evals = buffer.iter_scalars().collect::<Vec<_>>();
			let result_scalar = evaluate_inplace_scalars(scalar_evals, &point);

			assert_eq!(result_inplace, result_scalar, "mismatch at log_n={log_n}");
		}
	}

	#[test]
	fn evaluate_at_a_hypercube_vertex_reads_that_coefficient() {
		let mut rng = StdRng::seed_from_u64(0);

		// Every vertex of a small cube is cheap enough to check exhaustively.
		let n_vars = 8;
		let buffer = random_field_buffer::<F>(&mut rng, n_vars);

		for index in 0..1 << n_vars {
			let point = index_to_hypercube_point::<F>(n_vars, index);

			assert_eq!(evaluate(&buffer, &point), buffer.get(index), "mismatch at vertex {index}");
		}
	}

	#[test]
	fn evaluate_is_linear_in_every_coordinate() {
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 8;
		let buffer = random_field_buffer::<F>(&mut rng, n_vars);
		let mut point = random_scalars::<F>(&mut rng, n_vars);

		for coord_idx in 0..n_vars {
			// Three points differing only in this coordinate must have collinear evaluations.
			let coord_vals = random_scalars::<F>(&mut rng, 3);
			let evals = coord_vals
				.iter()
				.map(|&coord_val| {
					point[coord_idx] = coord_val;
					evaluate(&buffer, &point)
				})
				.collect::<Vec<_>>();

			// Collinearity of the three points, cross-multiplied so nothing is divided.
			let [x0, x1, x2] = [coord_vals[0], coord_vals[1], coord_vals[2]];
			let [y0, y1, y2] = [evals[0], evals[1], evals[2]];
			assert_eq!((y2 - y0) * (x1 - x0), (y1 - y0) * (x2 - x0));
		}
	}

	proptest! {
		#[test]
		fn the_two_evaluations_agree_with_the_definition(
			n_vars in 0..=MAX_VARS,
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let buffer = random_field_buffer::<P>(&mut rng, n_vars);
			let point = random_scalars::<F>(&mut rng, n_vars);

			// Pairing with the full expansion is the definition, and the cheapest reference.
			let reference =
				inner_product_par(&buffer, &Hypercube::One.expand(&point).build::<P>());

			prop_assert_eq!(evaluate(&buffer, &point), reference);

			// The in-place form consumes the coefficients, so it goes last.
			prop_assert_eq!(evaluate_inplace(buffer, &point), reference);
		}
	}
}
