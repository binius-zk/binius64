// Copyright 2025 Irreducible Inc.

pub mod eq;
pub mod eq_prefix;
pub mod evaluate;
pub mod fold;
pub mod hypercube;
pub mod sparse;

use binius_field::Field;

/// Evaluates the multilinear extension of the one-hot vector at `index` on `point`.
///
/// This is `eq(index, point)`, with `index` read as a hypercube vertex, low bit first:
///
/// ```text
/// prod_i (index_i ? point_i : 1 - point_i)
/// ```
pub fn eval_one_hot<F: Field>(index: usize, point: &[F]) -> F {
	debug_assert!(
		point.len() >= usize::BITS as usize || index >> point.len() == 0,
		"index {index} out of range for {} variables",
		point.len()
	);
	point
		.iter()
		.enumerate()
		.map(|(i, &x_i)| {
			if (index >> i) & 1 == 1 {
				x_i
			} else {
				F::ONE - x_i
			}
		})
		.product()
}

#[cfg(test)]
mod tests {
	use rand::prelude::*;

	use super::*;
	use crate::{
		FieldBuffer,
		test_utils::{B128, random_scalars},
	};

	#[test]
	fn eval_one_hot_matches_the_one_hot_buffer() {
		let mut rng = StdRng::seed_from_u64(0);
		let point = random_scalars::<B128>(&mut rng, 4);
		for index in 0..16 {
			let mut one_hot = FieldBuffer::<B128>::zeros(4);
			one_hot.set(index, B128::ONE);
			assert_eq!(eval_one_hot(index, &point), evaluate::evaluate(&one_hot, &point));
		}
	}
}
