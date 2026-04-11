// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{Field, field::FieldOps};
use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, univariate::evaluate_univariate};
use binius_spartan_frontend::constraint_system::{MulConstraint, WitnessIndex, WitnessSegment};

use crate::constraint_system::ConstraintSystemPadded;

/// Returns a closure that evaluates the wiring transparent polynomial at a given point.
///
/// The returned closure computes the expected evaluation of the wiring MLE batched with the
/// public input equality check, given a challenge point from the BaseFold opening.
pub fn eval_transparent<'a, G: Field, F: FieldOps + 'a>(
	constraint_system: &ConstraintSystemPadded<G>,
	r_x: &[F],
	lambda: F,
) -> binius_iop::channel::TransparentEvalFn<'a, F> {
	let r_x = r_x.to_vec();
	let mul_constraints = constraint_system.mul_constraints().to_vec();

	Box::new(move |r_y: &[F]| {
		evaluate_private_wiring_mle(&mul_constraints, lambda.clone(), &r_x, r_y)
	})
}

/// Evaluates the private wiring MLE at a point (r_x, r_y).
///
/// The r_y dimension corresponds to the private witness segment only (log_private variables).
/// Private wire indices are used directly as indices into r_y_tensor.
pub fn evaluate_private_wiring_mle<F: FieldOps>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	lambda: F,
	r_x: &[F],
	r_y: &[F],
) -> F {
	let mut acc = [F::zero(), F::zero(), F::zero()];

	let r_x_tensor = eq_ind_partial_eval_scalars(r_x);
	let r_y_tensor = eq_ind_partial_eval_scalars(r_y);
	for (r_x_tensor_i, MulConstraint { a, b, c }) in iter::zip(&r_x_tensor, mul_constraints) {
		for (dst, operand) in iter::zip(&mut acc, [a, b, c]) {
			let r_y_tensor_sum = operand
				.wires()
				.iter()
				.flat_map(|index| {
					if let WitnessSegment::Private = index.segment {
						Some(r_y_tensor[index.index as usize].clone())
					} else {
						None
					}
				})
				.sum::<F>();
			*dst += r_x_tensor_i.clone() * r_y_tensor_sum;
		}
	}

	evaluate_univariate(&acc, lambda)
}

pub fn evaluate_wiring_mle_public<F: FieldOps>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	log_public: usize,
	public: &[F],
	lambda: F,
	r_x: &[F],
) -> F {
	assert_eq!(public.len(), 1 << log_public);

	let mut acc = [F::zero(), F::zero(), F::zero()];
	let r_x_tensor = eq_ind_partial_eval_scalars(r_x);
	for (r_x_tensor_i, MulConstraint { a, b, c }) in iter::zip(&r_x_tensor, mul_constraints) {
		for (dst, operand) in iter::zip(&mut acc, [a, b, c]) {
			let public_sum = operand
				.wires()
				.iter()
				.flat_map(|index| {
					if let WitnessSegment::Public = index.segment {
						Some(public[index.index as usize].clone())
					} else {
						None
					}
				})
				.sum::<F>();
			*dst += r_x_tensor_i.clone() * public_sum;
		}
	}

	evaluate_univariate(&acc, lambda)
}
