// Copyright 2025 Irreducible Inc.

use std::{iter, rc::Rc};

use binius_field::{Field, field::FieldOps, util::FieldFn};
use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, univariate::evaluate_univariate};
use binius_spartan_frontend::constraint_system::{MulConstraint, WitnessIndex, WitnessSegment};

/// Returns a closure that evaluates the wiring transparent polynomial for a specific segment.
///
/// The closure evaluates the wiring MLE at a challenge point drawn during the BaseFold opening.
/// The three operand contributions are batched together with the challenge `lambda`.
/// The multiplication constraints and the eq-indicator evaluation at `r_x` are shared via `Rc`.
/// Sharing lets the closure own them and be `'static`.
/// The opening is deferred, so the closure must outlive this call.
pub fn eval_transparent<F: FieldOps + 'static>(
	mul_constraints: Rc<[MulConstraint<WitnessIndex>]>,
	segment: WitnessSegment,
	r_x_tensor: Rc<[F]>,
	lambda: F,
) -> binius_iop::channel::TransparentEvalFn<F> {
	Box::new(move |r_y: &[F]| {
		evaluate_segment_wiring_mle(&mul_constraints, segment, lambda.clone(), &r_x_tensor, r_y)
	})
}

/// Evaluates the wiring MLE for a specific segment at a point (r_x, r_y).
///
/// `r_x_tensor` is the eq-indicator partial evaluation at r_x, i.e.
/// `eq_ind_partial_eval_scalars(r_x)`. Accepting it as a parameter avoids redundant
/// computation when evaluating multiple segments with the same r_x.
pub fn evaluate_segment_wiring_mle<F: FieldOps>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	segment: WitnessSegment,
	lambda: F,
	r_x_tensor: &[F],
	r_y: &[F],
) -> F {
	let mut acc = [F::zero(), F::zero(), F::zero()];

	let r_y_tensor = eq_ind_partial_eval_scalars(r_y);
	for (r_x_tensor_i, MulConstraint { a, b, c }) in iter::zip(r_x_tensor, mul_constraints) {
		for (dst, operand) in iter::zip(&mut acc, [a, b, c]) {
			let r_y_tensor_sum = operand
				.wires()
				.iter()
				.flat_map(|index| {
					if index.segment == segment {
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

/// Evaluates the public segment's contribution to the wiring MLE.
///
/// `r_x_tensor` is the eq-indicator partial evaluation at r_x.
pub fn evaluate_wiring_mle_public<F: FieldOps>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	public: &[F],
	lambda: F,
	r_x_tensor: &[F],
) -> F {
	let mut acc = [F::zero(), F::zero(), F::zero()];
	for (r_x_tensor_i, MulConstraint { a, b, c }) in iter::zip(r_x_tensor, mul_constraints) {
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

/// The public segment's contribution to the batched operand evaluations, as a [`FieldFn`].
///
/// The inputs are the flat concatenation of three sections, in order:
/// - a block of public scalars;
/// - then the batching challenge `lambda`;
/// - then the eq-indicator partial evaluation at `r_x`.
pub struct PublicWiringEvalFn<'a> {
	/// The multiplication constraints whose operands the evaluation sums over.
	mul_constraints: &'a [MulConstraint<WitnessIndex>],
	/// The length of the leading block of public scalars.
	public_len: usize,
}

impl<'a> PublicWiringEvalFn<'a> {
	/// Builds the function from the constraints and the count of leading public inputs.
	pub const fn new(
		mul_constraints: &'a [MulConstraint<WitnessIndex>],
		public_len: usize,
	) -> Self {
		Self {
			mul_constraints,
			public_len,
		}
	}
}

impl<F: Field> FieldFn<F> for PublicWiringEvalFn<'_> {
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, inputs: &[E]) -> E {
		// Split the flat input back into its three parts.
		let public = &inputs[..self.public_len];
		let lambda = inputs[self.public_len].clone();
		let r_x_tensor = &inputs[self.public_len + 1..];

		evaluate_wiring_mle_public(self.mul_constraints, public, lambda, r_x_tensor)
	}
}
