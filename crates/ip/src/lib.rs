// Copyright 2026 The Binius Developers

use binius_field::Field;

pub mod channel;
pub mod fracaddcheck;
pub mod mlecheck;
pub mod prodcheck;
pub mod sumcheck;

/// A claim that a multilinear polynomial evaluates to a specific value at a point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearEvalClaim<F: Field> {
	/// The evaluation of the multilinear.
	pub eval: F,
	/// The evaluation point.
	pub point: Vec<F>,
}

/// A claim that a multilinear polynomial evaluation satisfies a rational equation.
///
/// This represents the output of an IOP opening protocol where the verification equation is:
/// ```text
/// eval_numerator == eval_denominator * transparent_poly_eval(point)
/// ```
///
/// The caller must compute `transparent_poly_eval(point)` based on the protocol context
/// and verify the equation holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearRationalEvalClaim<F: Field> {
	/// The numerator of the rational claim.
	pub eval_numerator: F,
	/// The denominator: evaluation of the committed polynomial at `point`.
	pub eval_denominator: F,
	/// The evaluation point.
	pub point: Vec<F>,
}
