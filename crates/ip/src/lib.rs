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
