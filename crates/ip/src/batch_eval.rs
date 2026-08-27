// Copyright 2026 The Binius Developers

//! Reducing many evaluation claims on one multilinear to a single claim.

use binius_field::{Field, field::FieldOps};
use binius_math::multilinear::eq::eq_ind;

use crate::{
	MultilinearEvalClaim,
	channel::IPVerifierChannel,
	sumcheck::{Error, batch_verify},
};

/// Reduces evaluation claims on one multilinear at several points to one claim at one point.
///
/// Every claim names the same multilinear.
/// So what comes back names it too, at a point none of them chose.
///
/// ```text
///     k claims at k points  ->  one claim at one point
/// ```
///
/// The claim that comes out has the same shape as the ones that went in.
/// So a caller may reduce again.
///
/// That is what lets an aggregation node carry one claim, however many it received.
///
/// # The reduction
///
/// An evaluation is a sum over the multilinear's own index space.
/// Its weight is an equality indicator at the point.
///
/// Batching the claims weights that sum by a combination of indicators.
/// That is a plain degree-two sumcheck.
///
/// So this is the batched sumcheck plus one reconstruction, and not a protocol of its own.
///
/// A single value comes off the channel: the multilinear's evaluation at the reduced point.
/// Every claim shares it, since every claim is about the same multilinear.
/// The verifier derives each indicator itself.
///
/// # Soundness
///
/// Suppose any claim given here is false.
/// Then the claim returned is false too, except with negligible probability.
///
/// For `k` claims over `n` variables that bound is about `(k - 1 + 3n) / |F|`.
///
/// Three terms make it up:
///
/// - the batching challenge,
/// - the sumcheck itself, at degree two,
/// - the chance the combined indicator vanishes where the sumcheck lands.
///
/// # Errors
///
/// Returns an error if the sumcheck fails, or if the reconstruction does not hold.
///
/// The reconstruction is asserted over the channel rather than compared.
/// So it becomes a constraint on a channel that carries wires.
/// That is what lets this run inside a circuit.
///
/// # Panics
///
/// Panics if no claim is given, or if the claims do not all span the same number of variables.
pub fn verify<F, C>(
	claims: impl IntoIterator<Item = MultilinearEvalClaim<C::Elem>>,
	channel: &mut C,
) -> Result<MultilinearEvalClaim<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F>,
{
	let claims = claims.into_iter().collect::<Vec<_>>();
	let n_vars = claims
		.first()
		.expect("precondition: at least one claim")
		.point
		.len();
	assert!(
		claims.iter().all(|claim| claim.point.len() == n_vars),
		"precondition: every claim must span the same variables"
	);

	// One sumcheck over the multilinear's index space, at degree two.
	// Sampling the batching challenge is part of it.
	let evals = claims
		.iter()
		.map(|claim| claim.eval.clone())
		.collect::<Vec<_>>();
	let output = batch_verify::<F, C>(n_vars, 2, &evals, channel)?;

	// The multilinear's evaluation is the one thing the verifier cannot derive.
	let eval = channel.recv_one()?;

	// The rounds bind the highest variable first, so the point reads back in reverse.
	let mut point = output.challenges;
	point.reverse();

	// The weight the sumcheck ran against, rebuilt at the point it landed on.
	//
	//     weight = sum_j batch^j * eq(point_j, point)
	let mut weight = C::Elem::zero();
	let mut batch_power = C::Elem::one();
	for claim in &claims {
		weight += eq_ind(&claim.point, &point) * &batch_power;
		batch_power *= &output.batch_coeff;
	}

	// The reduced claim is the product of the two factors the sumcheck ran over.
	channel.assert_zero(eval.clone() * &weight - output.eval)?;

	Ok(MultilinearEvalClaim { eval, point })
}
