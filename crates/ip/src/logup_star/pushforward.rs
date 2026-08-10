// Copyright 2026 The Binius Developers

//! The pushforward reduction that closes the table side.
//!
//! The table-side fractional-addition GKR runs to its leaf, which claims the pushforward `Y` at a
//! point `z`; its denominator half is the public `D = J - c`, which the verifier checks itself.
//! That leaves two claims that both read `Y` — the leaf claim `Y(z)` and the product claim
//! `<T, Y> = e` — and one batched `m`-variable sumcheck reduces them to a single evaluation point.

use binius_field::{BinaryField1b, ExtensionField, Field, field::FieldOps};
use binius_math::multilinear::eq::eq_ind;

use super::error::{Error, VerificationError};
use crate::{
	channel::IPVerifierChannel,
	sumcheck::{self, BatchSumcheckOutput},
};

/// The output of the pushforward reduction.
pub struct Pushforward<F> {
	/// The `m`-coordinate point at which both `T` and `Y` are evaluated.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of `T` at the point.
	pub table_eval_claim: F,
	/// The claimed evaluation of `Y` at the point.
	pub pushforward_eval_claim: F,
}

/// Verify the pushforward reduction.
///
/// Batches two sum claims over the `m`-variable cube:
///
/// ```text
///     S_1 = sum_x eq(x; z) * Y(x) = Y(z)      the GKR leaf claim, re-randomized
///     S_2 = sum_x (Y * T)(x)      = e         the product claim
/// ```
///
/// Both summands are degree 2 per variable — `eq` times a multilinear, and a product of two
/// multilinears — so the batched degree is 2. The reduction ends in one evaluation of `Y` and one
/// of `T` at the shared challenge point.
///
/// # Arguments
///
/// * `eval_claim` - The product claim `e = <T, Y>`.
/// * `pushforward_eval_claim` - The GKR leaf claim `Y(z)`.
/// * `pushforward_eval_point` - The leaf point `z`, of length `m`.
/// * `channel` - The verifier channel.
pub fn verify_pushforward<F, C>(
	eval_claim: C::Elem,
	pushforward_eval_claim: C::Elem,
	pushforward_eval_point: &[C::Elem],
	channel: &mut C,
) -> Result<Pushforward<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IPVerifierChannel<F>,
	C::Elem: From<F>,
{
	let m = pushforward_eval_point.len();

	// Batch the two sum claims with powers of a single coefficient and run the sumcheck.
	//
	//     sum = Y(z) + bc * e
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify::<F, C>(m, 2, &[pushforward_eval_claim, eval_claim], channel)?;

	// Read the pushforward and table evaluations at the sumcheck challenge point.
	let [y_eval, t_eval] = channel
		.recv_array()
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// Sumcheck binds variables highest-to-lowest; reverse to align with the low-to-high point z.
	challenges.reverse();
	let rho = challenges;

	// Reconstruct the batched summand at the challenge point and check it against the reduced eval.
	// Both claims read Y, so it factors out:
	//
	//     eq(rho; z) * Y(rho) + bc * Y(rho) * T(rho) = Y(rho) * (eq(rho; z) + bc * T(rho))
	let eq_eval = eq_ind::<C::Elem>(&rho, pushforward_eval_point);
	let expected = y_eval.clone() * (eq_eval + batch_coeff * t_eval.clone());
	channel
		.assert_zero(eval - expected)
		.map_err(|_| VerificationError::PushforwardMismatch)?;

	Ok(Pushforward {
		table_eval_point: rho,
		table_eval_claim: t_eval,
		pushforward_eval_claim: y_eval,
	})
}

/// Evaluate the negated table-side denominator multilinear at a point.
///
/// The logUp denominator is `c - J(x)` with the index embedding `J(x) = sum_{t} basis(t) * x_t`.
/// The table's fraction enters the sum of every instance negated, and that negation is carried on
/// the denominator, so this returns `J(x) - c`. Either way it is transparent: the verifier
/// evaluates it itself rather than taking the prover's word for the GKR leaf's denominator half.
///
/// In characteristic 2 subtraction is addition, but the field operations are written generically.
///
/// # Arguments
///
/// * `c` - The logUp challenge.
/// * `point` - The evaluation point, in low-to-high coordinate order.
pub fn denominator_eval<F, E>(c: &E, point: &[E]) -> E
where
	F: ExtensionField<BinaryField1b>,
	E: FieldOps + From<F>,
{
	let j = point
		.iter()
		.enumerate()
		.map(|(t, coord)| {
			let basis_t = E::from(<F as ExtensionField<BinaryField1b>>::basis(t));
			basis_t * coord.clone()
		})
		.fold(E::zero(), |acc, term| acc + term);

	j - c.clone()
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField1b, ExtensionField, Random, arch::OptimalB128 as B128};
	use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, test_utils::random_scalars};
	use rand::prelude::*;

	use super::*;

	// Embed a table position j into the field through the GF(2)-linear basis.
	//
	//     iota(j) = sum_{t : bit t of j is set} basis(t)
	fn iota(j: usize, m: usize) -> B128 {
		(0..m)
			.filter(|t| (j >> t) & 1 == 1)
			.map(<B128 as ExtensionField<BinaryField1b>>::basis)
			.fold(B128::ZERO, |acc, b| acc + b)
	}

	// Evaluate the multilinear `values` at `point` as the inner product with the eq tensor.
	fn evaluate_scalars(values: &[B128], point: &[B128]) -> B128 {
		let eq = eq_ind_partial_eval_scalars(point);
		values
			.iter()
			.zip(&eq)
			.map(|(v, e)| *v * *e)
			.fold(B128::ZERO, |acc, t| acc + t)
	}

	fn check_denominator_eval(m: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		let c = B128::random(&mut rng);
		let point = random_scalars::<B128>(&mut rng, m);

		// The explicitly built negated denominator multilinear D[j] = iota(j) - c over the table
		// cube.
		let d_values = (0..(1usize << m))
			.map(|j| iota(j, m) - c)
			.collect::<Vec<_>>();

		assert_eq!(
			denominator_eval::<B128, B128>(&c, &point),
			evaluate_scalars(&d_values, &point),
			"denominator evaluation mismatch for m = {m}"
		);
	}

	#[test]
	fn test_denominator_eval_matches_explicit_multilinear() {
		// m = 0 exercises the empty point; larger m exercise the basis sum over many coordinates.
		for m in 0..=6 {
			check_denominator_eval(m);
		}
	}
}
