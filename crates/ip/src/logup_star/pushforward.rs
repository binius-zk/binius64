// Copyright 2026 The Binius Developers

//! The pushforward reduction that closes the table side.
//!
//! Each table's fractional-addition GKR runs to its leaf, which claims that table's pushforward
//! `Y_t` at a point `z_t`; its denominator half is the public `D = J - c_t`, which the verifier
//! checks itself. That leaves two claims per table that both read `Y_t` — the leaf claim `Y_t(z_t)`
//! and the product claim `<T_t, Y_t> = e_t` — and one batched sumcheck reduces all of them to a
//! single evaluation point.

use std::iter;

use binius_field::{BinaryField1b, ExtensionField, Field, field::FieldOps};
use binius_math::{
	multilinear::eq::{eq_ind, eq_ind_zero},
	univariate::evaluate_univariate,
};

use super::error::{Error, VerificationError};
use crate::{
	channel::IPVerifierChannel,
	sumcheck::{self, BatchSumcheckOutput},
};

/// One table's two claims entering the pushforward reduction.
#[derive(Debug, Clone)]
pub struct TableClaim<'a, F> {
	/// The product claim `e_t = <T_t, Y_t>`.
	pub eval_claim: F,
	/// The fractional-addition leaf claim `Y_t(z_t)`.
	pub pushforward_eval_claim: F,
	/// The leaf point `z_t`, of length `m_t`.
	pub pushforward_eval_point: &'a [F],
}

/// The output of the pushforward reduction.
pub struct Pushforward<F> {
	/// The point spanning the widest table, at which every table's claims are taken.
	///
	/// Table `t`, over `m_t` variables, is claimed at the **first `m_t`** coordinates: the batch
	/// pads each table at its high coordinates.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluations of the tables `T_t`, each at its own prefix of the point.
	pub table_eval_claims: Vec<F>,
	/// The claimed evaluations of the pushforwards `Y_t`, each at the same prefix.
	pub pushforward_eval_claims: Vec<F>,
}

/// Verify the pushforward reduction over one or more tables.
///
/// Each table contributes two sum claims over its own `m_t`-variable cube:
///
/// ```text
///     S_1 = sum_x eq(x; z_t) * Y_t(x) = Y_t(z_t)      the GKR leaf claim, re-randomized
///     S_2 = sum_x (Y_t * T_t)(x)      = e_t           the product claim
/// ```
///
/// Both summands are degree 2 per variable — `eq` times a multilinear, and a product of two
/// multilinears — so the batched degree is 2.
///
/// Tables need not agree on a size. A table over `m_t < max_t m_t` variables is batched through
/// the padded claim `S * eq(0^nu; X_pad)` over the padding variables, which are the highest ones
/// and so are bound first; that equality factor sums to one over the cube, so the padded claim
/// holds exactly when the original does. The deepest table is never padded, so every round
/// polynomial in the batch still has degree 2.
///
/// The reduction ends in one evaluation of `Y_t` and one of `T_t` per table, all at the prefix of
/// one shared point.
///
/// # Arguments
///
/// * `tables` - The per-table claims; their leaf points may differ in length.
/// * `channel` - The verifier channel.
///
/// # Preconditions
///
/// - `tables` is non-empty.
pub fn verify_pushforward<'a, F, C>(
	tables: impl IntoIterator<Item = TableClaim<'a, C::Elem>>,
	channel: &mut C,
) -> Result<Pushforward<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IPVerifierChannel<F>,
	C::Elem: From<F> + 'a,
{
	// The claims are walked three times — for the deepest table, for the batched sums, and for the
	// reconstruction — so the iterator is materialized once here.
	let tables = tables.into_iter().collect::<Vec<_>>();
	assert!(!tables.is_empty(), "at least one table is required"); // precondition

	let max_m = tables
		.iter()
		.map(|table| table.pushforward_eval_point.len())
		.max()
		.expect("tables is non-empty");

	// Batch every table's two sum claims with powers of a single coefficient and run the sumcheck.
	// The claims are flattened in table order, two per table, which is the order the prover's
	// round polynomials arrive in.
	//
	//     sum = Y_0(z_0) + bc * e_0 + bc^2 * Y_1(z_1) + ...
	let sums = tables
		.iter()
		.flat_map(|table| {
			[
				table.pushforward_eval_claim.clone(),
				table.eval_claim.clone(),
			]
		})
		.collect::<Vec<_>>();
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify::<F, C>(max_m, 2, &sums, channel)?;

	// Read every table's pushforward and table evaluations at the sumcheck challenge point, in the
	// same table order.
	let evals: Vec<C::Elem> = channel
		.recv_many(2 * tables.len())
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;
	let (pushforward_eval_claims, table_eval_claims): (Vec<_>, Vec<_>) = evals
		.chunks_exact(2)
		.map(|pair| (pair[0].clone(), pair[1].clone()))
		.unzip();

	// Sumcheck binds variables highest-to-lowest; reverse to align with the low-to-high points z_t.
	challenges.reverse();
	let rho = challenges;

	// Reconstruct each table's pair of summands at the challenge point and check the batch against
	// the reduced evaluation. Both of a table's claims read Y_t, so it factors out:
	//
	//     eq(rho_t; z_t) * Y_t(rho_t) + bc * Y_t(rho_t) * T_t(rho_t)
	//         = Y_t(rho_t) * (eq(rho_t; z_t) + bc * T_t(rho_t))
	//
	// and a padded table's contribution carries the padding coordinates' equality weight.
	let contributions = iter::zip(&tables, iter::zip(&pushforward_eval_claims, &table_eval_claims))
		.flat_map(|(table, (y_eval, t_eval))| {
			let (own_point, padding_point) = rho.split_at(table.pushforward_eval_point.len());
			let weighted_y = y_eval.clone() * eq_ind_zero::<C::Elem>(padding_point);
			[
				weighted_y.clone() * eq_ind::<C::Elem>(own_point, table.pushforward_eval_point),
				weighted_y * t_eval.clone(),
			]
		})
		.collect::<Vec<_>>();
	let expected = evaluate_univariate(&contributions, &batch_coeff);
	channel
		.assert_zero(eval - expected)
		.map_err(|_| VerificationError::PushforwardMismatch)?;

	Ok(Pushforward {
		table_eval_point: rho,
		table_eval_claims,
		pushforward_eval_claims,
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
			let basis_t = E::from(F::basis(t));
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
