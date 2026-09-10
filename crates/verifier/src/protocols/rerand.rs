// Copyright 2026 The Binius Developers

//! The BitAnd sumcheck, batched with the multiplication reductions' operand-column MLE-checks.
//!
//! After the univariate skip, the BitAnd claim `g(z)` is the MLE-check
//!
//! ```text
//! g(z) = sum_x eq(x, r_x) * (A(z, x) * B(z, x) - C(z, x))
//! ```
//!
//! at the zerocheck point `r_x`. Each operand column of an IntMul or BinMul reduction carries one
//! claim per bit at that reduction's own point `rho`. The Lagrange weights at `z` fold them to one
//! MLE-check per column, `alpha = sum_x eq(x, rho) * Z(z, x)`, with no transcript traffic.
//!
//! One sumcheck in plain form proves all of them, batched by powers of one challenge `theta`,
//! BitAnd first. It runs over the longest summand's variables. A shorter summand is zero-padded on
//! its high variables, so its point is a prefix of the unified point `r*`.
//!
//! The round polynomials have degree 3: BitAnd's composition has degree 2, and its equality
//! indicator adds one.

use std::iter;

use binius_core::word::Word;
use binius_field::Field;
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{BatchSumcheckOutput, batch_verify},
};
use binius_math::{
	inner_product::inner_product,
	multilinear::eq::{eq_ind, eq_ind_zero},
	univariate::evaluate_univariate,
};

use crate::Error;

/// One reduction's per-bit evaluation claims on its operand columns, at that reduction's point.
pub struct OperandClaims<'a, E> {
	/// The reduction's constraint point. In batch mode: instances low, constraints high.
	pub point: &'a [E],
	/// One 64-entry array per operand column, in the shift reduction's operand order.
	pub columns: Vec<&'a [E; Word::BITS]>,
}

/// Degree of the batched round polynomials: the BitAnd summand's prime degree 2 plus its indicator.
pub const SUMCHECK_DEGREE: usize = 3;

/// The evaluation claims the batched sumcheck reduces to, all at prefixes of one point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerandOutput<F> {
	/// The unified point `r*`, low-to-high. Its length is the longest summand's variable count.
	pub eval_point: Vec<F>,
	/// The evaluations of `A`, `B` and `C`, folded at `z`, at `eval_point[..r_x.len()]`.
	pub bitand_evals: [F; 3],
	/// One evaluation per operand column, flattened in input order, each at
	/// `eval_point[..point.len()]` for its reduction's point.
	pub operand_evals: Vec<F>,
}

/// Verifies the BitAnd sumcheck batched with the operand-column MLE-checks.
///
/// The prover sends the round polynomials, then the evaluations `[a, b, c]`, then one evaluation
/// per operand column in input order.
///
/// # Arguments
///
/// * `zerocheck_challenges` - the BitAnd summand's point `r_x`.
/// * `bitand_claim` - the univariate-skip polynomial at `z`, `g(z)`.
/// * `lagrange` - the Lagrange weights at `z` on the 64-point domain.
/// * `operands` - the operand claims of each multiplication reduction.
/// * `channel` - the verifier channel.
///
/// # Soundness
///
/// The per-bit claims of `operands` must be in the transcript before `z` was drawn. Otherwise a
/// prover can choose them as a function of the weights that fold them.
///
/// # Errors
///
/// Returns an error if the sumcheck or its closing check fails.
pub fn verify<F, C>(
	zerocheck_challenges: &[C::Elem],
	bitand_claim: C::Elem,
	lagrange: &[C::Elem],
	operands: &[OperandClaims<'_, C::Elem>],
	channel: &mut C,
) -> Result<RerandOutput<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	// Fold each column's per-bit claims to one MLE-check claim at the reduction's point.
	let operand_claims = operands.iter().flat_map(|operand| {
		operand
			.columns
			.iter()
			.map(|column| inner_product(column.iter().cloned(), lagrange.iter().cloned()))
	});
	let sums = iter::once(bitand_claim)
		.chain(operand_claims)
		.collect::<Vec<_>>();

	let n_vars = operands
		.iter()
		.map(|operand| operand.point.len())
		.fold(zerocheck_challenges.len(), usize::max);
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		challenges: mut eval_point,
	} = batch_verify(n_vars, SUMCHECK_DEGREE, &sums, channel)?;
	eval_point.reverse();

	let bitand_evals = channel.recv_array::<3>()?;
	let operand_evals = channel.recv_many(sums.len() - 1)?;

	// A summand at `point` is weighted by its indicator on the low coordinates and by the zero
	// indicator on the padding coordinates above them.
	let weight = |point: &[C::Elem]| {
		let (low, padding) = eval_point.split_at(point.len());
		eq_ind(point, low) * eq_ind_zero(padding)
	};
	let [a, b, c] = bitand_evals.clone();
	let bitand_term = weight(zerocheck_challenges) * (a * b - c);
	let operand_weights = operands
		.iter()
		.flat_map(|operand| iter::repeat_n(weight(operand.point), operand.columns.len()));
	let operand_terms =
		iter::zip(operand_weights, &operand_evals).map(|(weight, eval)| weight * eval.clone());
	let terms = iter::once(bitand_term)
		.chain(operand_terms)
		.collect::<Vec<_>>();
	channel.assert_zero(evaluate_univariate(&terms, &batch_coeff) - eval)?;

	Ok(RerandOutput {
		eval_point,
		bitand_evals,
		operand_evals,
	})
}
