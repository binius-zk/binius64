// Copyright 2026 The Binius Developers

//! The top-level logUp* verification routine.

use std::iter;

use binius_field::{BinaryField1b, ExtensionField, Field, field::FieldOps, util::powers};
use binius_math::{
	multilinear::{eq::eq_ind, evaluate::evaluate_inplace_scalars},
	univariate::evaluate_univariate,
};

use super::{
	error::{Error, VerificationError},
	final_layer::{FinalLayer, verify_final_layer},
	output::LogupOutput,
};
use crate::{
	channel::IPVerifierChannel,
	fracaddcheck::{self, FracAddEvalClaim},
};

/// One looker's claim on its looked-up vector: `(I_j^* T)(eval_point) = eval_claim`.
#[derive(Debug, Clone)]
pub struct LookerClaim<'a, Elem> {
	/// The `n`-coordinate evaluation point of this looker's claim.
	pub eval_point: &'a [Elem],
	/// The claimed evaluation of this looker's looked-up vector at the point.
	pub eval_claim: Elem,
}

/// Verify a logUp* indexed-lookup reduction over one or more lookers sharing a table.
///
/// Reduces the claims `(I_j^* T)(r_j) = e_j` to the claims in [`LogupOutput`]. The lookers batch
/// by a random linear combination: the challenge `gamma` scales looker `j`'s numerator by
/// `gamma^j`, the pushforward is the gamma-weighted sum of the per-looker pushforwards, and the
/// product check binds `<T, Y>` to the gamma-combination of the claims. The looker side runs as
/// one GKR circuit over `k + n` variables (`k = ceil(log2 #lookers)`): the top `k` layers
/// fractionally add the per-looker circuit roots (padded with the zero fraction `0/1`), so the
/// reduced index claims share one evaluation point.
///
/// The caller samples `gamma` itself, before receiving the pushforward commitment: the prover
/// needs `gamma` to build the combined pushforward, so the commitment must come after.
///
/// The logUp challenge `c` is sampled against the committed `I_j`, `T`, and pushforward `Y`.
/// So the caller must absorb those commitments into the transcript before calling this routine.
///
/// # Arguments
///
/// * `gamma` - The looker batching challenge, sampled by the caller.
/// * `table_n_vars` - The number of variables `m` of the table multilinear (`2^m` entries).
/// * `lookers` - The looker claims; every evaluation point must have the same length `n`.
/// * `channel` - The verifier channel for receiving prover messages and sampling challenges.
///
/// # Transcript layout
///
/// The prover messages are consumed in this exact order:
///
/// ```text
///     1. sample c                                  (logUp challenge)
///     2. recv [num_L, den_L, num_R, den_R]         (root fractions of both GKR circuits)
///     3. looker-side GKR, k + n layers             (see fracaddcheck::verify)
///     4. recv per-looker index evaluations
///     5. table-side GKR, first m-1 layers          (see fracaddcheck::verify)
///     6. batched final layer:
///        a. recv e_0                               (partial product sum <Y_0, T_0>)
///        b. sample batch_coeff
///        c. m-1 rounds of degree-3 sumcheck
///        d. recv [Y_0, Y_1, T_0, T_1]              (leaf halves, split on the highest variable)
///        e. sample r                               (final line-fold)
/// ```
///
/// The sumchecks are assumed to bind variables from the highest index to the lowest.
/// This matches the convention of the fractional-addition GKR layers.
///
/// # Preconditions
///
/// - `table_n_vars` must be at least 1, so the table-side GKR has a variable to split on.
///
/// # Returns
///
/// The reduced [`LogupOutput`] claims on the table, pushforward, and index multilinears.
///
/// # Errors
///
/// Returns an error when the proof is malformed or any verification identity fails:
///
/// - the two lookup sums disagree,
/// - the batched `eq_r` evaluation is wrong,
/// - the index evaluations do not combine to the batched leaf denominator,
/// - the batched final layer is inconsistent.
pub fn verify_reduction<F, C>(
	gamma: C::Elem,
	table_n_vars: usize,
	lookers: &[LookerClaim<'_, C::Elem>],
	channel: &mut C,
) -> Result<LogupOutput<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IPVerifierChannel<F>,
	C::Elem: From<F>,
{
	// The table-side GKR circuit needs at least one variable to split on.
	assert!(table_n_vars > 0, "table must have at least one variable");
	assert!(!lookers.is_empty(), "at least one looker claim is required");

	let m = table_n_vars;
	let n = lookers[0].eval_point.len();
	assert!(
		lookers.iter().all(|looker| looker.eval_point.len() == n),
		"every looker evaluation point must have the same length"
	);
	let log_lookers = lookers.len().next_power_of_two().ilog2() as usize;

	// Sample the logUp challenge c that randomizes the logarithmic-derivative denominators.
	let c = channel.sample();

	// Read the root fractions of both fractional-addition GKR circuits.
	//
	//     looker side: num_L / den_L = sum_j gamma^j sum_{i in B_n} eq_{r_j}(i) / (c - I_j(i))
	//     table  side: num_R / den_R = sum_{v in B_m} Y(v) / (c - v)
	let [num_l, den_l, num_r, den_r] = channel
		.recv_array()
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// The lookup identity is the equality of the two fractional sums.
	//
	//     num_L / den_L = num_R / den_R   <=>   num_L * den_R - num_R * den_L = 0
	let sum_diff = num_l.clone() * den_r.clone() - num_r.clone() * den_l.clone();
	channel
		.assert_zero(sum_diff)
		.map_err(|_| VerificationError::LookupSumMismatch)?;

	// Looker side: run the full GKR down to the leaf claim. The top log_lookers layers
	// fractionally add the per-looker circuit roots (interpolated over the looker variables,
	// padded with the zero fraction 0/1), and the remaining n layers run the per-looker circuits
	// batched by the selector coordinates those top layers bind.
	let FracAddEvalClaim {
		num_eval: looker_num,
		den_eval: looker_den,
		point: leaf_point,
	} = fracaddcheck::verify::<F, C>(
		n + log_lookers,
		FracAddEvalClaim {
			num_eval: num_l,
			den_eval: den_l,
			point: Vec::new(),
		},
		channel,
	)?;

	// The leaf point splits into the selector coordinates and the shared content point.
	let (selector_coords, content_point) = leaf_point.split_at(log_lookers);

	// The per-looker leaf numerators are transparent scaled equality indicators, so the verifier
	// evaluates the batched leaf numerator by itself (padding numerators are zero):
	//
	//     N(leaf) = sum_j eq(selector_coords, j) * gamma^j * eq(r_j, content_point)
	let mut eq_evals = iter::zip(lookers, powers(gamma.clone()))
		.map(|(looker, power)| power * eq_ind::<C::Elem>(looker.eval_point, content_point))
		.collect::<Vec<_>>();
	eq_evals.resize(1 << log_lookers, C::Elem::zero());
	let expected_num = evaluate_inplace_scalars(eq_evals, selector_coords);
	channel
		.assert_zero(looker_num - expected_num)
		.map_err(|_| VerificationError::IncorrectXEvaluation)?;

	// Receive the per-looker index evaluations and check they combine to the batched leaf
	// denominator, whose per-looker leaves are c - I_j (padding denominators are one):
	//
	//     den(leaf) = sum_j eq(selector_coords, j) * (c - I_j(content_point))
	let index_eval_claims = channel
		.recv_many(lookers.len())
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;
	let mut den_evals = index_eval_claims
		.iter()
		.map(|eval| c.clone() - eval.clone())
		.collect::<Vec<_>>();
	den_evals.resize(1 << log_lookers, C::Elem::one());
	let expected_den = evaluate_inplace_scalars(den_evals, selector_coords);
	channel
		.assert_zero(looker_den - expected_den)
		.map_err(|_| VerificationError::IncorrectIndexEvaluation)?;

	// Table side: run the first m-1 GKR layers, stopping at the layer-1 claim.
	//
	//     num_1(Z) = sum_{x' in B_{m-1}} eq(x'; Z) * (Y_0 * D_1 + Y_1 * D_0)(x')
	//     den_1(Z) = sum_{x' in B_{m-1}} eq(x'; Z) * (D_0 * D_1)(x')
	//
	// where D = c - J is the table-side denominator and the halves split on the highest variable.
	let FracAddEvalClaim {
		num_eval: layer1_num,
		den_eval: layer1_den,
		point: layer1_point,
	} = fracaddcheck::verify::<F, C>(
		m - 1,
		FracAddEvalClaim {
			num_eval: num_r,
			den_eval: den_r,
			point: Vec::new(),
		},
		channel,
	)?;

	// Run the batched final layer.
	// It reduces the layer-1 claims and the product claim <T, Y> = e to a single shared evaluation.
	// The product check binds <T, Y> to the gamma-combination of the looker claims.
	let claims = lookers
		.iter()
		.map(|looker| looker.eval_claim.clone())
		.collect::<Vec<_>>();
	let combined_eval_claim = evaluate_univariate(&claims, gamma);
	let FinalLayer {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	} = verify_final_layer::<F, C>(
		m,
		c,
		combined_eval_claim,
		layer1_num,
		layer1_den,
		&layer1_point,
		channel,
	)?;

	Ok(LogupOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
		index_eval_point: content_point.to_vec(),
		index_eval_claims,
	})
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, arch::OptimalB128 as B128};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	use super::*;

	type StdChallenger = HasherChallenger<sha2::Sha256>;

	#[test]
	#[should_panic(expected = "table must have at least one variable")]
	fn test_empty_table_panics() {
		// A zero-variable table has no variable for the GKR circuit to split on.
		let transcript = ProverTranscript::new(StdChallenger::default());
		let mut verifier = transcript.into_verifier();

		// The precondition assertion fires before any transcript interaction.
		let claim = LookerClaim {
			eval_point: &[],
			eval_claim: B128::ZERO,
		};
		let _ = verify_reduction::<B128, _>(B128::ZERO, 0, &[claim], &mut verifier);
	}
}
