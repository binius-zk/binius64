// Copyright 2026 The Binius Developers

//! The top-level logUp* verification routine.

use binius_field::{BinaryField1b, ExtensionField, Field};
use binius_math::multilinear::eq::eq_ind;

use super::{
	error::{Error, VerificationError},
	final_layer::{FinalLayer, verify_final_layer},
	output::LogupOutput,
};
use crate::{
	channel::IPVerifierChannel,
	fracaddcheck::{self, FracAddEvalClaim},
};

/// Verify a logUp* indexed-lookup reduction.
///
/// Reduces the claim `(I^* T)(eval_point) = eval_claim` to the claims in [`LogupOutput`].
///
/// The logUp challenge `c` is sampled against the committed `I`, `T`, and pushforward `Y`.
/// So the caller must absorb those commitments into the transcript before calling this routine.
///
/// # Arguments
///
/// * `table_n_vars` - The number of variables `m` of the table multilinear (`2^m` entries).
/// * `eval_claim` - The claimed evaluation `e` of the looked-up vector.
/// * `eval_point` - The `n`-coordinate evaluation point `r`; its length defines `n`.
/// * `channel` - The verifier channel for receiving prover messages and sampling challenges.
///
/// # Transcript layout
///
/// The prover messages are consumed in this exact order:
///
/// ```text
///     1. sample c                                  (logUp challenge)
///     2. recv [num_L, den_L, num_R, den_R]         (root fractions of both GKR circuits)
///     3. looker-side GKR, n layers                 (see fracaddcheck::verify)
///     4. table-side GKR, first m-1 layers          (see fracaddcheck::verify)
///     5. batched final layer:
///        a. sample batch_coeff
///        b. m-1 rounds of degree-3 sumcheck
///        c. recv [Y_0, Y_1, T_0, T_1]              (leaf halves, split on the highest variable)
///        d. sample r                               (final line-fold)
/// ```
///
/// The batched final sumcheck is assumed to bind variables from the highest index to the lowest.
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
/// - the `eq_r` evaluation is wrong,
/// - the batched final layer is inconsistent.
pub fn verify<F, C>(
	table_n_vars: usize,
	eval_claim: C::Elem,
	eval_point: &[C::Elem],
	channel: &mut C,
) -> Result<LogupOutput<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IPVerifierChannel<F>,
	C::Elem: From<F>,
{
	// The table-side GKR circuit needs at least one variable to split on.
	assert!(table_n_vars > 0, "table must have at least one variable");

	let m = table_n_vars;
	let n = eval_point.len();

	// Sample the logUp challenge c that randomizes the logarithmic-derivative denominators.
	let c = channel.sample();

	// Read the root fractions of both fractional-addition GKR circuits.
	//
	//     looker side: num_l / den_l = sum_{i in B_n} eq_r(i) / (c - I(i))
	//     table  side: num_r / den_r = sum_{j in B_m} Y(j)    / (c - j)
	let [num_l, den_l, num_r, den_r] = channel
		.recv_array()
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// The lookup identity is the equality of the two fractional sums.
	//
	//     num_l / den_l = num_r / den_r   <=>   num_l * den_r - num_r * den_l = 0
	let sum_diff = num_l.clone() * den_r.clone() - num_r.clone() * den_l.clone();
	channel
		.assert_zero(sum_diff)
		.map_err(|_| VerificationError::LookupSumMismatch)?;

	// Looker side: run the full n-layer GKR to reach the leaf claim.
	//
	//     leaf numerator   = eq_r(point_l)
	//     leaf denominator = c - I(point_l)
	let FracAddEvalClaim {
		num_eval: looker_num,
		den_eval: looker_den,
		point: index_eval_point,
	} = fracaddcheck::verify::<F, C>(
		n,
		FracAddEvalClaim {
			num_eval: num_l,
			den_eval: den_l,
			point: Vec::new(),
		},
		channel,
	)?;

	// The looker numerator is the eq_r multilinear, which the verifier evaluates by itself.
	//
	//     eq_r(point_l) = eq(eval_point, point_l)
	let expected_eq = eq_ind::<C::Elem>(eval_point, &index_eval_point);
	channel
		.assert_zero(looker_num - expected_eq)
		.map_err(|_| VerificationError::IncorrectXEvaluation)?;

	// The looker denominator is c - I(point_l), so the index claim is I(point_l) = c - den.
	let index_eval_claim = c.clone() - looker_den;

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
	let FinalLayer {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	} = verify_final_layer::<F, C>(m, c, eval_claim, layer1_num, layer1_den, &layer1_point, channel)?;

	Ok(LogupOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
		index_eval_point,
		index_eval_claim,
	})
}

#[cfg(test)]
mod tests {
	use binius_field::arch::OptimalB128 as B128;
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
		let _ = verify::<B128, _>(0, B128::ZERO, &[], &mut verifier);
	}
}
