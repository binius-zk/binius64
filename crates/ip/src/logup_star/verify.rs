// Copyright 2026 The Binius Developers

//! The top-level logUp* verification routine.

use std::{iter, slice};

use binius_field::{BinaryField1b, ExtensionField, Field, field::FieldOps, util::powers};
use binius_math::{
	multilinear::{
		eq::{eq_ind, eq_ind_zero},
		evaluate::evaluate_inplace_scalars,
	},
	univariate::evaluate_univariate,
};

use super::{
	error::{Error, VerificationError},
	output::LogupOutput,
	pushforward::{Pushforward, TableClaim, denominator_eval, verify_pushforward},
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
/// product check binds `<T, Y>` to the gamma-combination of the claims.
///
/// Every fractional-addition circuit — one per looker `j` over `n_j` variables, plus the table's
/// over `m` — is an instance of **one** GKR of `k + max(max_j n_j, m)` layers, where
/// `k = ceil(log2(#lookers + 1))`. Its top `k` layers add the per-instance root fractions together
/// and its lower `max(max_j n_j, m)` run the instances in a batch, with every shallower instance
/// padded by zero fractions — that leaves a fractional sum unchanged and costs `O(1)` per round,
/// so the layer count depends only on the deepest instance.
///
/// The lookers need not agree on a length: a looker over `n_j` variables is simply padded by
/// `max(max_j n_j, m) - n_j`, and its reduced index claim lands on the **last `n_j`** coordinates
/// of [`LogupOutput::index_eval_point`].
///
/// The table's fraction enters that sum negated, so the circuit's root is
/// `sum_j num_j/den_j - num_r/den_r`, whose numerator vanishes exactly when the lookup identity
/// holds. The verifier therefore reads only the root *denominator* and supplies the zero numerator
/// itself: the identity is enforced by the shape of the claim rather than by a separate check.
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
/// * `lookers` - The looker claims; their evaluation points may differ in length.
/// * `channel` - The verifier channel for receiving prover messages and sampling challenges.
///
/// # Transcript layout
///
/// The prover messages are consumed in this exact order:
///
/// ```text
///     1. sample c                                  (logUp challenge)
///     2. recv den_root                             (the root denominator; its numerator is 0)
///     3. combined GKR, k + max(max_j n_j, m) layers (see fracaddcheck::verify)
///     4. recv per-looker index evaluations, then Y (the non-transparent leaf halves)
///     5. pushforward reduction:
///        a. sample batch_coeff
///        b. m rounds of degree-2 sumcheck
///        c. recv [Y, T]                            (evaluations at the challenge point)
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
/// - a GKR layer's reduction is inconsistent, which is where a violated lookup identity surfaces,
/// - the transparent leaf numerators do not interpolate to the batch's leaf numerator,
/// - the index and table denominators do not interpolate to the batch's leaf denominator,
/// - the pushforward reduction is inconsistent.
pub fn verify_reduction<F, C>(
	gamma: &C::Elem,
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
	// Lookers may differ in length; the batch pads each up to the deepest instance.
	let max_n = lookers
		.iter()
		.map(|looker| looker.eval_point.len())
		.max()
		.expect("lookers is non-empty");

	// Sample the logUp challenge c that randomizes the logarithmic-derivative denominators.
	let c = channel.sample();

	// Read the root denominator. The root numerator is not on the transcript: the whole circuit
	// sums the looker fractions against the negated table fraction, so its value is zero exactly
	// when the lookup identity holds, and the verifier supplies that zero itself.
	let root_den: C::Elem = channel
		.recv_one()
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// One GKR over the whole thing, from that single root fraction down to the leaves: k layers
	// interpolating the per-instance roots, then max(max_n, m) more over the instances. Looker j's
	// tree has depth n_j and the table tree depth m, so every shallower instance is padded by zero
	// fractions — the layer count reveals only the deepest one.
	let n_instances = lookers.len() + 1;
	let k = n_instances.next_power_of_two().ilog2() as usize;
	let n_layers = k + max_n.max(m);
	let FracAddEvalClaim {
		num_eval: leaf_num,
		den_eval: leaf_den,
		point: leaf_point,
	} = fracaddcheck::verify::<F, C>(
		n_layers,
		FracAddEvalClaim {
			num_eval: C::Elem::zero(),
			den_eval: root_den,
			point: Vec::new(),
		},
		channel,
	)?;

	// The leaf point splits into the selector coordinates and the shared node point.
	let (selector_coords, node_point) = leaf_point.split_at(k);

	// Read the claims the verifier cannot derive: the per-looker index evaluations and Y's.
	let index_eval_claims: Vec<C::Elem> = channel
		.recv_many(lookers.len())
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;
	let pushforward_eval: C::Elem = channel
		.recv_one()
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// Rebuild each circuit's padded leaf fraction and check they interpolate to the batch's leaf.
	//
	// The node point spans max(max_n, m) coordinates, so an instance over `d` variables is padded
	// by `max(max_n, m) - d` and its own content is the last `d` coordinates. Padding scales a
	// numerator by the padding coordinates' equality weight q and sends a denominator through
	// sel(q, .), so both halves follow from the claims above.
	let n_node_vars = node_point.len();

	// Every instance's padding weight is a prefix of the node point, so accumulate the prefixes
	// once: `pad_eqs[p] = eq(0^p; node_point[..p])`. With lookers of differing lengths there is one
	// weight per distinct depth, and this indexes them all in a single pass.
	let pad_eqs = iter::once(C::Elem::one())
		.chain(node_point.iter().scan(C::Elem::one(), |acc, coord| {
			*acc = acc.clone() * eq_ind_zero::<C::Elem>(slice::from_ref(coord));
			Some(acc.clone())
		}))
		.collect::<Vec<_>>();

	let table_pad = n_node_vars - m;
	let table_point = &node_point[table_pad..];

	// Looker numerators are transparent: gamma^j scales the equality indicator at r_j. The
	// denominators are c - I_j, from the index evaluations just read.
	let (mut leaf_nums, mut leaf_dens): (Vec<_>, Vec<_>) =
		iter::zip(iter::zip(lookers, powers(gamma.clone())), &index_eval_claims)
			.map(|((looker, power), index_eval)| {
				let pad = n_node_vars - looker.eval_point.len();
				let content = &node_point[pad..];
				let num = power * eq_ind::<C::Elem>(looker.eval_point, content);
				let den = c.clone() - index_eval.clone();
				fracaddcheck::pad_leaf_fraction((num, den), pad_eqs[pad].clone())
			})
			.unzip();

	// The table's numerator is Y, just read. Its denominator is the transparent J - c, the logUp
	// denominator negated — the table's fraction enters the sum that way, which is what makes the
	// root numerator vanish.
	let (table_num, table_den) = fracaddcheck::pad_leaf_fraction(
		(pushforward_eval.clone(), denominator_eval::<F, C::Elem>(&c, table_point)),
		pad_eqs[table_pad].clone(),
	);
	leaf_nums.push(table_num);
	leaf_dens.push(table_den);

	leaf_nums.resize(1 << k, C::Elem::zero());
	leaf_dens.resize(1 << k, C::Elem::one());
	channel
		.assert_zero(leaf_num - evaluate_inplace_scalars(leaf_nums, selector_coords))
		.map_err(|_| VerificationError::IncorrectXEvaluation)?;
	channel
		.assert_zero(leaf_den - evaluate_inplace_scalars(leaf_dens, selector_coords))
		.map_err(|_| VerificationError::IncorrectIndexEvaluation)?;

	// Reduce the leaf claim on Y and the product claim <T, Y> = e to a single shared evaluation.
	// The product check binds <T, Y> to the gamma-combination of the looker claims.
	let claims = lookers
		.iter()
		.map(|looker| looker.eval_claim.clone())
		.collect::<Vec<_>>();
	let combined_eval_claim = evaluate_univariate(&claims, gamma);
	let Pushforward {
		table_eval_point,
		table_eval_claims,
		pushforward_eval_claims,
	} = verify_pushforward::<F, C>(
		[TableClaim {
			eval_claim: combined_eval_claim,
			pushforward_eval_claim: pushforward_eval,
			pushforward_eval_point: table_point,
		}],
		channel,
	)?;
	// The reduction runs over the one shared table, so it returns one claim of each kind.
	let [table_eval_claim]: [_; 1] = table_eval_claims
		.try_into()
		.unwrap_or_else(|_| unreachable!("the reduction runs over one table"));
	let [pushforward_eval_claim]: [_; 1] = pushforward_eval_claims
		.try_into()
		.unwrap_or_else(|_| unreachable!("the reduction runs over one table"));

	Ok(LogupOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
		// Spans the deepest looker, not the whole node point: when the table is deeper than every
		// looker its extra coordinates belong to the table alone. Looker j reads the last n_j.
		index_eval_point: node_point[n_node_vars - max_n..].to_vec(),
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
		let _ = verify_reduction::<B128, _>(&B128::ZERO, 0, &[claim], &mut verifier);
	}
}
