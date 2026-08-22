// Copyright 2026 The Binius Developers

//! The top-level logUp* verification routine.

use std::{iter, slice};

use binius_field::{BinaryField1b, ExtensionField, Field, field::FieldOps};
use binius_math::{
	multilinear::{
		eq::{eq_ind, eq_ind_zero},
		evaluate::evaluate_inplace_scalars,
	},
	univariate::evaluate_univariate,
};
use itertools::izip;

use super::{
	error::{Error, VerificationError},
	output::{LogupOutput, LogupTableOutput},
	pushforward::{Pushforward, TableClaim, denominator_eval, verify_pushforward},
};
use crate::{
	channel::IPVerifierChannel,
	fracaddcheck::{self, FracAddEvalClaim},
};

/// One looker's claim on its looked-up vector: `(I^* T)(eval_point) = eval_claim`.
#[derive(Debug, Clone)]
pub struct LookerClaim<'a, Elem> {
	/// The `n`-coordinate evaluation point of this looker's claim.
	pub eval_point: &'a [Elem],
	/// The claimed evaluation of this looker's looked-up vector at the point.
	pub eval_claim: Elem,
}

/// One table together with the lookers that read it.
#[derive(Debug, Clone)]
pub struct TableLookup<'a, Elem> {
	/// The number of variables `m` of this table's multilinear (`2^m` entries).
	pub n_vars: usize,
	/// The claims of the lookers that read this table. Their evaluation points may differ in
	/// length, both from each other and from the table.
	pub lookers: Vec<LookerClaim<'a, Elem>>,
}

/// Verify a logUp* indexed-lookup reduction over one or more tables, each with its own lookers.
///
/// Reduces the claims `(I^* T)(r) = e` to the claims in [`LogupOutput`]. Each table batches its own
/// lookers by a random linear combination: its challenge `gamma` scales its looker `i`'s numerator
/// by `gamma^i`. A table's pushforward `Y` is the gamma-weighted sum of its lookers' pushforwards,
/// and its product check binds `<T, Y>` to the gamma-combination of its lookers' claims. Tables
/// share nothing but the batching machinery.
///
/// Every fractional-addition circuit — one per looker over `n` variables, plus one per table over
/// `m` — is an instance of **one** GKR of `k + max(max n, max m)` layers, where
/// `k = ceil(log2(#lookers + #tables))`. Its top `k` layers add the per-instance root fractions
/// together and its lower layers run the instances in a batch, with every shallower instance padded
/// by zero fractions — that leaves a fractional sum unchanged and costs `O(1)` per round, so the
/// layer count depends only on the deepest instance.
///
/// No column need agree with any other on a length: an instance over `d` variables is simply padded
/// by `max(max n, max m) - d`. A looker's reduced index claim lands on the **last `n`** coordinates
/// of [`LogupOutput::index_eval_point`].
///
/// Every table's fraction enters that sum negated, so the circuit's root is
/// `sum_lookers num/den - sum_tables num/den`, whose numerator vanishes exactly when the lookup
/// identities hold. The verifier therefore reads only the root *denominator* and supplies the zero
/// numerator itself: the identities are enforced by the shape of the claim rather than by a
/// separate check.
///
/// # Why one logUp challenge per table
///
/// Each table is randomized by its own `c`, so table `t`'s contribution to the root fraction,
///
/// ```text
///     f_t(c_t) = sum_i gamma_t^i sum_x eq_{r_i}(x)/(c_t - I_i(x))  -  sum_v Y_t(v)/(c_t - v)
/// ```
///
/// is a rational function of `c_t` alone, vanishing as `c_t` grows. A sum of such functions in
/// disjoint variables is identically zero only when every term is, so the one root check certifies
/// every table separately. Under a single shared challenge that argument fails: two tables could
/// miscount the same position in opposite directions and cancel inside the root numerator.
///
/// The logUp challenges are sampled against the committed `I`, `T`, and pushforwards `Y`.
/// So the caller must absorb those commitments into the transcript before calling this routine, and
/// must have sampled each table's `gamma` before its pushforward commitment.
///
/// # Arguments
///
/// * `tables` - One [`TableLookup`] per table, each carrying its batching challenge, its variable
///   count, and its lookers' claims.
/// * `channel` - The verifier channel for receiving prover messages and sampling challenges.
///
/// # Transcript layout
///
/// The prover messages are consumed in this exact order:
///
/// ```text
///     1. sample c                         (one logUp challenge per table)
///     2. recv den_root                    (the root denominator; its numerator is 0)
///     3. combined GKR, k + max(max n, max m) layers (see fracaddcheck::verify)
///     4. recv per-looker index evaluations, then per-table Y (the non-transparent leaf halves)
///     5. pushforward reduction:
///        a. sample batch_coeff
///        b. max m rounds of degree-2 sumcheck
///        c. recv per-table [Y, T]         (evaluations at the challenge point)
/// ```
///
/// The per-looker index evaluations arrive table by table, in the order the tables are given, and
/// within a table in its own looker order.
///
/// The sumchecks are assumed to bind variables from the highest index to the lowest.
/// This matches the convention of the fractional-addition GKR layers.
///
/// # Preconditions
///
/// - `tables` is non-empty, every table has at least one variable so its GKR has a variable to
///   split on, and every table has at least one looker.
///
/// # Returns
///
/// The reduced [`LogupOutput`] claims on the tables, pushforwards, and index multilinears.
///
/// # Errors
///
/// Returns an error when the proof is malformed or any verification identity fails:
///
/// - a GKR layer's reduction is inconsistent, which is where a violated lookup identity surfaces,
/// - the transparent leaf numerators do not interpolate to the batch's leaf numerator,
/// - the index and table denominators do not interpolate to the batch's leaf denominator,
/// - the pushforward reduction is inconsistent.
pub fn verify_reduction<'a, F, C>(
	gamma: &C::Elem,
	tables: impl IntoIterator<Item = TableLookup<'a, C::Elem>>,
	channel: &mut C,
) -> Result<LogupOutput<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IPVerifierChannel<F>,
	C::Elem: From<F> + 'a,
{
	let tables = tables.into_iter().collect::<Vec<_>>();
	assert!(!tables.is_empty(), "at least one table is required");
	// Each table-side GKR circuit needs at least one variable to split on.
	assert!(
		tables.iter().all(|table| table.n_vars > 0),
		"every table must have at least one variable"
	);
	assert!(
		tables.iter().all(|table| !table.lookers.is_empty()),
		"every table must have at least one looker"
	);

	let n_tables = tables.len();
	let n_lookers = tables
		.iter()
		.map(|table| table.lookers.len())
		.sum::<usize>();
	// No column need agree with any other on a length; the batch pads each up to the deepest
	// instance.
	let max_n = tables
		.iter()
		.flat_map(|table| &table.lookers)
		.map(|looker| looker.eval_point.len())
		.max()
		.expect("every table has at least one looker");
	let max_m = tables
		.iter()
		.map(|table| table.n_vars)
		.max()
		.expect("tables is non-empty");

	// Within a table, looker `i` is weighted by gamma^i. The same series serves every table: the
	// combination only has to bind the lookers inside one table, because the per-table denominator
	// challenges already separate the tables from each other. So only as many powers are needed as
	// the largest table has lookers.
	let max_table_lookers = tables
		.iter()
		.map(|table| table.lookers.len())
		.max()
		.expect("tables is non-empty");
	let looker_powers = gamma.powers().take(max_table_lookers).collect::<Vec<_>>();

	// Sample one logUp challenge per table. Distinct challenges are what make the single root check
	// certify every table separately: a table's contribution to the root fraction is a rational
	// function of its own c alone, so a sum of them vanishes only when each does. Under one shared
	// challenge two tables' errors could cancel inside the root numerator.
	let cs = channel.sample_many(n_tables);

	// Read the root denominator. The root numerator is not on the transcript: the whole circuit
	// sums the looker fractions against the negated table fractions, so its value is zero exactly
	// when every lookup identity holds, and the verifier supplies that zero itself.
	let root_den: C::Elem = channel
		.recv_one()
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// One GKR over the whole thing, from that single root fraction down to the leaves: k layers
	// interpolating the per-instance roots, then max(max_n, max_m) more over the instances. Looker
	// j's tree has depth n_j and table t's tree depth m_t, so every shallower instance is padded by
	// zero fractions — the layer count reveals only the deepest one.
	let n_instances = n_lookers + n_tables;
	let k = n_instances.next_power_of_two().ilog2() as usize;
	let n_layers = k + max_n.max(max_m);
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

	// Read the claims the verifier cannot derive: the per-looker index evaluations and the
	// per-table pushforward evaluations.
	let index_evals: Vec<C::Elem> = channel
		.recv_many(n_lookers)
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;
	let pushforward_evals: Vec<C::Elem> = channel
		.recv_many(n_tables)
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// Rebuild each circuit's padded leaf fraction and check they interpolate to the batch's leaf.
	//
	// The node point spans max(max_n, max_m) coordinates, so an instance over `d` variables is
	// padded by `max(max_n, max_m) - d` and its own content is the last `d` coordinates. Padding
	// scales a numerator by the padding coordinates' equality weight q and sends a denominator
	// through sel(q, .), so both halves follow from the claims above.
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

	// Each table's own content is the last m_t coordinates of the node point.
	let table_points = tables
		.iter()
		.map(|table| &node_point[n_node_vars - table.n_vars..])
		.collect::<Vec<_>>();

	// Looker numerators are transparent: its table's gamma^i scales the equality indicator at r_i.
	// The denominators are that table's c minus the index evaluation just read. The evaluations
	// arrive table by table, so they are split back into per-table groups here.
	let mut index_eval_claims = Vec::with_capacity(n_tables);
	let mut remaining_index_evals = index_evals.as_slice();
	let (mut leaf_nums, mut leaf_dens): (Vec<_>, Vec<_>) = izip!(&tables, &cs)
		.flat_map(|(table, c)| {
			let (table_evals, rest) = remaining_index_evals.split_at(table.lookers.len());
			remaining_index_evals = rest;
			index_eval_claims.push(table_evals.to_vec());
			izip!(&table.lookers, &looker_powers, table_evals).map(|(looker, power, index_eval)| {
				let pad = n_node_vars - looker.eval_point.len();
				let content = &node_point[pad..];
				let num = power.clone() * eq_ind::<C::Elem>(looker.eval_point, content);
				let den = c.clone() - index_eval.clone();
				fracaddcheck::pad_leaf_fraction((num, den), pad_eqs[pad].clone())
			})
		})
		.unzip();

	// A table's numerator is its Y_t, just read. Its denominator is the transparent J - c_t, the
	// logUp denominator negated — every table's fraction enters the sum that way, which is what
	// makes the root numerator vanish.
	for ((c, point), pushforward_eval) in
		iter::zip(iter::zip(&cs, &table_points), &pushforward_evals)
	{
		let pad = n_node_vars - point.len();
		let (num, den) = fracaddcheck::pad_leaf_fraction(
			(pushforward_eval.clone(), denominator_eval::<F, C::Elem>(c, point)),
			pad_eqs[pad].clone(),
		);
		leaf_nums.push(num);
		leaf_dens.push(den);
	}

	leaf_nums.resize(1 << k, C::Elem::zero());
	leaf_dens.resize(1 << k, C::Elem::one());
	channel
		.assert_zero(leaf_num - evaluate_inplace_scalars(leaf_nums, selector_coords))
		.map_err(|_| VerificationError::IncorrectXEvaluation)?;
	channel
		.assert_zero(leaf_den - evaluate_inplace_scalars(leaf_dens, selector_coords))
		.map_err(|_| VerificationError::IncorrectIndexEvaluation)?;

	// Reduce every table's leaf claim on Y_t and its product claim <T_t, Y_t> = e_t to one shared
	// evaluation point. A table's product check binds it to the gamma-combination of the claims of
	// the lookers that read it.
	let table_claims = izip!(&tables, pushforward_evals, &table_points).map(
		|(table, pushforward_eval_claim, &point)| {
			// Its lookers are weighted gamma^0, gamma^1, ..., so the combination is the univariate
			// evaluation of their claims at gamma.
			let claims = table
				.lookers
				.iter()
				.map(|looker| looker.eval_claim.clone())
				.collect::<Vec<_>>();
			let eval_claim = evaluate_univariate(&claims, gamma);
			TableClaim {
				eval_claim,
				pushforward_eval_claim,
				pushforward_eval_point: point,
			}
		},
	);
	let Pushforward {
		table_eval_point,
		table_eval_claims,
		pushforward_eval_claims,
	} = verify_pushforward::<F, C>(table_claims, channel)?;

	Ok(LogupOutput {
		table_eval_point,
		// Spans the deepest looker, not the whole node point: when a table is deeper than every
		// looker its extra coordinates belong to the tables alone. A looker reads the last n.
		index_eval_point: node_point[n_node_vars - max_n..].to_vec(),
		tables: izip!(table_eval_claims, pushforward_eval_claims, index_eval_claims)
			.map(|(eval_claim, pushforward_claim, index_eval_claims)| LogupTableOutput {
				eval_claim,
				pushforward_claim,
				index_eval_claims,
			})
			.collect(),
	})
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, arch::OptimalB128 as B128};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	use super::*;

	type StdChallenger = HasherChallenger<sha2::Sha256>;

	#[test]
	#[should_panic(expected = "every table must have at least one variable")]
	fn test_empty_table_panics() {
		// A zero-variable table has no variable for the GKR circuit to split on.
		let transcript = ProverTranscript::new(StdChallenger::default());
		let mut verifier = transcript.into_verifier();

		// The precondition assertion fires before any transcript interaction.
		let _ = verify_reduction::<B128, _>(
			&B128::ZERO,
			[TableLookup {
				n_vars: 0,
				lookers: vec![LookerClaim {
					eval_point: &[],
					eval_claim: B128::ZERO,
				}],
			}],
			&mut verifier,
		);
	}

	#[test]
	#[should_panic(expected = "every table must have at least one looker")]
	fn test_table_without_lookers_panics() {
		// A table nothing reads has no claim to prove, so it does not belong in the batch.
		let transcript = ProverTranscript::new(StdChallenger::default());
		let mut verifier = transcript.into_verifier();

		let _ = verify_reduction::<B128, _>(
			&B128::ZERO,
			[TableLookup {
				n_vars: 3,
				lookers: Vec::new(),
			}],
			&mut verifier,
		);
	}
}
