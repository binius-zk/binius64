// Copyright 2026 The Binius Developers

//! The top-level logUp* proving routine.

use binius_field::{BinaryField1b, ExtensionField, Field, PackedField};
use binius_ip::{MultilinearEvalClaim, logup_star::LogupOutput};
use binius_math::FieldBuffer;

use super::{
	error::Error,
	final_layer::{FinalLayerOutput, prove_final_layer},
	witness,
};
use crate::{
	channel::IPProverChannel,
	fracaddcheck::{FracAddCheckProver, FracEvalClaim},
};

/// Prove a logUp* indexed-lookup reduction.
///
/// This is the prover for [`binius_ip::logup_star::verify`].
/// It produces the transcript the verifier consumes and returns the same reduced claims.
///
/// The reduction proves the indexed lookup `(I^* T)(eval_point) = eval_claim`.
/// It never commits the looked-up vector `I^* T`, which would have `2^n` entries.
/// Instead it commits the pushforward `Y = I_* eq_r`, which has only `2^m` entries.
/// See [Soukhanov25] for the construction.
///
/// [Soukhanov25]: <https://eprint.iacr.org/2025/946>
///
/// # Arguments
///
/// * `table` - The table multilinear `T` over `m` variables (`2^m` entries).
/// * `index` - The index column, one table position per looker row.
///   - Its length defines `n` and must be `2^n`.
///   - Every entry must be less than `2^m`.
/// * `eval_point` - The `n`-coordinate evaluation point `r`.
/// * `eval_claim` - The claimed evaluation `e = (I^* T)(eval_point)`.
/// * `channel` - The prover channel for sending messages and sampling challenges.
///
/// The logUp challenge `c` is sampled against the committed `I`, `T`, and pushforward `Y`.
/// So the caller must absorb those commitments into the transcript before calling this routine.
///
/// # Preconditions
///
/// - The table must have at least one variable, so the table-side GKR has a variable to split on.
/// - `eval_claim` must equal `(I^* T)(eval_point)`, or the proof will not verify.
///
/// # Returns
///
/// The reduced claims on the table, the pushforward, and the index multilinears.
/// The caller verifies those three claims, which is out of scope here.
pub fn prove<F, P>(
	table: &FieldBuffer<P>,
	index: &[usize],
	eval_point: &[F],
	eval_claim: F,
	channel: &mut impl IPProverChannel<F>,
) -> Result<LogupOutput<F>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	P: PackedField<Scalar = F>,
{
	let m = table.log_len();
	let n = eval_point.len();

	// The table-side GKR circuit needs at least one variable to split on.
	assert!(m > 0, "table must have at least one variable");

	// The index column has one entry per looker row, i.e. one per point of the n-variable cube.
	let expected = 1usize << n;
	if index.len() != expected {
		return Err(Error::IndexLengthMismatch {
			got: index.len(),
			expected,
			n_vars: n,
		});
	}
	// Every index must address a real table position for the embedding and pushforward to be valid.
	// This is a precondition: the O(n) scan is compiled out of release builds.
	// An out-of-range index still panics in release, at the pushforward's scatter-add.
	debug_assert!(
		index.iter().all(|&j| j < 1usize << m),
		"every index entry must be less than the table size 2^m"
	);

	// Build the two witnesses that do not depend on the logUp challenge c.
	//
	//     eq_r = eq(eval_point, .)     the looker numerator
	//     Y    = I_* eq_r              the pushforward, scattered onto table positions
	let eq_r = witness::equality_indicator::<F, P>(eval_point);
	let pushforward = witness::pushforward::<F, P>(&eq_r, index, m);

	// The self-contained prover commits nothing.
	// It runs the reduction over the witnesses directly.
	prove_reduction(table, index, eval_claim, eq_r, &pushforward, channel)
}

/// Run the logUp* reduction over the pre-built witnesses `eq_r` and pushforward `Y`.
///
/// This is the reduction core of [`prove`], split out so a caller can build `Y` once and commit it.
/// The committing prover builds `eq_r` and `Y`, commits `Y`, then hands both here.
/// That way the scatter-add that forms `Y` runs only once.
///
/// # Arguments
///
/// * `table` - The table multilinear `T` over `m` variables.
/// * `index` - The index column, one table position per looker row.
/// * `eval_claim` - The claimed evaluation `e = (I^* T)(eval_point)`.
/// * `eq_r` - The looker numerator `eq(eval_point, .)` over `n` variables.
/// * `pushforward` - The pushforward `Y = I_* eq_r` over `m` variables.
/// * `channel` - The prover channel.
///
/// # Preconditions
///
/// - `table.log_len()` is at least 1.
/// - `index.len()` equals `2^{eq_r.log_len()}`, with every entry less than the table size.
/// - `pushforward` equals `I_* eq_r`.
pub fn prove_reduction<F, P>(
	table: &FieldBuffer<P>,
	index: &[usize],
	eval_claim: F,
	eq_r: FieldBuffer<P>,
	pushforward: &FieldBuffer<P>,
	channel: &mut impl IPProverChannel<F>,
) -> Result<LogupOutput<F>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	P: PackedField<Scalar = F>,
{
	let m = table.log_len();
	let n = eq_r.log_len();

	// Sample the logUp challenge c that randomizes the logarithmic-derivative denominators.
	// This is the prover's first transcript action, mirroring the verifier.
	// A committing caller must absorb the I, T, and Y commitments into the transcript before this.
	let c = channel.sample();

	// Build the denominators, which depend on c.
	//
	//     looker side: eq_r(i) / (c - I(i))   over n variables
	//     table  side: Y(j)    / (c - j)       over m variables
	let looker_den = witness::looker_denominator::<F, P>(c, index);
	let table_den = witness::table_denominator::<F, P>(c, m);

	// Build both fractional-addition circuits.
	// Constructing a circuit computes every layer and returns its single root fraction.
	let (looker_prover, looker_root) = FracAddCheckProver::new(n, (eq_r, looker_den));
	let (table_prover, table_root) =
		FracAddCheckProver::new(m, (FieldBuffer::clone(pushforward), table_den));

	// The two root fractions; their equality is the logUp identity the verifier checks.
	//
	//     num_l / den_l = sum_i eq_r(i) / (c - I(i))
	//     num_r / den_r = sum_j Y(j)    / (c - j)
	let num_l = looker_root.0.get(0);
	let den_l = looker_root.1.get(0);
	let num_r = table_root.0.get(0);
	let den_r = table_root.1.get(0);
	channel.send_many(&[num_l, den_l, num_r, den_r]);

	// Looker side: run the full n-layer GKR down to the leaf claim.
	//
	//     leaf numerator   = eq_r(point_l)        (verifier checks this against eq(eval_point, .))
	//     leaf denominator = c - I(point_l)
	let (looker_remaining, (_looker_num_claim, looker_den_claim)) =
		looker_prover.prove_layers(n, root_claim(num_l, den_l), channel)?;
	debug_assert!(
		looker_remaining.is_none_or(|prover| prover.n_layers() == 0),
		"the looker side runs all n layers"
	);

	// The looker denominator is c - I(point_l), so the index claim is I(point_l) = c - den.
	let index_eval_point = looker_den_claim.point;
	let index_eval_claim = c - looker_den_claim.eval;

	// Table side: run the first m-1 GKR layers, stopping at the layer-1 claim over m-1 variables.
	// The leaf layer is left on the prover, to be spliced into the batched final layer.
	let (table_remaining, layer1) =
		table_prover.prove_layers(m - 1, root_claim(num_r, den_r), channel)?;
	let table_leaf_prover = table_remaining.expect("m-1 < m layers leaves the leaf layer");

	// Batched final layer: reduce the layer-1 claims and <T, Y> = e to one shared evaluation point.
	let FinalLayerOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	} = prove_final_layer(eval_claim, table_leaf_prover, layer1, pushforward, table, channel)?;

	Ok(LogupOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
		index_eval_point,
		index_eval_claim,
	})
}

/// The root claim of a fractional-addition circuit, over zero variables.
///
/// The circuit collapses to one fraction `num / den` at its root, evaluated at the empty point.
const fn root_claim<F: Field>(num: F, den: F) -> FracEvalClaim<F> {
	(
		MultilinearEvalClaim {
			eval: num,
			point: Vec::new(),
		},
		MultilinearEvalClaim {
			eval: den,
			point: Vec::new(),
		},
	)
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField1b, ExtensionField, Field,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_ip::logup_star;
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// Embed a table position j into the field through the GF(2)-linear basis, as the protocol does.
	//
	//     iota(j) = sum_{t : bit t of j is set} basis(t)
	fn iota(j: usize, m: usize) -> F {
		(0..m)
			.filter(|t| (j >> t) & 1 == 1)
			.map(<F as ExtensionField<BinaryField1b>>::basis)
			.fold(F::ZERO, |acc, b| acc + b)
	}

	// Build a random instance and return (table, index, eval_point, eq_r scalars, true eval claim).
	fn random_instance(
		rng: &mut StdRng,
		n: usize,
		m: usize,
	) -> (FieldBuffer<P>, Vec<usize>, Vec<F>, Vec<F>, F) {
		let table = random_field_buffer::<P>(&mut *rng, m);
		let index = (0..(1usize << n))
			.map(|_| rng.random_range(0..(1usize << m)))
			.collect::<Vec<_>>();
		let eval_point = random_scalars::<F>(&mut *rng, n);

		// The looked-up evaluation: e = (I^* T)(r) = sum_i eq_r(i) * T[index[i]].
		let eq_r = eq_ind_partial_eval_scalars::<F>(&eval_point);
		let eval_claim = index
			.iter()
			.zip(&eq_r)
			.map(|(&j, &eq)| eq * table.get(j))
			.fold(F::ZERO, |acc, t| acc + t);

		(table, index, eval_point, eq_r, eval_claim)
	}

	fn check_prove_verify(n: usize, m: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let (table, index, eval_point, eq_r, eval_claim) = random_instance(&mut rng, n, m);

		// Prove, then replay the transcript through the verifier.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_out =
			prove::<F, P>(&table, &index, &eval_point, eval_claim, &mut prover_transcript)
				.expect("proving succeeds");

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_out =
			logup_star::verify::<F, _>(m, eval_claim, &eval_point, &mut verifier_transcript)
				.expect("verification succeeds");

		// The prover and verifier must derive identical reduced claims from the same transcript.
		assert_eq!(prover_out, verifier_out, "outputs disagree (n={n}, m={m})");

		// The reduced table claim must be the honest evaluation of T at the reduced point.
		assert_eq!(
			prover_out.table_eval_claim,
			evaluate(&table, &prover_out.table_eval_point),
			"table claim wrong (n={n}, m={m})"
		);

		// The pushforward claim must be the honest evaluation of Y = I_* eq_r at the same point.
		let mut pushforward = vec![F::ZERO; 1usize << m];
		for (&j, &eq) in index.iter().zip(&eq_r) {
			pushforward[j] += eq;
		}
		let pushforward = FieldBuffer::<P>::from_values(&pushforward);
		assert_eq!(
			prover_out.pushforward_eval_claim,
			evaluate(&pushforward, &prover_out.table_eval_point),
			"pushforward claim wrong (n={n}, m={m})"
		);

		// The index claim must be the honest evaluation of the embedded index column.
		let index_embedded = index.iter().map(|&j| iota(j, m)).collect::<Vec<_>>();
		let index_embedded = FieldBuffer::<P>::from_values(&index_embedded);
		assert_eq!(
			prover_out.index_eval_claim,
			evaluate(&index_embedded, &prover_out.index_eval_point),
			"index claim wrong (n={n}, m={m})"
		);
	}

	#[test]
	fn test_prove_verify_round_trip() {
		// A spread of shapes: m << n (the target regime), m == n, and a wide table.
		for (n, m) in [(6, 2), (5, 3), (4, 4), (3, 5), (7, 1)] {
			check_prove_verify(n, m, 0);
		}
	}

	#[test]
	fn test_prove_verify_single_table_variable() {
		// m = 1 exercises the batched final layer with an empty layer-1 point.
		check_prove_verify(4, 1, 1);
	}

	#[test]
	fn test_prove_verify_single_looker_row() {
		// n = 0 exercises the looker side with no GKR layers: the root is already the leaf claim.
		check_prove_verify(0, 3, 2);
	}

	#[test]
	fn test_verifier_rejects_wrong_eval_claim() {
		let mut rng = StdRng::seed_from_u64(3);
		let (table, index, eval_point, _eq_r, eval_claim) = random_instance(&mut rng, 5, 3);

		// Prove a false statement by perturbing the looked-up evaluation.
		let wrong_claim = eval_claim + F::ONE;
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prove::<F, P>(&table, &index, &eval_point, wrong_claim, &mut prover_transcript)
			.expect("proving a false claim still produces a transcript");

		// The product-check inconsistency must surface as a verification failure.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let result = logup_star::verify::<F, _>(
			table.log_len(),
			wrong_claim,
			&eval_point,
			&mut verifier_transcript,
		);
		assert!(result.is_err(), "verifier must reject a wrong eval claim");
	}

	#[test]
	#[should_panic(expected = "table must have at least one variable")]
	fn test_zero_variable_table_panics() {
		let mut rng = StdRng::seed_from_u64(0);

		// A zero-variable table has a single entry and no variable for the GKR to split on.
		// The precondition assertion must fire before any transcript interaction.
		let table = random_field_buffer::<P>(&mut rng, 0);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let _ = prove::<F, P>(&table, &[0], &[], F::ZERO, &mut transcript);
	}

	#[test]
	fn test_rejects_index_length_mismatch() {
		let mut rng = StdRng::seed_from_u64(0);
		let table = random_field_buffer::<P>(&mut rng, 3);
		let eval_point = random_scalars::<F>(&mut rng, 4);
		let mut transcript = ProverTranscript::new(StdChallenger::default());

		// eval_point has 4 coordinates, so the index column must have 2^4 = 16 entries, not 3.
		let err = prove::<F, P>(&table, &[0, 1, 2], &eval_point, F::ZERO, &mut transcript)
			.expect_err("a short index column is rejected");
		assert!(matches!(
			err,
			Error::IndexLengthMismatch {
				got: 3,
				expected: 16,
				n_vars: 4
			}
		));
	}

	#[test]
	#[should_panic(expected = "every index entry must be less than the table size")]
	fn test_out_of_range_index_panics() {
		let mut rng = StdRng::seed_from_u64(0);
		let table = random_field_buffer::<P>(&mut rng, 2);
		let eval_point = random_scalars::<F>(&mut rng, 1);
		let mut transcript = ProverTranscript::new(StdChallenger::default());

		// The table has 2^2 = 4 positions, so index value 4 is out of range.
		// The range check is a debug_assert precondition, so this panics in debug builds.
		let _ = prove::<F, P>(&table, &[0, 4], &eval_point, F::ZERO, &mut transcript);
	}
}
