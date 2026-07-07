// Copyright 2026 The Binius Developers

//! The top-level logUp* proving routine.

use binius_field::{BinaryField, Divisible, Field, PackedField};
use binius_ip::{MultilinearEvalClaim, logup_star::LogupOutput};
use binius_math::{FieldBuffer, univariate::evaluate_univariate};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use super::{
	final_layer::{FinalLayerOutput, prove_final_layer},
	witness,
};
use crate::{
	channel::IPProverChannel,
	fracaddcheck::{self, FracAddCheckProver, FracEvalClaim},
};

/// Prove a logUp* indexed-lookup reduction.
///
/// This is the prover for [`binius_ip::logup_star::verify_reduction`].
/// It produces the transcript the verifier consumes and returns the same reduced claims.
///
/// The reduction proves the indexed lookups `(I_j^* T)(r_j) = e_j` for one or more lookers
/// sharing the table. The lookers batch by a random linear combination: a challenge `gamma`
/// scales looker `j`'s equality-indicator numerator by `gamma^j`, and the pushforward is the
/// gamma-weighted sum of the per-looker pushforwards, still with only `2^m` entries. The
/// looked-up vectors are never committed.
/// See [Soukhanov25] for the construction.
///
/// [Soukhanov25]: <https://eprint.iacr.org/2025/946>
///
/// # Arguments
///
/// * `table` - The table multilinear `T` over `m` variables (`2^m` entries).
/// * `lookers` - The looker columns and claims; every evaluation point must have the same length
///   `n`, every index column must have `2^n` entries, and every index entry must be less than
///   `2^m`.
/// * `channel` - The prover channel for sending messages and sampling challenges.
///
/// The logUp challenge `c` is sampled against the committed `I_j`, `T`, and pushforward `Y`.
/// So the caller must absorb those commitments into the transcript before calling this routine.
///
/// # Preconditions
///
/// - The table must have at least one variable, so the table-side GKR has a variable to split on.
/// - Every `eval_claim` must equal `(I_j^* T)(r_j)`, or the proof will not verify.
///
/// # Returns
///
/// The reduced claims on the table, the pushforward, and the per-looker index multilinears, all
/// index claims sharing one evaluation point.
/// The caller verifies those claims, which is out of scope here.
/// One looker's column and claim: `(I^* T)(eval_point) = eval_claim` against the shared table.
pub struct Looker<'a, F> {
	/// The index column, one table position per looker row (`2^n` entries).
	pub index: &'a [usize],
	/// The `n`-coordinate evaluation point of this looker's claim.
	pub eval_point: &'a [F],
	/// The claimed evaluation of this looker's looked-up vector at the point.
	pub eval_claim: F,
}

pub fn prove<F, P>(
	table: &FieldBuffer<P>,
	lookers: &[Looker<'_, F>],
	channel: &mut impl IPProverChannel<F>,
) -> LogupOutput<F>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
{
	let m = table.log_len();

	// The table-side GKR circuit needs at least one variable to split on.
	assert!(m > 0, "table must have at least one variable");

	// Every index must address a real table position for the embedding and pushforward to be valid.
	// This is a precondition: the O(n) scan is compiled out of release builds.
	// An out-of-range index still panics in release, at the pushforward's scatter-add.
	debug_assert!(
		lookers
			.iter()
			.all(|looker| looker.index.iter().all(|&j| j < 1usize << m)),
		"every index entry must be less than the table size 2^m"
	);

	// Sample the batching challenge gamma that combines the looker claims, then build the two
	// witnesses that do not depend on the logUp challenge c.
	//
	//     gamma^j * eq_{r_j} = the per-looker scaled numerators
	//     Y = sum_j gamma^j * (I_j)_* eq_{r_j}     the combined pushforward
	let gamma = channel.sample();
	let (numerators, pushforward) = witness::combined_lookers::<F, P>(lookers, gamma, m);

	// The product check binds <T, Y> to the gamma-combination of the looker claims.
	let claims = lookers
		.iter()
		.map(|looker| looker.eval_claim)
		.collect::<Vec<_>>();
	let combined_eval_claim = evaluate_univariate(&claims, gamma);

	// The self-contained prover commits nothing.
	// It runs the reduction over the witnesses directly.
	prove_reduction(table, lookers, combined_eval_claim, numerators, &pushforward, channel)
}

/// Run the logUp* reduction over the pre-built witnesses `numerators` and pushforward `Y`.
///
/// This is the reduction core of [`prove`], split out so a caller can build `Y` once and commit it.
/// The committing prover builds the numerators and `Y`, commits `Y`, then hands both here.
/// That way the scatter-add that forms `Y` runs only once.
///
/// # Arguments
///
/// * `table` - The table multilinear `T` over `m` variables.
/// * `lookers` - The looker columns and claims (the claims are unused here; the caller combines
///   them into `eval_claim`).
/// * `eval_claim` - The gamma-combined claimed evaluation.
/// * `numerators` - The per-looker gamma-scaled numerators `gamma^j * eq_{r_j}` (see
///   [`witness::combined_lookers`]).
/// * `pushforward` - The combined pushforward `Y = sum_j gamma^j * (I_j)_* eq_{r_j}`, the scatter
///   of the numerators.
/// * `channel` - The prover channel.
///
/// # Preconditions
///
/// - `table.log_len()` is at least 1.
/// - `numerators` has one `n`-variable buffer per looker; every index column has `2^n` entries,
///   with every entry less than the table size.
/// - `pushforward` equals the scatter of the numerators.
#[tracing::instrument(
	skip_all,
	level = "debug",
	name = "logup* reduction",
	fields(n_lookers = lookers.len(), table_n_vars = table.log_len())
)]
pub fn prove_reduction<F, P>(
	table: &FieldBuffer<P>,
	lookers: &[Looker<'_, F>],
	eval_claim: F,
	numerators: Vec<FieldBuffer<P>>,
	pushforward: &FieldBuffer<P>,
	channel: &mut impl IPProverChannel<F>,
) -> LogupOutput<F>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
{
	let m = table.log_len();
	let n = lookers[0].eval_point.len();
	let log_lookers = log2_ceil_usize(lookers.len());

	// Sample the logUp challenge c that randomizes the logarithmic-derivative denominators.
	// This is the prover's first transcript action, mirroring the verifier.
	// A committing caller must absorb the I, T, and Y commitments into the transcript before this.
	let c = channel.sample();

	// Build the fractional-addition circuits, one per looker plus the table side.
	// Constructing a circuit computes every layer and returns its single root fraction.
	//
	//     looker j: gamma^j * eq_{r_j}(i) / (c - I_j(i))   over n variables
	//     table:    Y(v)                  / (c - v)         over m variables
	let circuits_guard = tracing::debug_span!("Build fracadd circuits").entered();
	let (looker_provers, looker_roots): (Vec<_>, Vec<_>) = std::iter::zip(lookers, numerators)
		.map(|(looker, numerator)| {
			let den = witness::looker_denominator::<F, P>(c, looker.index);
			let (prover, root) = FracAddCheckProver::new(n, (numerator, den));
			(prover, (root.0.get(0), root.1.get(0)))
		})
		.unzip();
	let table_den = witness::table_denominator::<F, P>(c, m);
	let (table_prover, table_root) =
		FracAddCheckProver::new(m, (FieldBuffer::clone(pushforward), table_den));
	let num_r = table_root.0.get(0);
	let den_r = table_root.1.get(0);

	// Top circuit: interpolate the per-looker root fractions into a multilinear pair over the
	// looker variables, padded with the zero fraction (numerators with 0, denominators with 1).
	// Its root is the fractional sum of every looker circuit, so the looker side runs as one
	// GKR circuit over n + log_lookers variables.
	let mut root_nums = looker_roots
		.iter()
		.map(|&(num_j, _)| num_j)
		.collect::<Vec<_>>();
	let mut root_dens = looker_roots
		.iter()
		.map(|&(_, den_j)| den_j)
		.collect::<Vec<_>>();
	root_nums.resize(1 << log_lookers, F::ZERO);
	root_dens.resize(1 << log_lookers, F::ONE);
	let (top_prover, top_root) = FracAddCheckProver::new(
		log_lookers,
		(FieldBuffer::<P>::from_values(&root_nums), FieldBuffer::<P>::from_values(&root_dens)),
	);
	let num_l = top_root.0.get(0);
	let den_l = top_root.1.get(0);
	drop(circuits_guard);

	// The two root fractions; their equality is the logUp identity the verifier checks.
	//
	//     num_l / den_l = sum_j gamma^j sum_i eq_{r_j}(i) / (c - I_j(i))
	//     num_r / den_r = sum_v Y(v) / (c - v)
	channel.send_many(&[num_l, den_l, num_r, den_r]);

	let looker_gkr_guard = tracing::debug_span!("Looker-side GKR").entered();
	// Looker side, first phase: run the top circuit over the looker variables to completion,
	// reducing its root to a claim on the interpolated root fractions at a selector point.
	let (top_remaining, (top_num_claim, _top_den_claim)) =
		top_prover.prove_layers(log_lookers, root_claim(num_l, den_l), channel);
	debug_assert!(
		top_remaining.is_none_or(|prover| prover.n_layers() == 0),
		"the top circuit runs all log_lookers layers"
	);
	let selector_point = top_num_claim.point;

	// Looker side, second phase: continue with the batched GKR over the per-looker circuits down
	// to the leaf claims, seeded at the selector point. With no looker variables there are no
	// layers to run: the roots are already the leaf fractions.
	let batch_output = if n == 0 {
		fracaddcheck::BatchProveOutput {
			eval_point: selector_point,
			fractions: looker_roots,
		}
	} else {
		fracaddcheck::batch_prove(looker_provers, looker_roots, selector_point, Vec::new(), channel)
	};
	drop(looker_gkr_guard);

	// The per-looker leaf denominators are c - I_j(content), so the index claims are their
	// c-complements. Send them so the verifier can check they combine to the batched leaf.
	let index_eval_point = batch_output.eval_point[log_lookers..].to_vec();
	let index_eval_claims = batch_output
		.fractions
		.iter()
		.map(|&(_num_leaf, den_leaf)| c - den_leaf)
		.collect::<Vec<_>>();
	channel.send_many(&index_eval_claims);

	// Table side: run the first m-1 GKR layers, stopping at the layer-1 claim over m-1 variables.
	// The leaf layer is left on the prover, to be spliced into the batched final layer.
	let table_gkr_guard = tracing::debug_span!("Table-side GKR layers").entered();
	let (table_remaining, layer1) =
		table_prover.prove_layers(m - 1, root_claim(num_r, den_r), channel);
	let table_leaf_prover = table_remaining.expect("m-1 < m layers leaves the leaf layer");
	drop(table_gkr_guard);

	// Batched final layer: reduce the layer-1 claims and <T, Y> = e to one shared evaluation point.
	let FinalLayerOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	} = prove_final_layer(eval_claim, table_leaf_prover, layer1, pushforward, table, channel);

	LogupOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
		index_eval_point,
		index_eval_claims,
	}
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
	use binius_ip::{channel::IPVerifierChannel, logup_star};
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
		let looker = Looker {
			index: &index,
			eval_point: &eval_point,
			eval_claim,
		};
		let prover_out = prove::<F, P>(&table, &[looker], &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let looker_claim = logup_star::LookerClaim {
			eval_point: &eval_point,
			eval_claim,
		};
		let gamma = IPVerifierChannel::<F>::sample(&mut verifier_transcript);
		let verifier_out = logup_star::verify_reduction::<F, _>(
			gamma,
			m,
			&[looker_claim],
			&mut verifier_transcript,
		)
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
			prover_out.index_eval_claims,
			vec![evaluate(&index_embedded, &prover_out.index_eval_point)],
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
		let looker = Looker {
			index: &index,
			eval_point: &eval_point,
			eval_claim: wrong_claim,
		};
		prove::<F, P>(&table, &[looker], &mut prover_transcript);

		// The product-check inconsistency must surface as a verification failure.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let looker_claim = logup_star::LookerClaim {
			eval_point: &eval_point,
			eval_claim: wrong_claim,
		};
		let gamma = IPVerifierChannel::<F>::sample(&mut verifier_transcript);
		let result = logup_star::verify_reduction::<F, _>(
			gamma,
			table.log_len(),
			&[looker_claim],
			&mut verifier_transcript,
		);
		assert!(result.is_err(), "verifier must reject a wrong eval claim");
	}

	#[test]
	fn test_multi_looker_round_trip() {
		let mut rng = StdRng::seed_from_u64(11);
		let (n, m) = (5, 3);
		let n_lookers = 3usize;

		let instances = (0..n_lookers)
			.map(|_| random_instance(&mut rng, n, m))
			.collect::<Vec<_>>();
		// All lookers share the first instance's table.
		let table = instances[0].0.clone();
		let lookers = instances
			.iter()
			.map(|(_, index, eval_point, eq_r, _)| {
				let eval_claim = index
					.iter()
					.zip(eq_r)
					.map(|(&j, &eq)| eq * table.get(j))
					.fold(F::ZERO, |acc, t| acc + t);
				(index, eval_point, eval_claim)
			})
			.collect::<Vec<_>>();

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_lookers = lookers
			.iter()
			.map(|(index, eval_point, eval_claim)| Looker {
				index,
				eval_point,
				eval_claim: *eval_claim,
			})
			.collect::<Vec<_>>();
		let prover_out = prove::<F, P>(&table, &prover_lookers, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let looker_claims = lookers
			.iter()
			.map(|(_, eval_point, eval_claim)| logup_star::LookerClaim {
				eval_point,
				eval_claim: *eval_claim,
			})
			.collect::<Vec<_>>();
		let gamma = IPVerifierChannel::<F>::sample(&mut verifier_transcript);
		let verifier_out = logup_star::verify_reduction::<F, _>(
			gamma,
			m,
			&looker_claims,
			&mut verifier_transcript,
		)
		.expect("verification succeeds");
		assert_eq!(prover_out, verifier_out);

		// The reduced table claim is the honest table evaluation.
		assert_eq!(prover_out.table_eval_claim, evaluate(&table, &prover_out.table_eval_point));

		// Every index claim is the honest evaluation of its looker's embedded index column at the
		// shared index point.
		for ((index, _, _), claim) in lookers.iter().zip(&prover_out.index_eval_claims) {
			let embedded = index.iter().map(|&j| iota(j, m)).collect::<Vec<_>>();
			let embedded = FieldBuffer::<P>::from_values(&embedded);
			assert_eq!(*claim, evaluate(&embedded, &prover_out.index_eval_point));
		}
	}

	#[test]
	#[should_panic(expected = "table must have at least one variable")]
	fn test_zero_variable_table_panics() {
		let mut rng = StdRng::seed_from_u64(0);

		// A zero-variable table has a single entry and no variable for the GKR to split on.
		// The precondition assertion must fire before any transcript interaction.
		let table = random_field_buffer::<P>(&mut rng, 0);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let looker = Looker {
			index: &[0],
			eval_point: &[],
			eval_claim: F::ZERO,
		};
		let _ = prove::<F, P>(&table, &[looker], &mut transcript);
	}

	#[test]
	#[should_panic(expected = "index column has 3 entries but 16 were expected for 4 variables")]
	fn test_rejects_index_length_mismatch() {
		let mut rng = StdRng::seed_from_u64(0);
		let table = random_field_buffer::<P>(&mut rng, 3);
		let eval_point = random_scalars::<F>(&mut rng, 4);
		let mut transcript = ProverTranscript::new(StdChallenger::default());

		// eval_point has 4 coordinates, so the index column must have 2^4 = 16 entries, not 3.
		let looker = Looker {
			index: &[0, 1, 2],
			eval_point: &eval_point,
			eval_claim: F::ZERO,
		};
		let _ = prove::<F, P>(&table, &[looker], &mut transcript);
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
		let looker = Looker {
			index: &[0, 4],
			eval_point: &eval_point,
			eval_claim: F::ZERO,
		};
		let _ = prove::<F, P>(&table, &[looker], &mut transcript);
	}
}
