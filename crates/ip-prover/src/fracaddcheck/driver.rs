// Copyright 2025-2026 The Binius Developers

//! The batched layer schedule: one uniform round dance over every tree in the batch.

use std::iter;

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::{mlecheck, sumcheck::RoundCoeffs};
use binius_math::{
	FieldBuffer, FieldVec, line::extrapolate_line, multilinear::hypercube::Hypercube,
};
use binius_utils::rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use itertools::izip;

use super::{FracAddCircuit, fraction::Fraction, padding::layer_provers};
use crate::{
	channel::IPProverChannel,
	sumcheck::{
		common::MleCheckProver,
		frac_add_mle,
		mle_store::MleStore,
		round_evaluator::{MleCheckRoundEvaluator, SharedMleCheckProver},
	},
};

/// Output of [`batch_prove_unequal_depths`].
///
/// After the full `n_layers` reduction, `fractions` holds each input tree's reduced fraction at
/// `eval_point`. The batched claim the verifier checks is the eq(selector)-weighted combination of
/// these fractions.
pub struct BatchProveOutput<F> {
	/// The reduced evaluation point (`selector ++ content`) at which the fractions are claimed.
	pub eval_point: Vec<F>,
	/// Each input prover's reduced `(num, den)` fraction at `eval_point`, in input order.
	pub fractions: Vec<Fraction<F>>,
}

/// Runs one batched fracaddcheck layer given its per-instance final-layer MLE-check provers.
///
/// The layer runs in four steps, one function each:
/// - [`prove_content_rounds`] folds the content variables of every instance in lockstep.
/// - [`finish_and_transpose`] turns the reduced halves into the four selector columns.
/// - [`prove_selector_rounds`] folds the `k` selector variables in one MLE-check.
/// - [`finalize_layer`] line-folds the merged evaluations into the next layer's claims.
///
/// Returns the per-instance fractions and the next evaluation point.
/// The fractions are padded to the `2^k` selector slots with the zero fraction.
///
/// One `batch_coeff` batches the layer's numerator and denominator claims.
/// The verifier's `batch_verify_mle` samples it once per layer, before the round polynomials.
/// The content rounds and the selector rounds reuse the same coefficient.
fn reduce_layer<A, F, P, MP>(
	alloc: &A,
	mut layer_provers: Vec<MP>,
	eval_point: &[F],
	k: usize,
	channel: &mut impl IPProverChannel<F>,
) -> (Vec<Fraction<F>>, Vec<F>)
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
	MP: MleCheckProver<F> + Send,
{
	// Split eval_point into outer (selector) and inner (content) coordinates.
	let (outer_coords, inner_coords) = eval_point.split_at(k);

	// eq weights for batching over instances: eq(i, outer_coords) for all i in B_k.
	let eq_weights = Hypercube::One.expand(outer_coords).build::<F>();

	let batch_coeff = channel.sample();

	let mut challenges = Vec::with_capacity(eval_point.len());

	prove_content_rounds(
		&mut layer_provers,
		eq_weights.as_ref(),
		inner_coords.len(),
		batch_coeff,
		&mut challenges,
		channel,
	);

	let (reduced_halves, selector_columns) =
		finish_and_transpose::<A, F, P, MP>(alloc, layer_provers, k);

	let merged_evals = prove_selector_rounds(
		alloc,
		selector_columns,
		eq_weights.as_ref(),
		outer_coords,
		batch_coeff,
		&mut challenges,
		channel,
	);

	finalize_layer(merged_evals, &reduced_halves, k, challenges, channel)
}

/// Folds the content variables of every instance in lockstep, one round polynomial per round.
///
/// Each round sends the eq(selector)-weighted sum of the per-instance round polynomials.
/// One instance's polynomial is its `[num, den]` pair batched with `batch_coeff`.
/// Every instance then folds on the challenge the round draws.
/// The challenges are appended to `challenges` in round order.
///
/// `eq_weights` holds one weight per selector slot, the instances taking the leading ones.
/// A slot past the last instance holds the constant fraction 0/1: numerator 0, denominator 1.
/// A constant composition has that same constant as its round polynomial.
/// So a padding slot's claims stay (0, 1) through every fold.
/// It contributes `eq_i * batch_coeff` to each round polynomial's constant coefficient.
fn prove_content_rounds<F, MP>(
	layer_provers: &mut [MP],
	eq_weights: &[F],
	n_rounds: usize,
	batch_coeff: F,
	challenges: &mut Vec<F>,
	channel: &mut impl IPProverChannel<F>,
) where
	F: Field,
	MP: MleCheckProver<F> + Send,
{
	let pad_eq_sum: F = eq_weights[layer_provers.len()..].iter().copied().sum();

	for _round in 0..n_rounds {
		// The instances are independent within a round, so their polynomials compute in parallel.
		//
		// One instance's round is too small a parallel region to fill the pool alone.
		let per_instance: Vec<RoundCoeffs<F>> = layer_provers
			.par_iter_mut()
			.map(|prover| RoundCoeffs::batch(prover.execute(), &batch_coeff))
			.collect();

		// Weight instance j's polynomial by eq_j and sum, in instance order.
		let real_coeffs: RoundCoeffs<F> = iter::zip(per_instance, eq_weights)
			.map(|(coeffs, &eq_i)| coeffs * eq_i)
			.sum();
		let round_coeffs = real_coeffs + &RoundCoeffs(vec![pad_eq_sum * batch_coeff]);

		channel.send_many(mlecheck::RoundProof::truncate(round_coeffs).coeffs());

		let challenge = channel.sample();
		challenges.push(challenge);

		for prover in layer_provers.iter_mut() {
			prover.fold(challenge);
		}
	}
}

/// Finishes the content provers and transposes their reduced halves into the selector columns.
///
/// Each instance finishes with the four evaluations `[num_0, num_1, den_0, den_1]` it reduced.
/// Instance `i` occupies slot `i` of each of the four returned columns.
/// The columns span the `k` selector variables, so they hold `2^k` slots.
///
/// Returns the per-instance evaluations and those columns.
/// The line-fold that closes the layer reduces the evaluations.
/// The selector MLE-check folds the columns.
///
/// Both children of a padding slot are the zero fraction 0/1.
/// So a slot past the last instance holds 0 in the numerator columns and 1 in the denominators.
fn finish_and_transpose<A, F, P, MP>(
	alloc: &A,
	layer_provers: Vec<MP>,
	k: usize,
) -> (Vec<[F; 4]>, [FieldVec<P, A>; 4])
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
	MP: MleCheckProver<F>,
{
	let reduced: Vec<[F; 4]> = layer_provers
		.into_iter()
		.map(|prover| {
			prover
				.finish()
				.try_into()
				.expect("fractional-addition prover has four multilinears")
		})
		.collect();

	// Each column starts as all padding, then one pass over the reduced halves writes the
	// instances into its leading slots.
	let pad = Fraction::<F>::ZERO;
	let mut columns = [pad.num, pad.num, pad.den, pad.den].map(|pad_half| {
		let mut column = FieldBuffer::zeros_in(alloc, k);
		for slot in reduced.len()..1 << k {
			column.set(slot, pad_half);
		}
		column
	});
	for (slot, evals) in reduced.iter().enumerate() {
		for (column, &eval) in iter::zip(&mut columns, evals) {
			column.set(slot, eval);
		}
	}

	(reduced, columns)
}

/// Folds the `k` selector variables of one layer in a single fractional-addition MLE-check.
///
/// The claim is what the content rounds reduced the layer to.
/// It is the eq(selector)-weighted sum of the fractional-addition composition of the four columns.
/// The rounds reuse `batch_coeff`, and their challenges are appended to `challenges`.
///
/// Returns the merged `[num_0, num_1, den_0, den_1]` evaluations at those challenges.
fn prove_selector_rounds<'a, A, F, P>(
	alloc: &'a A,
	columns: [FieldVec<P, A>; 4],
	eq_weights: &[F],
	outer_coords: &[F],
	batch_coeff: F,
	challenges: &mut Vec<F>,
	channel: &mut impl IPProverChannel<F>,
) -> [F; 4]
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	let k = outer_coords.len();

	let [num_0s, num_1s, den_0s, den_1s] = &columns;
	let num_eval: F = izip!(
		num_0s.iter_scalars(),
		num_1s.iter_scalars(),
		den_0s.iter_scalars(),
		den_1s.iter_scalars(),
		eq_weights
	)
	.map(|(n0, n1, d0, d1, &eq_i)| eq_i * (n0 * d1 + n1 * d0))
	.sum();
	let den_eval: F = izip!(den_0s.iter_scalars(), den_1s.iter_scalars(), eq_weights)
		.map(|(d0, d1, &eq_i)| eq_i * (d0 * d1))
		.sum();

	// The columns are freshly packed for this check, so the store owns them directly.
	let mut selector_store = MleStore::new(k, alloc);
	let selector_cols = columns.map(|column| selector_store.push_owned(column));
	let (selector_num, selector_den) = frac_add_mle::evaluators::<F, P>(selector_cols);
	let claims_with_evaluators: [(F, Box<dyn MleCheckRoundEvaluator<F, P> + 'a>); 2] = [
		(num_eval, Box::new(selector_num)),
		(den_eval, Box::new(selector_den)),
	];
	let mut selector_prover =
		SharedMleCheckProver::new(selector_store, claims_with_evaluators, outer_coords.to_vec());

	for _round in 0..k {
		let round_coeffs = RoundCoeffs::batch(selector_prover.execute(), &batch_coeff);
		channel.send_many(mlecheck::RoundProof::truncate(round_coeffs).coeffs());

		let challenge = channel.sample();
		challenges.push(challenge);
		selector_prover.fold(challenge);
	}

	selector_prover
		.finish()
		.try_into()
		.expect("fractional-addition prover has four multilinears")
}

/// Sends the merged child evaluations and line-folds them into the next layer's claims.
///
/// The verifier reads the four evaluations before it samples the doubling coordinate `r`.
/// So the fold happens only once the transcript holds them.
///
/// `reduced` holds the four child evaluations of each real instance, one entry per instance.
/// The `2^k - reduced.len()` padding slots hold the zero fraction.
/// A line-fold between two zero fractions leaves it unchanged.
/// Returning them keeps the output aligned with the next layer's `2^k` selector eq weights.
fn finalize_layer<F: Field>(
	merged_evals: [F; 4],
	reduced: &[[F; 4]],
	k: usize,
	challenges: Vec<F>,
	channel: &mut impl IPProverChannel<F>,
) -> (Vec<Fraction<F>>, Vec<F>) {
	channel.send_many(&merged_evals);

	let r = channel.sample();

	// Sumcheck binds variables high-to-low; reverse to low-to-high for the claim point.
	let mut next_point = challenges;
	next_point.reverse();
	next_point.push(r);

	let next_fractions = reduced
		.iter()
		.map(|&[num_0, num_1, den_0, den_1]| {
			Fraction::new(extrapolate_line(num_0, num_1, r), extrapolate_line(den_0, den_1, r))
		})
		.chain(iter::repeat_n(Fraction::ZERO, (1 << k) - reduced.len()))
		.collect();

	(next_fractions, next_point)
}

/// Runs a batched fractional-addition check for trees of *unequal* depths.
///
/// Every tree shallower than the deepest is proved over its zero-fraction-padded witness.
/// [`super::padding`] states the identity that such a padding satisfies.
/// The transcript is then exactly that of an equal-depth batch of the maximum depth.
/// The verifier runs the ordinary [`binius_ip::fracaddcheck::verify`] over `n_layers` layers.
/// It never learns the individual depths.
///
/// Every prover must reduce over *all* of its witness variables, so each fractional sum is a
/// scalar and there is no content point.
/// Dropping the content dimension keeps the padding bookkeeping to four scalars per layer.
///
/// The prover does not materialize the padded witnesses.
/// Each layer's per-tree reduction corrects the unpadded layer's messages in $O(1)$ per round.
///
/// # Arguments
///
/// * `provers` - The trees to batch, whose layer counts may differ.
/// * `claimed_fractions` - Each tree's claimed root fraction, one per prover.
/// * `selector_point` - Evaluation point for the selector variables.
/// * `channel` - The channel for sending prover messages and sampling challenges.
///
/// # Preconditions
/// * `provers` must be non-empty.
/// * Every prover's witness must have exactly `prover.n_layers()` variables. A tree of depth zero
///   is allowed — it is all padding, so its leaf claim is its root — but at least one tree must
///   have a layer.
/// * `2^selector_point.len() >= provers.len()`.
/// * `claimed_fractions.len() == provers.len()`.
///
/// # Returns
///
/// A [`BatchProveOutput`] whose `fractions` are each tree's leaf claim, in input order, at the
/// shared reduced `eval_point`.
///
/// Those leaf claims are on the *padded* witnesses.
/// [`super::unpad_leaf_claim`] reduces one to the claims on the tree's own witness.
pub fn batch_prove_unequal_depths<'a, A, F, P>(
	provers: Vec<FracAddCircuit<'a, A, P>>,
	claimed_fractions: Vec<Fraction<F>>,
	selector_point: Vec<F>,
	channel: &mut impl IPProverChannel<F>,
) -> BatchProveOutput<F>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert!(!provers.is_empty()); // precondition
	assert_eq!(claimed_fractions.len(), provers.len()); // precondition

	let k = selector_point.len();
	assert!(provers.len() <= (1 << k)); // precondition

	let alloc = provers[0].alloc;
	let mut provers = provers;
	let n_layers = provers
		.iter()
		.map(FracAddCircuit::n_layers)
		.max()
		.expect("provers is non-empty");
	assert!(n_layers >= 1); // precondition
	// How much depth each tree is padded by.
	let pad_lens = provers
		.iter()
		.map(|prover| n_layers - prover.n_layers())
		.collect::<Vec<_>>();

	let n_trees = provers.len();
	let mut claims = claimed_fractions;
	let mut eval_point = selector_point;

	// Each iteration reduces the layer whose node variables are the point's suffix past the
	// selector coordinates. A tree the batch has not yet reached contributes a padding layer.
	for _ in 0..n_layers {
		let layer_provers = layer_provers(&mut provers, &pad_lens, &claims, &eval_point[k..]);
		let (next_claims, next_point) =
			reduce_layer::<A, F, P, _>(alloc, layer_provers, &eval_point, k, channel);
		claims = next_claims;
		eval_point = next_point;
	}
	// A depth-0 tree is all padding, so it is passed through every round and never popped; every
	// tree that had a layer has spent them all.
	debug_assert!(
		provers.iter().all(|prover| prover.n_layers() == 0),
		"every tree with layers is exhausted after n_layers reductions"
	);

	// `reduce_layer` pads its output to the 2^k selector slots; only the real trees remain.
	let mut fractions = claims;
	fractions.truncate(n_trees);

	BatchProveOutput {
		eval_point,
		fractions,
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_ip::fracaddcheck;
	use binius_math::{
		inner_product::inner_product,
		multilinear::Multilinear,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::fracaddcheck::unpad_leaf_claim;

	type StdChallenger = HasherChallenger<sha2::Sha256>;

	/// A numerator/denominator witness pair.
	type Witness<P> = Fraction<FieldBuffer<P>>;

	/// One prover per entry of `depths`, each reducing over all of its witness variables.
	#[allow(clippy::type_complexity)]
	fn unequal_depth_provers<'a, P: PackedField>(
		rng: &mut impl rand::Rng,
		alloc: &'a GlobalAllocator,
		depths: &[usize],
	) -> (Vec<Witness<P>>, Vec<FracAddCircuit<'a, GlobalAllocator, P>>, Vec<Fraction<P::Scalar>>) {
		itertools::multiunzip(depths.iter().map(|&depth| {
			let witness = Fraction::new(
				random_field_buffer::<P>(&mut *rng, depth),
				random_field_buffer::<P>(&mut *rng, depth),
			);
			let (prover, sums) = FracAddCircuit::build(depth, alloc, witness.clone());
			assert_eq!(sums.num.log_len(), 0);
			(witness, prover, sums.as_ref().map(|buffer| buffer.get(0)))
		}))
	}

	/// The eq(selector)-weighted combination of per-tree fractions, as the verifier forms it.
	///
	/// The selector slots beyond the trees hold the zero fraction 0/1.
	fn combine_fractions<P: PackedField>(
		fractions: &[Fraction<P::Scalar>],
		selector_point: &[P::Scalar],
	) -> (P::Scalar, P::Scalar) {
		let n_slots = 1 << selector_point.len();
		let eq_weights = Hypercube::One.expand(selector_point).build::<P>();
		let num_eval = inner_product(
			fractions.iter().map(|f| f.num),
			(0..fractions.len()).map(|i| eq_weights.get(i)),
		);
		let den_eval = inner_product(
			fractions
				.iter()
				.map(|f| f.den)
				.chain(iter::repeat_n(P::Scalar::ONE, n_slots - fractions.len())),
			(0..n_slots).map(|i| eq_weights.get(i)),
		);
		(num_eval, den_eval)
	}

	/// Proves a batch of unequal-depth trees against the depth-oblivious verifier, then unpads each
	/// tree's leaf claims and checks them against that tree's own witness.
	fn test_unequal_depths_helper<P: PackedField>(depths: &[usize], seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let alloc = GlobalAllocator;

		let k = log2_ceil_usize(depths.len());
		let n_layers = *depths.iter().max().expect("depths is non-empty");

		let (witnesses, provers, claimed_fractions) =
			unequal_depth_provers::<P>(&mut rng, &alloc, depths);

		// The verifier's input claim is the eq(selector)-weighted combination of the fractions.
		let selector_point = random_scalars::<P::Scalar>(&mut rng, k);
		let (num_eval, den_eval) = combine_fractions::<P>(&claimed_fractions, &selector_point);
		let claim = fracaddcheck::FracAddEvalClaim {
			num_eval,
			den_eval,
			point: selector_point.clone(),
		};

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let BatchProveOutput {
			eval_point,
			fractions,
		} = batch_prove_unequal_depths(
			provers,
			claimed_fractions,
			selector_point,
			&mut prover_transcript,
		);

		// The verifier's control flow depends only on the maximum depth.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify(n_layers, claim, &mut verifier_transcript).unwrap();

		assert_eq!(verifier_output.point, eval_point);
		let (num_eval, den_eval) = combine_fractions::<P>(&fractions, &eval_point[..k]);
		assert_eq!(verifier_output.num_eval, num_eval);
		assert_eq!(verifier_output.den_eval, den_eval);

		// Each tree's reduced claims are on its *padded* witness; unpadding them yields claims on
		// the witness itself, at a suffix of the shared node point.
		for (i, (&depth, witness)) in iter::zip(depths, &witnesses).enumerate() {
			let leaf = unpad_leaf_claim(fractions[i], &eval_point[k..], n_layers - depth);
			assert_eq!(leaf.point.len(), depth);
			assert_eq!(leaf.num_eval, witness.num.evaluate(&leaf.point), "tree {i} numerator");
			assert_eq!(leaf.den_eval, witness.den.evaluate(&leaf.point), "tree {i} denominator");
		}
	}

	#[test]
	fn test_unequal_depths_mixed() {
		test_unequal_depths_helper::<Packed128b>(&[2, 4, 5], 11);
	}

	#[test]
	fn test_unequal_depths_single_prover() {
		test_unequal_depths_helper::<Packed128b>(&[3], 11);
	}

	#[test]
	fn test_unequal_depths_power_of_two_provers() {
		// The shallowest tree is padded by more than one layer, the deepest not at all.
		test_unequal_depths_helper::<Packed128b>(&[1, 2, 5, 5], 11);
	}

	#[test]
	fn test_unequal_depths_all_minimal() {
		// Depth 1 throughout: every tree retains its final layer immediately.
		test_unequal_depths_helper::<Packed128b>(&[1, 1, 1], 11);
	}

	#[test]
	fn test_unequal_depths_zero_depth_tree() {
		// A depth-0 tree never pops a layer: it is all padding, so its leaf claim is its root.
		test_unequal_depths_helper::<Packed128b>(&[0, 3], 11);
	}

	#[test]
	fn test_unequal_depths_maximal_padding() {
		// A single-layer tree beside a deep one: all but its last reduction is padding.
		test_unequal_depths_helper::<Packed128b>(&[1, 6], 11);
	}

	#[test]
	fn test_unequal_depths_equal_depths() {
		// Equal depths pad nothing, so every wrapper is a pass-through.
		test_unequal_depths_helper::<Packed128b>(&[4, 4, 4], 11);
	}

	proptest! {
		// A full batched prove-verify per case, so trade the default case count down for runtime.
		#![proptest_config(ProptestConfig::with_cases(64))]

		// Invariant: the batched round trip holds for any mix of tree depths.
		//
		// The cases above pin named edge shapes; this covers the space between them.
		// Padding is bookkept per tree, so it is the mix of depths that stresses it.
		#[test]
		fn unequal_depths_round_trip(
			seed in any::<u64>(),
			depths in prop::collection::vec(0usize..=6, 1..=5),
		) {
			// Batching needs at least one layer to reduce, so an all-depth-0 batch is not a case.
			prop_assume!(depths.iter().any(|&depth| depth > 0));

			test_unequal_depths_helper::<Packed128b>(&depths, seed);
		}
	}
}
