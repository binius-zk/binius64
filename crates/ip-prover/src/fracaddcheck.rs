// Copyright 2025-2026 The Binius Developers

use std::iter;

use binius_field::{Field, PackedField};
use binius_ip::{mlecheck, prodcheck::MultilinearEvalClaim, sumcheck::RoundCoeffs};
use binius_math::{
	FieldBuffer, line::extrapolate_line_packed, multilinear::eq::eq_ind_partial_eval,
};
use binius_utils::rayon::iter::{IntoParallelIterator, ParallelIterator};
use itertools::izip;

use crate::{
	channel::IPProverChannel,
	sumcheck::{
		batch::batch_prove_mle_and_write_evals,
		common::{MleCheckProver, SumcheckProver},
		frac_add_mle::{self, FractionalBuffer},
	},
};

/// The numerator and denominator evaluation claims of one fractional-addition layer.
///
/// Both claims share the same evaluation point, that of the layer they describe.
pub type FracEvalClaim<F> = (MultilinearEvalClaim<F>, MultilinearEvalClaim<F>);

/// Prover for the fractional addition protocol.
///
/// Each layer is a double of the numerator and denominator values of fractional terms. Each layer
/// represents the addition of siblings with respect to the fractional addition rule:
/// $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}$
pub struct FracAddCheckProver<P: PackedField> {
	layers: Vec<(FieldBuffer<P>, FieldBuffer<P>)>,
}

impl<F, P> FracAddCheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Creates a new [`FracAddCheckProver`].
	///
	/// Returns `(prover, sums)` where `sums` is the final layer containing the
	/// fractional additions over all `k` variables.
	///
	/// # Arguments
	/// * `k` - The number of variables over which the reduction is taken. Each reduction step
	///   reduces one variable by computing fractional additions of sibling terms.
	/// * `witness` - The witness numerator/denominator layers
	///
	/// # Preconditions
	/// * `witness.0.log_len() >= k`
	pub fn new(k: usize, witness: FractionalBuffer<P>) -> (Self, FractionalBuffer<P>) {
		let (witness_num, witness_den) = witness;
		assert_eq!(
			witness_num.log_len(),
			witness_den.log_len(),
			"numerator and denominator witnesses must have equal length"
		);
		assert!(witness_num.log_len() >= k);

		let mut layers = Vec::with_capacity(k + 1);
		layers.push((witness_num, witness_den));

		for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");

			let (num, den) = prev_layer;
			let (num_0, num_1) = num.split_half_ref();
			let (den_0, den_1) = den.split_half_ref();

			let (next_layer_num, next_layer_den) =
				(num_0.as_ref(), den_0.as_ref(), num_1.as_ref(), den_1.as_ref())
					.into_par_iter()
					.map(|(&a_0, &b_0, &a_1, &b_1)| (a_0 * b_1 + a_1 * b_0, b_0 * b_1))
					.collect::<(Vec<_>, Vec<_>)>();

			let next_layer = (
				FieldBuffer::new(num.log_len() - 1, next_layer_num.into_boxed_slice()),
				FieldBuffer::new(den.log_len() - 1, next_layer_den.into_boxed_slice()),
			);

			layers.push(next_layer);
		}

		let sums = layers.pop().expect("layers has k+1 elements");
		(Self { layers }, sums)
	}

	/// Returns the number of remaining layers to prove.
	pub const fn n_layers(&self) -> usize {
		self.layers.len()
	}

	/// Pops the last layer and returns a sumcheck prover for it.
	///
	/// Returns `(layer_prover, remaining)` where:
	/// - `layer_prover` is a sumcheck prover for the popped layer
	/// - `remaining` is `Some(self)` if there are more layers, `None` otherwise
	pub fn layer_prover(
		mut self,
		claim: FracEvalClaim<F>,
	) -> (impl MleCheckProver<F>, Option<Self>) {
		let (num_claim, den_claim) = claim;
		assert_eq!(
			num_claim.point, den_claim.point,
			"fractional claims must share the evaluation point"
		);

		let layer = self.layers.pop().expect("layers is non-empty");

		let remaining = if self.layers.is_empty() {
			None
		} else {
			Some(self)
		};

		let (num, den) = layer;
		let (num_0, num_1) = num.split_half_ref();
		let (den_0, den_1) = den.split_half_ref();
		let num_0 = FieldBuffer::new(num_0.log_len(), num_0.as_ref().into());
		let num_1 = FieldBuffer::new(num_1.log_len(), num_1.as_ref().into());
		let den_0 = FieldBuffer::new(den_0.log_len(), den_0.as_ref().into());
		let den_1 = FieldBuffer::new(den_1.log_len(), den_1.as_ref().into());
		let prover = frac_add_mle::new(
			[num_0, num_1, den_0, den_1],
			num_claim.point.clone(),
			[num_claim.eval, den_claim.eval],
		);

		(prover, remaining)
	}

	/// Runs the fractional addition check protocol and returns the final evaluation claims.
	///
	/// This consumes the prover and runs sumcheck reductions from the smallest layer back to
	/// the largest.
	///
	/// # Arguments
	/// * `claim` - The initial multilinear evaluation claims (numerator, denominator)
	/// * `channel` - The channel for sending prover messages and sampling challenges
	///
	/// # Preconditions
	/// * `claim.0.point.len() == witness.log_len() - k` (where k is the number of reduction layers)
	pub fn prove(
		self,
		claim: FracEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> FracEvalClaim<F> {
		// Proving the full circuit runs every layer, so delegate and drop the leftover prover.
		let n_layers = self.n_layers();
		let (remaining, claim) = self.prove_layers(n_layers, claim, channel);
		debug_assert!(
			remaining.is_none_or(|prover| prover.n_layers() == 0),
			"proving every layer leaves none unproved"
		);
		claim
	}

	/// Runs the first `n_layers` fractional-addition layers from a claim, returning the remainder.
	///
	/// Each layer adds one variable via a sumcheck and a line-fold.
	/// So starting from a claim over `d` variables, the returned claim is over `d + n_layers`.
	///
	/// This is the building block of [`Self::prove`], which runs every layer.
	/// Stopping early leaves the remaining prover on its untouched layers.
	/// A caller can splice the leaf layer into another reduction, as the logUp* final layer does.
	///
	/// # Arguments
	/// * `n_layers` - The number of layers to prove, at most [`Self::n_layers`].
	/// * `claim` - The initial numerator/denominator claims, sharing an evaluation point.
	/// * `channel` - The channel for sending prover messages and sampling challenges.
	///
	/// # Returns
	/// * `Some(self)` holding the untouched layers, or `None` if all were proved,
	/// * the reduced numerator/denominator claims after `n_layers` layers.
	///
	/// # Preconditions
	/// * `n_layers <= self.n_layers()`.
	pub fn prove_layers(
		self,
		n_layers: usize,
		claim: FracEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> (Option<Self>, FracEvalClaim<F>) {
		// Each layer consumes the prover and returns the remainder, so thread it through an Option.
		let mut prover_opt = Some(self);
		let mut claim = claim;

		for _ in 0..n_layers {
			let prover = prover_opt
				.take()
				.expect("precondition: n_layers <= self.n_layers()");
			let (sumcheck_prover, remaining) = prover.layer_prover(claim);
			prover_opt = remaining;

			let output = batch_prove_mle_and_write_evals(vec![sumcheck_prover], channel);

			let mut multilinear_evals = output.multilinear_evals;
			let evals = multilinear_evals.pop().expect("batch contains one prover");

			let [num_0, num_1, den_0, den_1] = evals
				.try_into()
				.expect("prover evaluates four multilinears");

			// Fold the highest variable to combine the two halves into the next layer's claim.
			let r = channel.sample();

			let next_num = extrapolate_line_packed(num_0, num_1, r);
			let next_den = extrapolate_line_packed(den_0, den_1, r);

			let mut next_point = output.challenges;
			next_point.push(r);

			let num_claim = MultilinearEvalClaim {
				eval: next_num,
				point: next_point.clone(),
			};
			let den_claim = MultilinearEvalClaim {
				eval: next_den,
				point: next_point,
			};

			claim = (num_claim, den_claim);
		}

		(prover_opt, claim)
	}
}

/// Output of [`batch_prove`].
///
/// After the full `n_layers` reduction, `fractions` holds each input prover's reduced
/// `(num, den)` fraction at `eval_point`. The batched claim the verifier checks is the
/// eq(selector)-weighted combination of these fractions.
pub struct BatchProveOutput<F> {
	/// The reduced evaluation point (`selector ++ content`) at which the fractions are claimed.
	pub eval_point: Vec<F>,
	/// Each input prover's reduced `(num, den)` fraction at `eval_point`, in input order.
	pub fractions: Vec<(F, F)>,
}

/// Runs a batched fractional-addition check for multiple independent fracaddcheck provers,
/// reducing all `n_layers` layers.
///
/// This is the fractional-addition analog of [`crate::prodcheck::batch_prove`]. It combines `n`
/// provers, each for an $m$-variate numerator/denominator pair, using multilinear interpolation
/// over `k = selector_point.len()` selector variables (where $n \le 2^k$). The combined claim is
/// the multilinear extrapolation of the individual claimed fractions (padded with zeros to $2^k$)
/// evaluated at `selector_point ++ content_point`.
///
/// The claimed fractions may themselves be evaluations of the $m$-variate fractional-sum
/// multilinears at a shared `content_point`. When the fractions are scalars (each prover reduces
/// over all of its variables), `content_point` is empty.
///
/// This delegates to [`batch_prove_until_final_layer`] and then runs the final layer's MLE-check,
/// returning the reduced per-input-prover fractions at the reduced evaluation point. The batched
/// claim is checked by the ordinary `binius_ip::fracaddcheck::verify` recursion over
/// `k + n_layers` variables (the eq(selector)-weighted combination of the returned fractions),
/// with the selector coordinates forming the first `k` coordinates of the claim point — there is
/// no separate batched verifier, mirroring prodcheck.
///
/// # Arguments
/// * `provers` - Vec of `n` fracaddcheck provers. All must have the same `n_layers()`, which is
///   $m$.
/// * `claimed_fractions` - Vec of `n` claimed `(num, den)` fraction values, one per prover. Each is
///   the corresponding prover's fractional-sum multilinears evaluated at `content_point`.
/// * `selector_point` - Evaluation point for the selector variables. Length is $k$.
/// * `content_point` - Shared evaluation point at which the claimed fractions are taken. Length is
///   the fractional-sum multilinear dimension (i.e. `witness.log_len() - n_layers`). Empty for
///   scalar fractions.
/// * `channel` - The channel for sending prover messages and sampling challenges.
///
/// # Preconditions
/// * `provers` must be non-empty.
/// * All provers must have the same `n_layers()` value.
/// * `2^selector_point.len() >= provers.len()`.
/// * `claimed_fractions.len() == provers.len()`.
/// * `content_point.len() == witness.log_len() - n_layers` for each prover.
pub fn batch_prove<F, P>(
	provers: Vec<FracAddCheckProver<P>>,
	claimed_fractions: Vec<(F, F)>,
	selector_point: Vec<F>,
	content_point: Vec<F>,
	channel: &mut impl IPProverChannel<F>,
) -> BatchProveOutput<F>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let n = provers.len();
	let k = selector_point.len();

	let BatchProveUntilFinalLayerOutput {
		eval_point,
		final_layer,
	} = batch_prove_until_final_layer(
		provers,
		claimed_fractions,
		selector_point,
		content_point,
		channel,
	);

	// Finish the retained final layer: run its per-instance content MLE-checks and the selector
	// merge, exactly as an interior reduction layer does.
	let layer_provers = final_layer
		.into_iter()
		.map(|(_frac, prover)| prover)
		.collect();
	let (mut fractions, eval_point) =
		reduce_layer::<F, P, _>(layer_provers, eval_point, k, channel);

	// Drop the padded (2^k) selector slots, keeping one reduced fraction per input prover.
	fractions.truncate(n);

	BatchProveOutput {
		eval_point,
		fractions,
	}
}

/// Output of [`batch_prove_until_final_layer`].
///
/// After running `n_layers - 1` reductions, holds — for each input prover, in input order — its
/// reduced `(num, den)` fraction paired with the [`MleCheckProver`] `MP` for its final (widest)
/// layer, at the shared `eval_point` (`selector ++ content`).
pub struct BatchProveUntilFinalLayerOutput<F, MP> {
	/// The reduced evaluation point (`selector ++ content`) at which the final layer is claimed.
	pub eval_point: Vec<F>,
	/// Each input prover's reduced `(num, den)` fraction and final-layer MLE-check prover.
	pub final_layer: Vec<((F, F), MP)>,
}

/// Runs a batched fractional-addition check up to (but not finishing) the final layer's MLE-check.
///
/// Runs `n_layers - 1` of the per-layer reductions, then — for each input prover — pops its final
/// (widest) layer as an [`MleCheckProver`] (via [`FracAddCheckProver::layer_prover`]), seeded at
/// the reduced content coordinates with the prover's reduced `(num, den)` fraction claim.
///
/// # Returns
/// * the reduced evaluation point (`selector ++ content`) at which the final layer is claimed, and
/// * for each input prover, its reduced `(num, den)` fraction paired with the final-layer
///   [`MleCheckProver`].
///
/// The caller finishes the final layer — e.g. [`batch_prove`] runs its MLE-check directly, or the
/// logUp* final layer splices these provers into another reduction.
///
/// Arguments and preconditions are as for [`batch_prove`].
pub fn batch_prove_until_final_layer<F, P, Channel>(
	provers: Vec<FracAddCheckProver<P>>,
	claimed_fractions: Vec<(F, F)>,
	selector_point: Vec<F>,
	content_point: Vec<F>,
	channel: &mut Channel,
) -> BatchProveUntilFinalLayerOutput<F, impl MleCheckProver<F> + use<F, P, Channel>>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	assert!(!provers.is_empty()); // precondition
	assert_eq!(claimed_fractions.len(), provers.len()); // precondition

	let k = selector_point.len();
	assert!(provers.len() <= (1 << k)); // precondition

	let n_layers = provers[0].n_layers();
	assert!(n_layers >= 1); // precondition
	assert!(provers.iter().all(|p| p.n_layers() == n_layers)); // precondition

	// Thread the content point as the initial inner (content) coordinates of the evaluation point.
	// `batch_prove_layer` splits `eval_point.split_at(k)` into (selector, content); on the first
	// layer this seeds each layer prover with a claim at `content_point`.
	let eval_point = [selector_point, content_point].concat();

	// Run `n_layers - 1` reductions, stopping one layer short so each prover retains its final
	// (widest) layer for the caller to finish.
	let (provers, claimed_fractions, eval_point) = (0..n_layers - 1).fold(
		(provers, claimed_fractions, eval_point),
		|(provers, claimed_fractions, eval_point), _| {
			batch_prove_layer(provers, claimed_fractions, eval_point, k, channel)
		},
	);

	// Pop each remaining single-layer prover's final layer as an MLE-check prover, seeded at the
	// content coordinates with its reduced fraction. `claimed_fractions` is padded to `2^k`; zip
	// with the `n` real provers keeps only the real (input-prover) entries.
	let inner_coords = eval_point[k..].to_vec();
	let final_layer = iter::zip(provers, claimed_fractions)
		.map(|(prover, (num, den))| {
			let (mle_prover, remaining) = prover.layer_prover((
				MultilinearEvalClaim {
					eval: num,
					point: inner_coords.clone(),
				},
				MultilinearEvalClaim {
					eval: den,
					point: inner_coords.clone(),
				},
			));
			debug_assert!(remaining.is_none(), "one retained layer per prover");
			((num, den), mle_prover)
		})
		.collect();

	BatchProveUntilFinalLayerOutput {
		eval_point,
		final_layer,
	}
}

/// Combines the per-claim round polynomials of one fracaddcheck layer prover into a single
/// polynomial by Horner-folding with `batch_coeff`, matching the `[num, den]` batching that
/// [`sumcheck::batch_verify_mle`](binius_ip::sumcheck::batch_verify_mle) performs on the verifier.
fn combine_claims<F: Field>(coeffs: Vec<RoundCoeffs<F>>, batch_coeff: F) -> RoundCoeffs<F> {
	coeffs
		.into_iter()
		.rfold(RoundCoeffs::default(), |acc, c| acc * batch_coeff + &c)
}

/// Runs one batched fracaddcheck layer given its per-instance final-layer MLE-check provers.
///
/// Folds the content variables of every instance in lockstep (eq(selector)-weighted, `[num, den]`-
/// batched), then the `k` selector variables via a single fractional-addition MLE-check over the
/// packed reduced halves, then the doubling line-fold. Returns the reduced per-instance fractions
/// (padded to `2^k` with zeros) and the next evaluation point.
///
/// The two (numerator, denominator) claims of every layer are batched via a single `batch_coeff`
/// that the verifier's `batch_verify_mle` samples once per layer, before the round polynomials; the
/// same coefficient is reused for the content and selector rounds.
fn reduce_layer<F, P, MP>(
	mut layer_provers: Vec<MP>,
	eval_point: Vec<F>,
	k: usize,
	channel: &mut impl IPProverChannel<F>,
) -> (Vec<(F, F)>, Vec<F>)
where
	F: Field,
	P: PackedField<Scalar = F>,
	MP: MleCheckProver<F>,
{
	// Split eval_point into outer (selector) and inner (content) coordinates.
	let (outer_coords, inner_coords) = eval_point.split_at(k);

	// eq weights for batching over instances: eq(i, outer_coords) for all i in B_k.
	let eq_weights = eq_ind_partial_eval::<F>(outer_coords);

	let batch_coeff = channel.sample();

	let mut challenges = Vec::with_capacity(eval_point.len());

	// Content rounds: fold the content variables of every instance in lockstep, sending the
	// eq(selector)-weighted sum of the per-instance (num, den)-batched round polynomials.
	for _round in 0..inner_coords.len() {
		let round_coeffs: RoundCoeffs<F> = iter::zip(&mut layer_provers, eq_weights.iter_scalars())
			.map(|(prover, eq_i)| combine_claims(prover.execute(), batch_coeff) * eq_i)
			.sum();

		channel.send_many(mlecheck::RoundProof::truncate(round_coeffs).coeffs());

		let challenge = channel.sample();
		challenges.push(challenge);

		for prover in layer_provers.iter_mut() {
			prover.fold(challenge);
		}
	}

	// Finish inner provers to get [num_0, num_1, den_0, den_1] evals per instance.
	let finished: Vec<[F; 4]> = layer_provers
		.into_iter()
		.map(|prover| {
			prover
				.finish()
				.try_into()
				.expect("fractional-addition prover has four multilinears")
		})
		.collect();

	// Split the reduced halves into per-multilinear vectors, padded to 2^k with zeros so they can
	// be packed into selector-variable buffers.
	let mut num_0s: Vec<F> = finished.iter().map(|e| e[0]).collect();
	let mut num_1s: Vec<F> = finished.iter().map(|e| e[1]).collect();
	let mut den_0s: Vec<F> = finished.iter().map(|e| e[2]).collect();
	let mut den_1s: Vec<F> = finished.iter().map(|e| e[3]).collect();
	for vals in [&mut num_0s, &mut num_1s, &mut den_0s, &mut den_1s] {
		vals.resize(1 << k, F::ZERO);
	}

	// The selector claim is the eq(selector)-weighted sum of the fractional-addition composition of
	// the reduced halves.
	let num_eval: F = izip!(&num_0s, &num_1s, &den_0s, &den_1s, eq_weights.as_ref())
		.map(|(&n0, &n1, &d0, &d1, &eq_i)| eq_i * (n0 * d1 + n1 * d0))
		.sum();
	let den_eval: F = izip!(&den_0s, &den_1s, eq_weights.as_ref())
		.map(|(&d0, &d1, &eq_i)| eq_i * (d0 * d1))
		.sum();

	// Selector rounds: fold the selector variables with a single fractional-addition MLE-check over
	// the packed reduced halves, reusing the same `batch_coeff`.
	let mut selector_prover = frac_add_mle::new(
		[
			FieldBuffer::<P>::from_values(&num_0s),
			FieldBuffer::<P>::from_values(&num_1s),
			FieldBuffer::<P>::from_values(&den_0s),
			FieldBuffer::<P>::from_values(&den_1s),
		],
		outer_coords.to_vec(),
		[num_eval, den_eval],
	);

	for _round in 0..k {
		let round_coeffs = combine_claims(selector_prover.execute(), batch_coeff);
		channel.send_many(mlecheck::RoundProof::truncate(round_coeffs).coeffs());

		let challenge = channel.sample();
		challenges.push(challenge);
		selector_prover.fold(challenge);
	}

	let [merged_num_0, merged_num_1, merged_den_0, merged_den_1]: [F; 4] = selector_prover
		.finish()
		.try_into()
		.expect("fractional-addition prover has four multilinears");

	// Finalize layer: send merged evals, sample r, compute next claims.
	channel.send_many(&[merged_num_0, merged_num_1, merged_den_0, merged_den_1]);

	let r = channel.sample();

	let mut next_point = challenges;
	next_point.reverse();
	next_point.push(r);

	// Reduce the (padded) selector halves to the next layer's fraction claims. Padding with the
	// selector buffers (not just the `n` real provers) keeps `fractions.len() == 2^k`, so it stays
	// aligned with the selector `eq` weights on subsequent layers; the padded entries are zero.
	let next_fractions = izip!(&num_0s, &num_1s, &den_0s, &den_1s)
		.map(|(&num_0, &num_1, &den_0, &den_1)| {
			(extrapolate_line_packed(num_0, num_1, r), extrapolate_line_packed(den_0, den_1, r))
		})
		.collect();

	(next_fractions, next_point)
}

/// Runs one interior batched fracaddcheck layer, returning the remaining provers, the reduced
/// per-instance fractions (padded to `2^k`), and the next evaluation point.
#[allow(clippy::type_complexity)]
fn batch_prove_layer<F, P>(
	provers: Vec<FracAddCheckProver<P>>,
	claimed_fractions: Vec<(F, F)>,
	eval_point: Vec<F>,
	k: usize,
	channel: &mut impl IPProverChannel<F>,
) -> (Vec<FracAddCheckProver<P>>, Vec<(F, F)>, Vec<F>)
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// Build a fractional-addition MLE-check prover per instance, seeded with a claim at the content
	// coordinates.
	let inner_coords = eval_point[k..].to_vec();
	let (layer_provers, next_provers): (Vec<_>, Vec<_>) = iter::zip(provers, &claimed_fractions)
		.map(|(prover, &(num, den))| {
			prover.layer_prover((
				MultilinearEvalClaim {
					eval: num,
					point: inner_coords.clone(),
				},
				MultilinearEvalClaim {
					eval: den,
					point: inner_coords.clone(),
				},
			))
		})
		.unzip();

	let (next_fractions, next_point) =
		reduce_layer::<F, P, _>(layer_provers, eval_point, k, channel);

	let next_provers = next_provers.into_iter().flatten().collect();

	(next_provers, next_fractions, next_point)
}

#[cfg(test)]
mod tests {
	use binius_field::PackedField;
	use binius_ip::fracaddcheck;
	use binius_math::{
		inner_product::inner_product,
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use binius_utils::checked_arithmetics::log2_ceil_usize;

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use rand::prelude::*;

	use super::*;

	fn test_frac_add_check_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// 1. Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// 2. Create prover (computes fractional-add layers)
		let (prover, sums) = FracAddCheckProver::new(k, (witness_num.clone(), witness_den.clone()));

		// 3. Generate random n-dimensional challenge point
		let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

		// 4. Evaluate sums at challenge point to createzz claims
		let sum_num_eval = evaluate(&sums.0, &eval_point);
		let sum_den_eval = evaluate(&sums.1, &eval_point);
		let prover_claim = (
			MultilinearEvalClaim {
				eval: sum_num_eval,
				point: eval_point.clone(),
			},
			MultilinearEvalClaim {
				eval: sum_den_eval,
				point: eval_point.clone(),
			},
		);
		let verifier_claim = fracaddcheck::FracAddEvalClaim {
			num_eval: sum_num_eval,
			den_eval: sum_den_eval,
			point: eval_point,
		};

		// 5. Run prover
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = prover.prove(prover_claim, &mut prover_transcript);

		// 6. Run verifier
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify(k, verifier_claim, &mut verifier_transcript).unwrap();

		// 7. Check outputs match
		assert_eq!(prover_output.0.point, prover_output.1.point);
		assert_eq!(prover_output.0.point, verifier_output.point);
		assert_eq!(prover_output.0.eval, verifier_output.num_eval);
		assert_eq!(prover_output.1.eval, verifier_output.den_eval);

		// 8. Verify multilinear evaluation of original witness
		let expected_num = evaluate(&witness_num, &verifier_output.point);
		let expected_den = evaluate(&witness_den, &verifier_output.point);
		assert_eq!(verifier_output.num_eval, expected_num);
		assert_eq!(verifier_output.den_eval, expected_den);
	}

	#[test]
	fn test_frac_add_check_prove_verify() {
		test_frac_add_check_prove_verify_helper::<Packed128b>(4, 3);
	}

	#[test]
	fn test_frac_add_check_full_prove_verify() {
		test_frac_add_check_prove_verify_helper::<Packed128b>(0, 4);
	}

	fn test_frac_add_check_layer_computation_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// Create prover (computes fractional-add layers)
		let (_prover, sums) =
			FracAddCheckProver::new(k, (witness_num.clone(), witness_den.clone()));

		// For each index i in the sums layer, verify it equals the fractional sum of witness values
		// at indices i + z * 2^n for z in 0..2^k (strided access, not contiguous)
		let stride = 1 << n;
		let num_terms = 1 << k;
		for i in 0..(1 << n) {
			let mut expected_num = witness_num.get(i);
			let mut expected_den = witness_den.get(i);
			for z in 1..num_terms {
				let idx = i + z * stride;
				let num_z = witness_num.get(idx);
				let den_z = witness_den.get(idx);
				expected_num = expected_num * den_z + num_z * expected_den;
				expected_den *= den_z;
			}
			let actual_num = sums.0.get(i);
			let actual_den = sums.1.get(i);
			assert_eq!(actual_num, expected_num, "Numerator mismatch at index {i}");
			assert_eq!(actual_den, expected_den, "Denominator mismatch at index {i}");
		}
	}

	#[test]
	fn test_frac_add_check_layer_computation() {
		test_frac_add_check_layer_computation_helper::<Packed128b>(4, 3);
	}

	// ==================== batch_prove tests ====================

	/// Combines the per-input-prover fractions returned by [`batch_prove`] into the single
	/// [`FracAddEvalClaim`] the verifier produces: the eq(selector)-weighted sum over the first `k`
	/// (selector) coordinates of the reduced evaluation point.
	fn combine_batch_prove<F: Field, P: PackedField<Scalar = F>>(
		output: BatchProveOutput<F>,
		log_n_provers: usize,
	) -> fracaddcheck::FracAddEvalClaim<F> {
		let BatchProveOutput {
			eval_point,
			fractions,
		} = output;
		let selector_weights = eq_ind_partial_eval::<P>(&eval_point[..log_n_provers]);
		let num_eval = inner_product(
			fractions.iter().map(|&(n, _)| n),
			(0..fractions.len()).map(|i| selector_weights.get(i)),
		);
		let den_eval = inner_product(
			fractions.iter().map(|&(_, d)| d),
			(0..fractions.len()).map(|i| selector_weights.get(i)),
		);
		fracaddcheck::FracAddEvalClaim {
			num_eval,
			den_eval,
			point: eval_point,
		}
	}

	/// Helper for testing `batch_prove` over `n_provers` fracaddcheck instances of `n_layers` each.
	///
	/// Each witness has exactly `n_layers` variables so that the fractional sums are scalars
	/// (0-variate).
	fn test_batch_prove_verify_helper<P: PackedField>(n_layers: usize, n_provers: usize) {
		let mut rng = StdRng::seed_from_u64(42);

		let log_n_provers = log2_ceil_usize(n_provers);

		// Each witness has exactly n_layers variables; fractional sums are scalars.
		let witnesses: Vec<(FieldBuffer<P>, FieldBuffer<P>)> = (0..n_provers)
			.map(|_| {
				(
					random_field_buffer::<P>(&mut rng, n_layers),
					random_field_buffer::<P>(&mut rng, n_layers),
				)
			})
			.collect();

		let (provers, individual_sums): (Vec<_>, Vec<_>) = witnesses
			.iter()
			.map(|witness| FracAddCheckProver::new(n_layers, witness.clone()))
			.unzip();

		// Fractions are 0-variate (scalars): just get the single (num, den) value.
		let claimed_fractions: Vec<(P::Scalar, P::Scalar)> = individual_sums
			.iter()
			.map(|(num, den)| {
				assert_eq!(num.log_len(), 0);
				(num.get(0), den.get(0))
			})
			.collect();

		// Combined verifier claim: eq(selector)-weighted sums of the claimed fractions, at point
		// selector.
		let selector_challenge = random_scalars::<P::Scalar>(&mut rng, log_n_provers);
		let eq_weights = eq_ind_partial_eval::<P>(&selector_challenge);
		let combined_num = inner_product(
			claimed_fractions.iter().map(|&(n, _)| n),
			(0..n_provers).map(|i| eq_weights.get(i)),
		);
		let combined_den = inner_product(
			claimed_fractions.iter().map(|&(_, d)| d),
			(0..n_provers).map(|i| eq_weights.get(i)),
		);

		let claim = fracaddcheck::FracAddEvalClaim {
			num_eval: combined_num,
			den_eval: combined_den,
			point: selector_challenge.clone(),
		};

		// Run batch_prove (scalar fractions: empty content point).
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let batch_output = batch_prove(
			provers,
			claimed_fractions,
			selector_challenge,
			Vec::new(),
			&mut prover_transcript,
		);
		assert_eq!(batch_output.fractions.len(), n_provers);
		let prover_output = combine_batch_prove::<_, P>(batch_output, log_n_provers);

		// Run verifier with n_layers layers.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify(n_layers, claim, &mut verifier_transcript).unwrap();

		assert_eq!(prover_output, verifier_output);

		// Verify the final fraction against the eq(selector)-weighted interpolation of the input
		// witnesses.
		let final_point = &verifier_output.point;
		assert_eq!(final_point.len(), log_n_provers + n_layers);

		let selector_challenges = &final_point[..log_n_provers];
		let content_challenges = &final_point[log_n_provers..];

		let selector_weights = eq_ind_partial_eval::<P>(selector_challenges);

		let expected_num = inner_product(
			(0..n_provers).map(|i| evaluate(&witnesses[i].0, content_challenges)),
			(0..n_provers).map(|i| selector_weights.get(i)),
		);
		let expected_den = inner_product(
			(0..n_provers).map(|i| evaluate(&witnesses[i].1, content_challenges)),
			(0..n_provers).map(|i| selector_weights.get(i)),
		);

		assert_eq!(verifier_output.num_eval, expected_num);
		assert_eq!(verifier_output.den_eval, expected_den);
	}

	#[test]
	fn test_batch_prove_power_of_two_provers() {
		// 4 provers, 3 layers.
		test_batch_prove_verify_helper::<Packed128b>(3, 4);
	}

	#[test]
	fn test_batch_prove_non_power_of_two_provers() {
		// 3 provers (non-power of 2, requires padding), 4 layers.
		test_batch_prove_verify_helper::<Packed128b>(4, 3);
	}

	#[test]
	fn test_batch_prove_single_prover() {
		// 1 prover (edge case), 5 layers.
		test_batch_prove_verify_helper::<Packed128b>(5, 1);
	}

	#[test]
	fn test_batch_prove_single_layer() {
		// n_layers=1 edge case: batch_prove_until_final_layer runs 0 reductions and the single
		// (final) layer is finished by batch_prove.
		test_batch_prove_verify_helper::<Packed128b>(1, 4);
	}

	/// Helper for testing `batch_prove` where the claimed fractions are non-scalar: each prover's
	/// fractional-sum multilinears are `content_len`-variate, claimed at a shared content point.
	fn test_batch_prove_with_content_helper<P: PackedField>(
		n_layers: usize,
		n_provers: usize,
		content_len: usize,
	) {
		let mut rng = StdRng::seed_from_u64(7);

		let log_n_provers = log2_ceil_usize(n_provers);

		// Each witness has log_len = content_len + n_layers; fractional sums are
		// content_len-variate.
		let witnesses: Vec<(FieldBuffer<P>, FieldBuffer<P>)> = (0..n_provers)
			.map(|_| {
				(
					random_field_buffer::<P>(&mut rng, content_len + n_layers),
					random_field_buffer::<P>(&mut rng, content_len + n_layers),
				)
			})
			.collect();

		let (provers, individual_sums): (Vec<_>, Vec<_>) = witnesses
			.iter()
			.map(|witness| FracAddCheckProver::new(n_layers, witness.clone()))
			.unzip();

		// Shared content point; each claimed fraction is its multilinears evaluated there.
		let content_point = random_scalars::<P::Scalar>(&mut rng, content_len);
		let claimed_fractions: Vec<(P::Scalar, P::Scalar)> = individual_sums
			.iter()
			.map(|(num, den)| {
				assert_eq!(num.log_len(), content_len);
				(evaluate(num, &content_point), evaluate(den, &content_point))
			})
			.collect();

		let selector_challenge = random_scalars::<P::Scalar>(&mut rng, log_n_provers);
		let eq_weights = eq_ind_partial_eval::<P>(&selector_challenge);
		let combined_num = inner_product(
			claimed_fractions.iter().map(|&(n, _)| n),
			(0..n_provers).map(|i| eq_weights.get(i)),
		);
		let combined_den = inner_product(
			claimed_fractions.iter().map(|&(_, d)| d),
			(0..n_provers).map(|i| eq_weights.get(i)),
		);

		let claim = fracaddcheck::FracAddEvalClaim {
			num_eval: combined_num,
			den_eval: combined_den,
			point: [selector_challenge.clone(), content_point.clone()].concat(),
		};

		// Run batch_prove with non-empty content point.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let batch_output = batch_prove(
			provers,
			claimed_fractions,
			selector_challenge,
			content_point,
			&mut prover_transcript,
		);
		assert_eq!(batch_output.fractions.len(), n_provers);
		let prover_output = combine_batch_prove::<_, P>(batch_output, log_n_provers);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify(n_layers, claim, &mut verifier_transcript).unwrap();

		assert_eq!(prover_output, verifier_output);

		let final_point = &verifier_output.point;
		assert_eq!(final_point.len(), log_n_provers + n_layers + content_len);

		let selector_challenges = &final_point[..log_n_provers];
		let witness_challenges = &final_point[log_n_provers..];

		let selector_weights = eq_ind_partial_eval::<P>(selector_challenges);

		let expected_num = inner_product(
			(0..n_provers).map(|i| evaluate(&witnesses[i].0, witness_challenges)),
			(0..n_provers).map(|i| selector_weights.get(i)),
		);
		let expected_den = inner_product(
			(0..n_provers).map(|i| evaluate(&witnesses[i].1, witness_challenges)),
			(0..n_provers).map(|i| selector_weights.get(i)),
		);

		assert_eq!(verifier_output.num_eval, expected_num);
		assert_eq!(verifier_output.den_eval, expected_den);
	}

	#[test]
	fn test_batch_prove_with_content() {
		// 3 provers (non power of 2), 4 layers, content_len = 2.
		test_batch_prove_with_content_helper::<Packed128b>(4, 3, 2);
	}
}
