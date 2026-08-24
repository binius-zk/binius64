// Copyright 2025-2026 The Binius Developers

use std::iter;

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::{fracaddcheck::FracAddEvalClaim, mlecheck, sumcheck::RoundCoeffs};
use binius_math::{
	FieldBuffer, FieldVec,
	line::extrapolate_line,
	multilinear::hypercube::{Hypercube, OneCube},
};
use binius_utils::{
	buffer::VecLike,
	rayon::{
		iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};
use itertools::izip;

use crate::{
	channel::IPProverChannel,
	sumcheck::{
		batch::batch_prove_mle,
		common::MleCheckProver,
		frac_add_mle,
		mle_store::MleStore,
		round_evaluator::{MleCheckRoundEvaluator, SharedMleCheckProver},
	},
};

pub mod fraction;
pub mod padding;
pub mod zero_pad_mle;

use fraction::Fraction;
use padding::layer_provers;
pub use padding::unpad_leaf_claim;

pub use crate::sumcheck::frac_add_mle::LayerProver;

/// Prover for the fractional addition protocol.
///
/// Each layer is a double of the numerator and denominator values of fractional terms. Each layer
/// represents the addition of siblings with respect to the fractional addition rule:
/// $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}$
pub struct FracAddCheckProver<'a, A: Allocator, P: PackedField> {
	layers: Vec<Fraction<FieldVec<P, A>>>,
	/// Allocator the layer buffers are drawn from.
	pub(crate) alloc: &'a A,
}

impl<A: Allocator, P: PackedField> Clone for FracAddCheckProver<'_, A, P>
where
	A::Vec<P>: Clone,
{
	fn clone(&self) -> Self {
		Self {
			layers: self.layers.clone(),
			alloc: self.alloc,
		}
	}
}

impl<'a, A, F, P> FracAddCheckProver<'a, A, P>
where
	A: Allocator,
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
	/// * `witness.num.log_len() >= k`
	pub fn new(
		k: usize,
		alloc: &'a A,
		witness: Fraction<FieldVec<P, A>>,
	) -> (Self, Fraction<FieldVec<P, A>>) {
		let Fraction {
			num: witness_num,
			den: witness_den,
		} = witness;
		assert_eq!(
			witness_num.log_len(),
			witness_den.log_len(),
			"numerator and denominator witnesses must have equal length"
		);
		assert!(witness_num.log_len() >= k);

		let mut layers = Vec::with_capacity(k + 1);
		layers.push(Fraction::new(witness_num, witness_den));

		for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");

			let Fraction { num, den } = prev_layer;
			let num_log_len = num.log_len() - 1;
			let den_log_len = den.log_len() - 1;
			let (num_0, num_1) = num.split_half_ref();
			let (den_0, den_1) = den.split_half_ref();

			// One packed word of the next layer from the sibling halves, written straight into
			// the pooled buffers:
			//     a_0/b_0 + a_1/b_1 = (a_0*b_1 + a_1*b_0) / (b_0*b_1)
			// Workers each take a contiguous run of words.
			// One word is three multiplies and an add, a few nanoseconds of work.
			// A run must therefore be long enough to pay back handing it off.
			let out_len = num_0.as_ref().len();
			let mut num_data = alloc.alloc::<P>(out_len);
			let mut den_data = alloc.alloc::<P>(out_len);
			(
				num_data.spare_capacity_mut(),
				den_data.spare_capacity_mut(),
				num_0.as_ref(),
				den_0.as_ref(),
				num_1.as_ref(),
				den_1.as_ref(),
			)
				.into_par_iter()
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|(num_out, den_out, &num_0, &den_0, &num_1, &den_1)| {
					num_out.write(num_0 * den_1 + num_1 * den_0);
					den_out.write(den_0 * den_1);
				});
			// Invariant: every zip input holds at least `out_len` words.
			//
			// A parallel zip yields as many items as its shortest input holds.
			// A shorter input would leave trailing slots uninitialized.
			//
			//     spare capacity:  >= out_len   allocated for at least that many
			//     sibling halves:  == out_len   halves of two equal-length buffers
			debug_assert!(
				num_data.spare_capacity_mut().len() >= out_len
					&& den_data.spare_capacity_mut().len() >= out_len,
				"allocated buffers must hold every claimed slot"
			);
			debug_assert!(
				[den_0.as_ref(), num_1.as_ref(), den_1.as_ref()]
					.iter()
					.all(|half| half.len() == out_len),
				"the four sibling halves must hold exactly one word per claimed slot"
			);
			// Safety: both length claims cover only initialized slots.
			// - The assertions above bound every zip input below by `out_len`.
			// - So the loop ran `out_len` items.
			// - Each item wrote one numerator slot and one denominator slot.
			unsafe {
				num_data.set_len(out_len);
				den_data.set_len(out_len);
			}
			let next_layer = Fraction::new(
				FieldBuffer::new(num_log_len, num_data),
				FieldBuffer::new(den_log_len, den_data),
			);

			layers.push(next_layer);
		}

		let sums = layers.pop().expect("layers has k+1 elements");
		(Self { layers, alloc }, sums)
	}

	/// Returns the number of remaining layers to prove.
	pub const fn n_layers(&self) -> usize {
		self.layers.len()
	}

	/// Pops the widest remaining layer as the MLE-check prover that reduces it.
	///
	/// The returned prover owns the popped buffers and borrows only the allocator, so it outlives
	/// this borrow: a caller drives one layer at a time and keeps the circuit in place.
	///
	/// # Preconditions
	/// * `self.n_layers() >= 1`
	pub fn pop_layer(&mut self, claim: FracAddEvalClaim<F>) -> LayerProver<'a, A, F, P> {
		let Fraction { num, den } = self
			.layers
			.pop()
			.expect("precondition: self.n_layers() >= 1");

		// The MLE-check reduces four multilinears: the low and high halves of the numerator buffer
		// and of the denominator buffer. The store takes ownership of the two popped buffers and
		// shares each between its halves, so the prover is self-contained with no up-front copy of
		// the popped layer.
		frac_add_mle::new_split_half(
			self.alloc,
			num,
			den,
			claim.point,
			[claim.num_eval, claim.den_eval],
		)
	}

	/// Runs the fractional addition check protocol and returns the final evaluation claims.
	///
	/// This consumes the prover and runs sumcheck reductions from the smallest layer back to
	/// the largest.
	///
	/// # Arguments
	/// * `claim` - The numerator and denominator claims at their shared evaluation point.
	/// * `channel` - The channel for sending prover messages and sampling challenges.
	///
	/// # Preconditions
	/// * `claim.point.len() == witness.log_len() - k`, for `k` the number of reduction layers.
	pub fn prove(
		self,
		claim: FracAddEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> FracAddEvalClaim<F> {
		// Proving the full circuit runs every layer, so delegate and drop the leftover prover.
		let n_layers = self.n_layers();
		let (remaining, claim) = self.prove_layers(n_layers, claim, channel);
		debug_assert_eq!(remaining.n_layers(), 0, "proving every layer leaves none unproved");
		claim
	}

	/// Runs the first `n_layers` fractional-addition layers from a claim, returning the remainder.
	///
	/// Each layer adds one variable via a sumcheck and a line-fold.
	/// So starting from a claim over `d` variables, the returned claim is over `d + n_layers`.
	///
	/// This is the layer loop of [`Self::prove`], which runs every layer.
	/// The returned prover still holds the layers that were not proved.
	///
	/// # Arguments
	/// * `n_layers` - The number of layers to prove, at most [`Self::n_layers`].
	/// * `claim` - The numerator and denominator claims at their shared evaluation point.
	/// * `channel` - The channel for sending prover messages and sampling challenges.
	///
	/// # Returns
	/// * the circuit, holding whatever layers were not proved,
	/// * the reduced numerator/denominator claims after `n_layers` layers.
	///
	/// # Preconditions
	/// * `n_layers <= self.n_layers()`.
	fn prove_layers(
		mut self,
		n_layers: usize,
		claim: FracAddEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> (Self, FracAddEvalClaim<F>) {
		let mut claim = claim;

		for _ in 0..n_layers {
			let sumcheck_prover = self.pop_layer(claim);

			// The driver draws the batching coefficient and Horner-folds the layer's two claims,
			// which is the polynomial the verifier's `batch_verify_mle` reconstructs.
			let output = batch_prove_mle(vec![sumcheck_prover], channel);
			output.send_evals(channel);

			let mut multilinear_evals = output.multilinear_evals;
			let evals = multilinear_evals.pop().expect("batch contains one prover");

			let [num_0, num_1, den_0, den_1] = evals
				.try_into()
				.expect("prover evaluates four multilinears");

			// Fold the highest variable to combine the two halves into the next layer's claim.
			let r = channel.sample();

			let next_num = extrapolate_line(num_0, num_1, r);
			let next_den = extrapolate_line(den_0, den_1, r);

			// Sumcheck binds variables high-to-low; reverse to low-to-high for the claim point.
			let mut next_point = output.challenges;
			next_point.reverse();
			next_point.push(r);

			claim = FracAddEvalClaim {
				num_eval: next_num,
				den_eval: next_den,
				point: next_point,
			};
		}

		(self, claim)
	}
}

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
	let eq_weights = OneCube::eq_ind_partial_eval::<F>(outer_coords);

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
			.map(|prover| combine_claims(prover.execute(), batch_coeff))
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
		let round_coeffs = combine_claims(selector_prover.execute(), batch_coeff);
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
/// [`padding`] states the identity that such a padding satisfies.
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
/// [`unpad_leaf_claim`] reduces one to the claims on the tree's own witness.
pub fn batch_prove_unequal_depths<'a, A, F, P>(
	provers: Vec<FracAddCheckProver<'a, A, P>>,
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
		.map(FracAddCheckProver::n_layers)
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
	use binius_field::{FieldOps, PackedField};
	use binius_ip::fracaddcheck;
	use binius_math::{
		inner_product::inner_product,
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use proptest::prelude::*;

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	type F = <Packed128b as FieldOps>::Scalar;
	use binius_compute::GlobalAllocator;
	use rand::prelude::*;

	use super::*;

	fn test_frac_add_check_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// 1. Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// 2. Create prover (computes fractional-add layers)
		let (prover, sums) = FracAddCheckProver::new(
			k,
			&alloc,
			Fraction::new(witness_num.clone(), witness_den.clone()),
		);

		// 3. Generate random n-dimensional challenge point
		let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

		// 4. Evaluate sums at challenge point to create claims
		let sum_num_eval = evaluate(&sums.num, &eval_point);
		let sum_den_eval = evaluate(&sums.den, &eval_point);
		// The prover and the verifier take the same claim type, so one claim serves both.
		let claim = FracAddEvalClaim {
			num_eval: sum_num_eval,
			den_eval: sum_den_eval,
			point: eval_point,
		};

		// 5. Run prover
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = prover.prove(claim.clone(), &mut prover_transcript);

		// 6. Run verifier
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output = fracaddcheck::verify(k, claim, &mut verifier_transcript).unwrap();

		// 7. Check outputs match
		assert_eq!(prover_output, verifier_output);

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

	fn check_all_layers<P: PackedField>(n: usize, k: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let alloc = GlobalAllocator;

		// Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// Create prover (computes fractional-add layers)
		let (prover, sums) = FracAddCheckProver::new(
			k,
			&alloc,
			Fraction::new(witness_num.clone(), witness_den.clone()),
		);

		// `new` pops the root off as `sums`, so the circuit is `layers` followed by it.
		for (j, layer) in prover.layers.iter().chain(iter::once(&sums)).enumerate() {
			// Entry i of layer j is the fractional sum of the 2^j witness values strided by that
			// layer's own width (strided access, not contiguous).
			let width = 1 << (n + k - j);
			let num_terms = 1 << j;
			for i in 0..width {
				let mut expected_num = witness_num.get(i);
				let mut expected_den = witness_den.get(i);
				for z in 1..num_terms {
					let idx = i + z * width;
					let num_z = witness_num.get(idx);
					let den_z = witness_den.get(idx);
					expected_num = expected_num * den_z + num_z * expected_den;
					expected_den *= den_z;
				}
				let actual_num = layer.num.get(i);
				let actual_den = layer.den.get(i);
				assert_eq!(actual_num, expected_num, "layer {j} numerator mismatch at index {i}");
				assert_eq!(actual_den, expected_den, "layer {j} denominator mismatch at index {i}");
			}
		}
	}

	proptest! {
		// Invariant: every layer of the circuit is the fractional-addition fold of the witness.
		//
		// Pinning each layer to that fold pins the sibling recurrence the layers are built from.
		// Only an end-to-end proof failure notices if `new` folds the wrong pairs.
		#[test]
		fn frac_add_check_layers_fold_the_witness(
			seed in any::<u64>(),
			n in 0usize..=4,
			k in 0usize..=4,
		) {
			check_all_layers::<Packed128b>(n, k, seed);
		}
	}

	// ==================== batch_prove_unequal_depths tests ====================

	/// A numerator/denominator witness pair.
	type Witness<P> = Fraction<FieldBuffer<P>>;

	/// One prover per entry of `depths`, each reducing over all of its witness variables.
	#[allow(clippy::type_complexity)]
	fn unequal_depth_provers<'a, P: PackedField>(
		rng: &mut impl rand::Rng,
		alloc: &'a GlobalAllocator,
		depths: &[usize],
	) -> (
		Vec<Witness<P>>,
		Vec<FracAddCheckProver<'a, GlobalAllocator, P>>,
		Vec<Fraction<P::Scalar>>,
	) {
		itertools::multiunzip(depths.iter().map(|&depth| {
			let witness = Fraction::new(
				random_field_buffer::<P>(&mut *rng, depth),
				random_field_buffer::<P>(&mut *rng, depth),
			);
			let (prover, sums) = FracAddCheckProver::new(depth, alloc, witness.clone());
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
		let eq_weights = OneCube::eq_ind_partial_eval::<P>(selector_point);
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
			assert_eq!(leaf.num_eval, evaluate(&witness.num, &leaf.point), "tree {i} numerator");
			assert_eq!(leaf.den_eval, evaluate(&witness.den, &leaf.point), "tree {i} denominator");
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

	proptest! {
		// Invariant: `unpad_leaf_claim` is the exact inverse of `pad_leaf_fraction`.
		//
		// The verifier pads a transparent leaf fraction, the prover unpads the claim it gets back.
		// Only an end-to-end proof failure notices if either map drifts from the other.
		#[test]
		fn unpad_leaf_claim_inverts_pad_leaf_fraction(
			seed in any::<u64>(),
			n_pad_vars in 0usize..=5,
			n_real_vars in 0usize..=5,
		) {
			let mut rng = StdRng::seed_from_u64(seed);

			// Splitting the point's length in two keeps `n_pad_vars <= point.len()` by construction.
			let point = random_scalars::<F>(&mut rng, n_pad_vars + n_real_vars);
			let halves = random_scalars::<F>(&mut rng, 2);
			let fraction = Fraction::new(halves[0], halves[1]);

			let pad_eq = point[..n_pad_vars]
				.iter()
				.map(|&coord| OneCube::eq_one_var(F::ZERO, coord))
				.product::<F>();
			// Unpadding asserts on a zero weight, which needs a padding coordinate equal to one.
			// Random 128-bit coordinates never are, so this rejects nothing.
			prop_assume!(pad_eq != F::ZERO);

			let padded = fracaddcheck::pad_leaf_fraction(fraction.into(), pad_eq);
			let claim = unpad_leaf_claim(padded.into(), &point, n_pad_vars);

			prop_assert_eq!(claim.num_eval, fraction.num);
			prop_assert_eq!(claim.den_eval, fraction.den);
			// The padding variables are the lowest ones, so unpadding strips them off the point.
			prop_assert_eq!(claim.point, point[n_pad_vars..].to_vec());
		}
	}
}
