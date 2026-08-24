// Copyright 2025-2026 The Binius Developers

use std::iter;

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::{fracaddcheck::FracAddEvalClaim, mlecheck, sumcheck::RoundCoeffs};
use binius_math::{
	FieldBuffer, FieldVec,
	batch_invert::BatchInversion,
	line::extrapolate_line,
	multilinear::eq::{eq_ind_partial_eval, eq_one_var},
};
use binius_utils::{
	buffer::VecLike,
	rayon::{
		iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};
use either::Either;
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
pub mod zero_pad_mle;

use fraction::Fraction;
use zero_pad_mle::{ConstantFraction, ZeroPadMleCheckProver};

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
	/// * `claim` - The initial multilinear evaluation claims (numerator, denominator)
	/// * `channel` - The channel for sending prover messages and sampling challenges
	///
	/// # Preconditions
	/// * `claim.0.point.len() == witness.log_len() - k` (where k is the number of reduction layers)
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
	/// * the circuit, holding whatever layers were not proved,
	/// * the reduced numerator/denominator claims after `n_layers` layers.
	///
	/// # Preconditions
	/// * `n_layers <= self.n_layers()`.
	pub fn prove_layers(
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
/// Folds the content variables of every instance in lockstep (eq(selector)-weighted, `[num, den]`-
/// batched), then the `k` selector variables via a single fractional-addition MLE-check over the
/// packed reduced halves, then the doubling line-fold. Returns the reduced per-instance fractions
/// (padded to `2^k` with zeros) and the next evaluation point.
///
/// The two (numerator, denominator) claims of every layer are batched via a single `batch_coeff`
/// that the verifier's `batch_verify_mle` samples once per layer, before the round polynomials; the
/// same coefficient is reused for the content and selector rounds.
fn reduce_layer<'a, A, F, P, MP>(
	alloc: &'a A,
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
	let eq_weights = eq_ind_partial_eval::<F>(outer_coords);

	// The padding slots beyond the real instances hold the constant fraction 0/1: the numerator
	// is the constant 0 function and the denominator the constant 1 function. A constant
	// composition has the constant prime round polynomial (0 for the numerator, 1 for the
	// denominator) and its claims stay (0, 1) through every fold, so the padding contributes
	// eq_i * batch_coeff to each batched round polynomial's constant coefficient.
	let pad_eq_sum: F = eq_weights.iter_scalars().skip(layer_provers.len()).sum();

	let batch_coeff = channel.sample();

	let mut challenges = Vec::with_capacity(eval_point.len());

	// Content rounds: fold the content variables of every instance in lockstep, sending the
	// eq(selector)-weighted sum of the per-instance (num, den)-batched round polynomials.
	for _round in 0..inner_coords.len() {
		// The instances are independent within a round, so their polynomials compute in parallel.
		//
		// One instance's round is too small a parallel region to fill the pool alone.
		let per_instance: Vec<RoundCoeffs<F>> = layer_provers
			.par_iter_mut()
			.map(|prover| combine_claims(prover.execute(), batch_coeff))
			.collect();

		// Weight instance j's polynomial by eq_j and sum, in instance order.
		let real_coeffs: RoundCoeffs<F> = iter::zip(per_instance, eq_weights.iter_scalars())
			.map(|(coeffs, eq_i)| coeffs * eq_i)
			.sum();
		let round_coeffs = real_coeffs + &RoundCoeffs(vec![pad_eq_sum * batch_coeff]);

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

	// Split the reduced halves into per-multilinear vectors, padded to 2^k so they can be packed
	// into selector-variable buffers. Both children of a padding slot are the zero fraction, which
	// is what makes the numerators fill with 0 and the denominators with 1.
	let mut num_0s: Vec<F> = finished.iter().map(|e| e[0]).collect();
	let mut num_1s: Vec<F> = finished.iter().map(|e| e[1]).collect();
	let mut den_0s: Vec<F> = finished.iter().map(|e| e[2]).collect();
	let mut den_1s: Vec<F> = finished.iter().map(|e| e[3]).collect();
	let pad = Fraction::<F>::ZERO;
	num_0s.resize(1 << k, pad.num);
	num_1s.resize(1 << k, pad.num);
	den_0s.resize(1 << k, pad.den);
	den_1s.resize(1 << k, pad.den);

	// The selector claim is the eq(selector)-weighted sum of the fractional-addition composition of
	// the reduced halves.
	let num_eval: F = izip!(&num_0s, &num_1s, &den_0s, &den_1s, eq_weights.as_ref())
		.map(|(&n0, &n1, &d0, &d1, &eq_i)| eq_i * (n0 * d1 + n1 * d0))
		.sum();
	let den_eval: F = izip!(&den_0s, &den_1s, eq_weights.as_ref())
		.map(|(&d0, &d1, &eq_i)| eq_i * (d0 * d1))
		.sum();

	// Selector rounds: fold the selector variables with a single fractional-addition MLE-check over
	// the packed reduced halves, reusing the same `batch_coeff`. The reduced halves are freshly
	// packed, so the store owns them directly.
	let mut selector_store = MleStore::new(k, alloc);
	let selector_cols = [
		FieldBuffer::from_values_in(alloc, &num_0s),
		FieldBuffer::from_values_in(alloc, &num_1s),
		FieldBuffer::from_values_in(alloc, &den_0s),
		FieldBuffer::from_values_in(alloc, &den_1s),
	]
	.map(|buffer| selector_store.push_owned(buffer));
	let (selector_num, selector_den) = frac_add_mle::evaluators::<F, P>(selector_cols);
	let selector_claims_with_evaluators: [(F, Box<dyn MleCheckRoundEvaluator<F, P> + 'a>); 2] = [
		(num_eval, Box::new(selector_num)),
		(den_eval, Box::new(selector_den)),
	];
	let mut selector_prover = SharedMleCheckProver::new(
		selector_store,
		selector_claims_with_evaluators,
		outer_coords.to_vec(),
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
	// aligned with the selector `eq` weights on subsequent layers; the padded entries are 0/1.
	let next_fractions = izip!(&num_0s, &num_1s, &den_0s, &den_1s)
		.map(|(&num_0, &num_1, &den_0, &den_1)| {
			Fraction::new(extrapolate_line(num_0, num_1, r), extrapolate_line(den_0, den_1, r))
		})
		.collect();

	(next_fractions, next_point)
}

/// Runs a batched fractional-addition check for trees of *unequal* depths.
///
/// Every tree shallower than the deepest is proved as a fracadd check over its witness padded with
/// zero fractions — the same witness with the extra depth filled by $0/1$ leaves, which leaves its
/// fractional sum unchanged. The transcript is then exactly that of an equal-depth batch of the
/// maximum depth: the verifier runs the ordinary [`binius_ip::fracaddcheck::verify`] over
/// `n_layers` layers and never learns the individual depths.
///
/// Every prover must reduce over *all* of its witness variables, so each fractional sum is a
/// scalar and there is no content point.
/// Dropping the content dimension keeps the padding bookkeeping to four scalars per layer.
///
/// The prover does not materialize the padded witnesses. Each layer's per-tree reduction runs
/// through [`zero_pad_mle`], which corrects the unpadded layer's messages in $O(1)$ per round.
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
/// [`unpad_leaf_claim`] reduces one to the claims on the tree's own
/// witness, given how much depth that tree was padded by.
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

/// The per-tree layer prover: either a real layer of the tree, or the padding layer it contributes
/// while the batch is still above it.
type PaddedLayerProver<'a, A, F, P> =
	ZeroPadMleCheckProver<F, Either<LayerProver<'a, A, F, P>, ConstantFraction<F>>>;

/// Builds one padded layer prover per tree, for the layer claimed at `node_point`.
///
/// The provers come back in input order, one per tree, so they stay aligned with `pad_lens` and
/// `claims`. A tree the batch has not reached yet keeps every layer it has.
fn layer_provers<'a, A, F, P>(
	provers: &mut [FracAddCheckProver<'a, A, P>],
	pad_lens: &[usize],
	claims: &[Fraction<F>],
	node_point: &[F],
) -> Vec<PaddedLayerProver<'a, A, F, P>>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	let node_len = node_point.len();

	// Every tree's padding segment is a prefix of this one node point, so a single table of prefix
	// products serves the whole batch.
	let pad_eq_prefixes = iter::once(F::ONE)
		.chain(node_point.iter().scan(F::ONE, |acc, &coord| {
			*acc *= eq_one_var(F::ZERO, coord);
			Some(*acc)
		}))
		.collect::<Vec<_>>();

	// De-padding a claim divides by the padding segment's equality weight, so the batch pays one
	// inversion rather than one per tree.
	let mut pad_eq_invs = pad_lens
		.iter()
		.map(|&pad_len| pad_eq_prefixes[pad_len.min(node_len)])
		.collect::<Vec<_>>();
	assert!(
		pad_eq_invs.iter().all(|&pad_eq| pad_eq != F::ZERO),
		"a padding coordinate of the claim point equals one"
	);
	BatchInversion::<F>::new(pad_eq_invs.len()).invert_nonzero(&mut pad_eq_invs);

	izip!(provers, pad_lens, claims, &pad_eq_invs)
		.map(|(prover, &tree_pad_len, &Fraction { num, den }, &pad_eq_inv)| {
			let pad_len = tree_pad_len.min(node_len);
			let point = node_point[pad_len..].to_vec();
			let [num_claim, den_claim] = zero_pad_mle::unpad_claims(pad_eq_inv, [num, den]);

			let inner = if node_len < tree_pad_len {
				// The batch is still above this tree, so every variable of its layer is a padding
				// variable and the de-padded claim is the tree's own fractional sum. The layer is
				// that fraction beside the zero fraction 0/1, and the tree keeps all of its layers.
				Either::Right(ConstantFraction::new(num_claim, den_claim))
			} else {
				Either::Left(prover.pop_layer(FracAddEvalClaim {
					num_eval: num_claim,
					den_eval: den_claim,
					point,
				}))
			};

			zero_pad_mle::new(pad_eq_prefixes[..=pad_len].to_vec(), node_point.to_vec(), inner)
		})
		.collect()
}

/// Reduces a leaf claim on a zero-fraction-padded witness to the claim on the witness itself.
///
/// A batched fractional-addition check over trees of unequal depths lifts each shallow tree to the
/// batch's depth by filling `n_pad_vars` extra leaf positions with the zero fraction $0/1$, which
/// leaves its fractional sum unchanged. [`binius_ip::fracaddcheck::verify`] is oblivious to that,
/// so the claims it outputs for such a tree are claims on the padded witness
///
/// $$
/// N'(X_\text{pad}, X_\text{real}) = N(X_\text{real}) \cdot \text{eq}(0^\nu; X_\text{pad}),
/// \qquad
/// D'(X_\text{pad}, X_\text{real}) = 1 + \bigl( D(X_\text{real}) - 1 \bigr) \cdot
/// \text{eq}(0^\nu; X_\text{pad}),
/// $$
///
/// whose padding variables are the lowest ones. This divides out their equality weight and drops
/// them from the point, leaving the claims on $N$ and $D$.
///
/// # Arguments
///
/// * `fraction` - The claimed numerator and denominator evaluations of the padded witness.
/// * `point` - The reduced evaluation point, with the batch's selector coordinates already
///   stripped.
/// * `n_pad_vars` - How much depth this tree was padded by: the batch's layer count less the tree's
///   own.
///
/// # Preconditions
/// * `point.len() >= n_pad_vars`
///
/// # Panics
///
/// Panics if the padding coordinates' equality weight is zero, which requires one of them to equal
/// one. They are the verifier's own challenges, so no prover can induce this; it happens with
/// probability at most $\nu / |K|$.
pub fn unpad_leaf_claim<F: Field>(
	fraction: Fraction<F>,
	point: &[F],
	n_pad_vars: usize,
) -> FracAddEvalClaim<F> {
	assert!(point.len() >= n_pad_vars); // precondition

	let pad_eq = point[..n_pad_vars]
		.iter()
		.map(|&coord| eq_one_var(F::ZERO, coord))
		.product::<F>();
	assert!(pad_eq != F::ZERO, "a padding coordinate equals one");
	let pad_eq_inv = pad_eq.invert_or_zero();

	let Fraction {
		num: num_eval,
		den: den_eval,
	} = fraction;
	FracAddEvalClaim {
		num_eval: num_eval * pad_eq_inv,
		den_eval: F::ONE + (den_eval - F::ONE) * pad_eq_inv,
		point: point[n_pad_vars..].to_vec(),
	}
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

		// 4. Evaluate sums at challenge point to createzz claims
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

	fn test_frac_add_check_layer_computation_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// Create prover (computes fractional-add layers)
		let (_prover, sums) = FracAddCheckProver::new(
			k,
			&alloc,
			Fraction::new(witness_num.clone(), witness_den.clone()),
		);

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
			let actual_num = sums.num.get(i);
			let actual_den = sums.den.get(i);
			assert_eq!(actual_num, expected_num, "Numerator mismatch at index {i}");
			assert_eq!(actual_den, expected_den, "Denominator mismatch at index {i}");
		}
	}

	#[test]
	fn test_frac_add_check_layer_computation() {
		test_frac_add_check_layer_computation_helper::<Packed128b>(4, 3);
	}

	// ==================== batch_prove_unequal_depths tests ====================

	/// A numerator/denominator witness pair.
	type Witness<P> = Fraction<FieldBuffer<P>>;

	/// One prover per entry of `depths`, each reducing over all of its witness variables.
	#[allow(clippy::type_complexity)]
	fn unequal_depth_provers<'a, P: PackedField>(
		rng: &mut impl Rng,
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
		let eq_weights = eq_ind_partial_eval::<P>(selector_point);
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
	fn test_unequal_depths_helper<P: PackedField>(depths: &[usize]) {
		let mut rng = StdRng::seed_from_u64(11);
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
		test_unequal_depths_helper::<Packed128b>(&[2, 4, 5]);
	}

	#[test]
	fn test_unequal_depths_single_prover() {
		test_unequal_depths_helper::<Packed128b>(&[3]);
	}

	#[test]
	fn test_unequal_depths_power_of_two_provers() {
		// The shallowest tree is padded by more than one layer, the deepest not at all.
		test_unequal_depths_helper::<Packed128b>(&[1, 2, 5, 5]);
	}

	#[test]
	fn test_unequal_depths_all_minimal() {
		// Depth 1 throughout: every tree retains its final layer immediately.
		test_unequal_depths_helper::<Packed128b>(&[1, 1, 1]);
	}

	#[test]
	fn test_unequal_depths_zero_depth_tree() {
		// A depth-0 tree never pops a layer: it is all padding, so its leaf claim is its root.
		test_unequal_depths_helper::<Packed128b>(&[0, 3]);
	}

	#[test]
	fn test_unequal_depths_maximal_padding() {
		// A single-layer tree beside a deep one: all but its last reduction is padding.
		test_unequal_depths_helper::<Packed128b>(&[1, 6]);
	}

	#[test]
	fn test_unequal_depths_equal_depths() {
		// Equal depths pad nothing, so every wrapper is a pass-through.
		test_unequal_depths_helper::<Packed128b>(&[4, 4, 4]);
	}
}
