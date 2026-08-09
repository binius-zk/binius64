// Copyright 2026 The Binius Developers

//! Round evaluators over a shared [`MleStore`] and the provers that drive them.
//!
//! A round evaluator holds the per-round-polynomial logic for one composite claim over store
//! columns. Evaluators hold [`ColId`](super::mle_store::ColId)s and receive column data by
//! argument; they hold no mutable
//! per-round state — they neither fold nor track the round claim. The driving prover owns the
//! `RoundState` machine (the claim ↔ coeffs alternation) for each evaluator, and the store folds
//! the columns (see the [`mle_store`](super::mle_store) module documentation).
//!
//! There are two evaluator traits, one per protocol:
//! - [`SumcheckRoundEvaluator`], driven by [`SharedSumcheckProver`], emits regular sumcheck round
//!   polynomials.
//! - [`MleCheckRoundEvaluator`], driven by [`SharedMleCheckProver`], emits the eq-factored prime
//!   round polynomials of the MLE-check protocol. Its claims all share one evaluation point, so the
//!   prover — not the evaluator — owns that point's eq-indicator tracker and passes the round's eq
//!   chunk and coordinate to the evaluator.
//!
//! Both provers adapt a store plus a list of evaluators — one per claim — to the [`SumcheckProver`]
//! interface (and [`SharedMleCheckProver`] additionally to [`MleCheckProver`]), so they batch
//! alongside standalone provers unchanged.

use std::iter;

use auto_impl::auto_impl;
use binius_compute::Allocator;
use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldSlice, multilinear::eq::eq_ind_partial_eval};

use super::{
	MleToSumCheckEvaluator,
	common::{MleCheckProver, SumcheckProver},
	mle_store::{EvaluationChunk, MleStore, RoundContext},
	round_state::RoundState,
};

/// Per-round-polynomial logic for one plain sumcheck claim over store columns.
///
/// The driving [`SharedSumcheckProver`] makes one parallel pass over column chunks per round; the
/// hot loops stay monomorphized inside each evaluator and only the per-chunk [`Self::accumulate`]
/// entry is virtual. Within a round the calls are: [`Self::accumulate`] from parallel workers into
/// per-worker accumulator slices, then [`Self::interpolate`] once on the slot-wise summed slice.
/// The driving prover, not the evaluator, holds the round claim and reduces the round polynomial
/// against the verifier challenge.
///
/// The driving prover owns the accumulator and sizes it from the degree.
/// - Accumulation writes wide, unreduced slots, so a per-chunk sum costs no reduction.
/// - The prover sums the workers' slices slot-wise, then reduces once per round.
/// - Interpolation reads the reduced slots.
///
/// An evaluator implements neither the allocation nor the merging.
/// It only writes its own slots and interpolates them.
/// The slot layout within one evaluator's run is private to it.
///
/// Evaluators are stateless across rounds: they hold no round claim and never fold. The prover
/// passes the round claim into [`Self::interpolate`] and recovers it back out of the emitted round
/// polynomial itself (as $R(0) + R(1)$), so the whole claim ↔ coeffs state machine lives in the
/// prover's `RoundState`.
///
/// The `auto_impl(Box)` derive forwards the trait through `Box`, so a heterogeneous group of
/// evaluators can drive a shared prover as `Vec<Box<dyn SumcheckRoundEvaluator<F, P>>>` while a
/// homogeneous group avoids boxing.
#[auto_impl(Box)]
pub trait SumcheckRoundEvaluator<F: Field, P: PackedField<Scalar = F>>: Send + Sync {
	/// The number of accumulator slots this evaluator's claim uses.
	///
	/// This is the count of sampled round-polynomial evaluations the accumulation pass collects;
	/// the remaining evaluation is recovered from the round's sum claim in [`Self::interpolate`].
	/// The driving prover reserves this many slots for the evaluator. It is the degree of the
	/// accumulated (prime/composite) polynomial, which for an eq-factored MLE-check evaluator is
	/// the prime degree, not the emitted round-polynomial degree.
	fn degree(&self) -> usize;

	/// Accumulates one chunk of the halved hypercube into `accum`.
	///
	/// The driving prover prepares `chunk` — the split, per-chunk column halves and eq-indicator
	/// expansions — so the evaluator only reads its columns by [`ColId`](super::mle_store::ColId)
	/// and eq trackers by
	/// [`EqId`](super::mle_store::EqId). `accum` is this evaluator's run of [`Self::degree`] wide
	/// slots, zero-initialized
	/// on the first chunk and carried across the worker's chunks.
	fn accumulate(&self, chunk: &EvaluationChunk<'_, P>, accum: &mut [<P as WideMul>::Output]);

	/// Interpolates this round's polynomial from the accumulator and the round claim.
	///
	/// # Arguments
	///
	/// * `ctx` - This round's scalar state: the unbound-variable count, and a registered tracker's
	///   equality coordinate and prefix.
	/// * `accum` - This evaluator's slots, summed across every worker and reduced.
	/// * `claim` - This evaluator's round claim.
	fn interpolate(&self, ctx: &RoundContext<'_, P>, accum: &[P], claim: F) -> RoundCoeffs<F>;
}

/// Per-round-polynomial logic for one MLE-check claim over store columns.
///
/// This is the MLE-check counterpart of [`SumcheckRoundEvaluator`].
/// It emits the prime, equality-factored round polynomials of the MLE-check protocol.
///
/// Every claim of one such prover shares a single evaluation point.
/// The prover, not the evaluator, owns that point's equality indicator.
/// Two arguments follow from that:
/// - Accumulation receives the round's equality-indicator chunk.
/// - Interpolation receives the round's equality coordinate.
///
/// So the evaluator stores no tracker identifier and no copy of the point.
/// The accumulator contract is the one above: wide slots in, reduced slots out.
#[auto_impl(Box)]
pub trait MleCheckRoundEvaluator<F: Field, P: PackedField<Scalar = F>>: Send + Sync {
	/// The number of accumulator slots this evaluator's claim uses. See
	/// [`SumcheckRoundEvaluator::degree`].
	fn degree(&self) -> usize;

	/// Accumulates one chunk of the halved hypercube into `accum`.
	///
	/// `eq_ind` is the round's eq-indicator chunk (the prover looks it up once and hands it to
	/// every evaluator); the evaluator weights its composition by it. Otherwise as
	/// [`SumcheckRoundEvaluator::accumulate`].
	fn accumulate(
		&self,
		chunk: &EvaluationChunk<'_, P>,
		eq_ind: FieldSlice<'_, P>,
		accum: &mut [<P as WideMul>::Output],
	);

	/// Interpolates this round's prime polynomial from the accumulator and the round claim.
	///
	/// As [`SumcheckRoundEvaluator::interpolate`], with one difference.
	/// The round's equality coordinate arrives as an argument rather than from a tracker.
	/// So `ctx` supplies only the unbound-variable count.
	fn interpolate(
		&self,
		ctx: &RoundContext<'_, P>,
		accum: &[P],
		claim: F,
		alpha: F,
	) -> RoundCoeffs<F>;
}

/// Maximum log2 chunk size of the parallel round pass.
///
/// Chunked accumulation keeps the equality-indicator chunk resident in L1 cache while all
/// evaluators read it, mirroring the chunking of the pre-store quadratic prover.
const MAX_CHUNK_VARS: usize = 12;

/// The state a group of claims shares, whichever protocol drives them.
///
/// One store of columns, one evaluator per claim, and each claim's position in the round cycle.
/// Each prover below is this, plus the round pass its protocol needs.
struct EvaluatorGroup<'a, A: Allocator, F: Field, P: PackedField<Scalar = F>, Evaluator> {
	store: MleStore<'a, A, P>,
	evaluators: Vec<Evaluator>,
	/// Each claim's position in the claim-to-coefficients cycle, parallel to the evaluators.
	///
	/// A claim is held directly until its round polynomial is produced.
	/// The coefficients are held afterwards, until the fold reduces them back to a claim.
	round_states: Vec<RoundState<RoundCoeffs<F>, F>>,
	/// A fold challenge whose store fold waits for the next round's read pass.
	///
	/// Deferring it lets the columns be touched once per round instead of twice.
	/// Only the store's fold waits here, as the round claims advance as soon as the challenge
	/// lands.
	buffered_challenge: Option<F>,
}

impl<'a, A, F, P, Evaluator> EvaluatorGroup<'a, A, F, P, Evaluator>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Creates a group from a store and the evaluators reading it, each with its initial claim.
	fn new(
		store: MleStore<'a, A, P>,
		claims_with_evaluators: impl IntoIterator<Item = (F, Evaluator)>,
	) -> Self {
		// Unzipping lets a caller pass an array of pairs rather than two parallel vectors.
		let (round_states, evaluators) = claims_with_evaluators
			.into_iter()
			.map(|(claim, evaluator)| (RoundState::Claim(claim), evaluator))
			.unzip();
		Self {
			store,
			evaluators,
			round_states,
			buffered_challenge: None,
		}
	}

	/// Returns the number of variables still to bind.
	///
	/// A buffered challenge is a fold the store has not seen yet.
	/// So the count sits one below the store's until the next round applies it.
	const fn n_vars(&self) -> usize {
		self.store.n_vars() - self.buffered_challenge.is_some() as usize
	}

	/// Adds one more claim, reading the same store, after the existing ones.
	fn push_claim(&mut self, claim: F, evaluator: Evaluator) {
		self.evaluators.push(evaluator);
		self.round_states.push(RoundState::Claim(claim));
	}

	/// Interpolates every claim's round polynomial, then records it for the coming fold.
	///
	/// The group serves both evaluator traits, so each protocol passes its own accessors in.
	///
	/// # Arguments
	///
	/// * `accum` - Every evaluator's slots, summed across workers and reduced.
	/// * `degree` - One evaluator's slot count.
	/// * `interpolate` - The protocol's call into one evaluator.
	fn interpolate_round(
		&mut self,
		accum: &[P],
		degree: impl Fn(&Evaluator) -> usize,
		interpolate: impl Fn(&Evaluator, &RoundContext<'_, P>, &[P], F) -> RoundCoeffs<F>,
	) -> Vec<RoundCoeffs<F>> {
		// The store has not folded this round, so its view carries this round's coordinates.
		let ctx = self.store.round_context();
		// Evaluators own consecutive runs of the accumulator, so one walk hands out every run.
		let mut rest = accum;
		let mut round_coeffs = Vec::with_capacity(self.evaluators.len());
		for (evaluator, state) in iter::zip(&self.evaluators, &self.round_states) {
			let (slots, tail) = rest.split_at(degree(evaluator));
			rest = tail;
			round_coeffs.push(interpolate(evaluator, &ctx, slots, *state.claim()));
		}
		debug_assert!(rest.is_empty(), "the runs must tile the accumulator exactly");
		// The coefficients become the round state the coming fold reduces.
		for (state, coeffs) in iter::zip(&mut self.round_states, &round_coeffs) {
			*state = RoundState::Coeffs(coeffs.clone());
		}
		round_coeffs
	}

	/// Reduces every round polynomial against the challenge, forming the next round's claims.
	///
	/// The store's own fold is deferred to the next round's read pass.
	fn fold(&mut self, challenge: F) {
		for state in &mut self.round_states {
			let claim = state.coeffs().evaluate(&challenge);
			*state = RoundState::Claim(claim);
		}
		debug_assert!(
			self.buffered_challenge.is_none(),
			"fold called twice without an intervening execute"
		);
		self.buffered_challenge = Some(challenge);
	}

	/// Applies any deferred fold, bringing the store's columns up to the claims' round.
	fn flush_fold(&mut self) {
		if let Some(challenge) = self.buffered_challenge.take() {
			self.store.fold(challenge);
		}
	}

	/// Applies any deferred fold, then emits every column's evaluation in push order.
	///
	/// The store owns each column once, so each evaluation is computed once however many claims
	/// read that column.
	fn finish(mut self) -> Vec<F> {
		self.flush_fold();
		self.store.final_evals()
	}

	/// Rebuilds the group with every evaluator replaced, carrying the store and claims across.
	fn map_evaluators<New>(
		self,
		wrap: impl FnMut(Evaluator) -> New,
	) -> EvaluatorGroup<'a, A, F, P, New> {
		EvaluatorGroup {
			store: self.store,
			evaluators: self.evaluators.into_iter().map(wrap).collect(),
			round_states: self.round_states,
			buffered_challenge: self.buffered_challenge,
		}
	}
}

/// A [`SumcheckProver`] over a shared [`MleStore`] and a list of [`SumcheckRoundEvaluator`]s, one
/// per claim.
///
/// Each round makes one parallel pass over the store's column chunks, feeding every evaluator,
/// and lists the round polynomials in evaluator registration order. [`Self::fold`] folds each
/// shared column and eq tracker once. [`Self::finish`] emits each store column's evaluation once,
/// computed a single time by the store no matter how many claims read the column.
pub struct SharedSumcheckProver<'a, A: Allocator, P: PackedField, Evaluator> {
	group: EvaluatorGroup<'a, A, P::Scalar, P, Evaluator>,
}

impl<'a, A, F, P, Evaluator> SharedSumcheckProver<'a, A, P, Evaluator>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: SumcheckRoundEvaluator<F, P>,
{
	/// Creates a prover from a store and the evaluators reading its columns, each paired with its
	/// initial claim — one `(claim, evaluator)` per claim.
	pub fn new(
		store: MleStore<'a, A, P>,
		claims_with_evaluators: impl IntoIterator<Item = (F, Evaluator)>,
	) -> Self {
		Self {
			group: EvaluatorGroup::new(store, claims_with_evaluators),
		}
	}

	/// Returns a shared reference to the underlying column store.
	pub const fn store(&self) -> &MleStore<'a, A, P> {
		&self.group.store
	}

	/// Returns an exclusive reference to the underlying column store.
	///
	/// Lets a caller extend the shared store with columns that a later-added evaluator reads: the
	/// logUp* final layer pushes the table halves onto it before adding its product evaluators.
	pub const fn store_mut(&mut self) -> &mut MleStore<'a, A, P> {
		&mut self.group.store
	}

	/// Adds one more evaluator — a claim reading the shared store, with its initial claim — to the
	/// group.
	///
	/// Its round polynomial is appended after the existing evaluators' in [`Self::execute`].
	pub fn add_evaluator(&mut self, claim: F, evaluator: Evaluator) {
		self.group.push_claim(claim, evaluator);
	}
}

impl<A, F, P, Evaluator> SumcheckProver<F> for SharedSumcheckProver<'_, A, P, Evaluator>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: SumcheckRoundEvaluator<F, P>,
{
	fn n_vars(&self) -> usize {
		self.group.n_vars()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let n_vars_remaining = self.group.n_vars();
		assert!(n_vars_remaining > 0);

		// One parallel pass over the halved hypercube feeds every evaluator, so shared columns
		// and eq-indicator chunks are read once per round while they are cache-resident.
		//
		// TODO: dynamically choose chunk size based on the number of columns and P byte-size,
		// based on estimated L1 cache size.
		let chunk_vars = (n_vars_remaining - 1).min(MAX_CHUNK_VARS.max(P::LOG_WIDTH));

		// Each evaluator owns a contiguous run of `degree` wide slots in one flat per-worker
		// buffer, laid out in registration order.
		let total_slots: usize = self
			.group
			.evaluators
			.iter()
			.map(|evaluator| evaluator.degree())
			.sum();

		// The store prepares one `EvaluationChunk` per chunk of the halved hypercube — the split
		// column halves and eq-indicator expansions each evaluator reads.
		let group = &mut self.group;
		let buffered_challenge = group.buffered_challenge.take();
		let evaluators = &group.evaluators;
		let store = &mut group.store;
		let map = |chunk: EvaluationChunk<'_, P>| {
			let mut accum = vec![Default::default(); total_slots];
			// One walk hands each evaluator its own run, so no offset table is built per round.
			let mut rest = accum.as_mut_slice();
			for evaluator in evaluators {
				let (slots, tail) = rest.split_at_mut(evaluator.degree());
				rest = tail;
				evaluator.accumulate(&chunk, slots);
			}
			debug_assert!(rest.is_empty(), "the runs must tile the accumulator exactly");
			accum
		};
		let reduce = |mut lhs: Vec<<P as WideMul>::Output>,
		              rhs: Vec<<P as WideMul>::Output>,
		              _level: usize| {
			// The only merge: sum the workers' slices slot-wise, generic over every evaluator.
			// Plain sumcheck has no eq factor, so the reduction level is unused.
			for (dst, src) in iter::zip(&mut lhs, rhs) {
				*dst += src;
			}
			lhs
		};
		let accum = match buffered_challenge {
			// The previous fold deferred its store fold; apply it and this round's read in one
			// pass.
			Some(challenge) => store.map_reduce_with_fold(chunk_vars, challenge, map, reduce),
			None => store.map_reduce(chunk_vars, map, reduce),
		};

		// The workers' wide sums are complete, so every slot is reduced once for the whole round.
		// Interpolation runs on reduced values.
		let accum = accum.into_iter().map(P::reduce).collect::<Vec<P>>();

		self.group.interpolate_round(
			&accum,
			|evaluator| evaluator.degree(),
			|evaluator, ctx, slots, claim| evaluator.interpolate(ctx, slots, claim),
		)
	}

	fn fold(&mut self, challenge: F) {
		self.group.fold(challenge);
	}

	fn finish(self) -> Vec<F> {
		self.group.finish()
	}
}

/// A [`MleCheckProver`] over a shared [`MleStore`] and a group of [`MleCheckRoundEvaluator`]s.
///
/// This is the MLE-check counterpart of [`SharedSumcheckProver`]: every evaluator's claim shares
/// the prover's evaluation point, and the round polynomials are the eq-factored prime polynomials
/// of the MLE-check protocol. Because the point is shared, the prover owns its eq-indicator tracker
/// and hands the round's eq chunk and coordinate to the evaluators, which therefore store no
/// tracker id.
pub struct SharedMleCheckProver<'a, A: Allocator, F: Field, P: PackedField<Scalar = F>, Evaluator> {
	group: EvaluatorGroup<'a, A, F, P, Evaluator>,
	/// The evaluation point every claim shares.
	///
	/// No eq-indicator tracker is registered for it.
	/// Each round expands only a chunk-wide prefix of this point and folds the higher coordinates
	/// through the eq-weighted reduction.
	eval_point: Vec<F>,
}

impl<'a, A, F, P, Evaluator> SharedMleCheckProver<'a, A, F, P, Evaluator>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: MleCheckRoundEvaluator<F, P>,
{
	/// Creates a prover from a store, the evaluators reading its columns each paired with its
	/// initial claim — one `(claim, evaluator)` per claim — and the evaluation point shared by all
	/// of the evaluators' claims.
	pub fn new(
		store: MleStore<'a, A, P>,
		claims_with_evaluators: impl IntoIterator<Item = (F, Evaluator)>,
		eval_point: Vec<F>,
	) -> Self {
		// precondition
		assert_eq!(
			eval_point.len(),
			store.n_vars(),
			"evaluation point length must equal the store's number of variables"
		);
		Self {
			group: EvaluatorGroup::new(store, claims_with_evaluators),
			eval_point,
		}
	}

	/// Returns the underlying column store, with any deferred fold applied first.
	///
	/// Lets a caller read the columns partway through the reduction, where they hold the
	/// multilinears specialized at every challenge bound so far: the padded product-check layer
	/// reads out its two children once their node variables are bound. Applying the deferred fold
	/// is what makes the columns match the rounds already run, at the cost of the pass over them
	/// that the next round would otherwise have fused in.
	pub fn store(&mut self) -> &MleStore<'a, A, P> {
		self.group.flush_fold();
		&self.group.store
	}

	/// Converts this MLE-check prover into a plain [`SharedSumcheckProver`] by folding each claim's
	/// equality factor into its emitted round polynomials — the [Gruen24] technique of
	/// [`MleToSumCheckEvaluator`].
	///
	/// This lets the eq-weighted fractional claims batch in one evaluator group alongside plain
	/// sumcheck claims over the same store: the logUp* final layer converts the popped table
	/// layer's prover, then adds its product evaluators to it. The store and its columns carry over
	/// untouched; only the evaluators are wrapped, sharing this prover's eq tracker.
	///
	/// [Gruen24]: <https://eprint.iacr.org/2024/108>
	pub fn into_shared_sumcheck(
		self,
	) -> SharedSumcheckProver<'a, A, P, Box<dyn SumcheckRoundEvaluator<F, P> + 'a>>
	where
		Evaluator: 'a,
	{
		let Self {
			mut group,
			eval_point,
		} = self;
		// The plain sumcheck prover this converts to has no eq machinery, so its wrappers read the
		// eq indicator from a store tracker. This prover kept none, so register one here for the
		// shared point.
		let eq_tracker = group.store.register_eq_tracker(&eval_point);
		// Wrap each MLE-check evaluator so it emits sumcheck round polynomials, handing it the
		// shared eq tracker the store folds. Conversion happens before proving starts, so — since
		// no variable has been folded — the equality prefix is one and each wrapper's sumcheck
		// claim equals the inner MLE-check claim it carries over unchanged.
		let group = group.map_evaluators(|evaluator| {
			Box::new(MleToSumCheckEvaluator::new(evaluator, eq_tracker))
				as Box<dyn SumcheckRoundEvaluator<F, P> + 'a>
		});
		SharedSumcheckProver { group }
	}
}

impl<A, F, P, Evaluator> MleCheckProver<F> for SharedMleCheckProver<'_, A, F, P, Evaluator>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: MleCheckRoundEvaluator<F, P>,
{
	fn n_vars(&self) -> usize {
		self.group.n_vars()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let n_vars_remaining = self.group.n_vars();
		assert!(n_vars_remaining > 0);

		// One parallel pass over the halved hypercube feeds every evaluator; see
		// [`SharedSumcheckProver::execute`].
		let chunk_vars = (n_vars_remaining - 1).min(MAX_CHUNK_VARS.max(P::LOG_WIDTH));
		let total_slots: usize = self
			.group
			.evaluators
			.iter()
			.map(|evaluator| evaluator.degree())
			.sum();

		// The eq indicator factors into a chunk part over the low `chunk_vars` coordinates and a
		// suffix part over the higher ones. Materialize only the chunk part, shared by every chunk
		// and evaluator; the suffix coordinates are folded in below through `reduce`. The highest
		// remaining coordinate is this round's `alpha`, folded into the interpolation, not the sum.
		let alpha = self.eval_point[n_vars_remaining - 1];
		let eq_chunk = eq_ind_partial_eval::<P>(&self.eval_point[..chunk_vars]);

		let eval_point = &self.eval_point;
		let group = &mut self.group;
		let buffered_challenge = group.buffered_challenge.take();
		let evaluators = &group.evaluators;
		let store = &mut group.store;
		let map = |chunk: EvaluationChunk<'_, P>| {
			let mut accum = vec![Default::default(); total_slots];
			// One walk hands each evaluator its own run, so no offset table is built per round.
			let mut rest = accum.as_mut_slice();
			for evaluator in evaluators {
				let (slots, tail) = rest.split_at_mut(evaluator.degree());
				rest = tail;
				evaluator.accumulate(&chunk, eq_chunk.to_ref(), slots);
			}
			debug_assert!(rest.is_empty(), "the runs must tile the accumulator exactly");
			// Reduce the wide accumulators so the eq-weighted `reduce` can linearly extrapolate on
			// P.
			accum.into_iter().map(P::reduce).collect::<Vec<P>>()
		};
		let reduce = |mut lhs: Vec<P>, rhs: Vec<P>, level: usize| {
			// Fold the `level`-th (suffix) coordinate: eq weights the low half by `1 - z` and the
			// high half by `z`, i.e. the linear extrapolation `lo + z * (hi - lo)`.
			let z = P::broadcast(eval_point[level]);
			for (lo, hi) in iter::zip(&mut lhs, rhs) {
				*lo += z * (hi - *lo);
			}
			lhs
		};
		let accum = match buffered_challenge {
			Some(challenge) => store.map_reduce_with_fold(chunk_vars, challenge, map, reduce),
			None => store.map_reduce(chunk_vars, map, reduce),
		};

		self.group.interpolate_round(
			&accum,
			|evaluator| evaluator.degree(),
			|evaluator, ctx, slots, claim| evaluator.interpolate(ctx, slots, claim, alpha),
		)
	}

	fn fold(&mut self, challenge: F) {
		self.group.fold(challenge);
	}

	fn finish(self) -> Vec<F> {
		self.group.finish()
	}

	fn eval_point(&self) -> &[F] {
		&self.eval_point[..self.group.n_vars()]
	}
}

// Prove-and-verify coverage for the shared store provers: a batched fractional-addition MLE-check,
// and the logUp* final-layer shape (eq-weighted fractional addition batched with plain product
// claims over shared columns).
#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::FieldOps;
	use binius_ip::sumcheck::{batch_verify, batch_verify_mle};
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_par,
		multilinear::{eq::eq_ind, evaluate::evaluate},
		test_utils::{Packed128b, random_field_buffer, random_scalars},
		univariate::evaluate_univariate,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{
		MleToSumCheckEvaluator,
		batch::{batch_prove, batch_prove_mle},
		bivariate_product_evaluator::BivariateProductEvaluator,
		frac_add_mle,
	};

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;
	type StdChallenger = HasherChallenger<sha2::Sha256>;
	type CompFn = fn([P; 4]) -> P;

	// The fractional-addition numerator composition, as a single-claim function.
	fn comp_num([num_a, num_b, den_a, den_b]: [P; 4]) -> P {
		num_a * den_b + num_b * den_a
	}

	// The fractional-addition denominator composition, as a single-claim function.
	fn comp_den([_num_a, _num_b, den_a, den_b]: [P; 4]) -> P {
		den_a * den_b
	}

	// Split a multilinear on its highest variable into owned low and high halves.
	fn owned_halves(buffer: &FieldBuffer<P>) -> [FieldBuffer<P>; 2] {
		let (lo, hi) = buffer.split_half_ref();
		[
			FieldBuffer::new(lo.log_len(), lo.as_ref().into()),
			FieldBuffer::new(hi.log_len(), hi.as_ref().into()),
		]
	}

	// Random fractional-addition instance: four columns, evaluation point, and honest claims.
	fn frac_instance(rng: &mut StdRng, n_vars: usize) -> ([FieldBuffer<P>; 4], Vec<F>, [F; 2]) {
		let cols: [FieldBuffer<P>; 4] =
			std::array::from_fn(|_| random_field_buffer::<P>(&mut *rng, n_vars));
		let eval_point = random_scalars::<F>(&mut *rng, n_vars);

		// The honest claim is each composition's MLE evaluated at the point.
		let claims = [comp_num as CompFn, comp_den as CompFn].map(|comp| {
			let vals = (0..1usize << n_vars)
				.map(|i| {
					let scalars = [
						cols[0].get(i),
						cols[1].get(i),
						cols[2].get(i),
						cols[3].get(i),
					];
					comp(scalars.map(P::broadcast))
						.iter()
						.next()
						.expect("packed field has at least one lane")
				})
				.collect::<Vec<_>>();
			evaluate(&FieldBuffer::<P>::from_values(&vals), &eval_point)
		});

		(cols, eval_point, claims)
	}

	// The store + evaluator MLE-check prover for the two fractional-addition claims, borrowing the
	// four shared columns.
	fn new_frac_prover<'a>(
		alloc: &'a GlobalAllocator,
		cols: &'a [FieldBuffer<P>; 4],
		eval_point: &[F],
		claims: [F; 2],
	) -> SharedMleCheckProver<'a, GlobalAllocator, F, P, Box<dyn MleCheckRoundEvaluator<F, P>>> {
		let mut store = MleStore::new(eval_point.len(), alloc);
		let col_ids = cols.each_ref().map(|col| store.push(col.to_ref()));
		let (num_ev, den_ev) = frac_add_mle::evaluators(col_ids);
		let claims_with_evaluators: [(F, Box<dyn MleCheckRoundEvaluator<F, P>>); 2] =
			[(claims[0], Box::new(num_ev)), (claims[1], Box::new(den_ev))];
		SharedMleCheckProver::new(store, claims_with_evaluators, eval_point.to_vec())
	}

	// Prove the two fractional-addition claims through the MLE-check batch driver, then verify.
	// The two claims share one store, so the four columns are folded and evaluated once.
	//
	// The 14- and 15-variable cases exceed `MAX_CHUNK_VARS`, so the eq indicator no longer fits in
	// one chunk prefix: they exercise `SharedMleCheckProver`'s suffix folding, where the higher eq
	// coordinates are linearly extrapolated in `reduce` — 15 in particular hits that path through
	// both `map_reduce` (round 0) and the fused `map_reduce_with_fold` (round 1).
	#[test]
	fn test_shared_frac_add_prove_verify() {
		for n_vars in [1, 2, 3, 8, 14, 15] {
			let mut rng = StdRng::seed_from_u64(0);
			let (cols, eval_point, claims) = frac_instance(&mut rng, n_vars);

			// Prove: one shared prover carries both claims over the four columns.
			let alloc = GlobalAllocator;
			let mut transcript = ProverTranscript::new(StdChallenger::default());
			let output = batch_prove_mle(
				vec![new_frac_prover(&alloc, &cols, &eval_point, claims)],
				&mut transcript,
			);

			// The shared prover emits the four column evaluations once, in push order.
			assert_eq!(output.multilinear_evals.len(), 1);
			let evals = output.multilinear_evals[0].clone();
			assert_eq!(evals.len(), 4);
			transcript.message().write_scalar_slice(&evals);

			// Verify: quadratic prime polynomials give degree-2 MLE-check rounds.
			let mut verifier = transcript.into_verifier();
			let sumcheck_output = batch_verify_mle(&eval_point, 2, &claims, &mut verifier).unwrap();
			let verified_evals: Vec<F> = verifier.message().read_vec(4).unwrap();
			assert_eq!(evals, verified_evals, "prover and verifier column evaluations must match");

			// The prover binds variables high-to-low; `evaluate` expects them low-to-high.
			let mut point = sumcheck_output.challenges.clone();
			point.reverse();

			// Each recovered column evaluation is the column's evaluation at the challenge point.
			for (col, &eval) in cols.iter().zip(&verified_evals) {
				assert_eq!(evaluate(col, &point), eval);
			}

			// The reduced evaluation is the batch combination of the two compositions at the evals.
			let packed = std::array::from_fn(|i| P::broadcast(verified_evals[i]));
			let composed = [comp_num, comp_den]
				.map(|comp| comp(packed).iter().next().expect("packed field has a lane"));
			let expected = evaluate_univariate(&composed, &sumcheck_output.batch_coeff);
			assert_eq!(expected, sumcheck_output.eval, "reduced evaluation must match the batch");

			assert_eq!(output.challenges, sumcheck_output.challenges);
		}
	}

	// The logUp* final-layer shape: an eq-weighted fractional addition (two claims) batched with
	// two plain product claims, all sharing the pushforward halves in one store. Prove through the
	// sumcheck batch driver, then verify the reduced evaluation against the batched compositions.
	#[test]
	fn test_shared_final_layer_prove_verify() {
		for m in [1, 2, 3, 6] {
			let mut rng = StdRng::seed_from_u64(0);

			// Three parent buffers of m variables; each splits into two m-1 variable halves.
			let pushforward = random_field_buffer::<P>(&mut rng, m);
			let denominator = random_field_buffer::<P>(&mut rng, m);
			let table = random_field_buffer::<P>(&mut rng, m);
			let [y_0, y_1] = owned_halves(&pushforward);
			let [d_0, d_1] = owned_halves(&denominator);
			let [t_0, t_1] = owned_halves(&table);

			// Fractional claims at z; product claims are the inner products of the pushforward and
			// table halves.
			let z = random_scalars::<F>(&mut rng, m - 1);
			let frac_claims: [F; 2] = [comp_num as CompFn, comp_den as CompFn].map(|comp| {
				let vals = (0..1usize << (m - 1))
					.map(|i| {
						let scalars = [y_0.get(i), y_1.get(i), d_0.get(i), d_1.get(i)];
						comp(scalars.map(P::broadcast))
							.iter()
							.next()
							.expect("packed field has at least one lane")
					})
					.collect::<Vec<_>>();
				evaluate(&FieldBuffer::<P>::from_values(&vals), &z)
			});
			let e_0 = inner_product_par(&y_0, &t_0);
			let e_1 = inner_product_par(&y_1, &t_1);

			// One store with six borrowed columns, in push order [Y_0, Y_1, D_0, D_1, T_0, T_1].
			let alloc = GlobalAllocator;
			let mut store = MleStore::new(m - 1, &alloc);
			let [y_0_col, y_1_col, d_0_col, d_1_col, t_0_col, t_1_col] =
				[&y_0, &y_1, &d_0, &d_1, &t_0, &t_1].map(|col| store.push(col.to_ref()));

			// The eq-weighted fractional evaluators, wrapped so they emit sumcheck round
			// polynomials. The wrappers, driven by a plain sumcheck prover, hold the shared eq
			// tracker themselves.
			let (num_evaluator, den_evaluator) =
				frac_add_mle::evaluators([y_0_col, y_1_col, d_0_col, d_1_col]);
			let eq_tracker = store.register_eq_tracker(&z);
			let num_evaluator = MleToSumCheckEvaluator::new(num_evaluator, eq_tracker);
			let den_evaluator = MleToSumCheckEvaluator::new(den_evaluator, eq_tracker);

			// The two plain product claims over the pushforward and table halves.
			let product_0 = BivariateProductEvaluator::new([y_0_col, t_0_col]);
			let product_1 = BivariateProductEvaluator::new([y_1_col, t_1_col]);

			// Claims paired with evaluators in order: the two fractional claims, then the two
			// product sums.
			let claims_with_evaluators: [(F, Box<dyn SumcheckRoundEvaluator<F, P>>); 4] = [
				(frac_claims[0], Box::new(num_evaluator)),
				(frac_claims[1], Box::new(den_evaluator)),
				(e_0, Box::new(product_0)),
				(e_1, Box::new(product_1)),
			];
			let shared = SharedSumcheckProver::new(store, claims_with_evaluators);

			// Prove and record the four claim sums in evaluator order.
			let mut transcript = ProverTranscript::new(StdChallenger::default());
			let output = batch_prove(vec![shared], &mut transcript);

			// The shared prover emits each store column's evaluation once, in push order.
			assert_eq!(output.multilinear_evals.len(), 1);
			let evals = output.multilinear_evals[0].clone();
			assert_eq!(evals.len(), 6);
			transcript.message().write_scalar_slice(&evals);

			// Verify: the eq-wrapped fractional rounds have degree 3, the product rounds degree 2,
			// so the batched round polynomial has degree 3.
			let claims = [frac_claims[0], frac_claims[1], e_0, e_1];
			let mut verifier = transcript.into_verifier();
			let sumcheck_output = batch_verify(m - 1, 3, &claims, &mut verifier).unwrap();
			let verified_evals: Vec<F> = verifier.message().read_vec(6).unwrap();
			assert_eq!(evals, verified_evals, "prover and verifier column evaluations must match");

			// The prover binds variables high-to-low; `evaluate` expects them low-to-high.
			let mut point = sumcheck_output.challenges.clone();
			point.reverse();

			// Each recovered column evaluation is the column's evaluation at the challenge point.
			for (col, &eval) in [&y_0, &y_1, &d_0, &d_1, &t_0, &t_1]
				.iter()
				.zip(&verified_evals)
			{
				assert_eq!(evaluate(col, &point), eval);
			}

			// The reduced evaluation batches the four claims' reduced compositions in evaluator
			// order. The fractional compositions carry the equality factor at z; the products do
			// not.
			let [y0, y1, d0, d1, t0, t1] =
				<[F; 6]>::try_from(verified_evals).expect("six column evaluations");
			let eq = eq_ind(&z, &point);
			let reduced = [(y0 * d1 + y1 * d0) * eq, (d0 * d1) * eq, y0 * t0, y1 * t1];
			let expected = evaluate_univariate(&reduced, &sumcheck_output.batch_coeff);
			assert_eq!(expected, sumcheck_output.eval, "reduced evaluation must match the batch");

			assert_eq!(output.challenges, sumcheck_output.challenges);
		}
	}
}
