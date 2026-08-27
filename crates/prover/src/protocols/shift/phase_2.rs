// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{cmp::max, iter};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_ip::sumcheck::{RoundCoeffs, SumcheckOutput};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		ProveSingleOutput, bivariate_product_prover, prove_single, round_evals::RoundEvals,
	},
};
use binius_math::{
	FieldVec,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_zero},
};
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};
use binius_verifier::protocols::shift::evaluate_words_mle;
use tracing::instrument;

use super::{
	SegmentWords, claims::PreparedOperatorClaims, key_collection::KeyCollection,
	phase_1::Phase1Output,
};
use crate::fold_word::BitAxisFolder;

/// Proves the second phase of the shift protocol reduction.
///
/// Folds the value-vector words by the bit-position challenge.
/// Builds the constraint-matrix multilinear's two segments.
/// Then runs a sumcheck between them, with a sparse first round over the segment selector.
///
/// # Arguments
///
/// - `key_collection`: the prover's key collection for the constraint system.
/// - `words`: the value-vector words.
/// - `prepared`: the prepared claim of each operation, indexed by the operation a key names.
/// - `phase_1_output`: the challenges and evaluation the first phase produced.
/// - `shift_ind_eval`: the scalar weighting every shift key.
/// - `epsilon`: the claim this phase's rounds prove.
/// - `channel`: the prover channel the interactive rounds run over.
/// - `alloc`: the allocator the intermediate buffers are drawn from.
///
/// `shift_ind_eval` is the product of the two indicator evaluations.
/// Those are what the earlier bit-index phases reduced to.
///
/// # Returns
///
/// The combined challenges with the witness evaluation, and the wiring multilinear's evaluation.
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "prove_phase_2")]
pub fn prove_phase_2<F, P, Channel, A>(
	key_collection: &KeyCollection,
	words: SegmentWords<'_>,
	prepared: &PreparedOperatorClaims<F>,
	phase_1_output: Phase1Output<F>,
	shift_ind_eval: F,
	epsilon: F,
	channel: &mut Channel,
	alloc: &A,
) -> ShiftOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	let Phase1Output {
		r_j,
		inner,
		outer,
		psi: _,
		gamma: _,
		g_eval: _,
	} = phase_1_output;

	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

	// Fold each segment separately.
	// The combined witness is never materialized.
	// Each fold is zero-padded to enough variables to cover its own segment's length.
	// Both columns fold against the same round tensor, so the tables are built once.
	let folder = BitAxisFolder::new(r_j_tensor.as_ref());
	let public_folded = folder.fold::<P, _>(alloc, words.public);
	let hidden_folded = folder.fold::<P, _>(alloc, words.hidden);

	let (public_monster, hidden_monster) =
		key_collection.build_monster_segments(alloc, prepared, shift_ind_eval, &inner, &outer);

	// Both halves of the sumcheck share one word-index space, spanning the wider segment.
	// The hidden segment is normally the wider one.
	// A system with more public words than private values inverts that.
	// So the hidden half is zero-extended to match.
	let log_segment_words = max(public_folded.log_len(), hidden_folded.log_len());
	let hidden_folded = hidden_folded.zero_extend_in(alloc, log_segment_words);
	let hidden_monster = hidden_monster.zero_extend_in(alloc, log_segment_words);

	run_sumcheck(
		&public_folded,
		hidden_folded,
		&public_monster,
		hidden_monster,
		shift_ind_eval,
		words.public,
		r_j,
		epsilon,
		channel,
		alloc,
	)
}

/// A witness or constraint-matrix buffer, split into the public and hidden segments the
/// phase-2 sumcheck's selector variable chooses between.
///
/// The hidden segment is normally the wider of the two, spanning the whole word-index space,
/// with the public segment sitting at its base.
struct SegmentPair<'a, P: PackedField, A: Allocator> {
	/// The public segment, at the base of the shared word-index space.
	public: &'a FieldVec<P, A>,
	/// The hidden segment, normally spanning the whole word-index space.
	hidden: FieldVec<P, A>,
}

impl<'a, P: PackedField, A: Allocator> SegmentPair<'a, P, A> {
	/// Pairs a public and hidden segment sharing one word-index space.
	const fn new(public: &'a FieldVec<P, A>, hidden: FieldVec<P, A>) -> Self {
		Self { public, hidden }
	}

	/// Folds the two segments at the selector challenge.
	///
	/// Consumes and overwrites the hidden buffer for memory efficiency.
	/// The result is `(1 - alpha) * public_padded + alpha * hidden`, exactly what folding the
	/// materialized combined buffer's highest variable would produce.
	fn fold<F>(self, alpha: F) -> FieldVec<P, A>
	where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let Self { public, mut hidden } = self;

		// Scale the dominant hidden segment in place, in parallel.
		let alpha_broadcast = P::broadcast(alpha);
		hidden
			.as_mut()
			.par_iter_mut()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|hidden_i| *hidden_i *= alpha_broadcast);

		// Add the small public prefix sequentially.
		// Its trailing partial packed element carries zero high lanes, so whole-element
		// updates are correct.
		let one_minus_alpha = P::broadcast(F::ONE - alpha);
		let n_public_packed = public.as_ref().len();
		for (value, &public_i) in
			iter::zip(&mut hidden.as_mut()[..n_public_packed], public.as_ref())
		{
			*value += public_i * one_minus_alpha;
		}

		hidden
	}
}

/// Computes the phase-2 first-round message: the degree-2 round polynomial that binds the
/// segment selector, evaluated sparsely without materializing the combined witness.
///
/// With `W(X, y) = (1 - X) * P_pad(y) + X * H(y)` and `M` likewise,
///
/// ```text
/// y_1 = sum_y H * M_h    y_inf = sum_y (P_pad + H) * (M_p_pad + M_h)
/// ```
///
/// The dense `H * M_h` pass dominates, and the `y_inf` corrections only have support on the
/// public prefix.
/// Both corrections run over whole packed elements, so past the public length the stray terms
/// the zero padding introduces cancel between the two sums.
fn first_round_coeffs<F, P: PackedField<Scalar = F>, A: Allocator>(
	witness: &SegmentPair<'_, P, A>,
	monster: &SegmentPair<'_, P, A>,
	gamma: F,
) -> RoundCoeffs<F>
where
	F: BinaryField,
{
	// The dense hidden-segment pass.
	let wide_dense = (witness.hidden.as_ref(), monster.hidden.as_ref())
		.into_par_iter()
		.with_min_task(WorkPerItem::FieldMuls)
		.map(|(&hidden_i, &monster_i)| P::wide_mul(hidden_i, monster_i))
		.reduce(<P as WideMul>::Output::default, |lhs, rhs| lhs + rhs);

	// The public-prefix corrections.
	let n_public_packed = witness.public.as_ref().len();
	let (wide_low_hidden, wide_low_cross) = iter::zip(
		iter::zip(witness.public.as_ref(), &witness.hidden.as_ref()[..n_public_packed]),
		iter::zip(monster.public.as_ref(), &monster.hidden.as_ref()[..n_public_packed]),
	)
	.map(|((&public_i, &hidden_i), (&public_monster_i, &hidden_monster_i))| {
		(
			P::wide_mul(hidden_i, hidden_monster_i),
			P::wide_mul(public_i + hidden_i, public_monster_i + hidden_monster_i),
		)
	})
	.fold(
		(<P as WideMul>::Output::default(), <P as WideMul>::Output::default()),
		|(acc_hidden, acc_cross), (hidden_term, cross_term)| {
			(acc_hidden + hidden_term, acc_cross + cross_term)
		},
	);

	let sum_lanes = |wide: <P as WideMul>::Output| P::reduce(wide).iter().sum::<F>();
	let y_1 = sum_lanes(wide_dense);
	let y_inf = y_1 + sum_lanes(wide_low_hidden) + sum_lanes(wide_low_cross);

	RoundEvals([y_1, y_inf]).interpolate(gamma)
}

/// Executes the phase-2 sumcheck over the witness, with a sparse first round.
///
/// # Overview
///
/// The witness and the constraint-matrix multilinear are each given as a (public, hidden)
/// segment pair.
/// The top word-index variable selects the segment.
///
/// The first round binds that selector without materializing the mostly-zero combined buffers.
/// After the selector challenge, the segment pairs fold into single dense buffers, and a
/// shared dense-product prover proves the remaining rounds.
/// So every round message is identical to what a fully dense prover would send.
///
/// After the sumcheck, this derives the witness evaluation from the combined evaluation: it
/// evaluates the public segment directly (cheap, like the verifier does), subtracts its
/// padded contribution, and scales.
///
/// It also divides the three bit-index factors back out of the constraint-matrix evaluation,
/// leaving the wiring evaluation the verifier's claim is about.
///
/// # Returns
///
/// The sumcheck's concatenated challenges with the witness evaluation, and the wiring
/// evaluation for the caller to send.
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "run_sumcheck")]
pub fn run_sumcheck<F, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>, A: Allocator>(
	public_folded: &FieldVec<P, A>,
	hidden_folded: FieldVec<P, A>,
	public_monster: &FieldVec<P, A>,
	hidden_monster: FieldVec<P, A>,
	shift_ind_eval: F,
	public_words: &[Word],
	r_j: Vec<F>,
	gamma: F,
	channel: &mut Channel,
	alloc: &A,
) -> ShiftOutput<F>
where
	F: BinaryField,
{
	// The hidden pair is the dense one every round iterates over.
	// So it spans the whole word-index space, and the public pair sits at its base.
	let log_hidden = hidden_folded.log_len();
	assert_eq!(hidden_monster.log_len(), log_hidden);
	assert_eq!(public_monster.log_len(), public_folded.log_len());
	assert!(public_folded.log_len() <= log_hidden);

	let witness = SegmentPair::<'_, P, A>::new(public_folded, hidden_folded);
	let monster = SegmentPair::<'_, P, A>::new(public_monster, hidden_monster);

	// Round 1: bind the segment selector.
	let round_coeffs = first_round_coeffs(&witness, &monster, gamma);
	channel.send_many(round_coeffs.clone().truncate().coeffs());
	let alpha = channel.sample();
	let round_sum = round_coeffs.evaluate(&alpha);

	// Fold the segment pairs at the selector challenge and run the remaining rounds with the
	// standard prover.
	let folded_witness = witness.fold(alpha);
	let folded_monster = monster.fold(alpha);
	let prover = bivariate_product_prover(alloc, [folded_witness, folded_monster], round_sum);

	let ProveSingleOutput {
		multilinear_evals,
		challenges,
	} = prove_single(prover, channel);

	let mut r_y = iter::once(alpha).chain(challenges).collect::<Vec<_>>();
	// Reverse the challenges to get the evaluation point.
	r_y.reverse();

	let [trace_eval, monster_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");

	// Every constraint-matrix entry carries the three bit-index factors.
	// Dividing them out leaves the bare wiring evaluation.
	//
	// Like the witness evaluation below, this makes the protocol incomplete with negligible
	// probability, when the scale is zero.
	let wiring_eval = monster_eval * shift_ind_eval.invert_or_zero();

	// Derive the witness evaluation from the combined evaluation: evaluate the public segment
	// directly (cheap, like the verifier does), subtract its padded contribution, and scale.
	//
	// This makes the protocol incomplete with negligible probability, when the segment
	// selector challenge is zero.
	let log_half = r_y.len() - 1;
	let r_segment = r_y[log_half];
	// Round the public word count up to a power of two: the segment spans that many word
	// slots.
	// The count itself need not be a power of two, and the missing words are read as zero.
	let log_public_words = log2_ceil_usize(public_words.len());
	let public_eval = evaluate_words_mle::<F, F>(public_words, &r_j, &r_y[..log_public_words]);
	let padded_public_eval = eq_ind_zero(&r_y[log_public_words..log_half]) * public_eval;
	let witness_eval =
		(trace_eval - (F::ONE - r_segment) * padded_public_eval) * r_segment.invert_or_zero();
	channel.send_one(witness_eval);

	ShiftOutput {
		sumcheck: SumcheckOutput {
			challenges: [r_j, r_y].concat(),
			eval: witness_eval,
		},
		wiring_eval,
	}
}

/// What the shift reduction leaves for its caller.
///
/// The wiring evaluation is not sent here: the verifier reads it after the public segment's
/// evaluation claim, which the caller proves, so the caller sends it at that point.
#[derive(Debug)]
pub struct ShiftOutput<F> {
	/// The sumcheck's challenges `[r_j, r_y]` and the witness evaluation.
	pub sumcheck: SumcheckOutput<F>,
	/// The wiring multilinear's evaluation at the reduced point.
	pub wiring_eval: F,
}
