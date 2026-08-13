// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{cmp::max, iter, ops::DerefMut};

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
	FieldBuffer, FieldVec,
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
	monster::build_monster_segments, phase_1::Phase1Output,
};
use crate::fold_word::fold_words;

/// Proves the second phase of the shift protocol reduction.
///
/// This function implements phase 2 of the shift protocol prover, which takes the output
/// from phase 1 and completes the shift reduction by proving the relationship between
/// the witness and the monster multilinear polynomial.
///
/// # Protocol Steps
/// 1. **Segment Folding**: Folds the public and hidden words using the `r_j` challenges
/// 2. **Monster Multilinear Construction**: Builds the monster segments from the key collection and
///    the prepared claims
/// 3. **Sumcheck Execution**: Runs the bivariate product sumcheck with a sparse first round over
///    the segment selector
///
/// # Parameters
/// - `key_collection`: Prover's key collection representing the constraint system
/// - `words`: The value vector words
/// - `prepared`: The prepared claim of each operation, indexed by the operation a key names
/// - `phase_1_output`: Challenges and evaluation from the first phase
/// - `shift_ind_eval`: the scalar weighting every shift key, the product of the bit-index rounds'
///   two factor evaluations ([`ShiftIndSumcheck`](super::ShiftIndSumcheck))
/// - `epsilon`: the claim these rounds prove, which those rounds reduced to
/// - `channel`: The prover's channel
///
/// # Returns
/// Returns `SumcheckOutput` containing the combined challenges `[r_j, r_y]` and the witness
/// evaluation, or an error if the protocol fails.
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "prove_phase_2")]
pub fn prove_phase_2<F, P: PackedField<Scalar = F>, Channel, A>(
	key_collection: &KeyCollection,
	words: SegmentWords<'_>,
	prepared: &PreparedOperatorClaims<F>,
	phase_1_output: Phase1Output<F>,
	shift_ind_eval: F,
	epsilon: F,
	channel: &mut Channel,
	alloc: &A,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	let Phase1Output {
		r_j,
		r_s,
		r_v,
		gamma: _,
		g_eval: _,
	} = phase_1_output;

	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

	// Fold each segment separately; the combined witness is never materialized. `fold_words`
	// zero-pads each fold to `log2_ceil(len)` variables, so each fold spans its own segment.
	let public_folded = fold_words::<_, P, _>(alloc, words.public, r_j_tensor.as_ref());
	let hidden_folded = fold_words::<_, P, _>(alloc, words.hidden, r_j_tensor.as_ref());

	let (public_monster, hidden_monster) =
		build_monster_segments(alloc, key_collection, prepared, shift_ind_eval, &r_s, &r_v);

	// Both halves of the sumcheck share one word-index address space, spanning the wider of the
	// two segments. The hidden segment is normally the wider one, but a system with more public
	// words than private values inverts that, and the hidden half zero-extends to match.
	let log_segment_words = max(public_folded.log_len(), hidden_folded.log_len());
	let hidden_folded = zero_extend(alloc, hidden_folded, log_segment_words);
	let hidden_monster = zero_extend(alloc, hidden_monster, log_segment_words);

	run_sumcheck(
		&public_folded,
		hidden_folded,
		&public_monster,
		hidden_monster,
		words.public,
		r_j,
		epsilon,
		channel,
		alloc,
	)
}

/// Zero-extends a segment buffer to span `log_len` word-index variables.
///
/// Returns the buffer untouched when it already spans that many, which is the common case: the
/// hidden segment is normally the wider of the two, so no copy happens.
///
/// ## Preconditions
/// * `log_len` is at least `buffer.log_len()`
fn zero_extend<F, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	buffer: FieldVec<P, A>,
	log_len: usize,
) -> FieldVec<P, A>
where
	F: BinaryField,
{
	assert!(log_len >= buffer.log_len());
	if log_len == buffer.log_len() {
		return buffer;
	}

	// Whole packed words copy across: a trailing partial word carries zero high lanes, which are
	// exactly the zeros the extension pads with.
	let mut extended = FieldVec::<P, A>::zeros_in(alloc, log_len);
	extended.as_mut()[..buffer.as_ref().len()].copy_from_slice(buffer.as_ref());
	extended
}

/// Computes the phase-2 first-round message: the degree-2 round polynomial that binds the segment
/// selector, evaluated sparsely without materializing the combined witness.
///
/// With `W(X, y) = (1 - X) * P_pad(y) + X * H(y)` and `M` likewise,
///
/// ```text
/// y_1 = sum_y H * M_h    y_inf = sum_y (P_pad + H) * (M_p_pad + M_h)
/// ```
///
/// The dense `H * M_h` pass dominates; the `y_inf` corrections have support on the public prefix
/// only. Both corrections run over whole packed elements: past the public length the `P`/`M_p`
/// entries are zero, so the stray `H * M_h` lane terms cancel between the two sums. The per-point
/// products accumulate in unreduced (wide) form and reduce once at the end.
fn first_round_coeffs<F, P: PackedField<Scalar = F>, Data: std::ops::Deref<Target = [P]>>(
	public_folded: &FieldBuffer<P, Data>,
	hidden_folded: &FieldBuffer<P, Data>,
	public_monster: &FieldBuffer<P, Data>,
	hidden_monster: &FieldBuffer<P, Data>,
	gamma: F,
) -> RoundCoeffs<F>
where
	F: BinaryField,
{
	// The dense hidden-segment pass.
	let wide_dense = (hidden_folded.as_ref(), hidden_monster.as_ref())
		.into_par_iter()
		.with_min_task(WorkPerItem::FieldMuls)
		.map(|(&hidden_i, &monster_i)| P::wide_mul(hidden_i, monster_i))
		.reduce(<P as WideMul>::Output::default, |lhs, rhs| lhs + rhs);

	// The public-prefix corrections.
	let n_public_packed = public_folded.as_ref().len();
	let (wide_low_hidden, wide_low_cross) = iter::zip(
		iter::zip(public_folded.as_ref(), &hidden_folded.as_ref()[..n_public_packed]),
		iter::zip(public_monster.as_ref(), &hidden_monster.as_ref()[..n_public_packed]),
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

/// Folds the two segment buffers of the witness at the selector challenge.
///
/// Consumes and overwrites `hidden` for memory efficiency, returning
/// `(1 - alpha) * public_padded + alpha * hidden` — exactly the result of
/// `fold_highest_var_inplace` on the materialized witness buffer.
fn fold_segments<F: Field, P: PackedField<Scalar = F>, Data: DerefMut<Target = [P]>>(
	public: &FieldBuffer<P, Data>,
	mut hidden: FieldBuffer<P, Data>,
	alpha: F,
) -> FieldBuffer<P, Data> {
	// Scale the dominant hidden segment in place, in parallel.
	let alpha_broadcast = P::broadcast(alpha);
	hidden
		.as_mut()
		.par_iter_mut()
		.with_min_task(WorkPerItem::FieldMuls)
		.for_each(|hidden_i| *hidden_i *= alpha_broadcast);

	// Add the small public prefix sequentially. Its trailing partial packed element carries zero
	// high lanes, so whole-element updates are correct.
	let one_minus_alpha = P::broadcast(F::ONE - alpha);
	let n_public_packed = public.as_ref().len();
	for (value, &public_i) in iter::zip(&mut hidden.as_mut()[..n_public_packed], public.as_ref()) {
		*value += public_i * one_minus_alpha;
	}

	hidden
}

/// Executes the phase-2 sumcheck over the witness, with a sparse first round.
///
/// The witness `W` and monster `M` are each given as a (public, hidden) segment pair; the top
/// word-index variable selects the segment. The first round (`first_round_coeffs`) binds that
/// selector without materializing the mostly-zero combined buffers. After the selector challenge
/// the segment pairs fold into single dense buffers (`fold_segments`) and the standard shared
/// bivariate-product prover (`bivariate_product_prover`) proves the remaining rounds, so every
/// round message is identical to the dense prover's.
///
/// After the sumcheck this derives the witness evaluation from the combined evaluation by
/// evaluating the public segment (cheap, like the verifier does), subtracting its padded
/// contribution off and scaling.
///
/// # Returns
/// Returns `SumcheckOutput` with concatenated challenges `[r_j, r_y]` and the witness
/// evaluation.
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "run_sumcheck")]
pub fn run_sumcheck<F, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>, A: Allocator>(
	public_folded: &FieldVec<P, A>,
	hidden_folded: FieldVec<P, A>,
	public_monster: &FieldVec<P, A>,
	hidden_monster: FieldVec<P, A>,
	public_words: &[Word],
	r_j: Vec<F>,
	gamma: F,
	channel: &mut Channel,
	alloc: &A,
) -> SumcheckOutput<F>
where
	F: BinaryField,
{
	// The hidden pair is the dense one both the first round and `fold_segments` iterate over, so
	// it spans the whole word-index space and the public pair sits at its base.
	let log_hidden = hidden_folded.log_len();
	assert_eq!(hidden_monster.log_len(), log_hidden);
	assert_eq!(public_monster.log_len(), public_folded.log_len());
	assert!(public_folded.log_len() <= log_hidden);

	// Round 1: bind the segment selector.
	let round_coeffs =
		first_round_coeffs(public_folded, &hidden_folded, public_monster, &hidden_monster, gamma);
	channel.send_many(round_coeffs.clone().truncate().coeffs());
	let alpha = channel.sample();
	let round_sum = round_coeffs.evaluate(&alpha);

	// Fold the segment pairs at the selector challenge and run the remaining rounds with the
	// standard prover.
	let folded_witness = fold_segments(public_folded, hidden_folded, alpha);
	let folded_monster = fold_segments(public_monster, hidden_monster, alpha);
	let prover = bivariate_product_prover(alloc, [folded_witness, folded_monster], round_sum);

	let ProveSingleOutput {
		multilinear_evals,
		challenges,
	} = prove_single(prover, channel);

	let mut r_y = iter::once(alpha).chain(challenges).collect::<Vec<_>>();
	// Reverse the challenges to get the evaluation point.
	r_y.reverse();

	let [trace_eval, _monster_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");

	// Derive the witness evaluation from the combined evaluation by evaluating the public
	// segment (cheap, like the verifier does), subtracting its padded contribution off and
	// scaling. This makes the protocol incomplete with negligible probability, when the segment
	// selector challenge is zero.
	let log_half = r_y.len() - 1;
	let r_segment = r_y[log_half];
	// Round the public word count up to a power of two: the segment spans that many word slots.
	// The count itself need not be a power of two, and the MLE reads the missing words as zero.
	let log_public_words = log2_ceil_usize(public_words.len());
	let public_eval = evaluate_words_mle::<F, F>(public_words, &r_j, &r_y[..log_public_words]);
	let padded_public_eval = eq_ind_zero(&r_y[log_public_words..log_half]) * public_eval;
	let witness_eval =
		(trace_eval - (F::ONE - r_segment) * padded_public_eval) * r_segment.invert_or_zero();
	channel.send_one(witness_eval);

	SumcheckOutput {
		challenges: [r_j, r_y].concat(),
		eval: witness_eval,
	}
}
