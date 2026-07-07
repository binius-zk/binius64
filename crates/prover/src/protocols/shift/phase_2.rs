// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_ip::sumcheck::{RoundCoeffs, SumcheckOutput};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		ProveSingleOutput, bivariate_product::BivariateProductSumcheckProver, prove_single,
		round_evals::RoundEvals2,
	},
};
use binius_math::{
	BinarySubspace, FieldBuffer,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_zero},
};
use binius_utils::{checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::{config::LOG_WORD_SIZE_BITS, protocols::shift::evaluate_words_mle};
use tracing::instrument;

use super::{
	key_collection::KeyCollection, monster::build_monster_segments, prove::PreparedOperatorData,
};
use crate::fold_word::fold_words;

/// Proves the second phase of the shift protocol reduction.
///
/// This function implements phase 2 of the shift protocol prover, which takes the output
/// from phase 1 and completes the shift reduction by proving the relationship between
/// the witness and the monster multilinear polynomial.
///
/// # Protocol Steps
/// 1. **Challenge Splitting**: Splits phase 1 challenges into `r_j` and `r_s` components
/// 2. **Segment Folding**: Folds the public and hidden words using the `r_j` challenges
/// 3. **Monster Multilinear Construction**: Builds the monster segments from the key collection and
///    operator data
/// 4. **Sumcheck Execution**: Runs the bivariate product sumcheck with a sparse first round over
///    the segment selector
///
/// # Parameters
/// - `key_collection`: Prover's key collection representing the constraint system
/// - `words`: The value vector words
/// - `bitand_data`: Operator data for bit multiplication constraints
/// - `intmul_data`: Operator data for integer multiplication constraints
/// - `phase_1_output`: Challenges and evaluation from the first phase
/// - `channel`: The prover's channel
///
/// # Returns
/// Returns `SumcheckOutput` containing the combined challenges `[r_j, r_y]` and the witness
/// evaluation, or an error if the protocol fails.
#[instrument(skip_all, name = "prove_phase_2")]
pub fn prove_phase_2<F, P: PackedField<Scalar = F>, Channel>(
	key_collection: &KeyCollection,
	words: &[Word],
	bitand_data: &PreparedOperatorData<F>,
	intmul_data: &PreparedOperatorData<F>,
	domain_subspace: &BinarySubspace<F>,
	phase_1_output: SumcheckOutput<F>,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	Channel: IPProverChannel<F>,
{
	let SumcheckOutput {
		challenges: mut r_jr_s,
		eval: gamma,
	} = phase_1_output;
	// Split challenges as r_j,r_s where r_j is the first LOG_WORD_SIZE_BITS
	// variables and r_s is the last LOG_WORD_SIZE_BITS variables
	// Thus r_s are the more significant variables.
	let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
	let r_j = r_jr_s;

	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

	// Fold each segment separately; the combined witness is never materialized. `fold_words`
	// zero-pads each fold to `log2_ceil(len)` variables, so the hidden fold already has
	// `log_witness_words` variables (its word count is the hidden segment length).
	let (public_words, hidden_words) = words.split_at(key_collection.public.n_words());
	let public_folded = fold_words::<_, P>(public_words, r_j_tensor.as_ref());
	let hidden_folded = fold_words::<_, P>(hidden_words, r_j_tensor.as_ref());

	let (public_monster, hidden_monster) = build_monster_segments(
		key_collection,
		bitand_data,
		intmul_data,
		domain_subspace,
		&r_j,
		&r_s,
	);

	run_sumcheck(
		public_folded,
		hidden_folded,
		public_monster,
		hidden_monster,
		public_words,
		r_j,
		gamma,
		channel,
	)
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
fn first_round_coeffs<F, P: PackedField<Scalar = F>>(
	public_folded: &FieldBuffer<P>,
	hidden_folded: &FieldBuffer<P>,
	public_monster: &FieldBuffer<P>,
	hidden_monster: &FieldBuffer<P>,
	gamma: F,
) -> RoundCoeffs<F>
where
	F: BinaryField,
{
	// The dense hidden-segment pass.
	let wide_dense = (hidden_folded.as_ref(), hidden_monster.as_ref())
		.into_par_iter()
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

	RoundEvals2 { y_1, y_inf }.interpolate(gamma)
}

/// Folds the two segment buffers of the witness at the selector challenge.
///
/// Consumes and overwrites `hidden` for memory efficiency, returning
/// `(1 - alpha) * public_padded + alpha * hidden` — exactly the result of
/// `fold_highest_var_inplace` on the materialized witness buffer.
fn fold_segments<F: Field, P: PackedField<Scalar = F>>(
	public: &FieldBuffer<P>,
	mut hidden: FieldBuffer<P>,
	alpha: F,
) -> FieldBuffer<P> {
	// Scale the dominant hidden segment in place, in parallel.
	let alpha_broadcast = P::broadcast(alpha);
	hidden
		.as_mut()
		.par_iter_mut()
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
/// the segment pairs fold into single dense buffers (`fold_segments`) and the standard
/// [`BivariateProductSumcheckProver`] proves the remaining rounds, so every round message is
/// identical to the dense prover's.
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
pub fn run_sumcheck<F, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>>(
	public_folded: FieldBuffer<P>,
	hidden_folded: FieldBuffer<P>,
	public_monster: FieldBuffer<P>,
	hidden_monster: FieldBuffer<P>,
	public_words: &[Word],
	r_j: Vec<F>,
	gamma: F,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField,
{
	let log_hidden = hidden_folded.log_len();
	assert_eq!(hidden_monster.log_len(), log_hidden);
	assert_eq!(public_monster.log_len(), public_folded.log_len());
	assert!(public_folded.log_len() <= log_hidden);

	// Round 1: bind the segment selector.
	let round_coeffs =
		first_round_coeffs(&public_folded, &hidden_folded, &public_monster, &hidden_monster, gamma);
	channel.send_many(round_coeffs.clone().truncate().coeffs());
	let alpha = channel.sample();
	let round_sum = round_coeffs.evaluate(alpha);

	// Fold the segment pairs at the selector challenge and run the remaining rounds with the
	// standard prover.
	let folded_witness = fold_segments(&public_folded, hidden_folded, alpha);
	let folded_monster = fold_segments(&public_monster, hidden_monster, alpha);
	let prover = BivariateProductSumcheckProver::new([folded_witness, folded_monster], round_sum);

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
	let log_public_words = checked_log_2(public_words.len());
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
