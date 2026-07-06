// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, Field, PackedField};
use binius_ip::sumcheck::{RoundCoeffs, SumcheckOutput};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		ProveSingleOutput, bivariate_product::BivariateProductSumcheckProver, prove_single,
	},
};
use binius_math::{
	FieldBuffer,
	multilinear::{
		eq::{eq_ind_partial_eval, eq_ind_zero},
		fold::fold_highest_var_inplace,
	},
};
use binius_utils::checked_arithmetics::checked_log_2;
use binius_verifier::{config::LOG_WORD_SIZE_BITS, protocols::shift::evaluate_words_mle};
use tracing::instrument;

use super::{
	key_collection::KeyCollection, monster::build_monster_multilinear, prove::PreparedOperatorData,
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
/// 3. **Monster Multilinear Construction**: Builds the monster multilinear from key collection and
///    operator data
/// 4. **Sumcheck Execution**: Runs bivariate product sumcheck to prove witness ×
///    monster_multilinear relationship
///
/// # Parameters
/// - `key_collection`: Prover's key collection representing the constraint system
/// - `words`: The witness words
/// - `bitand_data`: Operator data for bit multiplication constraints
/// - `intmul_data`: Operator data for integer multiplication constraints
/// - `phase_1_output`: Challenges and evaluation from the first phase
/// - `transcript`: The prover's transcript
///
/// # Returns
/// Returns `SumcheckOutput` containing the combined challenges `[r_j, r_y]` and witness evaluation,
/// or an error if the protocol fails.
#[instrument(skip_all, name = "prove_phase_2")]
pub fn prove_phase_2<F, P: PackedField<Scalar = F>, Channel>(
	key_collection: &KeyCollection,
	words: &[Word],
	bitand_data: &PreparedOperatorData<F>,
	intmul_data: &PreparedOperatorData<F>,
	phase_1_output: SumcheckOutput<F>,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField + From<AESTowerField8b>,
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

	// Fold each committed segment separately. The witness multilinear is, conceptually, the public
	// folded segment at the base of the low half-cube and the hidden folded segment at the base of
	// the high half-cube. Rather than materialize that mostly-zero combined buffer, the sumcheck's
	// special first round consumes the two segments directly (see `run_sumcheck`).
	let (public_words, hidden_words) = words.split_at(key_collection.public.n_words());
	let public_folded = fold_words::<_, P>(public_words, r_j_tensor.as_ref());
	let hidden_folded = fold_words::<_, P>(hidden_words, r_j_tensor.as_ref());

	let monster_multilinear =
		build_monster_multilinear(key_collection, bitand_data, intmul_data, &r_j, &r_s);

	run_sumcheck(
		public_folded,
		hidden_folded,
		hidden_words.len(),
		public_words,
		monster_multilinear,
		r_j,
		gamma,
		channel,
	)
}

/// Assembles the folded witness from its two folded segments.
///
/// Each segment sits at the base of its half-cube, zero-padded up to `2^log_half` entries. The
/// interim dense representation materializes the mostly-zero low half; a sparse special first
/// sumcheck round can remove this cost without changing the transcript.
pub fn assemble_witness<F: Field, P: PackedField<Scalar = F>>(
	public_folded: &FieldBuffer<P>,
	hidden_folded: &FieldBuffer<P>,
	log_half: usize,
) -> FieldBuffer<P> {
	let mut witness_folded = FieldBuffer::zeros(log_half + 1);
	{
		let mut split = witness_folded.split_half_mut();
		let (mut lo_half, mut hi_half) = split.halves();
		lo_half.as_mut()[..public_folded.as_ref().len()].copy_from_slice(public_folded.as_ref());
		hi_half.as_mut()[..hidden_folded.as_ref().len()].copy_from_slice(hidden_folded.as_ref());
	}
	witness_folded
}

/// Executes the bivariate product sumcheck for the witness and monster multilinear relationship,
/// with a special first round over the two committed segments.
///
/// The witness multilinear is, conceptually, the public folded segment at the base of the low
/// half-cube and the hidden folded segment at the base of the high half-cube, zero-padded to
/// `2^(log_half + 1)` scalars (see [`assemble_witness`]). Its first sumcheck round binds the
/// most-significant "segment selector" variable that distinguishes the two halves. Instead of
/// materializing that mostly-zero combined buffer and running a full-width first round, this round
/// is computed directly from the two segments, doing work proportional to `n_public_words +
/// n_hidden_words`. The remaining rounds run the generic bivariate product sumcheck over the
/// half-size folded buffers. The transcript is identical to the dense formulation.
///
/// After the sumcheck, sends the hidden-segment evaluation: the verifier reconstructs the full
/// witness evaluation from it and its own public words.
///
/// # Parameters
/// - `public_folded`: The public segment folded at challenges `r_j` (`n_public_words` scalars)
/// - `hidden_folded`: The hidden segment folded at challenges `r_j` (padded to `2^log_half`
///   scalars)
/// - `n_hidden_words`: The number of real hidden words; bounds the sparse first-round work
/// - `public_words`: The public segment words, for the witness evaluation derivation
/// - `monster_multilinear`: The monster multilinear polynomial constructed from constraints
/// - `r_j`: Challenge vector from phase 1 (first `LOG_WORD_SIZE_BITS` challenges)
/// - `gamma`: The claimed evaluation from phase 1
/// - `channel`: The prover's channel
///
/// # Returns
/// Returns `SumcheckOutput` with concatenated challenges `[r_j, r_y]` and the committed-half
/// evaluation.
#[instrument(skip_all, name = "run_sumcheck")]
#[allow(clippy::too_many_arguments)]
pub fn run_sumcheck<F, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>>(
	public_folded: FieldBuffer<P>,
	hidden_folded: FieldBuffer<P>,
	n_hidden_words: usize,
	public_words: &[Word],
	mut monster_multilinear: FieldBuffer<P>,
	r_j: Vec<F>,
	gamma: F,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField + From<AESTowerField8b>,
{
	let log_half = monster_multilinear.log_len() - 1;
	let half = 1usize << log_half;
	let n_public_words = public_words.len();

	#[cfg(debug_assertions)]
	let debug_witness_folded = assemble_witness(&public_folded, &hidden_folded, log_half);

	// Special first round (binds the segment selector). Over the combined witness `A` and monster
	// `B`, with `A_0`/`B_0` the low (public) halves and `A_1`/`B_1` the high (hidden) halves, the
	// round evaluations are `y_1 = Σ A_1·B_1` and `y_∞ = Σ (A_0+A_1)(B_0+B_1)`. The public folded
	// segment occupies `[0, n_public_words)` of the low half; the hidden folded segment occupies
	// `[0, n_hidden_words)` of the high half; everything else is zero. Since `n_hidden_words >=
	// n_public_words`, iterating `[0, n_hidden_words)` covers every non-zero term, so the work is
	// proportional to the two segment sizes rather than the full `2^(log_half + 1)` cube.
	let (mut y_1, mut y_inf) = (F::ZERO, F::ZERO);
	for i in 0..n_hidden_words {
		let a_0 = if i < n_public_words {
			public_folded.get(i)
		} else {
			F::ZERO
		};
		let a_1 = hidden_folded.get(i);
		let b_0 = monster_multilinear.get(i);
		let b_1 = monster_multilinear.get(half + i);
		y_1 += a_1 * b_1;
		y_inf += (a_0 + a_1) * (b_0 + b_1);
	}
	// Interpolate the degree-2 round polynomial from `(y_0 = gamma - y_1, y_1, y_∞)`.
	let y_0 = gamma - y_1;
	let round_coeffs = RoundCoeffs(vec![y_0, y_1 - y_0 - y_inf, y_inf]);
	channel.send_many(round_coeffs.clone().truncate().coeffs());
	let r_segment = channel.sample();
	let round_sum = round_coeffs.evaluate(r_segment);

	// Fold both multilinears on the selector. The witness fold is written sparsely into a
	// half-size buffer; the monster is folded in place.
	let mut witness_folded = FieldBuffer::<P>::zeros(log_half);
	let one_minus_r = F::ONE - r_segment;
	for i in 0..n_hidden_words {
		let a_0 = if i < n_public_words {
			public_folded.get(i)
		} else {
			F::ZERO
		};
		let a_1 = hidden_folded.get(i);
		witness_folded.set(i, one_minus_r * a_0 + r_segment * a_1);
	}
	fold_highest_var_inplace(&mut monster_multilinear, r_segment);

	// Run the remaining `log_half` rounds of the bivariate product sumcheck.
	let prover =
		BivariateProductSumcheckProver::new([witness_folded, monster_multilinear], round_sum);
	let ProveSingleOutput {
		multilinear_evals,
		challenges: r_rest,
	} = prove_single(prover, channel);

	// The full round challenges are `[r_segment, r_rest...]`; reverse to get the evaluation point.
	let mut r_y = Vec::with_capacity(log_half + 1);
	r_y.push(r_segment);
	r_y.extend(r_rest);
	r_y.reverse();

	let [trace_eval, _monster_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");

	#[cfg(debug_assertions)]
	{
		let r_y_tensor = eq_ind_partial_eval(&r_y);
		let expected_trace_eval =
			binius_math::inner_product::inner_product_buffers(&debug_witness_folded, &r_y_tensor);
		debug_assert_eq!(trace_eval, expected_trace_eval);
	}

	// Derive the witness evaluation from the trace evaluation by evaluating the public segment
	// (cheap, like the verifier does), subtracting its padded contribution off and scaling.
	// This makes the protocol incomplete with negligible probability, when the segment selector
	// challenge is zero.
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
