// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, Field, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		ProveSingleOutput, bivariate_product::BivariateProductSumcheckProver, prove_single,
	},
};
use binius_math::{
	FieldBuffer,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_zero},
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

	// Fold each segment separately and assemble the witness: the public words at the base of
	// the low half-cube, the hidden words at the base of the high half-cube.
	let (public_words, hidden_words) = words.split_at(key_collection.public.n_words());
	let public_folded = fold_words::<_, P>(public_words, r_j_tensor.as_ref());
	let hidden_folded = fold_words::<_, P>(hidden_words, r_j_tensor.as_ref());
	let witness_folded =
		assemble_witness(&public_folded, &hidden_folded, key_collection.log_witness_words());

	let monster_multilinear =
		build_monster_multilinear(key_collection, bitand_data, intmul_data, &r_j, &r_s);

	run_sumcheck(witness_folded, public_words, monster_multilinear, r_j, gamma, channel)
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

/// Executes the bivariate product sumcheck for the witness and monster multilinear
/// relationship.
///
/// This helper function runs the sumcheck protocol to prove the relationship between
/// the witness and monster multilinear, then sends the hidden-segment evaluation: the
/// verifier reconstructs the full witness evaluation from it and its own public words.
///
/// # Parameters
/// - `witness_folded`: The witness folded at challenges `r_j`
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
pub fn run_sumcheck<F, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>>(
	witness_folded: FieldBuffer<P>,
	public_words: &[Word],
	monster_multilinear: FieldBuffer<P>,
	r_j: Vec<F>,
	gamma: F,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField + From<AESTowerField8b>,
{
	#[cfg(debug_assertions)]
	let cloned_witness_folded_for_debugging = witness_folded.clone();

	// Run sumcheck on bivariate product
	let prover = BivariateProductSumcheckProver::new([witness_folded, monster_multilinear], gamma);

	let ProveSingleOutput {
		multilinear_evals,
		challenges: mut r_y,
	} = prove_single(prover, channel);

	// Reverse the challenges to get the evaluation point.
	r_y.reverse();

	let [trace_eval, _monster_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");

	#[cfg(debug_assertions)]
	{
		let r_y_tensor = eq_ind_partial_eval(&r_y);
		let expected_trace_eval = binius_math::inner_product::inner_product_buffers(
			&cloned_witness_folded_for_debugging,
			&r_y_tensor,
		);
		debug_assert_eq!(trace_eval, expected_trace_eval);
	}

	// Derive the witness evaluation from the trace evaluation by evaluating the public segment
	// (cheap, like the verifier does), subtracting its padded contribution off and scaling.
	// This makes the protocol incomplete with negligible probability, when the segment selector
	// challenge is zero.
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
