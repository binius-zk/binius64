// Copyright 2025 Irreducible Inc.

//! Batched BitAnd shift reduction prover for the M4 data-parallel proof system.
//!
//! M4 commits `K = 2^k` instances of one circuit, stacked instance-major.
//! The batched BitAnd reduction runs over `K * n_and` rows.
//! Its operand claim splits into a local constraint part `r_x` and an instance part `r_kappa`.
//!
//! The instance index enters the claim only through the committed witness bit.
//! The shift structure is shared by every instance, so the instance factor folds onto the witness:
//!
//! ```text
//!   W_tilde(j, y) = sum_kappa eq(r_kappa, kappa) * bit_j(w_kappa[y]) = W_hat(j, y, r_kappa)
//! ```
//!
//! Each 0/1 bit becomes one field element, folded across the batch rows.
//! The two-phase reduction then runs on this virtual instance over the flat committed witness.
//! Its claim is `W_hat(r_j, r_y, r_kappa)`, with `r_kappa` on the high (instance) coordinates.
//!
//! This handles BitAnd only; IntMul is out of scope for the initial M4 batch.
//! It reduces to the flat committed witness, matching the M4 batch commitment.
//!
//! The flat monster and phase-2 sumcheck here duplicate the segmented single-instance path.
//! This monster and the verifier's monster evaluation are independent implementations.
//! The round-trip tests over shifted operands cross-check them and catch any drift.

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		ProveSingleOutput, bivariate_product::BivariateProductSumcheckProver, prove_single,
	},
};
use binius_math::{
	BinarySubspace, FieldBuffer, multilinear::eq::eq_ind_partial_eval_scalars,
	univariate::lagrange_evals,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, LOG_WORDS_PER_ELEM, WORD_SIZE_BITS},
	protocols::shift::{BITAND_ARITY, SHIFT_VARIANT_COUNT, evaluate_h_op},
};
use tracing::instrument;

use super::{
	key_collection::{KeyCollection, KeySegment, Operation},
	monster::build_h_parts,
	phase_1::run_phase_1_sumcheck,
	prove::{OperatorData, PreparedOperatorData},
};

/// Number of variables in the phase-1 `g` and `h` multilinears: bit index `j` plus shift amount
/// `s`.
const LOG_LEN: usize = LOG_WORD_SIZE_BITS + LOG_WORD_SIZE_BITS;

/// Proves the batched BitAnd shift reduction over `K = 2^k` instances of one circuit.
///
/// The reduction leaves one operand claim per operand `(a, b, c)`.
/// Each claim sits at a point `(r_zhat, r_x, r_kappa)`: bit challenge, local constraint, instance.
/// It reduces the claims to a single evaluation of the committed batch witness.
///
/// # Algorithm
///
/// - Fold the batch onto one field-valued virtual instance: `W_tilde(j, y) = sum_kappa eq(r_kappa,
///   kappa) * bit_j(w_kappa[y])`.
/// - Phase 1 over `(j, s)` reduces the batched operand claim to a point `(r_j, r_s)`.
/// - Phase 2 over the word index `y` reduces to the witness evaluation.
///
/// The `g` parts and the phase-2 fold are linear in the witness.
/// So building them from the folded witness is exact, with no per-instance error.
///
/// # Arguments
///
/// - `key_collection`: the per-instance shift structure, shared by every instance.
/// - `instances`: one committed-word slice per instance, in instance order.
/// - `r_kappa`: the instance challenge of length `k`, the high coordinates of the reduction point.
/// - `bitand_data`: the BitAnd operand claim, with bit challenge `r_zhat`, local `r_x`, and evals.
/// - `channel`: the prover channel.
///
/// # Returns
///
/// The reduced claim: challenges `[r_j, r_y]` and the witness evaluation `W_tilde(r_j, r_y)`.
/// The caller appends `r_kappa` to form the committed-witness point `(r_j, r_y, r_kappa)`.
///
/// # Panics
///
/// Panics if the instance count is not `2^{r_kappa.len()}`.
/// Panics if any instance slice is shorter than the per-instance committed-word count.
/// In debug builds, panics if the key collection holds any non-BitAnd key.
#[instrument(skip_all, name = "batch_shift_prove")]
pub fn prove_batch<F, P, Channel>(
	key_collection: &KeyCollection,
	instances: &[&[Word]],
	r_kappa: &[F],
	bitand_data: OperatorData<F>,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField + From<AESTowerField8b>,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	// The batch is a clean hypercube of instances: exactly 2^k of them.
	let n_instances = 1usize << r_kappa.len();
	assert_eq!(instances.len(), n_instances, "instance count must be 2^r_kappa.len()");

	// This reduction covers BitAnd operands only.
	// A non-BitAnd key would be silently skipped, leaving its operand claim unreduced.
	debug_assert!(
		key_collection
			.public
			.keys
			.iter()
			.chain(&key_collection.hidden.keys)
			.all(|key| key.operation == Operation::BitwiseAnd),
		"batched shift reduction handles BitAnd only: the constraint system must have no MUL constraints"
	);

	// Every committed word of a single instance, across both value-vector segments.
	let n_words = key_collection.n_words();

	// Each instance slice must hold at least the whole per-instance committed witness.
	// The fold reads word y of every instance, for y in 0..n_words.
	assert!(
		instances.iter().all(|words| words.len() >= n_words),
		"each instance slice must hold at least the per-instance committed-word count"
	);

	// Sample the operand-batching coefficient for (a, b, c).
	// Only BitAnd is reduced here, so a single lambda suffices.
	let bitand_lambda = channel.sample();

	// The univariate bit challenge is shared by the fold and both phases.
	let r_zhat_prime = bitand_data.r_zhat_prime;
	let bitand_prep = PreparedOperatorData::new(bitand_data, bitand_lambda);

	// Fold the batch onto one virtual instance.
	//
	//     W_tilde[y][j] = sum_kappa eq(r_kappa, kappa) * bit_j(w_kappa[y])
	//
	// Each (bit, word) becomes one field element; the batch index is gone after this step.
	let w_tilde = fold_batch(instances, r_kappa, n_words);

	// Phase 1: the operand claim as a sum over (j, s) of g * h.
	//
	//     g_sigma(j, s) = sum_y W_tilde[y][j] * (constraint tensor weight of this word-key)
	//     h_sigma(j, s) = sum_i L_tilde_i(r_zhat) * shift_ind_sigma(i, j, s)   (public)
	let g_parts = build_batch_g_parts::<F, P>(&w_tilde, key_collection, &bitand_prep);
	let h_parts = build_h_parts::<F, P>(r_zhat_prime);

	let SumcheckOutput {
		challenges: mut r_jr_s,
		eval: gamma,
	} = run_phase_1_sumcheck(g_parts, h_parts, channel);

	// Split the phase-1 challenges: r_j is the low bit-index half, r_s the high shift-amount half.
	let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
	let r_j = r_jr_s;

	// Phase 2: pair the r_j-folded witness against the monster over the flat word index y.
	let monster = build_flat_monster::<F, P>(key_collection, &bitand_prep, &r_j, &r_s);
	let mut r_j_witness = fold_witness_at_r_j::<F, P>(&w_tilde, &r_j);

	// The monster rounds the word count up to at least one committed field element.
	// Pad the folded witness to the same variable count so the bivariate product lines up.
	if r_j_witness.log_len() < monster.log_len() {
		r_j_witness.zero_extend(monster.log_len());
	}

	run_flat_sumcheck(r_j_witness, monster, r_j, gamma, channel)
}

/// Folds the batch onto one field-valued virtual instance.
///
/// The result is indexed `[word][bit]`:
///
/// ```text
///   W_tilde[y][j] = sum_kappa eq(r_kappa, kappa) * bit_j(w_kappa[y])
/// ```
///
/// The words are independent, so the fold parallelizes over the word index.
fn fold_batch<F: BinaryField>(
	instances: &[&[Word]],
	r_kappa: &[F],
	n_words: usize,
) -> Vec<[F; WORD_SIZE_BITS]> {
	// eq(r_kappa, .) expanded over the 2^k instances: one weight per instance.
	let eps = eq_ind_partial_eval_scalars(r_kappa);

	let mut w_tilde = vec![[F::ZERO; WORD_SIZE_BITS]; n_words];
	w_tilde.par_iter_mut().enumerate().for_each(|(y, row)| {
		// Accumulate every instance's contribution to word y at its set bit positions.
		for (words, &eps_k) in instances.iter().zip(&eps) {
			let bits = words[y].0;
			for (j, slot) in row.iter_mut().enumerate() {
				// A set bit contributes the instance weight; a clear bit contributes nothing.
				if (bits >> j) & 1 == 1 {
					*slot += eps_k;
				}
			}
		}
	});
	w_tilde
}

/// Builds the phase-1 `g` multilinears from the folded witness.
///
/// A key's accumulated weight `acc` batches its operands by the lambda powers.
/// It lands in the `(variant, amount)` block chosen by the key, spread over the bit index `j`:
///
/// ```text
///   g[variant, amount][j] += W_tilde[y][j] * acc     for each word y and each key on it
/// ```
///
/// The scalar layout is `id * WORD_SIZE_BITS + j`, the same as the single-instance builder.
/// The block is `id = variant * WORD_SIZE_BITS + amount`, then bit index `j` in the low bits.
fn build_batch_g_parts<F, P>(
	w_tilde: &[[F; WORD_SIZE_BITS]],
	key_collection: &KeyCollection,
	bitand_prep: &PreparedOperatorData<F>,
) -> [FieldBuffer<P>; SHIFT_VARIANT_COUNT]
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	// One flat scalar buffer holding all 8 variant blocks of the (s, j) hypercube.
	// Heap-allocated on purpose: 8 * 2^12 field elements is far too large for the stack.
	#[allow(clippy::useless_vec)]
	let mut g_flat = vec![F::ZERO; SHIFT_VARIANT_COUNT << LOG_LEN];

	// The folded words are in flat value-vector order: the public segment, then the hidden one.
	let n_public = key_collection.public.n_words();
	let mut accumulate_segment = |segment: &KeySegment, word_offset: usize| {
		for w in 0..segment.n_words() {
			let row = &w_tilde[word_offset + w];
			for key in segment.word_keys(w) {
				// IntMul is out of scope; a prepared BitAnd-only system has only BitAnd keys
				// anyway.
				if key.operation != Operation::BitwiseAnd {
					continue;
				}

				// acc batches the key's operands by the lambda powers over its r_x tensor weights.
				let acc = key.accumulate(
					&segment.constraint_indices,
					bitand_prep.r_x_prime_tensor.as_ref(),
					&bitand_prep.lambda_powers,
				);

				// key.id selects the (variant, amount) block; j indexes the bit within the word.
				let base = key.id as usize * WORD_SIZE_BITS;
				for (j, &w_val) in row.iter().enumerate() {
					g_flat[base + j] += w_val * acc;
				}
			}
		}
	};
	accumulate_segment(&key_collection.public, 0);
	accumulate_segment(&key_collection.hidden, n_public);

	// Split the flat buffer into one multilinear per shift variant.
	g_flat
		.chunks(1 << LOG_LEN)
		.map(FieldBuffer::from_values)
		.collect::<Vec<_>>()
		.try_into()
		.expect("g_flat has SHIFT_VARIANT_COUNT blocks of 2^LOG_LEN scalars")
}

/// Builds the phase-2 monster multilinear over the flat committed word index.
///
/// The monster is witness-independent: a public function of the constraint system and challenges.
/// Word `y` holds the summed contribution of its keys:
///
/// ```text
///   monster[y] = sum_key sum_operand lambda^operand * h_op[variant](r_j, r_s) * eq(r_s, amount) * acc
/// ```
///
/// The words are in flat value-vector order: the public segment, then the hidden one.
fn build_flat_monster<F, P>(
	key_collection: &KeyCollection,
	bitand_prep: &PreparedOperatorData<F>,
	r_j: &[F],
	r_s: &[F],
) -> FieldBuffer<P>
where
	F: BinaryField + From<AESTowerField8b>,
	P: PackedField<Scalar = F>,
{
	// The shift kernels evaluated at (r_j, r_s), collapsed at the univariate bit challenge.
	let subspace = BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
	let l_tilde = lagrange_evals(&subspace, bitand_prep.r_zhat_prime);
	let h_ops = evaluate_h_op(l_tilde.as_ref(), r_j, r_s);
	let r_s_tensor = eq_ind_partial_eval_scalars(r_s);

	// The per-(operand, variant, amount) scalar folded into each key contribution.
	// Layout: operand is the top block, then variant, then amount in the low bits.
	// Heap-allocated on purpose: keep this off the stack.
	#[allow(clippy::useless_vec)]
	let mut scalars = vec![F::ZERO; BITAND_ARITY * SHIFT_VARIANT_COUNT * WORD_SIZE_BITS];
	for operand in 0..BITAND_ARITY {
		for variant in 0..SHIFT_VARIANT_COUNT {
			let operand_variant = bitand_prep.lambda_powers[operand] * h_ops[variant];
			for s in 0..WORD_SIZE_BITS {
				scalars[(operand * SHIFT_VARIANT_COUNT + variant) * WORD_SIZE_BITS + s] =
					operand_variant * r_s_tensor[s];
			}
		}
	}

	// The contribution of one word of a segment: sum over its keys and their operands.
	let word_scalar = |segment: &KeySegment, w: usize| -> F {
		segment
			.word_keys(w)
			.iter()
			.filter(|key| key.operation == Operation::BitwiseAnd)
			.map(|key| {
				key.accumulate_by_operand(&segment.constraint_indices, bitand_prep)
					.map(|(operand, acc)| {
						acc * scalars
							[key.id as usize + operand * SHIFT_VARIANT_COUNT * WORD_SIZE_BITS]
					})
					.sum::<F>()
			})
			.sum()
	};

	// Place each word at its flat index: public segment first, then the hidden segment.
	let n_words = key_collection.n_words();
	let log_len = log2_ceil_usize(n_words).max(LOG_WORDS_PER_ELEM);
	let mut monster = vec![F::ZERO; 1 << log_len];
	let n_public = key_collection.public.n_words();
	for w in 0..n_public {
		monster[w] = word_scalar(&key_collection.public, w);
	}
	for w in 0..key_collection.hidden.n_words() {
		monster[n_public + w] = word_scalar(&key_collection.hidden, w);
	}

	FieldBuffer::from_values(&monster)
}

/// Folds the field-valued witness at the bit-index challenge `r_j`.
///
/// The result is a multilinear over the flat word index:
///
/// ```text
///   folded[y] = sum_j eq(r_j, j) * W_tilde[y][j]
/// ```
fn fold_witness_at_r_j<F, P>(w_tilde: &[[F; WORD_SIZE_BITS]], r_j: &[F]) -> FieldBuffer<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	// eq(r_j, .) over the 64 bit positions.
	let r_j_tensor = eq_ind_partial_eval_scalars(r_j);

	// The word count need not be a power of two; the high words round up to zero.
	let n_words = w_tilde.len();
	let log_len = log2_ceil_usize(n_words);

	// One inner product per word collapses the bit dimension; padded words stay zero.
	let mut folded = vec![F::ZERO; 1 << log_len];
	folded[..n_words]
		.par_iter_mut()
		.zip(w_tilde.par_iter())
		.for_each(|(out, row)| {
			*out = row.iter().zip(&r_j_tensor).map(|(&w, &e)| w * e).sum();
		});

	FieldBuffer::from_values(&folded)
}

/// Runs the phase-2 bivariate product sumcheck and sends the witness evaluation.
///
/// It reduces `gamma = sum_y r_j_witness(y) * monster(y)` to a product at the sumcheck point.
/// It then sends the folded witness evaluation for the verifier's final check.
fn run_flat_sumcheck<F, P, Channel>(
	r_j_witness: FieldBuffer<P>,
	monster: FieldBuffer<P>,
	r_j: Vec<F>,
	gamma: F,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	let prover = BivariateProductSumcheckProver::new([r_j_witness, monster], gamma);

	let ProveSingleOutput {
		multilinear_evals,
		challenges: mut r_y,
	} = prove_single(prover, channel);

	// The evaluation point orders the word index from most to least significant.
	r_y.reverse();

	// The folded witness evaluation is the reduction's output claim.
	let [witness_eval, _monster_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");
	channel.send_one(witness_eval);

	SumcheckOutput {
		challenges: [r_j, r_y].concat(),
		eval: witness_eval,
	}
}
