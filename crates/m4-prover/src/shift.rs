// Copyright 2026 The Binius Developers

//! The batched shift-reduction prover for the data-parallel Binius64 M4 proof system.

use std::iter;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, multilinear::eq::eq_ind_partial_eval, univariate::lagrange_evals,
};
use binius_prover::{
	fold_word::fold_words,
	protocols::shift::{
		KeyCollection, KeySegment, OperatorClaims, PreparedOperatorClaims, ShiftChallengePoint,
		ShiftIndOutput, ShiftIndSumcheck, ShiftOutput,
		phase_1::{Phase1Output, SparseShiftRows},
		phase_2::run_sumcheck,
	},
};

use crate::witness::{FoldedWitness, FoldedWord};

/// Proves the batched shift-reduction, collapsing every operation's claims into one.
///
/// The result is a single multilinear claim on the batched witness.
///
/// This mirrors the single-instance shift reduction, with one difference.
/// The hidden witness enters already folded over the instance axis, as a [`FoldedWitness`].
/// The public words are constants shared by every instance, so they are passed as raw words.
///
/// The two phases call the single-instance prover's own subroutines.
/// Phase 1 accumulates the hidden segment's g rows with [`build_g_from_folded_words`].
/// It accumulates the public segment's with the single-instance `build_g`, then collects both
/// segments' rows into the sparse `g` the phase-1 sumcheck reads.
/// Phase 2 contracts the hidden witness along its bit axis with [`FoldedWitness::fold_bits`].
/// It folds the public segment the same way, at the same challenge `r_j`.
/// It then reuses `build_monster_segments` and `run_sumcheck` unchanged.
///
/// # Parameters
/// - `key_collection`: the prover's key collection for the constraint system.
/// - `public_words`: the public (constant) words, shared by every instance.
/// - `folded_witness`: the hidden witness, folded over the instance axis.
/// - `claims`: the operand evaluation claim of each operation.
/// - `domain_subspace`: the univariate evaluation domain.
/// - `channel`: the prover channel driving the interactive protocol.
/// - `alloc`: the allocator backing the reduction's intermediate buffers.
///
/// # Returns
/// The final challenges with the reduced witness evaluation, and the wiring multilinear's
/// evaluation for the caller to send.
pub fn prove<F, P, Channel, A>(
	key_collection: &KeyCollection,
	public_words: &[Word],
	folded_witness: &FoldedWitness<F, A>,
	claims: OperatorClaims<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
	alloc: &A,
) -> ShiftOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	// One batching coefficient per operation, drawn from the channel.
	// SOUNDNESS: `prepare` draws in the order the verifier draws in; do not reorder it.
	let prepared = claims.prepare(|| channel.sample());

	// Phase 1: accumulate the g rows once per key segment. The public words are constants shared
	// by every instance, so the single-instance builder folds them directly from their bits; the
	// hidden words are already folded over instances. This scalar path drives the single-instance
	// phase-1 sumcheck.
	let public = key_collection
		.public
		.build_g::<F, F>(public_words, &prepared);
	let hidden =
		build_g_from_folded_words(folded_witness.words(), &key_collection.hidden, &prepared);
	// `g` is the sum of its rows, so the two segments' rows concatenate rather than merge.
	let g = SparseShiftRows::from_segments([
		(&public, &key_collection.public.dense_shift_enc),
		(&hidden, &key_collection.hidden.dense_shift_enc),
	]);
	// The oblong weights, which phase 1 pushes through both shift slots and phase 3 runs its rounds
	// over directly.
	let oblong_weights = lagrange_evals(domain_subspace, prepared.bitand.r_zhat_prime);
	let Phase1Output {
		r_j,
		inner: inner_shift,
		outer: outer_shift,
		psi,
		gamma,
		g_eval,
	} = g.run_phase_1_sumcheck(oblong_weights.as_ref(), prepared.batched_eval(), channel, alloc);

	// Phases 2 and 3 bind the two bit indices the shift indicators chain through, working back up
	// the chain — see the single-instance prover for what each run carries.
	let inner = ShiftIndSumcheck::<P, _>::new(
		alloc,
		&psi,
		&ShiftChallengePoint::new(&r_j, &inner_shift),
		g_eval,
	);
	debug_assert_eq!(inner.beta(), gamma);
	let inner_output = inner.prove(channel, alloc);

	let outer = ShiftIndSumcheck::<P, _>::new(
		alloc,
		oblong_weights.as_ref(),
		&ShiftChallengePoint::new(&inner_output.point, &outer_shift),
		inner_output.ind_eval * g_eval,
	);
	debug_assert_eq!(outer.beta(), inner_output.eval);
	let ShiftIndOutput {
		weights_eval,
		ind_eval,
		eval: epsilon,
		point: _,
	} = outer.prove(channel, alloc);
	// The monster multilinear is scaled by the three bit-index factors the two runs reduced.
	let shift_ind_eval = weights_eval * ind_eval * inner_output.ind_eval;

	// Phase 4: the witness folded at the bit position challenge `r_j`, per segment.

	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);
	// The public fold is a raw-word fold; the hidden fold contracts the already-oblong bits.
	let public_folded = fold_words::<F, P, _>(alloc, public_words, r_j_tensor.as_ref());
	let hidden_folded = folded_witness.fold_bits::<P>(r_j_tensor.as_ref(), alloc);

	let (public_monster, hidden_monster) = key_collection.build_monster_segments::<F, P, _>(
		alloc,
		&prepared,
		shift_ind_eval,
		&inner_shift,
		&outer_shift,
	);

	run_sumcheck::<F, P, _, _>(
		&public_folded,
		hidden_folded,
		&public_monster,
		hidden_monster,
		shift_ind_eval,
		public_words,
		r_j,
		epsilon,
		channel,
		alloc,
	)
}

/// Accumulates the phase-1 "g" rows of one key segment, from instance-folded words.
///
/// This is the batched analogue of the single-instance key-segment method.
/// It consumes a key segment's words already folded over the instance axis.
/// So each word is a [`FoldedWord`], whose bits are field elements rather than a packed `u64`.
///
/// The single-instance builder scatters an accumulator onto a word's set bits by masking.
/// This one scales the accumulator by each folded bit with a field multiplication.
/// The two coincide when the folded bit is 0 or 1.
///
/// Use this for the hidden (committed) segment, whose words are folded over instances.
/// Use the single-instance method for the public segment, whose words are shared constants.
/// Collecting both results' rows gives the complete g multilinear.
///
/// # Arguments
///
/// - `folded_words`: the segment's words folded over instances, in `segment.key_ranges` order.
/// - `segment`: the shift keys of this segment, grouped by the word they act on.
/// - `prepared`: the per-operation claims, indexed by the operation each key names.
///
/// Words past the segment's key ranges are power-of-two padding, and are ignored.
///
/// # Returns
///
/// One row of `Word::BITS` scalars per shift the segment uses, in dense shift index order:
///
/// ```text
/// rows[key.dense_shift_idx * Word::BITS + bit] += folded_bit * acc(key)
/// ```
///
/// where `acc(key)` is the key's lambda-weighted partial evaluation tensor.
/// Every word carrying that key accumulates into the same row.
///
/// This scalar implementation ignores the single-instance builder's packing and parallelism.
pub fn build_g_from_folded_words<F: BinaryField>(
	folded_words: &[FoldedWord<F>],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
) -> Box<[F]> {
	// One zeroed row per shift the segment uses. A key names one such row, so the accumulation
	// below adds straight into it.
	let mut rows = vec![F::ZERO; segment.dense_shift_enc.len() * Word::BITS].into_boxed_slice();

	// Each folded word carries the keys `word_keys` names at its position; words past
	// `n_words()` are power-of-two padding.
	for (index, word) in folded_words.iter().enumerate().take(segment.n_words()) {
		let keys = segment.word_keys(index);
		for key in keys {
			let operator_data = &prepared[key.operation];

			// The lambda-weighted partial evaluation tensor for this shifted word.
			let acc = key.accumulate(
				&segment.constraint_indices,
				operator_data.r_x_prime_tensor.as_ref(),
				&operator_data.lambda_powers,
			);

			let base = key.dense_shift_idx as usize * Word::BITS;
			let slots = &mut rows[base..base + Word::BITS];

			// Scatter the accumulator across this key's bit slots, scaling each by the folded bit.
			for (slot, &folded_bit) in iter::zip(slots, word) {
				*slot += acc * folded_bit;
			}
		}
	}

	rows
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_core::{
		ValueTable,
		constraint_system::{AndConstraint, InoutSegment},
		word::Word,
	};
	use binius_field::{AESTowerField8b, Field, PackedBinaryGhash1x128b, Random};
	use binius_math::{
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::random_scalars,
		univariate::lagrange_evals_scalars,
	};
	use binius_prover::protocols::shift::{
		DenseShiftEncoding, OperatorData, build_key_collection, monster::shift_operator_row,
		outer::decode_shift,
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{
		config::{B128, StdChallenger},
		protocols::shift::{
			LOG_SHIFT_COUNT, OperatorData as VerifierOperatorData, SHIFT_COUNT, check_eval,
			evaluate_words_mle, verify,
		},
	};
	use rand::prelude::*;

	use super::*;
	use crate::{
		test_utils::{N_INPUT_WORDS, crc64_circuit, populate_crc64_witness},
		witness::OperandColumns,
	};

	// The oblong evaluation of each bitand operand column A, B, C at the shift challenges.
	//
	// Builds the batched AND witness, then for each column folds its word bits by the Lagrange
	// basis at r_z and evaluates the resulting row multilinear at the (instance, constraint) point
	// r_rho || r_x. The columns are constraint-major, so r_rho (low) indexes the instance within a
	// constraint and r_x (high) indexes the constraint.
	fn evaluate_and_witness<P: PackedField<Scalar = B128>>(
		table: &ValueTable,
		constants: &[Word],
		and_constraints: &[AndConstraint],
		domain_subspace: &BinarySubspace<B128>,
		r_z: B128,
		r_x: &[B128],
		r_rho: &[B128],
	) -> [B128; 3] {
		let columns = OperandColumns::build(table, constants, and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();
		let lagrange = lagrange_evals_scalars::<B128, B128>(domain_subspace, &r_z);
		let row_point: Vec<B128> = r_rho.iter().chain(r_x).copied().collect();
		let operand_eval = |column: &[Word]| {
			let folded_column = fold_words::<B128, P, _>(&GlobalAllocator, column, &lagrange);
			evaluate(&folded_column, &row_point)
		};
		// The batch witness stores only the `A` and `B` columns.
		// The reduction reads `C` as the word-by-word AND of the two.
		// Materialize that same derived column so its evaluation matches the reduction's claim.
		let c_column: Vec<Word> = iter::zip(a, b).map(|(&a, &b)| a & b).collect();
		[operand_eval(a), operand_eval(b), operand_eval(&c_column)]
	}

	// Folds a contiguous run of value-vector words over the instance axis, one FoldedWord per word.
	// This lets the public and hidden segments be folded separately, matching how `build_g`
	// consumes one segment at a time.
	fn fold_words_over_instances(
		table: &ValueTable,
		constants: &[Word],
		r_rho: &[B128],
		words: std::ops::Range<usize>,
	) -> Vec<FoldedWord<B128>> {
		let eq = eq_ind_partial_eval_scalars::<B128>(r_rho);
		let mut folded = vec![[B128::ZERO; Word::BITS]; words.len()];
		for (rho, &weight) in eq.iter().enumerate() {
			// Reconstruct this instance independently of the fold, then fold its chosen word range.
			let vv = table.instance_value_vec(rho, constants);
			for (word, out) in vv.combined_witness()[words.clone()].iter().zip(&mut folded) {
				for (b, out_b) in out.iter_mut().enumerate() {
					if (word.0 >> b) & 1 == 1 {
						*out_b += weight;
					}
				}
			}
		}
		folded
	}

	// The batched prove round-trips with the single-instance shift verifier: the two agree on the
	// reduced challenges and witness evaluation, and that evaluation equals the direct evaluation
	// of the instance-folded witness. The prover feeds the verifier's own subroutines, so the
	// transcript is exactly what the single-instance verifier expects.
	#[test]
	fn prove_and_verify_round_trip() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		let log_instances = 6;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		let key_collection = build_key_collection(&cs, InoutSegment::Hidden);

		// The univariate bit challenge, the constraint challenge, and the instance challenge.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, cs.log_and_constraints().unwrap_or(0));
		// The Zero claim closes at its own constraint point, as wide as the ZERO set. Its value is
		// zero at any point: a satisfied ZERO constraint array vanishes identically, so its
		// multilinear extension is the zero polynomial.
		let r_x_zero = random_scalars::<B128>(&mut rng, cs.log_zero_constraints().unwrap_or(0));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		// The hidden witness folded over instances (one FoldedWord per committed word), and the
		// public constants.
		let folded_witness =
			FoldedWitness::<B128, _>::fold_instances(&table, &r_rho, &GlobalAllocator);
		let public_words = &cs.constants;

		// The bitand operand evals at (r_z, r_x, r_rho); the circuit has no IMUL constraints, so
		// the intmul claim is the zero claim over an empty point.
		let bitand_evals = evaluate_and_witness::<P>(
			&table,
			public_words,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);
		let intmul_evals = [B128::ZERO; 4];

		// Prove.
		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let prover_output = prove::<B128, P, _, _>(
			&key_collection,
			public_words,
			&folded_witness,
			OperatorClaims {
				zero: OperatorData {
					evals: [B128::ZERO],
					r_zhat_prime: r_z,
					r_x_prime: r_x_zero.clone(),
				},
				bitand: OperatorData {
					evals: bitand_evals,
					r_zhat_prime: r_z,
					r_x_prime: r_x.clone(),
				},
				intmul: OperatorData::zero_claim(r_z),
				binmul: OperatorData::zero_claim(r_z),
			},
			&domain_subspace,
			&mut prover_transcript,
			&GlobalAllocator,
		);

		// The full reduction sends this after the public segment's evaluation claim; driving the
		// shift alone, it follows the reduction directly.
		prover_transcript.send_public_claim(prover_output.wiring_eval);

		// Verify against the single-instance shift verifier.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_zero = VerifierOperatorData::new(r_x_zero, [B128::ZERO]);
		let verifier_bitand = VerifierOperatorData::new(r_x, bitand_evals);
		let verifier_intmul = VerifierOperatorData::new(Vec::new(), intmul_evals);
		let verifier_bmul = VerifierOperatorData::new(Vec::new(), [B128::ZERO; 6]);
		let verifier_output = verify(
			&cs,
			InoutSegment::Hidden,
			&verifier_zero,
			&verifier_bitand,
			&verifier_intmul,
			&verifier_bmul,
			&mut verifier_transcript,
		)
		.unwrap();
		// The public segment over the shift's whole index space. The full reduction reads this
		// from the prover and ties it to the constants with a ring-switch; driving the shift
		// alone, evaluate it here.
		let public_eval = evaluate_words_mle::<B128, B128>(
			public_words,
			verifier_output.r_j(),
			verifier_output.r_y(),
		);

		let wiring_claim = check_eval(
			&cs,
			InoutSegment::Hidden,
			public_eval,
			&verifier_zero,
			&verifier_bitand,
			&verifier_intmul,
			&verifier_bmul,
			&domain_subspace,
			&r_z,
			&verifier_output,
			&mut verifier_transcript,
		)
		.unwrap();

		// Discharge the wiring claim the way the full reduction's caller does.
		wiring_claim.check_native().unwrap();
		verifier_transcript.finalize().unwrap();

		// The witness evaluation equals the instance-folded witness evaluated at the point, with
		// the segment's zero-padding contributing the (1 - r) factors above the folded length.
		let r_y = verifier_output.r_y();
		let log_folded = folded_witness.log_padded_words();
		let base = folded_witness.evaluate(verifier_output.r_j(), &r_y[..log_folded]);
		let expected_eval = r_y[log_folded..]
			.iter()
			.fold(base, |acc, &r_y_i| acc * (B128::ONE - r_y_i));
		assert_eq!(expected_eval, verifier_output.witness_eval);

		// Prover and verifier agree on the reduced challenges and the witness evaluation.
		let eval_point = [
			verifier_output.r_j(),
			r_y,
			std::slice::from_ref(&verifier_output.r_segment),
		]
		.concat();
		assert_eq!(prover_output.sumcheck.challenges, eval_point);
		assert_eq!(prover_output.sumcheck.eval, verifier_output.witness_eval);
	}

	// The phase-1 identity: summing the g·h inner products over the shift variants reconstructs the
	// lambda-batched operand evaluation claim.
	//
	// The g multilinear comes from the batched build_g on the full folded witness; h comes from
	// pushing the oblong weights at the same univariate challenge r_z through each term's two shift
	// slots. Their inner product must equal the lambda-powers scaling of the batched AND-check
	// operand evals (the intmul claim is empty, contributing nothing).
	#[test]
	fn phase_1_g_h_inner_product_matches_batched_evals() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		let log_instances = 6;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);
		let constants = &c.circuit.constraint_system().constants;

		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		let key_collection = build_key_collection(&cs, InoutSegment::Hidden);

		// The univariate bit challenge, the constraint challenge, and the instance challenge.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, cs.log_and_constraints().unwrap_or(0));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		// The batched AND-check operand evals at (r_z, r_x, r_rho), and the full folded witness at
		// the same r_rho, so g and the claim agree on the instance point.
		let bitand_evals = evaluate_and_witness::<P>(
			&table,
			constants,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);
		// The hidden segment spans value indices `[offset_inout, combined_len)`.
		let offset = table.layout().offset_inout();
		let combined = table.layout().combined_len();
		assert_eq!(table.n_hidden_words(), combined - offset);
		let public_words = &cs.constants;
		let hidden_folded = fold_words_over_instances(&table, constants, &r_rho, offset..combined);

		// Prepare the operator data: lambda batches the three operand claims. The circuit has no
		// IMUL or BMUL constraints, so those two are zero claims at an empty point.
		// The ZERO set has its own constraint point, as wide as the set itself.
		let claims = OperatorClaims {
			zero: OperatorData {
				evals: [B128::ZERO],
				r_zhat_prime: r_z,
				r_x_prime: random_scalars::<B128>(&mut rng, cs.log_zero_constraints().unwrap_or(0)),
			},
			bitand: OperatorData {
				evals: bitand_evals,
				r_zhat_prime: r_z,
				r_x_prime: r_x,
			},
			intmul: OperatorData::zero_claim(r_z),
			binmul: OperatorData::zero_claim(r_z),
		};
		let prepared = claims.prepare(|| B128::random(&mut rng));

		// The g rows: the public segment folds from raw constant words via the single-instance
		// builder, the hidden segment from the instance-folded words. The h multilinear comes
		// from the single-instance prover.
		let public = key_collection
			.public
			.build_g::<B128, B128>(public_words, &prepared);
		let hidden = build_g_from_folded_words(&hidden_folded, &key_collection.hidden, &prepared);
		let psi = lagrange_evals(&domain_subspace, r_z);

		// `g` is zero outside the rows the segments name, so the inner product is those rows
		// alone. A term names a shift *pair*, and its `h` row is the operator applied twice: the
		// oblong weights push through the outer slot, and that row pushes through the inner. The
		// composed table the two slots span holds `2^24` entries and is never formed — this walks
		// the one row per term instead, which is all the inner product reads.
		let segment_inner_product = |rows: &[B128], enc: &DenseShiftEncoding| {
			iter::zip(enc.shift_indices(), rows.chunks_exact(Word::BITS))
				.map(|(quadruple, row)| {
					let (outer_variant, outer_amount) = decode_shift(quadruple >> LOG_SHIFT_COUNT);
					let (inner_variant, inner_amount) = decode_shift(quadruple % SHIFT_COUNT);
					let mut eta_row = [B128::ZERO; Word::BITS];
					let mut h_row = [B128::ZERO; Word::BITS];
					shift_operator_row(outer_variant, outer_amount, &mut eta_row, psi.as_ref());
					shift_operator_row(inner_variant, inner_amount, &mut h_row, &eta_row);
					iter::zip(row, h_row)
						.map(|(&value, h)| value * h)
						.sum::<B128>()
				})
				.sum::<B128>()
		};
		let inner_product = segment_inner_product(&public, &key_collection.public.dense_shift_enc)
			+ segment_inner_product(&hidden, &key_collection.hidden.dense_shift_enc);

		// The lambda-powers scaling of the batched AND-check evals, plus the empty intmul claim.
		let expected = prepared.bitand.batched_eval + prepared.intmul.batched_eval;
		assert_eq!(inner_product, expected);
	}
}
