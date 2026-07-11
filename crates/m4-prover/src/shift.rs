// Copyright 2026 The Binius Developers

//! The batched shift-reduction prover for the data-parallel Binius64 M4 proof system.

#![allow(unused)]

use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer,
	inner_product::inner_product,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
};
use binius_prover::{
	fold_word::{WordFolder, fold_words},
	protocols::shift::{
		KeyCollection, KeySegment, Operation, OperatorData, PreparedOperatorData,
		monster::{build_h_parts, build_monster_segments},
		phase_1::{build_g_parts, run_phase_1_sumcheck},
		phase_2::run_sumcheck,
	},
};
use binius_utils::{checked_arithmetics::log2_strict_usize, rayon::prelude::*};
use binius_verifier::protocols::shift::SHIFT_VARIANT_COUNT;

use crate::ValueTable;

/// The number of variables in each "g" (and "h") multilinear of phase 1: one 6-bit shift-amount
/// axis and one 6-bit bit-position axis.
const LOG_LEN: usize = Word::LOG_BITS + Word::LOG_BITS;

/// A committed witness word after folding its bits into the field.
///
/// Each 64-bit word contributes one field element per bit position, so a folded word is the oblong
/// representation of that word: its bit axis expanded to full field elements.
pub type FoldedWord<F> = [F; Word::BITS];

/// Folds the committed witness of a batch value table along the instance axis.
///
/// The committed witness has three axes:
/// - the bits within each 64-bit word.
/// - the committed words within one instance.
/// - the instances themselves.
///
/// This collapses the instance axis by the equality-indicator weights of `r_rho`.
/// What remains is a multilinear over the other two axes.
///
/// For committed word `w` and bit `b`, the output element is
///
/// ```text
/// out[w][b] = sum_rho eq(r_rho, rho) * bit_b(word[rho][w])
/// ```
///
/// so each set bit contributes its instance's equality weight to a full field element.
///
/// The bit axis occupies the low coordinates and the word axis the high coordinates.
/// The result is a multilinear over `Word::LOG_BITS + log2(n_committed)` variables:
///
/// ```text
/// index = w * Word::BITS + b     (b occupies the low Word::LOG_BITS coordinates)
/// ```
///
/// The table stores exactly the committed (hidden) words, so nothing here is excluded.
/// The constants and public words live once on the constraint system, folded separately.
///
/// The wire-major layout makes this cheap: one wire's values across every instance are stored
/// contiguously, so each word position is a plain sub-slice rather than a strided gather.
///
/// Every word position folds against the same instance point `r_rho`.
/// So the lookup tables and per-chunk weights are built once and shared across all word positions.
/// The word positions are independent, so the fold runs in parallel, one output word per task.
///
/// # Panics
///
/// Panics if `r_rho.len()` does not equal the batch dimension.
pub fn fold_instances<F, P>(table: &ValueTable, r_rho: &[F]) -> FieldBuffer<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	assert_eq!(r_rho.len(), table.log_instances(), "r_rho must match the batch dimension");

	// The wire-major buffer holds one row per committed word, each row spanning all instances.
	let log_instances = table.log_instances();
	let data = table.as_words();

	// The committed word count sizes the word axis.
	// Word positions past it are the multilinear's zero padding up to `2^log_committed`.
	let n_committed = table.n_hidden_words();
	let log_committed = table.layout().log_witness_words();

	// Build the instance-fold tables once; the lookups and weights depend only on r_rho.
	let folder = WordFolder::<F>::new(r_rho);

	// Each output chunk holds one committed word position:
	//     out[w * Word::BITS + b] = sum_rho eq(r_rho, rho) * bit_b(word[rho][w]).
	// The word positions are independent, so fold them in parallel.
	// Positions beyond the committed count keep their zero padding.
	let mut out = vec![F::ZERO; 1 << (Word::LOG_BITS + log_committed)];
	out.par_chunks_mut(Word::BITS)
		.take(n_committed)
		.enumerate()
		.for_each(|(w, slot)| {
			// Word position w across every instance is row w of the wire-major buffer.
			// It is already contiguous, so the fold reads it in place with no gather or copy.
			let column = &data[w << log_instances..(w + 1) << log_instances];
			slot.copy_from_slice(&folder.fold(column));
		});

	FieldBuffer::from_values(&out)
}

/// Proves the batched shift-reduction, reducing the bitand and intmul evaluation claims to a single
/// multilinear claim on the batched witness.
///
/// This mirrors the single-instance shift reduction, but the hidden witness enters already folded
/// over the instance axis: `folded_witness` holds one [`FoldedWord`] per hidden (committed) word,
/// the oblong representation produced by [`fold_instances`]. The public words are constants shared
/// by every instance, so they are passed unfolded as raw words.
///
/// The two phases call the single-instance prover's own subroutines. Phase 1 builds the hidden g
/// parts with [`build_g_parts_from_folded_words`] and the public g parts with the single-instance
/// `build_g_parts`, then sums them. Phase 2 derives the hidden folded segment as a partial
/// evaluation of `folded_witness` along the bit (`r_j`) axis, folds the public segment the same
/// way, and reuses `build_monster_segments` and `run_sumcheck` unchanged.
///
/// # Parameters
/// - `key_collection`: the prover's key collection for the constraint system.
/// - `public_words`: the public (constant) words, shared by every instance.
/// - `folded_witness`: the hidden witness, folded over the instance axis, one word per entry.
/// - `bitand_data`: operator data for the bitand (AND) constraints.
/// - `intmul_data`: operator data for the intmul (MUL) constraints.
/// - `domain_subspace`: the univariate evaluation domain.
/// - `channel`: the prover channel driving the interactive protocol.
///
/// # Returns
/// The `SumcheckOutput` with the final challenges and the reduced witness evaluation.
pub fn prove<F, P, Channel>(
	key_collection: &KeyCollection,
	public_words: &[Word],
	folded_witness: &[FoldedWord<F>],
	bitand_data: OperatorData<F>,
	intmul_data: OperatorData<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	// Sample one batching lambda per operator, then prepare the operator data (tensor expansions
	// and lambda powers).
	let bitand_lambda = channel.sample();
	let intmul_lambda = channel.sample();
	let prepared_bitand = PreparedOperatorData::new(bitand_data, bitand_lambda);
	let prepared_intmul = PreparedOperatorData::new(intmul_data, intmul_lambda);

	// Phase 1: build the g parts once per key segment, then add them. The public words are
	// constants shared by every instance, so the single-instance builder folds them directly from
	// their bits; the hidden words are already folded over instances. This scalar path drives the
	// single-instance phase-1 sumcheck.
	let mut g_parts = build_g_parts::<F, F>(
		public_words,
		&key_collection.public,
		&prepared_bitand,
		&prepared_intmul,
	);
	let hidden_g_parts = build_g_parts_from_folded_words(
		folded_witness,
		&key_collection.hidden,
		&prepared_bitand,
		&prepared_intmul,
	);
	for (g, hidden_g) in g_parts.iter_mut().zip(&hidden_g_parts) {
		for (slot, add) in g.as_mut().iter_mut().zip(hidden_g.as_ref()) {
			*slot += *add;
		}
	}
	let h_parts = build_h_parts::<F, F>(domain_subspace, prepared_bitand.r_zhat_prime);
	let phase_1_output = run_phase_1_sumcheck::<F, F, _>(g_parts, h_parts, channel);

	// Phase 2: split the phase-1 challenges into the bit half `r_j` and the shift half `r_s`.
	let SumcheckOutput {
		challenges: mut r_jr_s,
		eval: gamma,
	} = phase_1_output;
	let r_s = r_jr_s.split_off(Word::LOG_BITS);
	let r_j = r_jr_s;
	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

	// The witness folded at `r_j`, per segment. The public fold is a raw-word fold; the hidden fold
	// is a partial evaluation of `folded_witness` along the bit axis, contracting each word's
	// folded bits against the `r_j` tensor.
	let public_folded = fold_words::<F, P>(public_words, r_j_tensor.as_ref());
	let hidden_scalars: Vec<F> = folded_witness
		.iter()
		.map(|word| inner_product(word.iter().copied(), r_j_tensor.as_ref().iter().copied()))
		.collect();
	let hidden_folded = FieldBuffer::<P>::from_values(&hidden_scalars);

	let (public_monster, hidden_monster) = build_monster_segments::<F, P>(
		key_collection,
		&prepared_bitand,
		&prepared_intmul,
		domain_subspace,
		&r_j,
		&r_s,
	);

	run_sumcheck::<F, P, _>(
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

/// Constructs the phase-1 "g" multilinear parts, one per shift variant, from instance-folded words.
///
/// This is the batched analogue of the single-instance [`build_g_parts`]: it consumes a key
/// segment's words already folded over the instance axis, so each word is a [`FoldedWord`] whose
/// bits are full field elements rather than a packed `u64`. Where the single-instance builder
/// scatters an accumulator onto a word's set bits by masking, this scales the accumulator by each
/// folded bit with a field multiplication, which coincides with masking when the folded bit is 0 or
/// 1.
///
/// Use this for the hidden (committed) segment, whose words are folded over instances, and the
/// single-instance [`build_g_parts`] for the public segment, whose words are constants; add the two
/// results to obtain the complete g parts. `folded_words` is paired with `segment.key_ranges` in
/// order, so any power-of-two padding beyond the segment's word count is ignored.
///
/// The result is a flat accumulator split into `SHIFT_VARIANT_COUNT` multilinears of [`LOG_LEN`]
/// variables each. Each multilinear is indexed by `(shift amount, bit position)`: for shift key
/// `id = (variant << Word::LOG_BITS) | amount`, the slot at `id * Word::BITS + bit`
/// accumulates, over every word carrying that key, the word's folded bit times the key's
/// lambda-weighted partial evaluation tensor.
///
/// This scalar implementation ignores the packed-field and parallelism optimizations of the
/// single-instance builder.
pub fn build_g_parts_from_folded_words<F: BinaryField>(
	folded_words: &[FoldedWord<F>],
	segment: &KeySegment,
	bitand_operator_data: &PreparedOperatorData<F>,
	intmul_operator_data: &PreparedOperatorData<F>,
) -> [FieldBuffer<F>; SHIFT_VARIANT_COUNT] {
	// One flat accumulator holding SHIFT_VARIANT_COUNT multilinears of LOG_LEN variables each, laid
	// out variant-major. Kept on the heap rather than a stack array: it is thousands of elements.
	#[allow(clippy::useless_vec)]
	let mut multilinears = vec![F::ZERO; SHIFT_VARIANT_COUNT << LOG_LEN];

	// Each folded word carries the keys named by the segment-relative range at its position.
	for (word, range) in folded_words.iter().zip(&segment.key_ranges) {
		let keys = &segment.keys[range.start as usize..range.end as usize];
		for key in keys {
			let operator_data = match key.operation {
				Operation::BitwiseAnd => bitand_operator_data,
				Operation::IntegerMul => intmul_operator_data,
			};

			// The lambda-weighted partial evaluation tensor for this shifted word.
			let acc = key.accumulate(
				&segment.constraint_indices,
				operator_data.r_x_prime_tensor.as_ref(),
				&operator_data.lambda_powers,
			);

			// Scatter the accumulator across this key's bit slots, scaling each by the folded bit.
			let bit_base = key.id as usize * Word::BITS;
			for (bit, &folded_bit) in word.iter().enumerate() {
				multilinears[bit_base + bit] += acc * folded_bit;
			}
		}
	}

	// Split the flat accumulator into one multilinear per shift variant.
	multilinears
		.chunks(1 << LOG_LEN)
		.map(|chunk| FieldBuffer::new(LOG_LEN, chunk.to_vec().into_boxed_slice()))
		.collect::<Vec<_>>()
		.try_into()
		.expect("chunks yield SHIFT_VARIANT_COUNT parts of size 1 << LOG_LEN")
}

#[cfg(test)]
mod tests {
	use binius_core::{constraint_system::AndConstraint, verify::verify_constraints, word::Word};
	use binius_field::{AESTowerField8b, Field, PackedBinaryGhash1x128b, Random};
	use binius_math::{
		inner_product::inner_product_buffers, multilinear::evaluate::evaluate,
		test_utils::random_scalars, univariate::lagrange_evals_scalars,
	};
	use binius_prover::{
		fold_word::fold_words,
		protocols::shift::{build_key_collection, monster::build_h_parts},
	};
	use binius_transcript::ProverTranscript;
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use binius_verifier::{
		config::{B128, StdChallenger},
		protocols::shift::{OperatorData as VerifierOperatorData, check_eval, verify},
	};
	use rand::prelude::*;

	use super::*;
	use crate::{BatchAndCheckWitness, test_utils::*};

	#[test]
	fn circuit_matches_reference() {
		let c = crc64_circuit();

		// A handful of fixed inputs, checked against the standalone reference implementation.
		let cases: [[u64; N_INPUT_WORDS]; 3] = [
			[0, 0, 0, 0],
			[1, 2, 3, 4],
			[
				0x0123456789abcdef,
				0xfedcba9876543210,
				0xdeadbeefcafebabe,
				0x00ff00ff00ff00ff,
			],
		];

		for words in cases {
			let mut filler = c.circuit.new_witness_filler();
			for (wire, &w) in c.input.iter().zip(&words) {
				filler[*wire] = Word(w);
			}
			c.circuit.populate_wire_witness(&mut filler).unwrap();

			assert_eq!(filler[c.output], Word(crc64_iso_reference(&words)));
		}
	}

	#[test]
	fn populate_batch_of_random_instances() {
		let c = crc64_circuit();

		// A batch of 2^10 instances, each with an independent random message.
		let log_instances = 10;
		let n_instances = 1usize << log_instances;

		// Sample every instance's inputs up front so the fill closure is a pure lookup and the
		// reference check below sees the same words.
		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();

		let table = populate_crc64_witness(&c, &inputs);
		let constants = &c.circuit.constraint_system().constants;

		// Shape: 2^10 instances, one hidden-word row per committed word.
		let n_hidden_words = c
			.circuit
			.constraint_system()
			.value_vec_layout
			.n_hidden_words;
		assert_eq!(table.log_instances(), log_instances);
		assert_eq!(table.n_instances(), n_instances);
		assert_eq!(table.n_hidden_words(), n_hidden_words);
		assert_eq!(table.as_words().len(), n_hidden_words * n_instances);

		// Spot-check a few instances: each reconstructs to a valid single-instance witness whose
		// output word is the reference CRC of its inputs.
		let output_index = c.circuit.witness_index(c.output);
		for i in [0, 1, 42, n_instances - 1] {
			let vv = table.instance_value_vec(i, constants);
			verify_constraints(c.circuit.constraint_system(), &vv)
				.unwrap_or_else(|e| panic!("instance {i} failed verification: {e}"));
			assert_eq!(vv[output_index], Word(crc64_iso_reference(&inputs[i])));
		}
	}

	// Folding the batch over the instance axis and then evaluating over the (bit, word) axes agrees
	// with folding each word's bits first and then evaluating over the (word, instance) axes.
	//
	// Both routes compute the same triple sum, just associated differently:
	//
	//     sum_{rho, w, b} eq(r_rho, rho) * eq(r_wire, w) * eq(r_bit, b) * bit_b(word[rho][w])
	//
	// The evaluation point `r` is fresh and unrelated to the reduction's own r_z / r_x challenges;
	// its low Word::LOG_BITS coordinates are the bit axis and its high coordinates are the word
	// axis, matching the layout `fold_instances` produces.
	#[test]
	fn fold_instances_commutes_with_evaluation() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();
		let mut rng = StdRng::seed_from_u64(0);

		// Cover every chunk regime of the per-column fold.
		// A sub-chunk batch (< CHUNK_SIZE instances), exactly one chunk, and several chunks.
		for log_instances in [3, Word::LOG_BITS, Word::LOG_BITS + 2] {
			let n_instances = 1usize << log_instances;

			let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
				.map(|_| std::array::from_fn(|_| rng.random()))
				.collect();
			let table = populate_crc64_witness(&c, &inputs);
			let constants = &c.circuit.constraint_system().constants;

			// The committed witness segment, whose word count fixes the word (x) axis.
			let layout = table.layout();
			let offset = layout.offset_witness;
			let n_committed = layout.combined_len() - offset;
			let log_committed = log2_strict_usize(n_committed);

			// The instance-fold point, and a fresh point over the (bit, word) axes.
			let r_rho = random_scalars::<B128>(&mut rng, log_instances);
			let r = random_scalars::<B128>(&mut rng, Word::LOG_BITS + log_committed);

			// Route A: fold the instance axis, then evaluate the resulting (bit, word) multilinear
			// at r.
			let folded = fold_instances::<B128, P>(&table, &r_rho);
			let lhs = evaluate(&folded, &r);

			// Route B: fold each word's bits by the tensor expansion of the bit coordinates, then
			// evaluate the resulting (word, instance) multilinear over the word and instance axes.
			let (r_bit, r_wire) = r.split_at(Word::LOG_BITS);
			let bit_tensor = eq_ind_partial_eval_scalars::<B128>(r_bit);

			// Gather the committed words of every instance, instance-major: index = rho *
			// n_committed + w. Each instance is reconstructed independently of the fold under test.
			let mut committed = Vec::with_capacity(n_instances * n_committed);
			for rho in 0..n_instances {
				let vv = table.instance_value_vec(rho, constants);
				committed.extend_from_slice(&vv.combined_witness()[offset..]);
			}
			let folded_words = fold_words::<B128, P>(&committed, &bit_tensor);

			let mut point = r_wire.to_vec();
			point.extend_from_slice(&r_rho);
			let rhs = evaluate(&folded_words, &point);

			assert_eq!(lhs, rhs, "mismatch at log_instances = {log_instances}");
		}
	}

	// The oblong evaluation of each bitand operand column A, B, C at the shift challenges.
	//
	// Builds the batched AND witness, then for each column folds its word bits by the Lagrange
	// basis at r_z and evaluates the resulting row multilinear at the (constraint, instance) point
	// r_x || r_rho. The columns are instance-major, so r_x (low) indexes the constraint within an
	// instance and r_rho (high) indexes the instance.
	fn evaluate_and_witness<P: PackedField<Scalar = B128>>(
		table: &ValueTable,
		constants: &[Word],
		and_constraints: &[AndConstraint],
		domain_subspace: &BinarySubspace<B128>,
		r_z: B128,
		r_x: &[B128],
		r_rho: &[B128],
	) -> [B128; 3] {
		let witness = BatchAndCheckWitness::build(table, constants, and_constraints);
		let lagrange = lagrange_evals_scalars::<B128, B128>(domain_subspace, r_z);
		let row_point: Vec<B128> = r_x.iter().chain(r_rho).copied().collect();
		let operand_eval = |column: &[Word]| {
			let folded_column = fold_words::<B128, P>(column, &lagrange);
			evaluate(&folded_column, &row_point)
		};
		[
			operand_eval(witness.a()),
			operand_eval(witness.b()),
			operand_eval(witness.c()),
		]
	}

	// Folds a contiguous run of value-vector words over the instance axis, one FoldedWord per word.
	// This lets the public and hidden segments be folded separately, matching how `build_g_parts`
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

	// Evaluates the instance-folded witness at the oblong point `(r_j, r_y)`: fold each word's
	// folded bits against the `r_j` tensor, then contract the resulting per-word multilinear at
	// `r_y`.
	fn evaluate_folded_witness(folded: &[FoldedWord<B128>], r_j: &[B128], r_y: &[B128]) -> B128 {
		let r_j_tensor = eq_ind_partial_eval::<B128>(r_j);
		let per_word: Vec<B128> = folded
			.iter()
			.map(|word| inner_product(word.iter().copied(), r_j_tensor.as_ref().iter().copied()))
			.collect();
		let r_y_tensor = eq_ind_partial_eval::<B128>(r_y);
		inner_product(per_word.iter().copied(), r_y_tensor.as_ref().iter().copied())
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

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		let key_collection = build_key_collection(&cs);

		// The univariate bit challenge, the constraint challenge, and the instance challenge.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, log2_strict_usize(cs.n_and_constraints()));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		// The hidden witness folded over instances (reshaped to one FoldedWord per word), and the
		// public constants.
		let folded = fold_instances::<B128, P>(&table, &r_rho);
		let scalars: Vec<B128> = folded.iter_scalars().collect();
		let folded_witness: Vec<FoldedWord<B128>> = scalars
			.chunks_exact(Word::BITS)
			.map(|chunk| chunk.try_into().unwrap())
			.collect();
		let offset = table.layout().offset_witness;
		let public_words = &cs.constants;

		// The bitand operand evals at (r_z, r_x, r_rho); the circuit has no MUL constraints, so the
		// intmul claim is the zero claim over an empty point.
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
		let prover_output = prove::<B128, P, _>(
			&key_collection,
			public_words,
			&folded_witness,
			OperatorData {
				evals: bitand_evals.to_vec(),
				r_zhat_prime: r_z,
				r_x_prime: r_x.clone(),
			},
			OperatorData {
				evals: intmul_evals.to_vec(),
				r_zhat_prime: r_z,
				r_x_prime: Vec::new(),
			},
			&domain_subspace,
			&mut prover_transcript,
		);

		// Verify against the single-instance shift verifier.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_bitand = VerifierOperatorData::new(r_x, bitand_evals);
		let verifier_intmul = VerifierOperatorData::new(Vec::new(), intmul_evals);
		let verifier_output =
			verify(&cs, &verifier_bitand, &verifier_intmul, &mut verifier_transcript).unwrap();
		check_eval(
			&cs,
			public_words,
			&verifier_bitand,
			&verifier_intmul,
			&domain_subspace,
			r_z,
			&verifier_output,
			&mut verifier_transcript,
		)
		.unwrap();
		verifier_transcript.finalize().unwrap();

		// The witness evaluation equals the instance-folded witness evaluated at the point, with
		// the segment's zero-padding contributing the (1 - r) factors above the folded length.
		let r_y = verifier_output.r_y();
		let log_folded = log2_ceil_usize(folded_witness.len());
		let base =
			evaluate_folded_witness(&folded_witness, verifier_output.r_j(), &r_y[..log_folded]);
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
		assert_eq!(prover_output.challenges, eval_point);
		assert_eq!(prover_output.eval, verifier_output.witness_eval);
	}

	// The phase-1 identity: summing the g·h inner products over the shift variants reconstructs the
	// lambda-batched operand evaluation claim.
	//
	// The g parts come from the batched build_g_parts on the full folded witness; the h parts come
	// from the single-instance prover's build_h_parts at the same univariate challenge r_z. Their
	// inner product must equal the lambda-powers scaling of the batched AND-check operand evals
	// (the intmul claim is empty, contributing nothing).
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

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		let key_collection = build_key_collection(&cs);

		// The univariate bit challenge, the constraint challenge, and the instance challenge.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, log2_strict_usize(cs.n_and_constraints()));
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
		// The hidden segment spans value indices `[offset_witness, combined_len)`.
		let offset = table.layout().offset_witness;
		let combined = table.layout().combined_len();
		let public_words = &cs.constants;
		let hidden_folded = fold_words_over_instances(&table, constants, &r_rho, offset..combined);

		// Prepare the operator data: lambda batches the three operand claims. The circuit has no
		// MUL constraints, so the intmul claim is empty.
		let prepared_bitand = PreparedOperatorData::new(
			OperatorData {
				evals: bitand_evals.to_vec(),
				r_zhat_prime: r_z,
				r_x_prime: r_x,
			},
			B128::random(&mut rng),
		);
		let prepared_intmul = PreparedOperatorData::new(
			OperatorData {
				evals: Vec::new(),
				r_zhat_prime: r_z,
				r_x_prime: Vec::new(),
			},
			B128::random(&mut rng),
		);

		// The g parts: the public segment folds from raw constant words via the single-instance
		// builder, the hidden segment from the instance-folded words. Add them. The h parts come
		// from the single-instance prover.
		let mut g_parts = build_g_parts::<B128, B128>(
			public_words,
			&key_collection.public,
			&prepared_bitand,
			&prepared_intmul,
		);
		let hidden_g_parts = build_g_parts_from_folded_words(
			&hidden_folded,
			&key_collection.hidden,
			&prepared_bitand,
			&prepared_intmul,
		);
		for (g, hidden_g) in g_parts.iter_mut().zip(&hidden_g_parts) {
			for (slot, add) in g.as_mut().iter_mut().zip(hidden_g.as_ref()) {
				*slot += *add;
			}
		}
		let h_parts = build_h_parts::<B128, B128>(&domain_subspace, r_z);
		let inner_product: B128 = g_parts
			.iter()
			.zip(&h_parts)
			.map(|(g, h)| inner_product_buffers(g, h))
			.sum();

		// The lambda-powers scaling of the batched AND-check evals, plus the empty intmul claim.
		let expected = prepared_bitand.batched_eval() + prepared_intmul.batched_eval();
		assert_eq!(inner_product, expected);
	}
}
