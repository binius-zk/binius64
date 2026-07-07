// Copyright 2026 The Binius Developers

//! The batched shift-reduction prover for the data-parallel Binius64 M4 proof system.

#![allow(unused)]

use binius_core::consts::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS};
use binius_field::{BinaryField, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{BinarySubspace, FieldBuffer, multilinear::eq::eq_ind_partial_eval_scalars};
use binius_prover::protocols::shift::{
	KeyCollection, KeySegment, Operation, OperatorData, PreparedOperatorData,
};
use binius_utils::checked_arithmetics::log2_strict_usize;
use binius_verifier::protocols::shift::SHIFT_VARIANT_COUNT;

use crate::ValueTable;

/// The number of variables in each "g" (and "h") multilinear of phase 1: one 6-bit shift-amount
/// axis and one 6-bit bit-position axis.
const LOG_LEN: usize = LOG_WORD_SIZE_BITS + LOG_WORD_SIZE_BITS;

/// A committed witness word after folding its bits into the field.
///
/// Each 64-bit word contributes one field element per bit position, so a folded word is the oblong
/// representation of that word: its bit axis expanded to full field elements.
pub type FoldedWord<F> = [F; WORD_SIZE_BITS];

/// Folds the committed witness of a batch value table along the instance axis.
///
/// The batch value table is a three-axis object: the bits within each 64-bit word, the committed
/// words within one instance, and the instances themselves. This collapses the instance axis by the
/// equality-indicator weights of `r_rho`, leaving a multilinear over the other two axes.
///
/// For committed word `w` and bit `b`, the output element is
///
/// ```text
/// out[w][b] = sum_rho eq(r_rho, rho) * bit_b(word[rho][w])
/// ```
///
/// so each message bit contributes its instance's equality weight to a full field element. The
/// result is laid out with the bit axis in the low coordinates and the word axis in the high
/// coordinates, making it a multilinear over `LOG_WORD_SIZE_BITS + log2(n_committed)` variables:
///
/// ```text
/// index = w * WORD_SIZE_BITS + b     (b occupies the low LOG_WORD_SIZE_BITS coordinates)
/// ```
///
/// The public words at the front of each instance are excluded; only the committed witness words
/// are folded, so the word axis has `combined_len - offset_witness` entries.
///
/// This implementation is intentionally naive: it walks every committed word of every instance and
/// scans its bits one at a time. A faster method-of-four-Russians version will replace it later.
///
/// # Panics
///
/// Panics if `r_rho.len()` does not equal the batch dimension, or if the committed word count is
/// not a power of two.
pub fn fold_instances<F, P>(table: &ValueTable, r_rho: &[F]) -> FieldBuffer<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	assert_eq!(r_rho.len(), table.log_instances(), "r_rho must match the batch dimension");

	// The committed witness words occupy the tail of each instance, after the public segment.
	let layout = table.layout();
	let offset = layout.offset_witness;
	let log_committed = layout.log_witness_words();

	// One equality weight per instance: eq(r_rho, rho).
	let eq = eq_ind_partial_eval_scalars::<F>(r_rho);

	// Accumulate every committed word bit into its field element, weighted by its instance. Walking
	// instances outermost lets each instance's slice and weight be read once.
	let mut out = vec![F::ZERO; 1 << (LOG_WORD_SIZE_BITS + log_committed)];
	for (rho, &weight) in eq.iter().enumerate() {
		let words = &table.instance(rho)[offset..];
		for (w, word) in words.iter().enumerate() {
			let base = w << LOG_WORD_SIZE_BITS;
			for b in 0..WORD_SIZE_BITS {
				if (word.0 >> b) & 1 == 1 {
					out[base + b] += weight;
				}
			}
		}
	}

	FieldBuffer::from_values(&out)
}

/// Proves the batched shift-reduction, reducing the bitand and intmul evaluation claims to a single
/// multilinear claim on the batched witness.
///
/// This mirrors the single-instance shift reduction, but the witness enters already folded over the
/// instance axis: `folded_witness` holds one [`FoldedWord`] per committed word, the oblong
/// representation produced by [`fold_instances`]. The two operator claims and the domain subspace
/// play the same roles as in the single-instance prover.
///
/// # Parameters
/// - `key_collection`: the prover's key collection for the constraint system.
/// - `folded_witness`: the batched witness, folded over the instance axis, one word per entry.
/// - `bitand_data`: operator data for the bitand (AND) constraints.
/// - `intmul_data`: operator data for the intmul (MUL) constraints.
/// - `domain_subspace`: the univariate evaluation domain.
/// - `channel`: the prover channel driving the interactive protocol.
///
/// # Returns
/// The `SumcheckOutput` with the final challenges and the reduced witness evaluation.
// `P` is unused only while the body is a stub; the reduction's sumcheck rounds are generic over it.
#[allow(clippy::extra_unused_type_parameters)]
pub fn prove<F, P, Channel>(
	key_collection: &KeyCollection,
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
	todo!()
}

/// Constructs the phase-1 "g" multilinear parts, one per shift variant, for a single key segment.
///
/// This is the batched analogue of the single-instance `build_g_parts`: it consumes the segment's
/// words already folded over the instance axis, so each word is a [`FoldedWord`] whose bits are
/// full field elements rather than a packed `u64`. Where the single-instance builder scatters an
/// accumulator onto a word's set bits by masking, this scales the accumulator by each folded bit
/// with a field multiplication, which coincides with masking when the folded bit is 0 or 1.
///
/// Because the value vector's public and hidden words are folded by different means, the collection
/// splits into a public and a hidden [`KeySegment`]. Call this once per segment with that segment's
/// folded words, then add the two results to obtain the complete g parts. `folded_words` is paired
/// with `segment.key_ranges` in order, so any power-of-two padding beyond the segment's word count
/// is ignored.
///
/// The result is a flat accumulator split into `SHIFT_VARIANT_COUNT` multilinears of [`LOG_LEN`]
/// variables each. Each multilinear is indexed by `(shift amount, bit position)`: for shift key
/// `id = (variant << LOG_WORD_SIZE_BITS) | amount`, the slot at `id * WORD_SIZE_BITS + bit`
/// accumulates, over every word carrying that key, the word's folded bit times the key's
/// lambda-weighted partial evaluation tensor.
///
/// This scalar implementation ignores the packed-field and parallelism optimizations of the
/// single-instance builder.
pub fn build_g_parts<F: BinaryField>(
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
			let bit_base = key.id as usize * WORD_SIZE_BITS;
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

/// A CRC-64/GO-ISO circuit and reference implementation used to build shift-heavy witnesses for the
/// shift-prover tests. These helpers will likely move to a crate-level module once the prover needs
/// them outside of tests.
#[cfg(test)]
mod crc64 {
	use binius_core::word::Word;
	use binius_frontend::{Circuit, CircuitBuilder, Wire};

	use crate::ValueTable;

	/// CRC-64/GO-ISO parameters.
	///
	/// The generator polynomial is `x^64 + x^4 + x^3 + x + 1`, normal form `0x1b`. Both input and
	/// output are reflected, so the polynomial enters the register in its bit-reversed form.
	const POLY_REFLECTED: u64 = 0xd800000000000000;
	/// The register is preset to all ones before absorbing the message.
	const INIT: u64 = 0xffffffffffffffff;
	/// The final register is XORed with all ones before being returned.
	const XOR_OUT: u64 = 0xffffffffffffffff;

	/// The number of 64-bit input words the CRC circuit consumes.
	pub const N_INPUT_WORDS: usize = 4;

	/// Computes CRC-64/GO-ISO over `words`, absorbing bits least-significant-first.
	///
	/// Each input word contributes its 64 bits in order from bit 0 up to bit 63, and the words are
	/// absorbed in index order. This is the reflected bitwise algorithm: for every message bit, the
	/// register's low bit is combined with the message bit, the register is shifted right by one,
	/// and the polynomial is conditionally mixed in.
	///
	/// The `Circuit` counterpart mirrors this loop gate for gate, so the two agree bit for bit.
	pub fn crc64_iso_reference(words: &[u64; N_INPUT_WORDS]) -> u64 {
		let mut crc = INIT;
		for &word in words {
			for i in 0..64 {
				let bit = (word >> i) & 1;
				let mix = (crc ^ bit) & 1;
				crc >>= 1;
				if mix != 0 {
					crc ^= POLY_REFLECTED;
				}
			}
		}
		crc ^ XOR_OUT
	}

	/// A circuit computing CRC-64/GO-ISO over four private witness words.
	///
	/// The four inputs are ordinary witness wires, not public inout wires, so the whole computation
	/// lives in the private witness. The output wire is force-committed: without an assertion or a
	/// public output reading it, dead-code elimination would otherwise prune the entire CRC.
	pub struct Crc64Circuit {
		pub circuit: Circuit,
		pub input: [Wire; N_INPUT_WORDS],
		pub output: Wire,
	}

	/// Builds the CRC-64/GO-ISO circuit, mirroring [`crc64_iso_reference`] gate for gate.
	pub fn crc64_circuit() -> Crc64Circuit {
		let builder = CircuitBuilder::new();

		// The four message words are private witnesses supplied by the prover.
		let input = std::array::from_fn(|_| builder.add_witness());

		// The register starts at the all-ones preset and the polynomial is a constant.
		let mut crc = builder.add_constant_64(INIT);
		let poly = builder.add_constant_64(POLY_REFLECTED);

		for word in input {
			for i in 0..64 {
				// Isolate message bit `i` into the low bit; the higher bits are junk we discard.
				let bit = if i == 0 { word } else { builder.shr(word, i) };

				// The low bit that decides whether the polynomial is mixed in this step.
				let mixed = builder.bxor(crc, bit);

				// Broadcast that low bit across the whole word: all ones iff it is set, else zero.
				// Shifting it up to bit 63 then arithmetic-shifting back fills every bit from it.
				let to_msb = builder.shl(mixed, 63);
				let mask = builder.sar(to_msb, 63);
				let poly_term = builder.band(mask, poly);

				// Advance the register: shift right by one, then conditionally mix the polynomial.
				let shifted = builder.shr(crc, 1);
				crc = builder.bxor(shifted, poly_term);
			}
		}

		// Apply the final output XOR to produce the CRC value.
		let output = builder.bxor(crc, builder.add_constant_64(XOR_OUT));

		// Pin the output so the constraint compiler keeps the CRC computation alive.
		builder.force_commit(output);

		Crc64Circuit {
			circuit: builder.build(),
			input,
			output,
		}
	}

	/// Populates a batch value table with one instance per input tuple.
	///
	/// The instance count is the number of tuples, which must be a power of two. Each instance's
	/// four message words are the corresponding tuple, and circuit evaluation derives the rest.
	pub fn populate_crc64_witness(c: &Crc64Circuit, inputs: &[[u64; N_INPUT_WORDS]]) -> ValueTable {
		let log_instances = inputs.len().ilog2() as usize;
		ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			for (wire, &w) in c.input.iter().zip(&inputs[i]) {
				filler[*wire] = Word(w);
			}
		})
		.unwrap()
	}
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
	use binius_verifier::config::{B128, StdChallenger};
	use rand::prelude::*;

	use super::{crc64::*, *};
	use crate::BatchAndCheckWitness;

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

		// Shape: 2^10 instances, one committed witness per instance.
		let stride = c
			.circuit
			.constraint_system()
			.value_vec_layout
			.combined_len();
		assert_eq!(table.log_instances(), log_instances);
		assert_eq!(table.n_instances(), n_instances);
		assert_eq!(table.instance_stride(), stride);

		// Spot-check a few instances: each reconstructs to a valid single-instance witness whose
		// output word is the reference CRC of its inputs.
		let output_index = c.circuit.witness_index(c.output);
		for i in [0, 1, 42, n_instances - 1] {
			let vv = table.instance_value_vec(i);
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
	// its low LOG_WORD_SIZE_BITS coordinates are the bit axis and its high coordinates are the word
	// axis, matching the layout `fold_instances` produces.
	#[test]
	fn fold_instances_commutes_with_evaluation() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		// A modest batch keeps the naive fold quick while still exercising a non-trivial rho axis.
		let log_instances = 6;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		// The committed witness segment, whose word count fixes the word (x) axis.
		let layout = table.layout();
		let offset = layout.offset_witness;
		let n_committed = layout.combined_len() - offset;
		let log_committed = log2_strict_usize(n_committed);

		// The instance-fold point, and a fresh point over the (bit, word) axes.
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);
		let r = random_scalars::<B128>(&mut rng, LOG_WORD_SIZE_BITS + log_committed);

		// Route A: fold the instance axis, then evaluate the resulting (bit, word) multilinear at
		// r.
		let folded = fold_instances::<B128, P>(&table, &r_rho);
		let lhs = evaluate(&folded, &r);

		// Route B: fold each word's bits by the tensor expansion of the bit coordinates, then
		// evaluate the resulting (word, instance) multilinear over the word and instance axes.
		let (r_bit, r_wire) = r.split_at(LOG_WORD_SIZE_BITS);
		let bit_tensor = eq_ind_partial_eval_scalars::<B128>(r_bit);

		// Gather the committed words of every instance, instance-major: index = rho * n_committed +
		// w.
		let mut committed = Vec::with_capacity(n_instances * n_committed);
		for rho in 0..n_instances {
			committed.extend_from_slice(&table.instance(rho)[offset..]);
		}
		let folded_words = fold_words::<B128, P>(&committed, &bit_tensor);

		let mut point = r_wire.to_vec();
		point.extend_from_slice(&r_rho);
		let rhs = evaluate(&folded_words, &point);

		assert_eq!(lhs, rhs);
	}

	// The oblong evaluation of each bitand operand column A, B, C at the shift challenges.
	//
	// Builds the batched AND witness, then for each column folds its word bits by the Lagrange
	// basis at r_z and evaluates the resulting row multilinear at the (constraint, instance) point
	// r_x || r_rho. The columns are instance-major, so r_x (low) indexes the constraint within an
	// instance and r_rho (high) indexes the instance.
	fn evaluate_and_witness<P: PackedField<Scalar = B128>>(
		table: &ValueTable,
		and_constraints: &[AndConstraint],
		domain_subspace: &BinarySubspace<B128>,
		r_z: B128,
		r_x: &[B128],
		r_rho: &[B128],
	) -> [B128; 3] {
		let witness = BatchAndCheckWitness::build(table, and_constraints);
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
		r_rho: &[B128],
		words: std::ops::Range<usize>,
	) -> Vec<FoldedWord<B128>> {
		let eq = eq_ind_partial_eval_scalars::<B128>(r_rho);
		let mut folded = vec![[B128::ZERO; WORD_SIZE_BITS]; words.len()];
		for (rho, &weight) in eq.iter().enumerate() {
			for (word, out) in table.instance(rho)[words.clone()].iter().zip(&mut folded) {
				for (b, out_b) in out.iter_mut().enumerate() {
					if (word.0 >> b) & 1 == 1 {
						*out_b += weight;
					}
				}
			}
		}
		folded
	}

	// Expands the public words into FoldedWords without folding over the instance axis.
	//
	// The public words are constants, identical across every instance, so their instance fold is
	// `bit * sum_rho eq(r_rho, rho) = bit` (the equality weights sum to one). Folding is therefore
	// unnecessary: take the first instance's public words and embed each bit as F::ONE or F::ZERO.
	// The assertion pins the "identical across instances" premise this relies on.
	fn expand_public_words(table: &ValueTable) -> Vec<FoldedWord<B128>> {
		let offset = table.layout().offset_witness;
		let public = &table.instance(0)[..offset];
		for rho in 1..table.n_instances() {
			assert_eq!(
				&table.instance(rho)[..offset],
				public,
				"public words differ across instances"
			);
		}

		public
			.iter()
			.map(|word| {
				std::array::from_fn(|b| {
					if (word.0 >> b) & 1 == 1 {
						B128::ONE
					} else {
						B128::ZERO
					}
				})
			})
			.collect()
	}

	// The batched prove is still a stub; feeding it a real folded witness must reach the `todo!()`.
	// This pins the plumbing that produces `prove`'s inputs (folded witness, operator data) even
	// before the reduction itself exists.
	#[test]
	#[should_panic(expected = "not yet implemented")]
	fn prove_reaches_unimplemented_reduction() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();

		// Same setup as the fold commutativity test: a modest batch of random instances.
		let log_instances = 6;
		let n_instances = 1usize << log_instances;

		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		// The prepared constraint system feeds the key collection, the AND-constraint count, and
		// the per-instance AND constraints shared by every instance.
		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		let key_collection = build_key_collection(&cs);
		let n_and = cs.n_and_constraints();
		let log_and_constraints = log2_strict_usize(n_and);

		// The univariate bit challenge r_z and the multilinear constraint challenge r_x. The domain
		// subspace is also the one r_z's Lagrange basis is taken over.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, log_and_constraints);
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		// Fold the instance axis, then reshape the resulting multilinear into one folded word per
		// committed word: `fold_instances` lays the 64 bit elements of each word contiguously.
		let folded = fold_instances::<B128, P>(&table, &r_rho);
		let scalars: Vec<B128> = folded.iter_scalars().collect();
		let folded_witness: Vec<FoldedWord<B128>> = scalars
			.chunks_exact(WORD_SIZE_BITS)
			.map(|chunk| chunk.try_into().unwrap())
			.collect();

		// The bitand operator claim: the oblong evaluation of each operand column A, B, C at the
		// challenges, plus the challenges themselves.
		let bitand_evals = evaluate_and_witness::<P>(
			&table,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);
		let bitand_data = OperatorData {
			evals: bitand_evals.to_vec(),
			r_zhat_prime: r_z,
			r_x_prime: r_x,
		};

		// The circuit has no MUL constraints, so the intmul claim is empty: no evals and an empty
		// point, whose tensor expansion is the trivial one-element FieldBuffer.
		let intmul_data = OperatorData {
			evals: Vec::new(),
			r_zhat_prime: r_z,
			r_x_prime: Vec::new(),
		};

		let mut transcript = ProverTranscript::<StdChallenger>::default();

		prove::<B128, P, _>(
			&key_collection,
			&folded_witness,
			bitand_data,
			intmul_data,
			&domain_subspace,
			&mut transcript,
		);
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

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		let key_collection = build_key_collection(&cs);

		// The univariate bit challenge, the constraint challenge, and the instance challenge.
		let domain_subspace =
			BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
		let r_z = B128::random(&mut rng);
		let r_x = random_scalars::<B128>(&mut rng, log2_strict_usize(cs.n_and_constraints()));
		let r_rho = random_scalars::<B128>(&mut rng, log_instances);

		// The batched AND-check operand evals at (r_z, r_x, r_rho), and the full folded witness at
		// the same r_rho, so g and the claim agree on the instance point.
		let bitand_evals = evaluate_and_witness::<P>(
			&table,
			&cs.and_constraints,
			&domain_subspace,
			r_z,
			&r_x,
			&r_rho,
		);
		let offset = table.layout().offset_witness;
		let stride = table.instance_stride();
		let public_folded = expand_public_words(&table);
		let hidden_folded = fold_words_over_instances(&table, &r_rho, offset..stride);

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

		// The g parts: build each segment's contribution separately and add them. The h parts come
		// from the single-instance prover.
		let mut g_parts = build_g_parts(
			&public_folded,
			&key_collection.public,
			&prepared_bitand,
			&prepared_intmul,
		);
		let hidden_g_parts = build_g_parts(
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
