// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, ops::Range};

use binius_core::word::Word;
use binius_field::{
	AESTowerField8b, BinaryField, Field, PackedField, UnderlierType, WithUnderlier,
};
use binius_ip::sumcheck::{SumcheckOutput, common::RoundCoeffs};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{bivariate_product::BivariateProductSumcheckProver, common::SumcheckProver},
};
use binius_math::{FieldBuffer, inner_product::inner_product_buffers};
use binius_utils::rayon::prelude::*;
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS, WORD_SIZE_BYTES},
	protocols::shift::SHIFT_VARIANT_COUNT,
};
use bytemuck::zeroed_vec;
use itertools::izip;
use tracing::instrument;

use super::{
	key_collection::{KeyCollection, Operation},
	monster::build_h_parts,
	prove::PreparedOperatorData,
	wiring::{WiringCollection, WiringEntry, WiringInfo},
};

// This is the number of variables in the g (and h) multilinears of phase 1.
const LOG_LEN: usize = LOG_WORD_SIZE_BITS + LOG_WORD_SIZE_BITS;

/// Constructs the "g" multilinear parts for both BITAND and INTMUL operations.
/// Proves the first phase of the shift reduction.
/// Computes the g and h multilinears and performs the sumcheck.
#[instrument(skip_all, name = "prover_phase_1")]
pub fn prove_phase_1<F, P, Channel>(
	key_collection: &KeyCollection,
	words: &[Word],
	bitand_data: &PreparedOperatorData<F>,
	intmul_data: &PreparedOperatorData<F>,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField + From<AESTowerField8b>,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	let g_parts = build_g_parts::<_, P>(words, key_collection, bitand_data, intmul_data);

	// BitAnd and IntMul share the same `r_zhat_prime`.
	let h_parts = build_h_parts(bitand_data.r_zhat_prime);

	run_phase_1_sumcheck(g_parts, h_parts, channel)
}

/// Runs the phase 1 sumcheck protocol for shift constraint verification.
///
/// Executes a sumcheck over bivariate products of g and h multilinear parts for each
/// operation (BITAND, INTMUL). The protocol proves that the sum of g·h products across
/// all shift variants equals the claimed batched evaluation.
///
/// # Protocol Structure
///
/// For each operation, creates 3 bivariate product sumcheck provers (one per shift variant):
/// - g_sll · h_sll with claim `sll_sum`
/// - g_srl · h_srl with claim `srl_sum`
/// - g_sra · h_sra with claim `sar_sum = total_sum - sll_sum - srl_sum`
///
/// The g parts incorporate batching randomness (lambda weighting), while h parts
/// encode the shift operation behavior at the univariate challenge points.
///
/// # Parameters
///
/// - `g_parts`: g multilinear parts for each operation (witness-dependent)
/// - `h_parts`: h multilinear parts for each operation (challenge-dependent)
/// - `sums`: Expected total sums for each operation from lambda-weighted evaluation claims
///
/// # Returns
///
/// `SumcheckOutput` containing the challenge vector and final evaluation `gamma`
#[instrument(skip_all, name = "run_sumcheck")]
pub fn run_phase_1_sumcheck<F: Field, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>>(
	g_parts: [FieldBuffer<P>; SHIFT_VARIANT_COUNT],
	h_parts: [FieldBuffer<P>; SHIFT_VARIANT_COUNT],
	channel: &mut Channel,
) -> SumcheckOutput<F> {
	// Build `BivariateProductSumcheckProver` provers.
	let mut provers = iter::zip(g_parts, h_parts)
		.map(|(g_part, h_part)| {
			let sum = inner_product_buffers(&g_part, &h_part);
			BivariateProductSumcheckProver::new([g_part, h_part], sum)
		})
		.collect::<Vec<_>>();

	// Perform the sumcheck rounds, collecting challenges.
	let n_vars = 2 * LOG_WORD_SIZE_BITS;
	let mut challenges = Vec::with_capacity(n_vars);
	for _ in 0..n_vars {
		let mut all_round_coeffs = Vec::new();
		for prover in &mut provers {
			all_round_coeffs.extend(prover.execute());
		}

		let summed_round_coeffs = all_round_coeffs
			.into_iter()
			.rfold(RoundCoeffs::default(), |acc, coeffs| acc + &coeffs);

		let round_proof = summed_round_coeffs.truncate();

		channel.send_many(round_proof.coeffs());

		let challenge = channel.sample();
		challenges.push(challenge);

		for prover in &mut provers {
			prover.fold(challenge);
		}
	}
	challenges.reverse();

	let multilinear_evals = provers
		.into_iter()
		.map(|prover| prover.finish())
		.collect::<Vec<Vec<F>>>();

	// Evaluate the composition polynomial to compute `gamma`.
	let gamma = multilinear_evals
		.into_iter()
		.map(|prover_evals| {
			assert_eq!(prover_evals.len(), 2);
			let h_eval = prover_evals[0];
			let g_eval = prover_evals[1];
			h_eval * g_eval
		})
		.sum();

	SumcheckOutput {
		challenges,
		eval: gamma,
	}
}

/// Constructs the "g" multilinear parts for both BITAND and INTMUL operations.
///
/// This function builds the g multilinear polynomials used in phase 1 of the shift protocol.
/// For each operation (BITAND and INTMUL), it constructs three multilinear polynomials
/// corresponding to the three shift variants (SLL, SRL, SRA).
///
/// # Construction Process
///
/// 1. **Parallel Processing**: Words are processed in parallel chunks for efficiency
/// 2. **Key Processing**: For each word, iterate through its associated keys from the key
///    collection
/// 3. **Accumulation**: For each key, accumulate its contribution weighted by the r_x' tensor
/// 4. **Word Expansion**: Expand each witness word bitwise to populate the g multilinears
/// 5. **Lambda Weighting**: Apply lambda powers to weight different operand positions
///
/// # Returns
///
/// An array of multilinear extensions of each shift variant part.
///
/// # Usage
///
/// Used in phase 1 to construct the constant size g multilinears
/// that will participate in the phase 1 sumcheck protocol.
#[instrument(skip_all, name = "build_g_parts")]
pub fn build_g_parts<F: BinaryField, P: PackedField<Scalar = F>>(
	words: &[Word],
	key_collection: &KeyCollection,
	bitand_operator_data: &PreparedOperatorData<F>,
	intmul_operator_data: &PreparedOperatorData<F>,
) -> [FieldBuffer<P>; SHIFT_VARIANT_COUNT] {
	let acc_size: usize = SHIFT_VARIANT_COUNT << (LOG_LEN.saturating_sub(P::LOG_WIDTH));

	assert!(
		P::WIDTH <= 8,
		"the optimizations below work only when the width of `P` is less than 8 (which is true for all packed 128b fields we use for now)"
	);

	// Map from a u8 with `P::WIDTH` meaningful bits to the lane mask selecting exactly those lanes,
	// precomputed once and reused across every accumulator below.
	let packed_masks_map = (0..1 << P::WIDTH)
		.map(|i| P::make_mask((0..P::WIDTH).map(|bit_index| (i >> bit_index) & 1 == 1)))
		.collect::<Vec<_>>();
	// A mask for low `P::WIDTH` bits.
	let low_bits_mask = (1u8 << P::WIDTH) - 1;

	// Process the public and hidden segments in absolute value-vector order: the public
	// words are the prefix of `words`, and each segment's key ranges are segment-relative.
	let (public_words, hidden_words) = words.split_at(key_collection.public.n_words());
	let public_iter = public_words
		.par_iter()
		.zip(key_collection.public.key_ranges.par_iter())
		.map(|(word, range)| (word, range, &key_collection.public));
	let hidden_iter = hidden_words
		.par_iter()
		.zip(key_collection.hidden.key_ranges.par_iter())
		.map(|(word, range)| (word, range, &key_collection.hidden));

	let multilinears = public_iter
		.chain(hidden_iter)
		.fold(
			|| zeroed_vec::<P>(acc_size).into_boxed_slice(),
			|mut multilinears, (word, Range { start, end }, segment)| {
				let keys = &segment.keys[*start as usize..*end as usize];

				for key in keys {
					let operator_data = match key.operation {
						Operation::BitwiseAnd => bitand_operator_data,
						Operation::IntegerMul => intmul_operator_data,
					};

					let acc = key.accumulate(&segment.constraint_indices, operator_data);
					let acc_packed = P::broadcast(acc);

					// The following loop is an optimized version of the following
					// for i in 0..WORD_SIZE_BITS {
					//     if get_bit(word, i) {
					//         values[start + i] += acc;
					//     }
					// }
					let start = key.id as usize * (WORD_SIZE_BITS >> P::LOG_WIDTH);
					let word_bytes = word.0.to_le_bytes();
					for (&byte, values) in word_bytes.iter().zip(
						multilinears[start..start + (WORD_SIZE_BITS >> P::LOG_WIDTH)]
							.chunks_exact_mut(WORD_SIZE_BYTES >> P::LOG_WIDTH),
					) {
						for value_index in 0..(8 >> P::LOG_WIDTH) {
							unsafe {
								let packed_mask_index =
									((byte >> (value_index * P::WIDTH)) & low_bits_mask) as usize;

								// Safety:
								// - `packed_masks_map` is guaranteed to have enough elements to be
								//   indexed with a `P::WIDTH`-bits value.
								let packed_mask = packed_masks_map.get_unchecked(packed_mask_index);

								// Safety:
								// - `values` is guaranteed to be (8 >> P::LOG_WIDTH) elements long
								//   due to the chunking
								// - `value_index` is always in bounds because we iterate over 0..(8
								//   >> P::LOG_WIDTH)
								*values.get_unchecked_mut(value_index) +=
									acc_packed.select(packed_mask);
							}
						}
					}
				}

				multilinears
			},
		)
		.reduce(
			|| zeroed_vec::<P>(acc_size).into_boxed_slice(),
			|mut acc, local| {
				izip!(acc.iter_mut(), local.iter()).for_each(|(acc, local)| {
					*acc += *local;
				});
				acc
			},
		);

	build_multilinear_parts(&multilinears)
}

/// Builds the multilinear parts for a single operation by combining its operand multilinears.
///
/// Takes the raw multilinears for all operands and shift variants of an operation,
/// applies lambda weighting to each operand, and combines them into parts.
/// Each operand of index `i` gets weighted by λ^(i+1).
#[instrument(skip_all, name = "build_multilinear_parts")]
fn build_multilinear_parts<P: PackedField>(
	multilinears: &[P],
) -> [FieldBuffer<P>; SHIFT_VARIANT_COUNT] {
	assert!(
		P::LOG_WIDTH < LOG_LEN,
		"P::WIDTH is not supposed to exceed 8, so this statement must hold"
	);

	multilinears
		.chunks(1 << (LOG_LEN - P::LOG_WIDTH))
		.map(|chunk| FieldBuffer::new(LOG_LEN, chunk.to_vec().into_boxed_slice()))
		.collect::<Vec<_>>()
		.try_into()
		.expect("chunk has SHIFT_VARIANT_COUNT parts of size 1 << LOG_LEN")
}

/// Alternate construction of the phase-1 "g" multilinear parts, over the [`WiringCollection`]
/// sparse-matrix layout instead of the per-word [`KeyCollection`].
///
/// Produces output byte-for-byte identical to [`build_g_parts`], but transposes the iteration
/// order: rather than iterating witness words and scattering a per-key scalar into a large
/// per-thread accumulator, it parallelizes over the `SHIFT_VARIANT_COUNT * WORD_SIZE_BITS`
/// independent output rows. Each `(shift_variant, shift_amount)` task owns a single 64-scalar
/// output row with no shared mutable state, so there is no fold/reduce, no packed bit-scatter, and
/// no `P::WIDTH <= 8` restriction.
///
/// For each `(variant, amount)` row and each `(operation, operand)`, the wiring matrix is scanned:
/// every `(constraint, word)` entry scatters the operand's tensor value into the bit positions set
/// in that word. The per-matrix accumulator is scaled once by `lambda_powers[operand]` and folded
/// into the row. The 8 × 64 × 64 scalars are repacked at the very end, with row `(variant, amount)`
/// occupying scalar indices `amount*64 .. amount*64+64` of `output[variant]` — the same layout
/// [`build_g_parts`] produces.
///
/// This is the BINIUS-228 experiment sibling of [`build_g_parts`]; the two are benchmarked
/// head-to-head. See [`WiringInfo`] for the layout.
#[instrument(skip_all, name = "build_g_parts_wiring")]
pub fn build_g_parts_wiring<F: BinaryField + WithUnderlier, P: PackedField<Scalar = F>>(
	words: &[Word],
	wiring: &WiringCollection,
	bitand_operator_data: &PreparedOperatorData<F>,
	intmul_operator_data: &PreparedOperatorData<F>,
) -> [FieldBuffer<P>; SHIFT_VARIANT_COUNT] {
	// The public words are the prefix of `words`; each segment's `word_index` values are
	// segment-relative, matching the split `build_g_parts` performs over `KeyCollection`.
	let (public_words, hidden_words) = words.split_at(wiring.public.n_words);

	// One scalar per (variant, amount, bit), laid out variant-major so that each variant's
	// `WORD_SIZE_BITS * WORD_SIZE_BITS` contiguous scalars repack into one `FieldBuffer`. Heap
	// allocated (~512 KiB for 128b `F`), not a stack array.
	let mut scalars =
		vec![F::ZERO; SHIFT_VARIANT_COUNT * WORD_SIZE_BITS * WORD_SIZE_BITS].into_boxed_slice();

	// Parallelize over the `SHIFT_VARIANT_COUNT * WORD_SIZE_BITS` output rows; each
	// `WORD_SIZE_BITS` chunk is one `(variant, amount)` row with no shared state.
	scalars
		.par_chunks_mut(WORD_SIZE_BITS)
		.enumerate()
		.for_each(|(task_index, row)| {
			let variant = task_index / WORD_SIZE_BITS;
			let amount = task_index % WORD_SIZE_BITS;

			for (segment, segment_words) in [
				(&wiring.public, public_words),
				(&wiring.hidden, hidden_words),
			] {
				accumulate_segment_row(
					row,
					variant,
					amount,
					segment,
					segment_words,
					bitand_operator_data,
					intmul_operator_data,
				);
			}
		});

	scalars
		.chunks_exact(WORD_SIZE_BITS * WORD_SIZE_BITS)
		.map(FieldBuffer::<P>::from_values)
		.collect::<Vec<_>>()
		.try_into()
		.expect("scalars splits into SHIFT_VARIANT_COUNT variant chunks of 1 << LOG_LEN scalars")
}

/// Accumulates one segment's contribution to a single `(variant, amount)` output row.
///
/// Adds, for both operations and every operand, the lambda-weighted scatter of each wiring matrix
/// entry into `row` (one scalar per word bit).
#[inline]
fn accumulate_segment_row<F: BinaryField + WithUnderlier>(
	row: &mut [F],
	variant: usize,
	amount: usize,
	segment: &WiringInfo,
	segment_words: &[Word],
	bitand_operator_data: &PreparedOperatorData<F>,
	intmul_operator_data: &PreparedOperatorData<F>,
) {
	for (operand_matrices, operator_data) in [
		(&segment.bitand[..], bitand_operator_data),
		(&segment.intmul[..], intmul_operator_data),
	] {
		let tensor = operator_data.r_x_prime_tensor.as_ref();
		for (operand_index, matrices) in operand_matrices.iter().enumerate() {
			let matrix = &matrices[variant][amount];
			if matrix.is_empty() {
				continue;
			}

			// Accumulate the operand's contribution into a bit-indexed row temp, then scale by its
			// lambda power once — a whole matrix shares one operand.
			let mut acc = [F::ZERO; WORD_SIZE_BITS];
			for &WiringEntry {
				constraint_index,
				word_index,
			} in matrix
			{
				let t = tensor[constraint_index as usize].to_underlier();
				let word = segment_words[word_index as usize].0;
				// Constant-time scatter: iterate all `WORD_SIZE_BITS` positions and add `t` masked
				// by each bit (an all-ones/all-zero underlier mask), so the work is independent of
				// the word's set-bit pattern. Field add is XOR.
				for (bit, acc_scalar) in acc.iter_mut().enumerate() {
					let mask = F::Underlier::fill_with_bit((word >> bit) as u8 & 1);
					*acc_scalar += F::from_underlier(t & mask);
				}
			}

			let lambda = operator_data.lambda_powers[operand_index];
			for (row_scalar, &acc_scalar) in row.iter_mut().zip(acc.iter()) {
				*row_scalar += acc_scalar * lambda;
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{
		ShiftVariant,
		constraint_system::{
			AndConstraint, ConstraintSystem, MulConstraint, Operand, ShiftedValueIndex, ValueIndex,
			ValueVecLayout,
		},
	};
	use binius_field::{
		BinaryField128bGhash, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
		PackedBinaryGhash4x128b, Random,
	};
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use binius_verifier::protocols::shift::{BITAND_ARITY, INTMUL_ARITY};
	use proptest::prelude::*;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::protocols::shift::{OperatorData, build_key_collection, build_wiring_info};

	type F = BinaryField128bGhash;

	const SHIFT_VARIANTS: [ShiftVariant; SHIFT_VARIANT_COUNT] = [
		ShiftVariant::Sll,
		ShiftVariant::Slr,
		ShiftVariant::Sar,
		ShiftVariant::Rotr,
		ShiftVariant::Sll32,
		ShiftVariant::Srl32,
		ShiftVariant::Sra32,
		ShiftVariant::Rotr32,
	];

	/// A random operand: a XOR of up to three shifted references into random words.
	fn random_operand(rng: &mut StdRng, committed_total_len: usize) -> Operand {
		let n_terms = rng.random_range(0..=3);
		(0..n_terms)
			.map(|_| ShiftedValueIndex {
				value_index: ValueIndex(rng.random_range(0..committed_total_len) as u32),
				shift_variant: SHIFT_VARIANTS[rng.random_range(0..SHIFT_VARIANT_COUNT)],
				amount: rng.random_range(0..WORD_SIZE_BITS) as u8,
			})
			.collect()
	}

	/// Random prepared operator data whose tensor is large enough to index every constraint.
	fn random_prepared_data(
		rng: &mut StdRng,
		arity: usize,
		n_constraints: usize,
	) -> PreparedOperatorData<F> {
		let log_len = if n_constraints == 0 {
			0
		} else {
			log2_ceil_usize(n_constraints)
		};
		let operator_data = OperatorData {
			evals: (0..arity).map(|_| F::random(&mut *rng)).collect(),
			r_zhat_prime: F::random(&mut *rng),
			r_x_prime: (0..log_len).map(|_| F::random(&mut *rng)).collect(),
		};
		PreparedOperatorData::new(operator_data, F::random(rng))
	}

	/// Builds a random (witness-inconsistent, but structurally valid) constraint system, witness,
	/// and operator data. The g-parts output is a deterministic function of these inputs, so the
	/// two `build_g_parts` variants must agree regardless of whether the witness satisfies the
	/// constraints.
	fn random_case(
		seed: u64,
	) -> (ConstraintSystem, Vec<Word>, PreparedOperatorData<F>, PreparedOperatorData<F>) {
		let mut rng = StdRng::seed_from_u64(seed);

		// Public segment is a power of two; the hidden segment fills out the rest.
		let n_public_words = 1usize << rng.random_range(1..=3);
		let committed_total_len = n_public_words + rng.random_range(1..=24);

		let n_and = rng.random_range(0..=32);
		let n_mul = rng.random_range(0..=16);

		let and_constraints = (0..n_and)
			.map(|_| AndConstraint {
				a: random_operand(&mut rng, committed_total_len),
				b: random_operand(&mut rng, committed_total_len),
				c: random_operand(&mut rng, committed_total_len),
			})
			.collect();
		let mul_constraints = (0..n_mul)
			.map(|_| MulConstraint {
				a: random_operand(&mut rng, committed_total_len),
				b: random_operand(&mut rng, committed_total_len),
				hi: random_operand(&mut rng, committed_total_len),
				lo: random_operand(&mut rng, committed_total_len),
			})
			.collect();

		let value_vec_layout = ValueVecLayout {
			n_const: 0,
			n_inout: 0,
			n_witness: 0,
			n_internal: 0,
			offset_inout: 0,
			offset_witness: n_public_words,
			committed_total_len,
			n_scratch: 0,
		};
		let cs = ConstraintSystem {
			value_vec_layout,
			constants: Vec::new(),
			and_constraints,
			mul_constraints,
		};

		let words = (0..committed_total_len)
			.map(|_| Word(rng.random()))
			.collect();
		let bitand = random_prepared_data(&mut rng, BITAND_ARITY, n_and);
		let intmul = random_prepared_data(&mut rng, INTMUL_ARITY, n_mul);

		(cs, words, bitand, intmul)
	}

	fn assert_g_parts_identity<P: PackedField<Scalar = F>>(
		words: &[Word],
		key_collection: &KeyCollection,
		wiring: &WiringCollection,
		bitand: &PreparedOperatorData<F>,
		intmul: &PreparedOperatorData<F>,
	) {
		let expected = build_g_parts::<F, P>(words, key_collection, bitand, intmul);
		let actual = build_g_parts_wiring::<F, P>(words, wiring, bitand, intmul);
		for (variant, (expected_part, actual_part)) in
			expected.iter().zip(actual.iter()).enumerate()
		{
			assert_eq!(
				expected_part.as_ref(),
				actual_part.as_ref(),
				"g-parts mismatch in shift variant {variant} (P::WIDTH = {})",
				P::WIDTH,
			);
		}
	}

	proptest! {
		#[test]
		fn wiring_g_parts_matches_key_collection(seed in any::<u64>()) {
			let (cs, words, bitand, intmul) = random_case(seed);
			let key_collection = build_key_collection(&cs);
			let wiring = build_wiring_info(&cs);

			// A couple of `P` widths: the `KeyCollection` path packs the bit-scatter differently
			// per width, while the wiring path is width-agnostic until the final repack.
			assert_g_parts_identity::<PackedBinaryGhash1x128b>(
				&words, &key_collection, &wiring, &bitand, &intmul,
			);
			assert_g_parts_identity::<PackedBinaryGhash2x128b>(
				&words, &key_collection, &wiring, &bitand, &intmul,
			);
			assert_g_parts_identity::<PackedBinaryGhash4x128b>(
				&words, &key_collection, &wiring, &bitand, &intmul,
			);
		}
	}
}
