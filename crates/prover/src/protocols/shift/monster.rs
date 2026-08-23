// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_compute::{Allocator, VecLike};
use binius_core::{ShiftVariant, constraint_system::Shift, word::Word};
use binius_field::{BinaryField, Field, PackedField};
use binius_math::{FieldBuffer, FieldVec, multilinear::eq::eq_ind_partial_eval};
use tracing::instrument;

use super::{phase_1::SHIFT_OPERATOR_LOG_LEN, shift_ind::ShiftChallenge};

/// The width the half-word (`*32`) shift variants act over.
const HALF_WORD_BITS: usize = 32;

/// The phase-2 scalar weights of one key segment, one table per operation.
///
/// A table holds the `arity` weights of each shift sequence the segment uses, in dense shift index
/// order, so a key's weights are the chunk at `key.dense_shift_idx * arity`.
pub(super) struct ScalarTables<F> {
	pub(super) zero: Vec<F>,
	pub(super) bitand: Vec<F>,
	pub(super) intmul: Vec<F>,
	pub(super) binmul: Vec<F>,
}

/// The equality-indicator weights of a shift sequence's outer slot.
///
/// The sequence weight factorizes across its two slots, so each slot carries its own two tensors:
/// one over the variant axis, one over the amount axis.
/// Keeping them apart holds the weights at `2 * SHIFT_COUNT` entries rather than `SHIFT_COUNT^2`.
pub(super) struct OuterSlotWeights<F: Field> {
	/// The equality indicator over the outer variant axis, one weight per shift variant.
	variant: FieldBuffer<F>,
	/// The equality indicator over the outer amount axis, one weight per shift amount.
	amount: FieldBuffer<F>,
}

impl<F: Field> OuterSlotWeights<F> {
	/// The equality indicators of the outer slot's challenge point.
	pub(super) fn new(outer: &ShiftChallenge<F>) -> Self {
		Self {
			variant: eq_ind_partial_eval::<F>(&outer.variant),
			amount: eq_ind_partial_eval::<F>(&outer.amount),
		}
	}

	/// The weight one outer shift contributes to its sequence's scalar.
	#[inline]
	pub(super) fn weight(&self, shift: Shift) -> F {
		self.variant.as_ref()[shift.variant as usize] * self.amount.as_ref()[shift.amount as usize]
	}
}

/// Writes the row of a shift operator table that one `(variant, amount)` pair contributes.
///
/// The row holds, for each bit position, the weight that pair moves there:
///
/// ```text
///     row[j] = sum_k psi(k) * shift-ind_variant(k, j, amount)
/// ```
///
/// This is the one place that says what a variant does to a weight vector:
/// - Logical left and logical right move the weights and leave zeros behind.
/// - Arithmetic right piles every weight that falls off the end onto the sign position.
/// - Rotate wraps them around instead.
/// - The half-word forms apply the same rule to each 32-bit half, reading only the low 5 bits of
///   the amount.
///
/// # Arguments
///
/// The amount is an index over the reduction's amount axis rather than a validated
/// [`Shift`] amount: that axis spans `Word::BITS` for every
/// variant, and a half-word variant reads it modulo its own 32-bit width.
///
/// Every cell of `row` is written, so the caller need not zero it first. A caller reading one
/// slice at a time can therefore carry a single scratch row across every pair it visits.
///
/// # Panics
///
/// Panics unless the row and the weights each hold one entry per bit position of a word.
pub fn shift_operator_row<F: Field>(
	variant: ShiftVariant,
	amount: usize,
	row: &mut [F],
	psi: &[F],
) {
	assert_eq!(row.len(), Word::BITS, "the row is indexed by bit position");
	assert_eq!(psi.len(), Word::BITS, "the weights are indexed by bit position");

	// A half-word variant repeats the full-width rule over each half, so both share one closure.
	let halves = |row: &mut [F], rule: fn(usize, &mut [F], &[F])| {
		let amount = amount % HALF_WORD_BITS;
		for (row_half, psi_half) in
			iter::zip(row.chunks_mut(HALF_WORD_BITS), psi.chunks(HALF_WORD_BITS))
		{
			rule(amount, row_half, psi_half);
		}
	};

	fn sll<F: Field>(amount: usize, row: &mut [F], psi: &[F]) {
		let width = row.len();
		row[..width - amount].copy_from_slice(&psi[amount..]);
		// The positions the weights vacate take no weight at all.
		row[width - amount..].fill(F::ZERO);
	}
	fn srl<F: Field>(amount: usize, row: &mut [F], psi: &[F]) {
		let width = row.len();
		row[..amount].fill(F::ZERO);
		row[amount..].copy_from_slice(&psi[..width - amount]);
	}
	fn sar<F: Field>(amount: usize, row: &mut [F], psi: &[F]) {
		let width = row.len();
		srl(amount, row, psi);
		// Every position past the shift reads the sign bit, so their weights pile onto it.
		row[width - 1] += psi[width - amount..].iter().sum::<F>();
	}
	fn rotr<F: Field>(amount: usize, row: &mut [F], psi: &[F]) {
		let width = row.len();
		row[..amount].copy_from_slice(&psi[width - amount..]);
		row[amount..].copy_from_slice(&psi[..width - amount]);
	}

	match variant {
		ShiftVariant::Sll => sll(amount, row, psi),
		ShiftVariant::Slr => srl(amount, row, psi),
		ShiftVariant::Sar => sar(amount, row, psi),
		ShiftVariant::Rotr => rotr(amount, row, psi),
		ShiftVariant::Sll32 => halves(row, sll),
		ShiftVariant::Srl32 => halves(row, srl),
		ShiftVariant::Sra32 => halves(row, sar),
		ShiftVariant::Rotr32 => halves(row, rotr),
	}
}

/// Pushes one weight vector through every shift.
///
/// A shift indicator says whether an output bit reads a given input bit at a given amount.
/// This contracts it on the output index, against the supplied weights:
///
/// ```text
///     T[psi](j, s, o) = sum_k psi(k) * shift-ind_op(o)(k, j, s)
/// ```
///
/// At most one input bit feeds each output bit.
/// So a slice at fixed `(s, o)` is the weights moved by that shift, not a matrix applied to them.
///
/// The reduction contracts the indicator once per slot of a shift sequence.
/// Both contractions are this operator:
///
/// - the first carries the oblong weights to the bits of the intermediate word;
/// - the second carries that result down to the witness bit.
///
/// # Returns
///
/// One multilinear over [`SHIFT_OPERATOR_LOG_LEN`] variables, indexed from the low variables up:
///
/// ```text
///     low     Word::LOG_BITS             the bit position
///     middle  Word::LOG_BITS             the shift amount
///     high    LOG_SHIFT_VARIANT_COUNT    the shift variant
/// ```
///
/// # Performance
///
/// Every entry is one copy or one accumulation.
/// So the whole table costs `O(2^15)` field operations, and a single slice `O(2^6)`.
/// A caller needing one slice rather than the whole table calls [`shift_operator_row`] itself.
///
/// # Panics
///
/// Panics unless the weights hold one entry per bit position of a word.
#[instrument(skip_all, name = "shift_operator_table")]
pub fn shift_operator_table<F, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	psi: &[F],
) -> FieldVec<P, A>
where
	F: BinaryField,
{
	assert_eq!(psi.len(), Word::BITS, "the weights are indexed by bit position");
	assert_eq!(
		Word::BITS % P::WIDTH,
		0,
		"a row of Word::BITS weights must be packed-element aligned"
	);

	// One row of `Word::BITS` weights per `(variant, amount)`, variant most significant, packed
	// `P::WIDTH` scalars at a time straight into the destination buffer. The scratch row is
	// reused across every pair rather than collected into a second full-size buffer.
	let row_packed_len = Word::BITS / P::WIDTH;
	let packed_len = 1 << SHIFT_OPERATOR_LOG_LEN.saturating_sub(P::LOG_WIDTH);
	let mut values = alloc.alloc::<P>(packed_len);
	let mut row = [F::ZERO; Word::BITS];
	for (variant, block) in iter::zip(
		ShiftVariant::ALL,
		values
			.spare_capacity_mut()
			.chunks_exact_mut(Word::BITS * row_packed_len),
	) {
		for (amount, packed_row) in block.chunks_exact_mut(row_packed_len).enumerate() {
			shift_operator_row(variant, amount, &mut row, psi);
			for (slot, chunk) in iter::zip(packed_row, row.chunks_exact(P::WIDTH)) {
				slot.write(P::from_scalars(chunk.iter().copied()));
			}
		}
	}
	// Safety: the loop above wrote every one of the `packed_len` slots.
	unsafe { values.set_len(packed_len) };

	FieldBuffer::new(SHIFT_OPERATOR_LOG_LEN, values)
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{AESTowerField8b, BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
	use binius_math::{
		BinarySubspace, inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval,
		test_utils::random_scalars, univariate::subspace_lagrange_evals,
	};
	use binius_verifier::protocols::shift::LOG_SHIFT_VARIANT_COUNT;
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::{
		super::{ShiftChallenge, ShiftChallengePoint, ShiftIndSumcheck},
		*,
	};

	/// Phase 1's h multilinear and the claim phase 3 starts from must agree.
	///
	/// Phase 3 sums the shift indicators over the bit index, weighted by the Lagrange evaluations
	/// and by the constant it carries; the multilinear holds those sums over the whole shift axis.
	/// So with the carried constant set to one, evaluating the multilinear at `(r_j, r_s, r_v)`
	/// must give phase 3's claim.
	#[test]
	fn h_op_consistency() {
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash2x128b;

		let mut rng = StdRng::seed_from_u64(0);

		let num_random_tests = 10;

		for test_case in 0..num_random_tests {
			let r_zhat_prime = F::random(&mut rng);

			let r_j = random_scalars::<F>(&mut rng, Word::LOG_BITS);
			let r_s = random_scalars::<F>(&mut rng, Word::LOG_BITS);
			let r_v = random_scalars::<F>(&mut rng, LOG_SHIFT_VARIANT_COUNT);
			let shift = ShiftChallenge::new(r_s.clone(), r_v.clone());

			// Method 1: the claim phase 3 starts from, with the carried constant set to one.
			let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
			let l_tilde = subspace_lagrange_evals(&subspace, r_zhat_prime);
			let claimed = ShiftIndSumcheck::<P, _>::new(
				&GlobalAllocator,
				l_tilde.as_ref(),
				&ShiftChallengePoint::new(&r_j, &shift),
				F::ONE,
			)
			.beta();

			// Method 2: evaluate the built multilinear at the whole point.
			let h = shift_operator_table::<F, P, _>(&GlobalAllocator, l_tilde.as_ref());
			let evaluation_point = [r_j, r_s, r_v].concat();
			let tensor = eq_ind_partial_eval::<P>(&evaluation_point);
			let direct = inner_product_buffers(&h, &tensor);

			assert_eq!(
				claimed, direct,
				"H-op evaluation mismatch (test_case={test_case}): claimed != direct",
			);
		}
	}

	/// Whether output bit `k` of `variant` at `amount` reads input bit `j`.
	///
	/// Read off the word operation itself.
	/// Shifting a word with only bit `j` set leaves bits exactly where that bit is read.
	fn reads_input_bit(variant: ShiftVariant, k: usize, j: usize, amount: usize) -> bool {
		let shifted = variant.apply(Word(1u64 << j), amount);
		(shifted.as_u64() >> k) & 1 == 1
	}

	/// The operator table computed straight from the indicator definition, one entry at a time.
	fn reference_table<F: Field>(psi: &[F]) -> Vec<F> {
		let mut table = vec![F::ZERO; 1 << SHIFT_OPERATOR_LOG_LEN];
		for (variant_idx, variant) in ShiftVariant::ALL.into_iter().enumerate() {
			for amount in 0..Word::BITS {
				for j in 0..Word::BITS {
					// Contract on the indicator's output-bit index, which is the summed one.
					let entry = (0..Word::BITS)
						.filter(|&k| reads_input_bit(variant, k, j, amount))
						.map(|k| psi[k])
						.sum();
					table[(variant_idx * Word::BITS + amount) * Word::BITS + j] = entry;
				}
			}
		}
		table
	}

	proptest! {
		// Invariant: every entry is the contraction the definition names.
		//
		// The eight variants and all 64 amounts are enumerated in full.
		// Only the weights are sampled, since the operator is linear in them.
		//
		// This is what pins `sra`.
		// Its vacated positions all read bit 63, so several weights land in one entry.
		// That is the only slice which is not a plain move of the weights.
		#[test]
		fn shift_operator_table_matches_the_indicator_definition(seed: u64) {
			type F = BinaryField128bGhash;

			let mut rng = StdRng::seed_from_u64(seed);
			let psi = random_scalars::<F>(&mut rng, Word::BITS);

			let table = shift_operator_table::<F, F, _>(&GlobalAllocator, &psi);
			let reference = reference_table(&psi);
			prop_assert_eq!(table.as_ref(), reference.as_slice());
		}

		// Invariant: the operator is linear in the weights.
		//
		//     T[a * psi_1 + b * psi_2] == a * T[psi_1] + b * T[psi_2]
		//
		// The reduction folds the weights between its two contractions.
		// Linearity is what lets it fold first and contract after.
		#[test]
		fn shift_operator_table_is_linear_in_the_weights(seed: u64) {
			type F = BinaryField128bGhash;

			let mut rng = StdRng::seed_from_u64(seed);
			let psi_1 = random_scalars::<F>(&mut rng, Word::BITS);
			let psi_2 = random_scalars::<F>(&mut rng, Word::BITS);
			let (a, b) = (F::random(&mut rng), F::random(&mut rng));

			// The combination pushed through the operator.
			let combined = iter::zip(&psi_1, &psi_2)
				.map(|(&x, &y)| a * x + b * y)
				.collect::<Vec<F>>();
			let lhs = shift_operator_table::<F, F, _>(&GlobalAllocator, &combined);

			// The two tables combined afterwards.
			let table_1 = shift_operator_table::<F, F, _>(&GlobalAllocator, &psi_1);
			let table_2 = shift_operator_table::<F, F, _>(&GlobalAllocator, &psi_2);
			let rhs = iter::zip(table_1.as_ref(), table_2.as_ref())
				.map(|(&x, &y)| a * x + b * y)
				.collect::<Vec<F>>();

			prop_assert_eq!(lhs.as_ref(), rhs.as_slice());
		}
	}

	// Invariant: the row builder writes exactly the slice the table holds for that pair.
	//
	// The reduction's outer phase reads one slice at a time rather than building the table, so the
	// two paths have to agree entry for entry.
	//
	// The scratch row is carried across every pair and starts out non-zero, which is what pins the
	// full-write contract: a builder that left cells alone would leak the previous pair's weights.
	#[test]
	fn shift_operator_row_matches_its_slice_of_the_table() {
		type F = BinaryField128bGhash;

		let mut rng = StdRng::seed_from_u64(0);
		let psi = random_scalars::<F>(&mut rng, Word::BITS);

		let table = shift_operator_table::<F, F, _>(&GlobalAllocator, &psi);
		let mut row = vec![F::ONE; Word::BITS];
		for (variant_idx, variant) in ShiftVariant::ALL.into_iter().enumerate() {
			for amount in 0..Word::BITS {
				shift_operator_row(variant, amount, &mut row, &psi);
				let offset = (variant_idx * Word::BITS + amount) * Word::BITS;
				assert_eq!(
					row.as_slice(),
					&table.as_ref()[offset..offset + Word::BITS],
					"{variant:?} at amount {amount}"
				);
			}
		}
	}

	// Invariant: the amount-zero slice hands back the weights untouched.
	//
	// A zero amount is the identity for every variant.
	// That is what makes a single shift the special case of a sequence with a zero outer amount.
	//
	// The half-word forms read the amount modulo 32, so they are the identity at zero as well.
	#[test]
	fn the_zero_amount_slice_returns_the_weights_unchanged() {
		type F = BinaryField128bGhash;

		let mut rng = StdRng::seed_from_u64(0);
		let psi = random_scalars::<F>(&mut rng, Word::BITS);

		let table = shift_operator_table::<F, F, _>(&GlobalAllocator, &psi);
		for (variant_idx, variant) in ShiftVariant::ALL.into_iter().enumerate() {
			// Amount zero sits at the front of the variant's block of rows.
			let row = &table.as_ref()[variant_idx * Word::BITS * Word::BITS..][..Word::BITS];
			assert_eq!(row, psi.as_slice(), "{variant:?} at amount zero is not the identity");
		}
	}
}
