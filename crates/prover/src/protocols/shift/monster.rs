// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, ops::Index};

use binius_compute::{Allocator, VecLike};
use binius_core::{ShiftVariant, constraint_system::Shift, word::Word};
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_math::{
	BinarySubspace, FieldBuffer, FieldVec, inner_product::inner_product,
	multilinear::eq::eq_ind_partial_eval, univariate::lagrange_evals,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use binius_verifier::protocols::shift::{
	BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, LOG_SHIFT_VARIANT_COUNT, ZERO_ARITY, evaluate_h_op,
	evaluate_inner_h_op,
};
use bytemuck::zeroed_vec;
use tracing::instrument;

use super::{
	claims::PreparedOperatorClaims,
	key_collection::{DenseShiftEncoding, KeyCollection, KeySegment, Operation},
	slot_phase::SLOT_PHASE_LOG_LEN,
};

/// The width the half-word (`*32`) shift variants act over.
const HALF_WORD_BITS: usize = 32;

/// The words phase's scalar weights of one key segment, one table per operation.
///
/// A table holds the `arity` weights of each shift sequence the segment uses, in dense shift index
/// order, so a key's weights are the chunk at `key.dense_shift_idx * arity`.
struct ScalarTables<F> {
	zero: Vec<F>,
	bitand: Vec<F>,
	intmul: Vec<F>,
	binmul: Vec<F>,
}

/// The equality-indicator weights of one shift slot, over its two axes.
///
/// A sequence's weight factorizes across its two slots, so each slot carries its own two tensors:
/// one over the variant axis, one over the amount axis.
/// Keeping them apart holds the weights at `2 * SHIFT_COUNT` entries rather than `SHIFT_COUNT^2`.
/// The cost is one extra multiply per key.
pub struct SlotWeights<F: Field> {
	/// The equality indicator over this slot's variant axis, one weight per shift variant.
	variant: FieldBuffer<F>,
	/// The equality indicator over this slot's amount axis, one weight per shift amount.
	amount: FieldBuffer<F>,
}

impl<F: Field> SlotWeights<F> {
	/// The weights at the amount and variant points a phase bound for this slot.
	///
	/// # Panics
	///
	/// Panics if the points are not the widths of the two axes.
	pub fn at(r_s: &[F], r_v: &[F]) -> Self {
		assert_eq!(r_s.len(), Word::LOG_BITS);
		assert_eq!(r_v.len(), LOG_SHIFT_VARIANT_COUNT);
		Self {
			variant: eq_ind_partial_eval::<F>(r_v),
			amount: eq_ind_partial_eval::<F>(r_s),
		}
	}

	/// The weights that select the identity and reject every other shift in the slot.
	///
	/// This is the equality indicator at the all-zero point, where a reduction stands on a slot it
	/// never bound: no challenges were drawn over those axes.
	/// Every key's shift there is [`Shift::IDENTITY`], spelled `(Sll, 0)`.
	/// So a well-formed key weighs one, and the scalars match what a single-slot reduction builds.
	pub fn identity_selecting() -> Self {
		Self {
			variant: eq_ind_partial_eval::<F>(&[F::ZERO; LOG_SHIFT_VARIANT_COUNT]),
			amount: eq_ind_partial_eval::<F>(&[F::ZERO; Word::LOG_BITS]),
		}
	}

	/// The weight one shift in this slot contributes to its sequence's scalar.
	#[inline]
	pub fn weight(&self, shift: Shift) -> F {
		self.variant.as_ref()[shift.variant as usize] * self.amount.as_ref()[shift.amount as usize]
	}
}

/// One scalar per operation, as the words phase weights that operation's keys by.
///
/// Every key of an operation shares the weight of the phase that closed the operand claim's bit
/// axis.
/// So it is a single factor rather than part of the per-shift tables.
#[derive(Debug, Clone, Copy)]
pub struct OperationScalars<F> {
	/// The scalar shared by every ZERO key.
	pub zero: F,
	/// The scalar shared by every AND key.
	pub bitand: F,
	/// The scalar shared by every IMUL key.
	pub intmul: F,
	/// The scalar shared by every BMUL key.
	pub binmul: F,
}

impl<F: Field> OperationScalars<F> {
	/// Scales every operation's scalar by a factor the whole reduction shares.
	///
	/// The inner phase's weight carries no oblong factor.
	/// So it is one value for all four operations, and folds in here rather than per shift.
	pub fn scaled_by(self, factor: F) -> Self {
		Self {
			zero: self.zero * factor,
			bitand: self.bitand * factor,
			intmul: self.intmul * factor,
			binmul: self.binmul * factor,
		}
	}
}

impl<F> Index<Operation> for OperationScalars<F> {
	type Output = F;

	fn index(&self, operation: Operation) -> &F {
		match operation {
			Operation::Zero => &self.zero,
			Operation::BitwiseAnd => &self.bitand,
			Operation::IntegerMul => &self.intmul,
			Operation::BinMul => &self.binmul,
		}
	}
}

/// Fills one row of the phase-1 "h" multilinear: the shift indicator of one `(variant, amount)`
/// pair, contracted against the Lagrange evaluations at the univariate challenge.
///
/// This is the one place that says what a variant does to the challenge weights:
/// - Logical left and logical right move the weights and leave zeros behind.
/// - Arithmetic right piles every weight that falls off the end onto the sign position.
/// - Rotate wraps them around instead.
/// - The half-word forms apply the same rule to each 32-bit half, reading only the low 5 bits of
///   the amount.
fn fill_h_row<F: Field>(variant: ShiftVariant, amount: usize, row: &mut [F], l_tilde: &[F]) {
	// A half-word variant repeats the full-width rule over each half, so both share one closure.
	let halves = |row: &mut [F], rule: fn(usize, &mut [F], &[F])| {
		let amount = amount % HALF_WORD_BITS;
		for (row_half, l_tilde_half) in
			iter::zip(row.chunks_mut(HALF_WORD_BITS), l_tilde.chunks(HALF_WORD_BITS))
		{
			rule(amount, row_half, l_tilde_half);
		}
	};

	fn sll<F: Field>(amount: usize, row: &mut [F], l_tilde: &[F]) {
		let width = row.len();
		row[..width - amount].copy_from_slice(&l_tilde[amount..]);
	}
	fn srl<F: Field>(amount: usize, row: &mut [F], l_tilde: &[F]) {
		let width = row.len();
		row[amount..].copy_from_slice(&l_tilde[..width - amount]);
	}
	fn sar<F: Field>(amount: usize, row: &mut [F], l_tilde: &[F]) {
		let width = row.len();
		srl(amount, row, l_tilde);
		// Every position past the shift reads the sign bit, so their weights pile onto it.
		row[width - 1] += l_tilde[width - amount..].iter().sum::<F>();
	}
	fn rotr<F: Field>(amount: usize, row: &mut [F], l_tilde: &[F]) {
		let width = row.len();
		row[..amount].copy_from_slice(&l_tilde[width - amount..]);
		row[amount..].copy_from_slice(&l_tilde[..width - amount]);
	}

	match variant {
		ShiftVariant::Sll => sll(amount, row, l_tilde),
		ShiftVariant::Slr => srl(amount, row, l_tilde),
		ShiftVariant::Sar => sar(amount, row, l_tilde),
		ShiftVariant::Rotr => rotr(amount, row, l_tilde),
		ShiftVariant::Sll32 => halves(row, sll),
		ShiftVariant::Srl32 => halves(row, srl),
		ShiftVariant::Sra32 => halves(row, sar),
		ShiftVariant::Rotr32 => halves(row, rotr),
	}
}

/// Constructs the "h" multilinear of a phase from its per-output-bit weights.
///
/// See the paper for the definition of the h polynomials. There is one per shift variant, and this
/// returns all of them as a single multilinear over [`SLOT_PHASE_LOG_LEN`] variables: the shift
/// variant occupies the high
/// [`LOG_SHIFT_VARIANT_COUNT`](binius_verifier::protocols::shift::LOG_SHIFT_VARIANT_COUNT)
/// variables, the shift amount the middle
/// [`Word::LOG_BITS`], and the bit position the low [`Word::LOG_BITS`].
///
/// Both phases contract each variant's shift indicator against a weight per output bit position.
/// Only the weights differ.
/// The verifier evaluates the same object succinctly rather than building it.
///
/// # Usage in Protocol
///
/// A phase runs one sumcheck over this multilinear and its "g" counterpart, so the variant axis is
/// folded by the sumcheck rather than summed across separate provers.
#[instrument(skip_all, name = "build_h_from_weights")]
pub fn build_h_from_weights<F, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	l_tilde: &[F],
) -> FieldVec<P, A>
where
	F: BinaryField,
{
	assert_eq!(l_tilde.len(), Word::BITS);

	// One row of `Word::BITS` scalars per `(variant, amount)`, variant most significant.
	let mut data = zeroed_vec::<F>(1 << SLOT_PHASE_LOG_LEN);
	for (variant, block) in
		iter::zip(ShiftVariant::ALL, data.chunks_exact_mut(Word::BITS * Word::BITS))
	{
		for (amount, row) in block.chunks_exact_mut(Word::BITS).enumerate() {
			fill_h_row(variant, amount, row, l_tilde);
		}
	}

	FieldBuffer::from_values_in(alloc, &data)
}

/// Constructs the outer phase's "H" multilinear at a univariate challenge point.
///
/// The weights are the oblong factor `delta_D(r_zhat_prime, .)`: the Lagrange basis of the bit axis
/// at the univariate challenge.
/// This closes the operand claim's bit axis, which is why it belongs to the phase binding the
/// *output* end of the shift sequence.
#[instrument(skip_all, name = "build_h")]
pub fn build_h<F, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	domain_subspace: &BinarySubspace<F>,
	r_zhat_prime: F,
) -> FieldVec<P, A>
where
	F: BinaryField,
{
	let l_tilde = lagrange_evals(domain_subspace, r_zhat_prime);
	build_h_from_weights(alloc, l_tilde.as_ref())
}

/// Constructs the inner phase's "h" multilinear at the intermediate point the outer phase produced.
///
/// The weights are the equality indicator of the intermediate point, so this is the bare shift
/// indicator there.
/// It carries no oblong factor, since the outer phase already summed the output bit index out.
///
/// # Panics
///
/// Panics if `r_k` does not have `Word::LOG_BITS` coordinates.
#[instrument(skip_all, name = "build_inner_h")]
pub fn build_inner_h<F, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	r_k: &[F],
) -> FieldVec<P, A>
where
	F: BinaryField,
{
	assert_eq!(r_k.len(), Word::LOG_BITS);
	let eq_r_k = eq_ind_partial_eval::<F>(r_k);
	build_h_from_weights(alloc, eq_r_k.as_ref())
}

/// The outer phase's `H` evaluation for each operation, at the point that phase bound.
///
/// This weight carries the oblong factor `delta_D(r_zhat_prime, .)`.
/// Each operation claims at its own univariate challenge, so there is one evaluation per operation
/// rather than one shared value.
///
/// # Arguments
///
/// - `r_out`: the bit point the phase bound. Without an outer phase this is the source bit point,
///   since one phase then closes both the bit axis and the words.
/// - `r_s`, `r_v`: the amount and variant points of the slot that phase bound.
pub fn outer_h_evals<F: BinaryField>(
	prepared: &PreparedOperatorClaims<F>,
	domain_subspace: &BinarySubspace<F>,
	r_out: &[F],
	r_s: &[F],
	r_v: &[F],
) -> OperationScalars<F> {
	// The phase folded the variant axis into its sumcheck.
	// So the weight is one scalar per operation: its eight indicators interpolated at `r_v`.
	let r_v_tensor = eq_ind_partial_eval::<F>(r_v);
	let eval_at = |r_zhat_prime: F| {
		let l_tilde = lagrange_evals(domain_subspace, r_zhat_prime);
		let h_ops = evaluate_h_op(l_tilde.as_ref(), r_out, r_s);
		inner_product(h_ops, r_v_tensor.as_ref().iter().copied())
	};
	OperationScalars {
		zero: eval_at(prepared.zero.r_zhat_prime),
		bitand: eval_at(prepared.bitand.r_zhat_prime),
		intmul: eval_at(prepared.intmul.r_zhat_prime),
		binmul: eval_at(prepared.binmul.r_zhat_prime),
	}
}

/// The inner phase's `h` evaluation, at the point that phase bound.
///
/// This carries no oblong factor, since the outer phase already summed the output bit index out.
/// So it is one value shared by every operation, folded into the per-operation scalars.
///
/// # Arguments
///
/// - `r_k`: the intermediate bit point the outer phase produced.
/// - `r_j`, `r_s`, `r_v`: the source bit point and the inner slot's amount and variant points.
pub fn inner_h_eval<F: BinaryField>(r_k: &[F], r_j: &[F], r_s: &[F], r_v: &[F]) -> F {
	let r_v_tensor = eq_ind_partial_eval::<F>(r_v);
	let h_ops = evaluate_inner_h_op(r_k, r_j, r_s);
	inner_product(h_ops, r_v_tensor.as_ref().iter().copied())
}

/// Constructs the "monster multilinear" that combines all shift operations into a single
/// multilinear.
///
/// This function builds a comprehensive multilinear polynomial that encapsulates the AND, IMUL and
/// BMUL constraints with their associated shift operations. For each witness word, it computes the
/// contribution from all constraints involving that word, weighted by the appropriate h-polynomial
/// evaluations and lambda powers.
///
/// # Construction Process
///
/// 1. **Compute lambda powers**: Powers λ^(i+1) for each operand index in both operations
/// 2. **Evaluate h-polynomials**: Compute h_op evaluations for SLL, SRL, SRA at challenge points
/// 3. **Build scalar matrix**: Create scalars combining lambda powers, h-evaluations, and r_s
///    tensor
/// 4. **Process keys in parallel**: For each word, accumulate contributions from all its
///    constraints
///
/// # Formula
///
/// For each word w, computes:
/// ```text
/// ∑_{key ∈ keys[w]} key.accumulate(constraint_indices, tensor, scalars[key.dense_shift_idx])
/// ```
/// where a key's entry is the contiguous per-operand chunk encoding
///
/// ```text
/// lambda^(operand + 1) * operation_scalar * inner_weight(s_1) * outer_weight(s_2)
/// ```
///
/// with `(s_1, s_2)` the shift sequence the key's segment decodes its dense shift index to.
///
/// # Arguments
///
/// - `operation_scalars`: the `h` evaluation each operation's claim contributes to all of its keys.
/// - `inner`, `outer`: the equality weights of the two shift slots, at the points the phases bound.
///
/// # Usage
///
/// Used in the words phase of the shift protocol. The two returned buffers are the segments of the
/// witness's monster multilinear: the public piece over `log_public_words` variables and the
/// hidden piece over `log_witness_words` variables (the hidden words at the base, zeros above).
/// The sparse first sumcheck round consumes them without materializing the combined buffer.
#[instrument(skip_all, name = "build_monster_segments")]
pub fn build_monster_segments<F, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	key_collection: &KeyCollection,
	prepared: &PreparedOperatorClaims<F>,
	operation_scalars: OperationScalars<F>,
	inner: &SlotWeights<F>,
	outer: &SlotWeights<F>,
) -> (FieldVec<P, A>, FieldVec<P, A>)
where
	F: BinaryField,
{
	// The scalars of one operation, laid out with the operand index innermost so that the `arity`
	// weights for one `key.dense_shift_idx` form a contiguous chunk that [`Key::accumulate_wide`]
	// can index directly by operand index.
	//
	// A key's sequence selects itself through an equality indicator over both slots' axes.
	// The h evaluation is one factor shared by every key of the operation.
	let build_scalars =
		|arity: usize, lambda_powers: &[F], h_eval: F, dense_shift_enc: &DenseShiftEncoding| {
			let mut scalars = vec![F::ZERO; arity * dense_shift_enc.len()];
			for (dense_shift_idx, shift_seq) in dense_shift_enc.iter().enumerate() {
				let [inner_shift, outer_shift] = shift_seq;
				let shift_scalar = h_eval * inner.weight(inner_shift) * outer.weight(outer_shift);
				for operand_idx in 0..arity {
					scalars[dense_shift_idx * arity + operand_idx] =
						lambda_powers[operand_idx] * shift_scalar;
				}
			}
			scalars
		};

	// Each segment has its own dense shift encoding, so it has its own scalar tables.
	let build_scalar_tables = |dense_shift_enc: &DenseShiftEncoding| ScalarTables {
		zero: build_scalars(
			ZERO_ARITY,
			&prepared.zero.lambda_powers,
			operation_scalars.zero,
			dense_shift_enc,
		),
		bitand: build_scalars(
			BITAND_ARITY,
			&prepared.bitand.lambda_powers,
			operation_scalars.bitand,
			dense_shift_enc,
		),
		intmul: build_scalars(
			INTMUL_ARITY,
			&prepared.intmul.lambda_powers,
			operation_scalars.intmul,
			dense_shift_enc,
		),
		binmul: build_scalars(
			BINMUL_ARITY,
			&prepared.binmul.lambda_powers,
			operation_scalars.binmul,
			dense_shift_enc,
		),
	};

	// The scalar for one word of a segment: the accumulated contribution of all its keys. The
	// per-key wide accumulations are summed unreduced and reduced once at the end.
	let word_scalar = |segment: &KeySegment, tables: &ScalarTables<F>, index: usize| {
		let wide = segment
			.word_keys(index)
			.iter()
			.map(|key| {
				// The scalar table is per operation, and its stride is that operation's arity.
				let (scalars, arity) = match key.operation {
					Operation::Zero => (&tables.zero, ZERO_ARITY),
					Operation::BitwiseAnd => (&tables.bitand, BITAND_ARITY),
					Operation::IntegerMul => (&tables.intmul, INTMUL_ARITY),
					Operation::BinMul => (&tables.binmul, BINMUL_ARITY),
				};
				let base = key.dense_shift_idx as usize * arity;
				key.accumulate_wide(
					&segment.constraint_indices,
					prepared[key.operation].r_x_prime_tensor.as_ref(),
					&scalars[base..base + arity],
				)
			})
			.sum::<<F as WideMul>::Output>();
		F::reduce(wide)
	};

	// Each segment sits at the base of its buffer: the public piece fills its power-of-two
	// length exactly, the hidden piece is zero-padded up to the hidden segment length.
	let build_segment = |segment: &KeySegment, log_len: usize| {
		let tables = build_scalar_tables(&segment.dense_shift_enc);
		let capacity = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		let n_words = segment.n_words();
		// Full packed elements: each maps exactly `P::WIDTH` words, so `from_scalars` sees a
		// statically-sized iterator. The trailing partial element is filled separately below.
		let n_full = n_words / P::WIDTH;
		// Allocate the backing buffer up front from the allocator, then fill the `n_full` aligned
		// packed elements in parallel through its spare capacity — the single allocation happens
		// before the parallel region, which only writes.
		let mut values = alloc.alloc::<P>(capacity);
		values.spare_capacity_mut()[..n_full]
			.par_iter_mut()
			.enumerate()
			.for_each(|(chunk_index, slot)| {
				let start = chunk_index * P::WIDTH;
				slot.write(P::from_scalars(
					(0..P::WIDTH).map(|i| word_scalar(segment, &tables, start + i)),
				));
			});
		// Safety: the parallel loop above initialized every one of the `n_full` slots.
		unsafe { values.set_len(n_full) };
		if !n_words.is_multiple_of(P::WIDTH) {
			let start = n_full * P::WIDTH;
			values.push(P::from_scalars(
				(start..n_words).map(|word_index| word_scalar(segment, &tables, word_index)),
			));
		}
		values.resize(capacity, P::default());
		FieldBuffer::new(log_len, values)
	};

	// The segment word count need not be a power of two; the monster spans the rounded-up count.
	let log_public_words = log2_ceil_usize(key_collection.public.n_words());
	let public_monster = build_segment(&key_collection.public, log_public_words);
	let hidden_monster = build_segment(&key_collection.hidden, key_collection.log_witness_words());

	(public_monster, hidden_monster)
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{AESTowerField8b, BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
	use binius_math::{
		inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval,
		test_utils::random_scalars,
	};
	use binius_verifier::protocols::shift::{LOG_SHIFT_VARIANT_COUNT, evaluate_h_op};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	/// The built h multilinear and the succinct `evaluate_h_op` must agree.
	///
	/// The variant axis is folded by the sumcheck now, so evaluating the whole multilinear at
	/// `(r_j, r_s, r_v)` gives the interpolation of the eight succinct evaluations over `r_v` —
	/// which is exactly the single scalar phase 2 weights its keys by.
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

			// Method 1: the succinct per-variant evaluations, interpolated over the variant axis.
			let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
			let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
			let succinct_evaluations = evaluate_h_op(l_tilde.as_ref(), &r_j, &r_s);
			let r_v_tensor = eq_ind_partial_eval::<F>(&r_v);
			let succinct = inner_product(succinct_evaluations, r_v_tensor.as_ref().iter().copied());

			// Method 2: evaluate the built multilinear at the whole point.
			let h = build_h(&GlobalAllocator, &subspace, r_zhat_prime);
			let evaluation_point = [r_j, r_s, r_v].concat();
			let tensor = eq_ind_partial_eval::<P>(&evaluation_point);
			let direct = inner_product_buffers(&h, &tensor);

			assert_eq!(
				succinct, direct,
				"H-op evaluation mismatch (test_case={test_case}): succinct != direct",
			);
		}
	}
}
