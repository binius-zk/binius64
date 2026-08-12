// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_compute::{Allocator, VecLike};
use binius_core::{ShiftVariant, word::Word};
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_math::{
	BinarySubspace, FieldBuffer, FieldVec, multilinear::eq::eq_ind_partial_eval,
	univariate::lagrange_evals,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use binius_verifier::protocols::shift::{BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, ZERO_ARITY};
use bytemuck::zeroed_vec;
use tracing::instrument;

use super::{
	claims::PreparedOperatorClaims,
	key_collection::{DenseShiftEncoding, KeyCollection, KeySegment, Operation},
	phase_1::PHASE_1_LOG_LEN,
};

/// The width the half-word (`*32`) shift variants act over.
const HALF_WORD_BITS: usize = 32;

/// The phase-2 scalar weights of one key segment, one table per operation.
///
/// A table holds the `arity` weights of each shift the segment uses, in dense shift index order,
/// so a key's weights are the chunk at `key.dense_shift_idx * arity`.
struct ScalarTables<F> {
	zero: Vec<F>,
	bitand: Vec<F>,
	intmul: Vec<F>,
	binmul: Vec<F>,
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

/// Constructs the "h" multilinear for shift operations at a univariate challenge point.
///
/// See the paper for the definition of the h polynomials. There is one per shift variant, and this
/// returns all of them as a single multilinear over [`PHASE_1_LOG_LEN`] variables: the shift
/// variant occupies the high
/// [`LOG_SHIFT_VARIANT_COUNT`](binius_verifier::protocols::shift::LOG_SHIFT_VARIANT_COUNT)
/// variables, the shift amount the middle
/// [`Word::LOG_BITS`], and the bit position the low [`Word::LOG_BITS`].
///
/// # Usage in Protocol
///
/// Phase 1 runs one sumcheck over this multilinear and its "g" counterpart, so the variant axis is
/// folded by the sumcheck rather than summed across separate provers.
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
	let l_tilde = l_tilde.as_ref();

	// One row of `Word::BITS` scalars per `(variant, amount)`, variant most significant.
	let mut data = zeroed_vec::<F>(1 << PHASE_1_LOG_LEN);
	for (variant, block) in
		iter::zip(ShiftVariant::ALL, data.chunks_exact_mut(Word::BITS * Word::BITS))
	{
		for (amount, row) in block.chunks_exact_mut(Word::BITS).enumerate() {
			fill_h_row(variant, amount, row, l_tilde);
		}
	}

	FieldBuffer::from_values_in(alloc, &data)
}

/// Constructs the "monster multilinear" that combines all shift operations into a single
/// multilinear.
///
/// This function builds a comprehensive multilinear polynomial that encapsulates the AND, IMUL and
/// BMUL constraints with their associated shift operations. For each witness word, it computes the
/// contribution from all constraints involving that word, weighted by the h evaluation and lambda
/// powers.
///
/// # Construction Process
///
/// 1. **Compute lambda powers**: Powers λ^(i+1) for each operand index in both operations
/// 2. **Build scalar matrix**: Create scalars combining lambda powers, the h evaluation, and the
///    `r_s` and `r_v` tensors
/// 3. **Process keys in parallel**: For each word, accumulate contributions from all its
///    constraints
///
/// # Formula
///
/// For each word w, computes:
/// ```text
/// ∑_{key ∈ keys[w]} key.accumulate(constraint_indices, tensor, scalars[key.dense_shift_idx])
/// ```
/// where `scalars[key.dense_shift_idx]` is the contiguous per-operand chunk encoding
/// `λ^(operand_idx+1) × h_eval × r_v_tensor[shift_variant] × r_s_tensor[shift_amount]` for operand
/// index `operand_idx`, and `(shift_variant, shift_amount)` the pair the key's segment decodes its
/// dense shift index to. The h evaluation comes from
/// [`ShiftIndSumcheck`](super::ShiftIndSumcheck), which proves it.
///
/// # Usage
///
/// Used in phase 2 of the shift protocol. The two returned buffers are the segments of the
/// witness's monster multilinear: the public piece over `log_public_words` variables and the
/// hidden piece over `log_witness_words` variables (the hidden words at the base, zeros above).
/// The sparse first sumcheck round consumes them without materializing the combined buffer.
#[instrument(skip_all, name = "build_monster_segments")]
pub fn build_monster_segments<F, P: PackedField<Scalar = F>, A: Allocator>(
	alloc: &A,
	key_collection: &KeyCollection,
	prepared: &PreparedOperatorClaims<F>,
	h_eval: F,
	r_s: &[F],
	r_v: &[F],
) -> (FieldVec<P, A>, FieldVec<P, A>)
where
	F: BinaryField,
{
	let r_v_tensor = eq_ind_partial_eval::<F>(r_v);
	let r_s_tensor = eq_ind_partial_eval::<F>(r_s);

	// The scalars of one operation, laid out with the operand index innermost so that the `arity`
	// weights for one `key.dense_shift_idx` form a contiguous chunk that [`Key::accumulate_wide`]
	// can index directly by operand index.
	//
	// A key's shift now selects itself through a pure equality indicator over both of phase 1's
	// shift axes, and the h evaluation is a single factor shared by every key.
	let build_scalars =
		|arity: usize, lambda_powers: &[F], dense_shift_enc: &DenseShiftEncoding| {
			let mut scalars = vec![F::ZERO; arity * dense_shift_enc.len()];
			for (dense_shift_idx, (variant, amount)) in dense_shift_enc.iter().enumerate() {
				let shift_scalar = h_eval
					* r_v_tensor.as_ref()[variant as usize]
					* r_s_tensor.as_ref()[amount as usize];
				for operand_idx in 0..arity {
					scalars[dense_shift_idx * arity + operand_idx] =
						lambda_powers[operand_idx] * shift_scalar;
				}
			}
			scalars
		};

	// Each segment has its own dense shift encoding, so it has its own scalar tables.
	let build_scalar_tables = |dense_shift_enc: &DenseShiftEncoding| ScalarTables {
		zero: build_scalars(ZERO_ARITY, &prepared.zero.lambda_powers, dense_shift_enc),
		bitand: build_scalars(BITAND_ARITY, &prepared.bitand.lambda_powers, dense_shift_enc),
		intmul: build_scalars(INTMUL_ARITY, &prepared.intmul.lambda_powers, dense_shift_enc),
		binmul: build_scalars(BINMUL_ARITY, &prepared.binmul.lambda_powers, dense_shift_enc),
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
	use binius_verifier::protocols::shift::LOG_SHIFT_VARIANT_COUNT;
	use rand::{SeedableRng, rngs::StdRng};

	use super::{super::ShiftIndSumcheck, *};

	/// Phase 1's h multilinear and the h evaluation the last sumcheck claims must agree.
	///
	/// The claim sums the shift indicators over the bit index, weighted by the Lagrange
	/// evaluations; the multilinear holds those sums over the whole shift axis. So evaluating the
	/// multilinear at `(r_j, r_s, r_v)` must give the claim — which is the single scalar phase 2
	/// weights its keys by.
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

			// Method 1: the sum the last sumcheck claims.
			let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
			let claimed = ShiftIndSumcheck::<P, _>::new(
				&GlobalAllocator,
				&subspace,
				r_zhat_prime,
				&r_j,
				&r_s,
				&r_v,
			)
			.h_eval();

			// Method 2: evaluate the built multilinear at the whole point.
			let h = build_h(&GlobalAllocator, &subspace, r_zhat_prime);
			let evaluation_point = [r_j, r_s, r_v].concat();
			let tensor = eq_ind_partial_eval::<P>(&evaluation_point);
			let direct = inner_product_buffers(&h, &tensor);

			assert_eq!(
				claimed, direct,
				"H-op evaluation mismatch (test_case={test_case}): claimed != direct",
			);
		}
	}
}
