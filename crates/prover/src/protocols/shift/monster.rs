// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_math::{
	BinarySubspace, FieldBuffer, multilinear::eq::eq_ind_partial_eval, univariate::lagrange_evals,
};
use binius_utils::{checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::protocols::shift::{BITAND_ARITY, INTMUL_ARITY, evaluate_h_op};
use bytemuck::zeroed_vec;
use tracing::instrument;

use super::{
	SHIFT_VARIANT_COUNT,
	key_collection::{KeyCollection, KeySegment, Operation},
	prove::PreparedOperatorData,
};

/// Constructs the three "h" multilinear polynomials for shift operations at a
/// univariate challenge point. See the paper for definition of h polynomials.
///
/// There is one h multilinear for each shift variant. For each operation, there is one univariate
/// challenge `r_zhat_prime` at which to construct the h parts.
///
/// # Usage in Protocol
///
/// Used in phase 1, thus returning an array of multilinear evaluations.
#[instrument(skip_all, name = "build_h_parts")]
pub fn build_h_parts<F, P: PackedField<Scalar = F>>(
	domain_subspace: &BinarySubspace<F>,
	r_zhat_prime: F,
) -> [FieldBuffer<P>; SHIFT_VARIANT_COUNT]
where
	F: BinaryField,
{
	let l_tilde = lagrange_evals(domain_subspace, r_zhat_prime);
	let l_tilde = l_tilde.as_ref();

	fn build_part<F: Field, P: PackedField<Scalar = F>>(
		fill: impl Fn(usize, &mut [F; Word::BITS]),
	) -> FieldBuffer<P> {
		let mut data = zeroed_vec::<[F; Word::BITS]>(Word::BITS);
		for (s, chunk) in data.iter_mut().enumerate() {
			fill(s, chunk);
		}
		FieldBuffer::from_values(&data.into_flattened())
	}

	let sll = build_part(|s, sll_s| {
		sll_s[..Word::BITS - s].copy_from_slice(&l_tilde[s..]);
	});

	let srl = build_part(|s, srl_s| {
		srl_s[s..].copy_from_slice(&l_tilde[..Word::BITS - s]);
	});

	let sra = build_part(|s, sra_s| {
		sra_s[s..].copy_from_slice(&l_tilde[..Word::BITS - s]);
		sra_s[Word::BITS - 1] += l_tilde[Word::BITS - s..].iter().sum::<F>();
	});

	let rotr = build_part(|s, rotr_s| {
		rotr_s[..s].copy_from_slice(&l_tilde[Word::BITS - s..]);
		rotr_s[s..].copy_from_slice(&l_tilde[..Word::BITS - s]);
	});

	let sll32 = build_part(|s, sll32_s| {
		let s = s % 32;
		for (l_tilde_i, sll_s_i) in iter::zip(l_tilde.chunks(32), sll32_s.chunks_mut(32)) {
			sll_s_i[..32 - s].copy_from_slice(&l_tilde_i[s..]);
		}
	});

	let srl32 = build_part(|s, srl32_s| {
		let s = s % 32;
		for (l_tilde_i, srl32_s_i) in iter::zip(l_tilde.chunks(32), srl32_s.chunks_mut(32)) {
			srl32_s_i[s..].copy_from_slice(&l_tilde_i[..32 - s]);
		}
	});

	let sra32 = build_part(|s, sra32_s| {
		let s = s % 32;
		for (l_tilde_i, sra32_s_i) in iter::zip(l_tilde.chunks(32), sra32_s.chunks_mut(32)) {
			sra32_s_i[s..].copy_from_slice(&l_tilde_i[..32 - s]);
			sra32_s_i[32 - 1] += l_tilde_i[32 - s..].iter().sum::<F>();
		}
	});

	let rotr32 = build_part(|s, rotr32_s| {
		let s = s % 32;
		for (l_tilde_i, rotr32_s_i) in iter::zip(l_tilde.chunks(32), rotr32_s.chunks_mut(32)) {
			rotr32_s_i[..s].copy_from_slice(&l_tilde_i[32 - s..]);
			rotr32_s_i[s..].copy_from_slice(&l_tilde_i[..32 - s]);
		}
	});

	[sll, srl, sra, rotr, sll32, srl32, sra32, rotr32]
}

/// Constructs the "monster multilinear" that combines all shift operations into a single
/// multilinear.
///
/// This function builds a comprehensive multilinear polynomial that encapsulates both AND and IMUL
/// constraints with their associated shift operations. For each witness word, it computes the
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
/// ∑_{key ∈ keys[w]} key.accumulate(constraint_indices, tensor, scalars[key.id])
/// ```
/// where `scalars[key.id]` is the contiguous per-operand chunk encoding
/// `λ^(operand_idx+1) × h_op[shift_variant] × r_s_tensor[shift_amount]` for operand index
/// `operand_idx` and `shift_variant` in {SLL, SRL, SRA} and `shift_amount` in [0, Word::BITS).
///
/// # Usage
///
/// Used in phase 2 of the shift protocol. The two returned buffers are the segments of the
/// witness's monster multilinear: the public piece over `log_public_words` variables and the
/// hidden piece over `log_witness_words` variables (the hidden words at the base, zeros above).
/// The sparse first sumcheck round consumes them without materializing the combined buffer.
#[instrument(skip_all, name = "build_monster_segments")]
pub fn build_monster_segments<F, P: PackedField<Scalar = F>>(
	key_collection: &KeyCollection,
	bitand_operator_data: &PreparedOperatorData<F>,
	intmul_operator_data: &PreparedOperatorData<F>,
	domain_subspace: &BinarySubspace<F>,
	r_j: &[F],
	r_s: &[F],
) -> (FieldBuffer<P>, FieldBuffer<P>)
where
	F: BinaryField,
{
	// Compute h evaluations
	let [bitand_h_ops, intmul_h_ops] = [
		bitand_operator_data.r_zhat_prime,
		intmul_operator_data.r_zhat_prime,
	]
	.map(|r_zhat_prime| {
		let l_tilde = lagrange_evals(domain_subspace, r_zhat_prime);
		evaluate_h_op(l_tilde.as_ref(), r_j, r_s)
	});

	let r_s_tensor = eq_ind_partial_eval::<F>(r_s);

	// Allocate and populate the scalars, laid out with the operand index innermost so that the
	// `arity` weights for one `key.id` (= `op * Word::BITS + s`) form a contiguous chunk that
	// [`Key::accumulate`] can index directly by operand index.
	let mut bitand_scalars = vec![F::ZERO; BITAND_ARITY * SHIFT_VARIANT_COUNT * Word::BITS];
	let mut intmul_scalars = vec![F::ZERO; INTMUL_ARITY * SHIFT_VARIANT_COUNT * Word::BITS];

	let populate_scalars = |scalars: &mut [F], arity: usize, lambda_powers: &[F], h_ops: &[F]| {
		for op in 0..SHIFT_VARIANT_COUNT {
			for s in 0..Word::BITS {
				let key_id = op * Word::BITS + s;
				let op_s_scalar = h_ops[op] * r_s_tensor.as_ref()[s];
				for operand_idx in 0..arity {
					scalars[key_id * arity + operand_idx] =
						lambda_powers[operand_idx] * op_s_scalar;
				}
			}
		}
	};

	populate_scalars(
		&mut bitand_scalars,
		BITAND_ARITY,
		&bitand_operator_data.lambda_powers,
		&bitand_h_ops,
	);
	populate_scalars(
		&mut intmul_scalars,
		INTMUL_ARITY,
		&intmul_operator_data.lambda_powers,
		&intmul_h_ops,
	);

	// The scalar for one word of a segment: the accumulated contribution of all its keys. The
	// per-key wide accumulations are summed unreduced and reduced once at the end.
	let word_scalar = |segment: &KeySegment, index: usize| {
		let wide = segment
			.word_keys(index)
			.iter()
			.map(|key| {
				let (operator_data, scalars, arity) = match key.operation {
					Operation::BitwiseAnd => (bitand_operator_data, &bitand_scalars, BITAND_ARITY),
					Operation::IntegerMul => (intmul_operator_data, &intmul_scalars, INTMUL_ARITY),
				};
				let base = key.id as usize * arity;
				key.accumulate_wide(
					&segment.constraint_indices,
					operator_data.r_x_prime_tensor.as_ref(),
					&scalars[base..base + arity],
				)
			})
			.sum::<<F as WideMul>::Output>();
		F::reduce(wide)
	};

	// Each segment sits at the base of its buffer: the public piece fills its power-of-two
	// length exactly, the hidden piece is zero-padded up to the hidden segment length.
	let build_segment = |segment: &KeySegment, log_len: usize| {
		let capacity = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		let n_words = segment.n_words();
		// Full packed elements: each maps exactly `P::WIDTH` words, so `from_scalars` sees a
		// statically-sized iterator. The trailing partial element is filled separately below.
		let n_full = n_words / P::WIDTH;
		let mut values = Vec::<P>::with_capacity(capacity);
		(0..n_full)
			.into_par_iter()
			.map(|chunk_index| {
				let start = chunk_index * P::WIDTH;
				P::from_scalars((0..P::WIDTH).map(|i| word_scalar(segment, start + i)))
			})
			.collect_into_vec(&mut values);
		if !n_words.is_multiple_of(P::WIDTH) {
			let start = n_full * P::WIDTH;
			values.push(P::from_scalars(
				(start..n_words).map(|word_index| word_scalar(segment, word_index)),
			));
		}
		values.resize(capacity, P::default());
		FieldBuffer::new(log_len, values.into_boxed_slice())
	};

	let log_public_words = checked_log_2(key_collection.public.n_words());
	let public_monster = build_segment(&key_collection.public, log_public_words);
	let hidden_monster = build_segment(&key_collection.hidden, key_collection.log_witness_words());

	(public_monster, hidden_monster)
}

#[cfg(test)]
mod tests {
	use binius_field::{AESTowerField8b, BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
	use binius_math::{inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval};
	use binius_verifier::protocols::shift::evaluate_h_op;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	/// Test consistency between direct multilinear evaluation of h
	/// multilinears and the succinct `evaluate_h_op` implementation
	#[test]
	fn h_op_consistency() {
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash2x128b;

		let mut rng = StdRng::seed_from_u64(0);

		let num_random_tests = 10;

		for test_case in 0..num_random_tests {
			let r_zhat_prime = F::random(&mut rng);

			let r_j: Vec<F> = (0..6).map(|_| F::random(&mut rng)).collect();
			let r_s: Vec<F> = (0..6).map(|_| F::random(&mut rng)).collect();

			// Method 1: Succinct evaluation using `evaluate_h_op`
			let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
			let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
			let succinct_evaluations = evaluate_h_op(l_tilde.as_ref(), &r_j, &r_s);

			// Method 2: Direct evaluation via multilinear part
			let h_parts = build_h_parts(&subspace, r_zhat_prime);
			let evaluation_point: Vec<F> = [r_j.clone(), r_s.clone()].concat();
			let tensor = eq_ind_partial_eval::<P>(&evaluation_point);
			let direct_evaluations = h_parts.map(|buf| inner_product_buffers(&buf, &tensor));

			assert_eq!(
				succinct_evaluations, direct_evaluations,
				"H-op evaluation mismatch (test_case={test_case}): succinct != direct",
			);
		}
	}
}
