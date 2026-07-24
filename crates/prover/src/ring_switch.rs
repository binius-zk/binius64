// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, ops::Deref};

use binius_field::{
	BinaryField, Divisible, ExtensionField, Field, PackedField, cast_base_mut,
	linear_transformation::{
		BytewiseLookupTransformationFactory, InputWrappingTransformationFactory,
		LinearTransformationFactory, OutputWrappingTransformationFactory, Transformation,
	},
	util::expand_subset_sums_array,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	FieldBuffer, inner_product::inner_product, multilinear::eq::eq_ind_partial_eval,
	tensor_algebra::TensorAlgebra,
};
use binius_utils::{
	checked_arithmetics::{checked_int_div, checked_log_2},
	rayon::prelude::*,
};
use binius_verifier::config::{B1, B128};
use itertools::izip;

/// Transforms a [`FieldBuffer`] by mapping every scalar to the inner product of its B1 components
/// and a given vector of field elements.
///
/// ## Arguments
///
/// * `elems` - the buffer of `F` elements to transform
/// * `vec` - the vector of `F` field elements (must have length equal to extension degree)
///
/// ## Returns
///
/// The transformed buffer with each element replaced by its inner product result
///
/// ## Preconditions
///
/// * `vec` has length equal to the extension degree of `F` over `B1`
pub fn fold_elems_inplace<F, P>(mut elems: FieldBuffer<P>, vec: &FieldBuffer<F>) -> FieldBuffer<P>
where
	F: BinaryField,
	F::Underlier: Divisible<u8>,
	P: PackedField<Scalar = F>,
{
	assert_eq!(vec.log_len(), F::LOG_DEGREE); // precondition

	// Create transformation factory with proper wrapping
	let factory = OutputWrappingTransformationFactory::new(
		InputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory),
	);

	// Create the transformation from the vector
	let transform = factory.create(vec.as_ref());

	// Apply transformation to each scalar in each packed element
	elems.as_mut().par_iter_mut().for_each(|packed_elem| {
		*packed_elem = P::from_scalars(
			packed_elem
				.into_iter()
				.map(|scalar| transform.transform(&scalar)),
		);
	});

	elems
}

/// Optimized version of [`fold_elems_inplace`] specifically for B128 fields.
///
/// Every scalar is replaced by the inner product of its 128 bits and the given vector.
/// The map is applied through per-byte lookup tables, in place, one parallel pass.
///
/// ## Arguments
///
/// * `elems` - the buffer of B128 elements to transform
/// * `vec` - the vector of B128 field elements (must have length 128 = B128 extension degree)
///
/// ## Returns
///
/// The transformed buffer with each element replaced by its inner product result
///
/// ## Preconditions
///
/// * `vec` must have log length equal to B128's extension degree over B1 (7)
pub fn fold_b128_elems_inplace<P>(
	mut elems: FieldBuffer<P>,
	vec: &FieldBuffer<B128>,
) -> FieldBuffer<P>
where
	P: PackedField<Scalar = B128>,
{
	let lookup_tables = row_batch_lookup_tables(vec);

	elems.as_mut().par_iter_mut().for_each(|packed_elem| {
		*packed_elem = P::from_scalars(
			packed_elem
				.into_iter()
				.map(|scalar| apply_row_batch_tables(&lookup_tables, scalar)),
		);
	});

	elems
}

/// Builds the per-byte lookup tables of a bitwise fold by a 128-element vector.
///
/// Folding a scalar means replacing it with the inner product of its 128 bits and the vector.
/// That map is F_2-linear.
/// So byte `i` of the input contributes a subset sum of the vector's entries 8i..8i+8.
/// The 16 tables of 256 entries each cover every byte position and value.
fn row_batch_lookup_tables(vec: &FieldBuffer<B128>) -> Vec<[B128; 256]> {
	assert_eq!(vec.len(), B128::N_BITS); // precondition

	// Group bits into bytes for the lookups.
	const CHUNK_BITS: usize = 8;

	let lookup_tables = vec
		.as_ref()
		.chunks(CHUNK_BITS)
		.map(|chunk| {
			let chunk = <[B128; CHUNK_BITS]>::try_from(chunk)
				.expect("vec.len() == 128; thus, chunks must be exact CHUNK_BITS in size");
			expand_subset_sums_array::<_, CHUNK_BITS, { 1 << CHUNK_BITS }>(chunk)
		})
		.collect::<Vec<_>>();

	assert_eq!(lookup_tables.len(), checked_int_div(B128::N_BITS, CHUNK_BITS));

	lookup_tables
}

/// Folds one scalar through the per-byte tables of [`row_batch_lookup_tables`].
#[inline]
fn apply_row_batch_tables(lookup_tables: &[[B128; 256]], scalar: B128) -> B128 {
	let bytes = u128::from(scalar.val()).to_le_bytes();
	bytes
		.into_iter()
		.enumerate()
		.map(|(i, byte)| {
			// Safety: i is in the range 0..16, and byte is in range 0..256
			unsafe { lookup_tables.get_unchecked(i).get_unchecked(byte as usize) }
		})
		.sum()
}

/// Base-2 log of the nibble width the row folds group bits into for their lookups.
const LOG_FOLD_CHUNK_BITS: usize = 2;
/// Nibble width of the row folds' subset-sum lookups.
const FOLD_CHUNK_BITS: usize = 1 << LOG_FOLD_CHUNK_BITS;

/// Base-2 log of the low-factor length the tensor split targets.
///
/// The split fold builds one subset-sum table per nibble of the low factor.
/// At 2^10 low entries that is 256 tables of 256 bytes, 64 KiB in total.
/// That footprint stays resident in a performance core's L1 data cache across every hi-block.
/// Each hi-block then pays 128 multiplies to scale its result, well under one per row.
const LOG_SPLIT_BLOCK: usize = 10;

/// Optimized version of folding 1-bit rows specifically for B128 fields.
///
/// This function computes the linear combination of the rows of a B1 matrix by B128 extension
/// field coefficient vectors. It uses the Method of Four Russians optimization to achieve better
/// performance for B128 fields.
///
/// The optimization works by:
/// 1. Processing 4 elements at a time (2^2 chunks) for better cache locality
/// 2. Precomputing a lookup table of 16 partial sums for 4-bit chunks
/// 3. Bit-transpose 4-bit matrix chunks to get lookup indices
/// 4. Using the lookup table to compute dot products via table lookups instead of multiplications
///
/// ## Arguments
///
/// * `mat` - the [`B1`] matrix packed into B128 elements, with 128 columns
/// * `vec` - the row coefficients as B128 elements
///
/// ## Returns
///
/// A buffer containing the linear combination result
///
/// ## Preconditions
///
/// * `mat` and `vec` must have the same log length
pub fn fold_1b_rows_for_b128<P, Data>(
	mat: &FieldBuffer<P, Data>,
	vec: &FieldBuffer<P>,
) -> FieldBuffer<B128>
where
	P: PackedField<Scalar = B128>,
	Data: Deref<Target = [P]>,
{
	let log_scalar_bit_width = <B128 as ExtensionField<B1>>::LOG_DEGREE;
	assert_eq!(mat.log_len(), vec.log_len()); // precondition

	(vec.as_ref().par_chunks(FOLD_CHUNK_BITS), mat.as_ref().par_chunks(FOLD_CHUNK_BITS))
		.into_par_iter()
		.fold(
			|| FieldBuffer::zeros(log_scalar_bit_width),
			|mut acc, (vec_chunk, mat_chunk)| {
				let mut vec_chunk_iter = P::iter_slice(vec_chunk);
				let mut mat_chunk_iter = P::iter_slice(mat_chunk);

				for _ in 0..P::WIDTH {
					// Copy from slices to arrays. This works even when the inputs are less than the
					// chunk size.
					let mut vec_scalars = [B128::ZERO; FOLD_CHUNK_BITS];
					iter::zip(&mut vec_scalars, &mut vec_chunk_iter)
						.for_each(|(dst, src)| *dst = src);

					let mut mat_scalars = [B128::ZERO; FOLD_CHUNK_BITS];
					iter::zip(&mut mat_scalars, &mut mat_chunk_iter)
						.for_each(|(dst, src)| *dst = src);

					// Build the lookup table of subset sums of the vector chunk elements.
					let lookup =
						expand_subset_sums_array::<_, FOLD_CHUNK_BITS, { 1 << FOLD_CHUNK_BITS }>(
							vec_scalars,
						);

					square_transpose_const_size::<_, LOG_FOLD_CHUNK_BITS, FOLD_CHUNK_BITS>(
						mat_scalars.each_mut().map(cast_base_mut::<B1, _>),
					);

					{
						let acc = acc.as_mut();
						for (j, mat_elem) in mat_scalars.iter().enumerate() {
							let elem_bytes = u128::from(mat_elem.val()).to_le_bytes();
							for (i, &byte) in elem_bytes.iter().enumerate() {
								acc[(i << 3) | j] += lookup[byte as usize & 0x0F];
								acc[(i << 3) | (1 << 2) | j] += lookup[byte as usize >> 4];
							}
						}
					}
				}

				acc
			},
		)
		.reduce(
			|| FieldBuffer::zeros(log_scalar_bit_width),
			|mut lhs, rhs| {
				for (lhs_i, &rhs_i) in izip!(lhs.as_mut(), rhs.as_ref()) {
					*lhs_i += rhs_i;
				}
				lhs
			},
		)
}

/// [`fold_1b_rows_for_b128`] over a tensor given as two factors, never materialized in full.
///
/// # Overview
///
/// The equality tensor over `n_lo + n_hi` coordinates factors:
///
/// ```text
///     tensor[hi << n_lo | lo] = eq_lo[lo] * eq_hi[hi]
/// ```
///
/// The fold splits along that structure.
/// Each hi-block of the matrix folds against the low factor into a 128-entry block result.
/// The block result is scaled once by the high factor's entry and merged.
/// Field addition and multiplication are exact.
/// So the output is bit-identical to folding against the materialized tensor.
///
/// # Why this beats the dense fold
///
/// - The `2^(n_lo + n_hi)` tensor is never written or read; only the two small factors are.
/// - The nibble subset-sum tables cover only the low factor. They are built once and stay
///   cache-resident across every hi-block. The dense fold rebuilds an equivalent table per 4 rows
///   instead.
/// - The price is 128 multiplies per hi-block, amortized to well under one per row.
///
/// ## Preconditions
///
/// * `mat.log_len()` must equal `eq_lo.log_len() + eq_hi.log_len()`
/// * `eq_lo.log_len()` must be at least `max(2, P::LOG_WIDTH)`
pub fn fold_1b_rows_for_b128_split<P, Data>(
	mat: &FieldBuffer<P, Data>,
	eq_lo: &FieldBuffer<B128>,
	eq_hi: &FieldBuffer<B128>,
) -> FieldBuffer<B128>
where
	P: PackedField<Scalar = B128>,
	Data: Deref<Target = [P]>,
{
	let log_scalar_bit_width = <B128 as ExtensionField<B1>>::LOG_DEGREE;
	assert_eq!(mat.log_len(), eq_lo.log_len() + eq_hi.log_len()); // precondition
	assert!(eq_lo.log_len() >= LOG_FOLD_CHUNK_BITS.max(P::LOG_WIDTH)); // precondition

	// Subset-sum tables over the low factor, built once and reused by every hi-block.
	let lo_tables = eq_lo
		.as_ref()
		.chunks(FOLD_CHUNK_BITS)
		.map(|chunk| {
			let chunk = <[B128; FOLD_CHUNK_BITS]>::try_from(chunk)
				.expect("eq_lo.len() is a power of two >= FOLD_CHUNK_BITS");
			expand_subset_sums_array::<_, FOLD_CHUNK_BITS, { 1 << FOLD_CHUNK_BITS }>(chunk)
		})
		.collect::<Vec<_>>();

	let block_packed_len = eq_lo.len() >> P::LOG_WIDTH;

	(mat.as_ref().par_chunks(block_packed_len), eq_hi.as_ref().par_iter())
		.into_par_iter()
		.fold(
			|| FieldBuffer::zeros(log_scalar_bit_width),
			|mut acc, (mat_block, &eq_hi_val)| {
				// Fold this hi-block against the low factor alone.
				//
				//     block_acc[i] = sum_lo eq_lo[lo] * bit_i(mat_block[lo])
				//
				// The 2 KiB block accumulator lives on the stack.
				let mut block_acc = [B128::ZERO; B128::N_BITS];
				let mut mat_iter = P::iter_slice(mat_block);

				for lookup in &lo_tables {
					// Each table covers 4 consecutive rows of the block.
					let mut mat_scalars = [B128::ZERO; FOLD_CHUNK_BITS];
					iter::zip(&mut mat_scalars, &mut mat_iter).for_each(|(dst, src)| *dst = src);

					square_transpose_const_size::<_, LOG_FOLD_CHUNK_BITS, FOLD_CHUNK_BITS>(
						mat_scalars.each_mut().map(cast_base_mut::<B1, _>),
					);

					for (j, mat_elem) in mat_scalars.iter().enumerate() {
						let elem_bytes = u128::from(mat_elem.val()).to_le_bytes();
						for (i, &byte) in elem_bytes.iter().enumerate() {
							block_acc[(i << 3) | j] += lookup[byte as usize & 0x0F];
							block_acc[(i << 3) | (1 << 2) | j] += lookup[byte as usize >> 4];
						}
					}
				}

				// Scale the block result by the high factor's entry: 128 multiplies per block.
				for (acc_i, block_i) in iter::zip(acc.as_mut(), block_acc) {
					*acc_i += eq_hi_val * block_i;
				}
				acc
			},
		)
		.reduce(
			|| FieldBuffer::zeros(log_scalar_bit_width),
			|mut lhs, rhs| {
				for (lhs_i, &rhs_i) in izip!(lhs.as_mut(), rhs.as_ref()) {
					*lhs_i += rhs_i;
				}
				lhs
			},
		)
}

/// Builds the ring-switching equality indicator directly from the tensor's two factors.
///
/// # Overview
///
/// The indicator is the suffix equality tensor with every scalar folded bitwise by the
/// row-batching query.
/// The dense route writes the `2^n` tensor, then reads and rewrites it through the fold.
/// This route produces each entry in one pass:
///
/// ```text
///     out[hi << n_lo | lo] = fold(eq_lo[lo] * eq_hi[hi])
/// ```
///
/// The tensor expansion already costs one multiply per entry.
/// So the fused product changes no operation count.
/// It removes one full read pass and one full write pass over the `2^n` buffer.
/// The output is bit-identical to the dense route.
///
/// ## Preconditions
///
/// * `row_batch_query.len()` must equal 128, the extension degree of B128 over B1
/// * `eq_lo.log_len()` must be at least `P::LOG_WIDTH`
pub fn rs_eq_ind_from_factors<P>(
	eq_lo: &FieldBuffer<B128>,
	eq_hi: &FieldBuffer<B128>,
	row_batch_query: &FieldBuffer<B128>,
) -> FieldBuffer<P>
where
	P: PackedField<Scalar = B128>,
{
	assert!(eq_lo.log_len() >= P::LOG_WIDTH); // precondition

	let lookup_tables = row_batch_lookup_tables(row_batch_query);

	let log_len = eq_lo.log_len() + eq_hi.log_len();
	let packed_len = 1usize << (log_len - P::LOG_WIDTH);
	let block_packed_len = eq_lo.len() >> P::LOG_WIDTH;

	// The buffer is written exactly once, block by block, so it starts uninitialized.
	let mut out = Vec::<P>::with_capacity(packed_len);
	(
		out.spare_capacity_mut()[..packed_len].par_chunks_mut(block_packed_len),
		eq_hi.as_ref().par_iter(),
	)
		.into_par_iter()
		.for_each(|(out_block, &eq_hi_val)| {
			// Each slot holds P::WIDTH consecutive entries of this hi-block.
			// Every entry is one product of the two factors, folded and written once.
			let lo_chunks = eq_lo.as_ref().chunks(P::WIDTH);
			for (slot, lo_chunk) in iter::zip(out_block, lo_chunks) {
				slot.write(P::from_scalars(
					lo_chunk
						.iter()
						.map(|&lo| apply_row_batch_tables(&lookup_tables, lo * eq_hi_val)),
				));
			}
		});
	// SAFETY: the block partition covers all `packed_len` slots and every slot was written.
	unsafe { out.set_len(packed_len) };

	FieldBuffer::new(log_len, out)
}

/// Transpose square blocks of elements within packed field elements in place.
///
/// This is similar to [`binius_field::transpose::square_transpose`] but uses const generic
/// parameters for the array size and block dimension. The const generics enable the compiler
/// to unroll loops and optimize the transpose operation more aggressively.
///
/// ## Type Parameters
///
/// * `P` - The packed field type
/// * `LOG_N` - Base-2 logarithm of the dimension of the square blocks to transpose
/// * `S` - Size of the array (must be a power of 2)
///
/// ## Arguments
///
/// * `elems` - Array of packed field elements to transpose in place
///
/// ## Preconditions
///
/// * `S` must be a power of two
/// * `LOG_N` must be less than or equal to `P::LOG_WIDTH`
/// * `LOG_N` must be less than or equal to `log2(S)`
fn square_transpose_const_size<P: PackedField, const LOG_N: usize, const S: usize>(
	elems: [&mut P; S],
) {
	let log_size = checked_log_2(S);

	assert!(LOG_N <= P::LOG_WIDTH);
	assert!(LOG_N <= log_size);

	let log_w = log_size - LOG_N;

	// See Hacker's Delight, Section 7-3.
	// https://dl.acm.org/doi/10.5555/2462741
	for i in 0..LOG_N {
		for j in 0..1 << (LOG_N - i - 1) {
			for k in 0..1 << (log_w + i) {
				let idx0 = (j << (log_w + i + 1)) | k;
				let idx1 = idx0 | (1 << (log_w + i));

				let v0 = *elems[idx0];
				let v1 = *elems[idx1];
				let (v0, v1) = v0.interleave(v1, i);
				*elems[idx0] = v0;
				*elems[idx1] = v1;
			}
		}
	}
}

/// Output of ring-switching prover.
pub struct RingSwitchOutput<P: PackedField> {
	/// The ring-switching equality indicator MLE (transparent poly for BaseFold).
	pub rs_eq_ind: FieldBuffer<P>,
	/// The sumcheck claim.
	pub sumcheck_claim: P::Scalar,
}

/// Prove the ring-switching reduction.
///
/// Takes the packed witness and evaluation point from shift reduction, and:
/// 1. Computes partial evaluations s_hat_v
/// 2. Sends s_hat_v to verifier via channel
/// 3. Samples row-batching challenges
/// 4. Computes the ring-switching equality indicator and sumcheck claim
///
/// Returns the transparent polynomial and sumcheck claim for BaseFold.
///
/// ## Arguments
///
/// * `packed_witness` - the packed witness buffer (B1 polynomial packed into P elements)
/// * `eval_point` - the evaluation point from shift reduction
/// * `channel` - the prover channel for sending/sampling
///
/// ## Preconditions
///
/// * `packed_witness.log_len() + log_packing == eval_point.len()` where log_packing is the base-2
///   log of the extension degree of B128 over B1 (= 7)
pub fn prove<P, Channel>(
	packed_witness: &FieldBuffer<P>,
	eval_point: &[B128],
	channel: &mut Channel,
) -> RingSwitchOutput<P>
where
	P: PackedField<Scalar = B128>,
	Channel: IPProverChannel<B128>,
{
	let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
	assert_eq!(packed_witness.log_len() + log_packing, eval_point.len());

	// Factor the suffix equality tensor instead of materializing its 2^n entries.
	//
	//     tensor[hi << n_lo | lo] = eq_lo[lo] * eq_hi[hi]
	//
	// Both passes below consume the two small factors directly.
	// The low factor is capped so its fold tables stay cache-resident (see the split fold).
	let eval_point_suffix = &eval_point[log_packing..];
	let split_at = eval_point_suffix
		.len()
		.min(LOG_SPLIT_BLOCK)
		.max(LOG_FOLD_CHUNK_BITS.max(P::LOG_WIDTH));

	if eval_point_suffix.len() < split_at {
		// The suffix is too short to split; take the dense route over the tiny tensor.
		return prove_dense(packed_witness, eval_point_suffix, channel);
	}
	let (suffix_lo, suffix_hi) = eval_point_suffix.split_at(split_at);
	let (eq_lo, eq_hi) = tracing::debug_span!("Expand evaluation suffix factors").in_scope(|| {
		(eq_ind_partial_eval::<B128>(suffix_lo), eq_ind_partial_eval::<B128>(suffix_hi))
	});

	// Ring-switching partial evaluations (Method of Four Russians)
	let s_hat_v = tracing::debug_span!("Compute ring-switching partial evaluations")
		.in_scope(|| fold_1b_rows_for_b128_split(packed_witness, &eq_lo, &eq_hi));
	channel.send_many(s_hat_v.as_ref());

	// Basis transpose
	let s_hat_u = TensorAlgebra::<B1, B128>::new(s_hat_v.as_ref().to_vec())
		.transpose()
		.elems;

	// Sample row-batching challenges
	let r_double_prime = channel.sample_many(log_packing);
	let eq_r_double_prime = eq_ind_partial_eval::<B128>(&r_double_prime);

	// Compute sumcheck claim
	let sumcheck_claim = inner_product(s_hat_u, eq_r_double_prime.as_ref().iter().copied());

	// Compute ring-switching equality indicator (transparent poly)
	let rs_eq_ind = tracing::debug_span!("Compute ring-switching equality indicator")
		.in_scope(|| rs_eq_ind_from_factors::<P>(&eq_lo, &eq_hi, &eq_r_double_prime));

	RingSwitchOutput {
		rs_eq_ind,
		sumcheck_claim,
	}
}

/// [`prove`] over the materialized suffix tensor, for suffixes too short to split.
///
/// Produces the same transcript messages and outputs as the factored route.
/// The factored route needs a low factor covering one packed element and one lookup nibble.
/// Suffixes below that floor take this dense route instead.
fn prove_dense<P, Channel>(
	packed_witness: &FieldBuffer<P>,
	eval_point_suffix: &[B128],
	channel: &mut Channel,
) -> RingSwitchOutput<P>
where
	P: PackedField<Scalar = B128>,
	Channel: IPProverChannel<B128>,
{
	let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;

	// Expand evaluation suffix with eq_ind
	let suffix_tensor = eq_ind_partial_eval::<P>(eval_point_suffix);

	// Ring-switching partial evaluations (Method of Four Russians)
	let s_hat_v = fold_1b_rows_for_b128(packed_witness, &suffix_tensor);
	channel.send_many(s_hat_v.as_ref());

	// Basis transpose
	let s_hat_u = TensorAlgebra::<B1, B128>::new(s_hat_v.as_ref().to_vec())
		.transpose()
		.elems;

	// Sample row-batching challenges
	let r_double_prime = channel.sample_many(log_packing);
	let eq_r_double_prime = eq_ind_partial_eval::<B128>(&r_double_prime);

	// Compute sumcheck claim
	let sumcheck_claim = inner_product(s_hat_u, eq_r_double_prime.as_ref().iter().copied());

	// Compute ring-switching equality indicator (transparent poly)
	let rs_eq_ind = fold_b128_elems_inplace(suffix_tensor, &eq_r_double_prime);

	RingSwitchOutput {
		rs_eq_ind,
		sumcheck_claim,
	}
}

#[cfg(test)]
mod test {
	use binius_field::{
		BinaryField128bGhash, ExtensionField, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b,
		PackedField, PackedSubfield, cast_ext,
	};
	use binius_math::{
		FieldBuffer,
		inner_product::{inner_product_buffers, inner_product_subfield},
		multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate_inplace},
		test_utils::{index_to_hypercube_point, random_field_buffer, random_scalars},
	};
	use binius_verifier::{config::B1, ring_switch::eval_rs_eq};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	type F = BinaryField128bGhash;

	// The split fold must reproduce the dense fold bit for bit.
	//
	//     dense:  fold(mat, eq(full suffix))
	//     split:  fold(mat, eq(lo), eq(hi)) with the tensor never materialized
	//
	// Field addition and multiplication are exact, so any split position must agree.
	fn check_split_fold_matches_dense<P: PackedField<Scalar = F>>(log_len: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let mat = random_field_buffer::<P>(&mut rng, log_len);
		let suffix: Vec<F> = random_scalars(&mut rng, log_len);

		let full_tensor = eq_ind_partial_eval::<P>(&suffix);
		let dense = fold_1b_rows_for_b128(&mat, &full_tensor);

		// Sweep every legal split position, including an empty high factor.
		for split_at in LOG_FOLD_CHUNK_BITS.max(P::LOG_WIDTH)..=log_len {
			let (suffix_lo, suffix_hi) = suffix.split_at(split_at);
			let eq_lo = eq_ind_partial_eval::<F>(suffix_lo);
			let eq_hi = eq_ind_partial_eval::<F>(suffix_hi);

			let split = fold_1b_rows_for_b128_split(&mat, &eq_lo, &eq_hi);
			assert_eq!(split.as_ref(), dense.as_ref(), "split_at={split_at}");
		}
	}

	#[test]
	fn test_split_fold_matches_dense() {
		check_split_fold_matches_dense::<F>(6, 0);
		check_split_fold_matches_dense::<PackedBinaryGhash2x128b>(7, 1);
		check_split_fold_matches_dense::<PackedBinaryGhash4x128b>(8, 2);
	}

	// The factored indicator must reproduce the dense expand-then-fold route bit for bit.
	//
	//     dense:    fold_elems(eq(full suffix), q)     two passes over 2^n entries
	//     factored: fold(eq_lo[lo] * eq_hi[hi], q)     one pass, tensor never materialized
	fn check_rs_eq_ind_from_factors_matches_dense<P: PackedField<Scalar = F>>(
		log_len: usize,
		seed: u64,
	) {
		let mut rng = StdRng::seed_from_u64(seed);
		let suffix: Vec<F> = random_scalars(&mut rng, log_len);
		let row_batching_challenges: Vec<F> =
			random_scalars(&mut rng, <F as ExtensionField<B1>>::LOG_DEGREE);
		let row_batch_query = eq_ind_partial_eval::<F>(&row_batching_challenges);

		let dense = fold_b128_elems_inplace(eq_ind_partial_eval::<P>(&suffix), &row_batch_query);

		for split_at in P::LOG_WIDTH..=log_len {
			let (suffix_lo, suffix_hi) = suffix.split_at(split_at);
			let eq_lo = eq_ind_partial_eval::<F>(suffix_lo);
			let eq_hi = eq_ind_partial_eval::<F>(suffix_hi);

			let factored = rs_eq_ind_from_factors::<P>(&eq_lo, &eq_hi, &row_batch_query);
			assert_eq!(factored.as_ref(), dense.as_ref(), "split_at={split_at}");
		}
	}

	#[test]
	fn test_rs_eq_ind_from_factors_matches_dense() {
		check_rs_eq_ind_from_factors_matches_dense::<F>(6, 3);
		check_rs_eq_ind_from_factors_matches_dense::<PackedBinaryGhash2x128b>(7, 4);
		check_rs_eq_ind_from_factors_matches_dense::<PackedBinaryGhash4x128b>(8, 5);
	}

	#[test]
	fn test_consistent_with_tensor_alg() {
		let mut rng = StdRng::from_seed([0; 32]);

		let n_vars_big_field = 3;

		let z_vals: Vec<F> = random_scalars(&mut rng, n_vars_big_field);

		let row_batching_challenges: Vec<F> =
			random_scalars(&mut rng, <F as ExtensionField<B1>>::LOG_DEGREE);

		let row_batching_expanded_query = eq_ind_partial_eval(&row_batching_challenges);

		// Build the indicator the way the prover does: fold the tensor-expanded z_vals point by
		// the tensor-expanded row-batching query.
		let rs_eq =
			fold_elems_inplace(eq_ind_partial_eval::<F>(&z_vals), &row_batching_expanded_query);

		// test all points points in the boolean hypercube
		for hypercube_point in 0..1 << 3 {
			let evaluated_at_pt = eval_rs_eq::<F>(
				&z_vals,
				&index_to_hypercube_point::<F>(3, hypercube_point),
				row_batching_expanded_query.as_ref(),
			);

			assert_eq!(rs_eq.get(hypercube_point), evaluated_at_pt);
		}
	}

	#[test]
	fn test_out_of_range_evaluation() {
		let mut rng = StdRng::from_seed([0; 32]);

		let n_vars_big_field = 3;

		// setup ring switch eq mle
		let z_vals: Vec<F> = random_scalars(&mut rng, n_vars_big_field);

		let row_batching_challenges: Vec<F> =
			random_scalars(&mut rng, <F as ExtensionField<B1>>::LOG_DEGREE);

		let row_batching_expanded_query: FieldBuffer<F> =
			eq_ind_partial_eval(&row_batching_challenges);

		let rs_eq =
			fold_elems_inplace(eq_ind_partial_eval::<F>(&z_vals), &row_batching_expanded_query);

		// out of range eval point
		let eval_point: Vec<F> = random_scalars(&mut rng, n_vars_big_field);

		// compare eval against inner product w/ eq ind mle of eval point

		let tensor_expanded_eval_point = eq_ind_partial_eval::<F>(&eval_point);
		let expected_eval = inner_product_buffers(&rs_eq, &tensor_expanded_eval_point);

		let actual_eval =
			eval_rs_eq::<F>(&z_vals, &eval_point, row_batching_expanded_query.as_ref());

		assert_eq!(expected_eval, actual_eval);
	}

	#[test]
	fn test_fold_tensor_product() {
		let mut rng = StdRng::seed_from_u64(0);

		type P = PackedBinaryGhash2x128b;

		// Parameters
		let n = 10;
		let log_degree = <F as ExtensionField<B1>>::LOG_DEGREE;

		// Generate a random B1 bit matrix with 2^(n + log_degree) bits
		let bit_matrix = random_field_buffer::<PackedSubfield<P, B1>>(&mut rng, n + log_degree);

		// Generate a random evaluation point with n + log_degree coordinates
		let eval_point: Vec<F> = random_scalars(&mut rng, n + log_degree);

		// Split the evaluation point into prefix and suffix
		let (prefix, suffix) = eval_point.split_at(log_degree);

		// Method 1 (Reference): Tensor expand the full challenge and compute inner product
		let full_tensor = eq_ind_partial_eval::<F>(&eval_point);
		let reference_result = inner_product_subfield(
			PackedField::iter_slice(bit_matrix.as_ref()),
			PackedField::iter_slice(full_tensor.as_ref()),
		);

		// Method 2: Tensor expand prefix, call fold_elems_inplace, then evaluate_inplace on suffix
		let prefix_tensor = eq_ind_partial_eval::<F>(prefix);
		let bit_matrix_packed = FieldBuffer::new(
			n,
			bit_matrix
				.as_ref()
				.iter()
				.map(|&bits_packed| cast_ext::<B1, P>(bits_packed))
				.collect(),
		);
		let folded_method2 = fold_elems_inplace(bit_matrix_packed, &prefix_tensor);
		let method2_result = evaluate_inplace(folded_method2, suffix);

		// Compare all three results
		assert_eq!(reference_result, method2_result, "Method 2 does not match reference");
	}

	#[test]
	fn test_fold_b128_elems_consistency() {
		let mut rng = StdRng::seed_from_u64(0);
		type P = PackedBinaryGhash4x128b;

		// Parameters - test with various sizes
		for n in [4, 6, 8, 10] {
			// Generate a random buffer of B128 elements
			let elems = random_field_buffer::<P>(&mut rng, n);

			// Generate a random vector of 128 B128 field elements (for the inner product)
			let vec =
				random_field_buffer::<B128>(&mut rng, <B128 as ExtensionField<B1>>::LOG_DEGREE);

			// Call the generic fold_elems_inplace function
			let result_generic = fold_elems_inplace(elems.clone(), &vec);

			// Call the specialized fold_b128_elems_inplace function
			let result_specialized = fold_b128_elems_inplace(elems.clone(), &vec);

			// Both results should be identical
			assert_eq!(
				result_generic.as_ref(),
				result_specialized.as_ref(),
				"fold_b128_elems_inplace does not match fold_elems_inplace for n = {}",
				n
			);
		}
	}
}
