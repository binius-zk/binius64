// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	iter,
	ops::{Deref, DerefMut},
};

use binius_compute::{Allocator, VecLike};
use binius_field::{
	BinaryField, Divisible, ExtensionField, Field, PackedBinaryField128x1b, PackedField, cast_base,
	cast_bases_mut,
	linear_transformation::{
		BytewiseLookupTransformationFactory, InputWrappingTransformationFactory,
		LinearTransformationFactory, OutputWrappingTransformationFactory, Transformation,
	},
	util::expand_subset_sums_array,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec,
	inner_product::inner_product,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_in},
	tensor_algebra::TensorAlgebra,
};
use binius_utils::rayon::prelude::*;
use binius_verifier::config::{B1, B128};
use itertools::izip;

use crate::fold_word::{fold_row_group, row_fold_tables, square_transpose_const_size};

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
pub fn fold_elems_inplace<F, P, Data>(
	mut elems: FieldBuffer<P, Data>,
	vec: &FieldBuffer<F>,
) -> FieldBuffer<P, Data>
where
	F: BinaryField,
	F::Underlier: Divisible<u8>,
	P: PackedField<Scalar = F>,
	Data: DerefMut<Target = [P]>,
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

/// Base-2 log of the row group one subset-sum table covers.
///
/// Eight rows is the widest group whose lookup index still fits one byte.
/// One table load then replaces eight conditional additions.
const LOG_SPLIT_CHUNK_BITS: usize = 3;

/// Base-2 log of the low-factor length the tensor split targets.
///
/// The row fold consumes a chunk of as many rows as a row has columns.
/// So the low factor spans exactly that many rows.
/// For 128 columns that is 16 tables of 256 field elements, or 64 KiB.
/// That footprint stays resident in a performance core's L1 data cache across every hi-block.
/// A wider factor needs proportionally more tables and starts missing that cache.
pub const LOG_SPLIT_BLOCK: usize = <B128 as ExtensionField<B1>>::LOG_DEGREE;

/// Number of subset-sum tables one chunk of the split fold uses.
const N_ROW_TABLES: usize = 1 << (LOG_SPLIT_BLOCK - LOG_SPLIT_CHUNK_BITS);
/// Rows one subset-sum table covers.
const ROW_GROUP: usize = 1 << LOG_SPLIT_CHUNK_BITS;

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
pub fn fold_1b_rows_for_b128<P, MatData, VecData>(
	mat: &FieldBuffer<P, MatData>,
	vec: &FieldBuffer<P, VecData>,
) -> FieldBuffer<B128>
where
	P: PackedField<Scalar = B128>,
	MatData: Deref<Target = [P]>,
	VecData: Deref<Target = [P]>,
{
	let log_scalar_bit_width = <B128 as ExtensionField<B1>>::LOG_DEGREE;
	assert_eq!(mat.log_len(), vec.log_len()); // precondition

	// Group bits into 4-bit nibbles for the lookups.
	// This fold rebuilds its table for every group, so a wider group would cost more to build than
	// the lookups it saves.
	const LOG_CHUNK_BITS: usize = 2;
	const CHUNK_BITS: usize = 1 << LOG_CHUNK_BITS;

	(vec.as_ref().par_chunks(CHUNK_BITS), mat.as_ref().par_chunks(CHUNK_BITS))
		.into_par_iter()
		.fold(
			|| FieldBuffer::zeros(log_scalar_bit_width),
			|mut acc, (vec_chunk, mat_chunk)| {
				let mut vec_chunk_iter = P::iter_slice(vec_chunk);
				let mut mat_chunk_iter = P::iter_slice(mat_chunk);

				for _ in 0..P::WIDTH {
					// Copy from slices to arrays. This works even when the inputs are less than the
					// chunk size.
					let mut vec_scalars = [B128::ZERO; CHUNK_BITS];
					iter::zip(&mut vec_scalars, &mut vec_chunk_iter)
						.for_each(|(dst, src)| *dst = src);

					let mut mat_scalars = [B128::ZERO; CHUNK_BITS];
					iter::zip(&mut mat_scalars, &mut mat_chunk_iter)
						.for_each(|(dst, src)| *dst = src);

					// Build the lookup table of subset sums of the vector chunk elements.
					let lookup =
						expand_subset_sums_array::<_, CHUNK_BITS, { 1 << CHUNK_BITS }>(vec_scalars);

					// Viewing each element as its 128 single-bit components is a reinterpretation,
					// so the transpose runs on the same memory.
					let mat_bits = cast_bases_mut::<B1, _>(&mut mat_scalars);
					square_transpose_const_size::<_, LOG_CHUNK_BITS, CHUNK_BITS>(
						mat_bits
							.try_into()
							.expect("cast preserves the array length"),
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

/// Folds the 1-bit rows of a matrix against an equality tensor supplied as two factors.
///
/// # Overview
///
/// The equality tensor over `n_lo + n_hi` coordinates factors along any coordinate split:
///
/// ```text
///     tensor[hi << n_lo | lo] = eq_lo[lo] * eq_hi[hi]
/// ```
///
/// Multiplication distributes over the sum, so the fold regroups exactly:
///
/// ```text
///     sum_x tensor[x] * row[x]
///       = sum_hi eq_hi[hi] * ( sum_lo eq_lo[lo] * row[hi, lo] )
/// ```
///
/// Each block of the matrix folds against the low factor alone.
/// Its result is then scaled once by that block's high-factor entry and merged.
/// Field addition and multiplication are exact.
/// So the result is bit-identical to folding against the materialized tensor.
///
/// # Why the factored form is faster
///
/// Both effects follow from the tables covering only the low factor, so being built once.
///
/// Memory traffic:
///
/// - The full tensor is never written or read, so only the matrix streams through.
/// - Folding against a materialized tensor reads one tensor entry per row alongside it.
/// - That doubles the stream for the same arithmetic.
///
/// Lookup count:
///
/// - A table built once costs nothing per row, so it can cover eight rows instead of four.
/// - Eight-row groups halve the lookups and accumulator updates.
/// - Those updates are what the fold is bound on, not arithmetic or bandwidth.
/// - A table rebuilt per group cannot widen: 256 entries per group costs more than it saves.
///
/// The price is one multiply per row, to scale each block by its high-factor entry.
///
/// # Preconditions
///
/// * The matrix must have as many rows as the two factors have entries together.
/// * The low factor must span at least one packed element, so blocks align to packed boundaries.
/// * The low factor must span at most one row fold chunk, which is 128 rows.
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
	assert!(eq_lo.log_len() >= P::LOG_WIDTH); // precondition
	assert!(eq_lo.log_len() <= LOG_SPLIT_BLOCK); // precondition

	// One subset-sum table per group of eight rows, built once and reused by every hi-block.
	// A low factor shorter than a full block leaves the trailing tables zero.
	// Those pair with rows past the end of the block, which read as zero, so nothing is added.
	let lo_tables = row_fold_tables::<B128, N_ROW_TABLES>(eq_lo.as_ref());

	let block_packed_len = eq_lo.len() >> P::LOG_WIDTH;

	(mat.as_ref().par_chunks(block_packed_len), eq_hi.as_ref().par_iter())
		.into_par_iter()
		.fold(
			|| FieldBuffer::zeros(log_scalar_bit_width),
			|mut acc, (mat_block, &eq_hi_val)| {
				// Fold this block against the low factor alone:
				//
				//     block[col] = sum_lo eq_lo[lo] * bit_col(mat_block[lo])
				//
				// A matrix row is 128 single-bit columns, which is one full-width packed row.
				let mut rows = P::iter_slice(mat_block);
				let mut columns = [[B128::ZERO; ROW_GROUP]; N_ROW_TABLES];

				for table in &lo_tables {
					// Gather this group's rows out of the packed elements they sit in.
					// Rows past the end of the block stay zero and contribute nothing.
					// A field element and a 128-bit row of single-bit scalars share one underlier,
					// so each view is free.
					let mut group = [PackedBinaryField128x1b::default(); ROW_GROUP];
					iter::zip(&mut group, &mut rows)
						.for_each(|(dst, src)| *dst = cast_base::<B1, _>(src));

					fold_row_group(&group, table, &mut columns);
				}

				// Scale by this block's high-factor entry and merge, unpacking the nesting into
				// bit-position order.
				// That is 128 multiplies per block, one per row.
				{
					let acc = acc.as_mut();
					for (i, group) in columns.iter().enumerate() {
						for (j, &column) in group.iter().enumerate() {
							acc[(i << LOG_SPLIT_CHUNK_BITS) | j] += eq_hi_val * column;
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

/// Builds the ring-switching equality indicator directly from the tensor's two factors.
///
/// # Overview
///
/// The indicator is the suffix equality tensor, folded bitwise by the row-batching query.
/// The dense route writes the `2^n` tensor, then [`fold_elems_inplace`] reads and rewrites it.
/// This route produces each entry in one pass, through the same bitwise fold:
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
/// ## Arguments
///
/// * `alloc` - the allocator the returned indicator is drawn from
/// * `eq_lo` - the low tensor factor
/// * `eq_hi` - the high tensor factor
/// * `row_batch_query` - the vector every entry is folded bitwise by
///
/// ## Preconditions
///
/// * `row_batch_query.len()` must equal 128, the extension degree of B128 over B1
/// * `eq_lo.log_len()` must be at least `P::LOG_WIDTH`
pub fn rs_eq_ind_from_factors<A, P>(
	alloc: &A,
	eq_lo: &FieldBuffer<B128>,
	eq_hi: &FieldBuffer<B128>,
	row_batch_query: &FieldBuffer<B128>,
) -> FieldVec<P, A>
where
	A: Allocator,
	P: PackedField<Scalar = B128>,
{
	assert!(eq_lo.log_len() >= P::LOG_WIDTH); // precondition
	assert_eq!(row_batch_query.log_len(), <B128 as ExtensionField<B1>>::LOG_DEGREE); // precondition

	// The same bitwise fold [`fold_elems_inplace`] applies, built once and shared by every block.
	let transform = OutputWrappingTransformationFactory::new(
		InputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory),
	)
	.create(row_batch_query.as_ref());

	let log_len = eq_lo.log_len() + eq_hi.log_len();
	let packed_len = 1usize << (log_len - P::LOG_WIDTH);
	let block_packed_len = eq_lo.len() >> P::LOG_WIDTH;

	// The buffer is written exactly once, block by block, so it starts uninitialized.
	let mut out = alloc.alloc::<P>(packed_len);
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
						.map(|&lo| transform.transform(&(lo * eq_hi_val))),
				));
			}
		});
	// SAFETY: the block partition covers all `packed_len` slots and every slot was written.
	unsafe { out.set_len(packed_len) };

	FieldBuffer::new(log_len, out)
}

/// The suffix equality tensor, held in whichever form the fold and the indicator can consume.
///
/// Both consumers prefer the two factors: neither then reads or writes the `2^n` expansion.
/// The factored form needs a low factor spanning one lookup group and one packed element.
/// A suffix shorter than that floor cannot supply one, so it is expanded in full instead.
enum SuffixTensor<A: Allocator, P: PackedField> {
	/// The low and high factors of the tensor, in that order.
	Factored(FieldBuffer<B128>, FieldBuffer<B128>),
	/// The full `2^n`-entry expansion.
	Dense(FieldVec<P, A>),
}

impl<A: Allocator, P: PackedField<Scalar = B128>> SuffixTensor<A, P> {
	/// Expands `point` into the form the fold and the indicator will consume.
	///
	/// The low factor spans one row fold chunk.
	/// It is floored at one packed element, so every block covers a whole number of them.
	/// A point too short to reach that floor cannot be split, so it is expanded in full.
	fn expand(alloc: &A, point: &[B128]) -> Self {
		if point.len() < P::LOG_WIDTH {
			return Self::Dense(eq_ind_partial_eval_in::<A, P>(alloc, point));
		}

		let split_at = point.len().min(LOG_SPLIT_BLOCK).max(P::LOG_WIDTH);
		let (point_lo, point_hi) = point.split_at(split_at);
		Self::Factored(eq_ind_partial_eval::<B128>(point_lo), eq_ind_partial_eval::<B128>(point_hi))
	}
}

/// Output of ring-switching prover.
pub struct RingSwitchOutput<A: Allocator, P: PackedField> {
	/// The ring-switching equality indicator MLE (transparent poly for BaseFold).
	pub rs_eq_ind: FieldVec<P, A>,
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
/// * `alloc` - the allocator the ring-switching equality indicator is drawn from
/// * `packed_witness` - the packed witness buffer (B1 polynomial packed into P elements)
/// * `eval_point` - the evaluation point from shift reduction
/// * `channel` - the prover channel for sending/sampling
///
/// ## Preconditions
///
/// * `packed_witness.log_len() + log_packing == eval_point.len()` where log_packing is the base-2
///   log of the extension degree of B128 over B1 (= 7)
pub fn prove<A, P, Channel>(
	alloc: &A,
	packed_witness: FieldSlice<P>,
	eval_point: &[B128],
	channel: &mut Channel,
) -> RingSwitchOutput<A, P>
where
	A: Allocator,
	P: PackedField<Scalar = B128>,
	Channel: IPProverChannel<B128>,
{
	let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
	assert_eq!(packed_witness.log_len() + log_packing, eval_point.len());

	let eval_point_suffix = &eval_point[log_packing..];
	let suffix_tensor = tracing::debug_span!("Expand evaluation suffix query")
		.in_scope(|| SuffixTensor::<A, P>::expand(alloc, eval_point_suffix));

	// Ring-switching partial evaluations (Method of Four Russians)
	let s_hat_v =
		tracing::debug_span!("Compute ring-switching partial evaluations").in_scope(|| {
			match &suffix_tensor {
				SuffixTensor::Factored(eq_lo, eq_hi) => {
					fold_1b_rows_for_b128_split(&packed_witness, eq_lo, eq_hi)
				}
				SuffixTensor::Dense(tensor) => fold_1b_rows_for_b128(&packed_witness, tensor),
			}
		});
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
	let rs_eq_ind =
		tracing::debug_span!("Compute ring-switching equality indicator").in_scope(|| {
			match suffix_tensor {
				SuffixTensor::Factored(eq_lo, eq_hi) => {
					rs_eq_ind_from_factors::<A, P>(alloc, &eq_lo, &eq_hi, &eq_r_double_prime)
				}
				SuffixTensor::Dense(tensor) => fold_elems_inplace(tensor, &eq_r_double_prime),
			}
		});

	RingSwitchOutput {
		rs_eq_ind,
		sumcheck_claim,
	}
}

#[cfg(test)]
mod test {
	use binius_compute::GlobalAllocator;
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

		// Sweep every legal low-factor width.
		// The floor is one packed element, so every block covers a whole number of them.
		// The ceiling is one row's worth of rows, which is the chunk the row fold consumes.
		for split_at in P::LOG_WIDTH..=log_len.min(LOG_SPLIT_BLOCK) {
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

		let dense = fold_elems_inplace(eq_ind_partial_eval::<P>(&suffix), &row_batch_query);

		for split_at in P::LOG_WIDTH..=log_len {
			let (suffix_lo, suffix_hi) = suffix.split_at(split_at);
			let eq_lo = eq_ind_partial_eval::<F>(suffix_lo);
			let eq_hi = eq_ind_partial_eval::<F>(suffix_hi);

			let factored =
				rs_eq_ind_from_factors::<_, P>(&GlobalAllocator, &eq_lo, &eq_hi, &row_batch_query);
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
		let bit_matrix_packed = FieldBuffer::<P>::new(
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
}
