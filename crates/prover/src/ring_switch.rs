// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, ops::Deref};

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::{
	ExtensionField, Field, PackedBinaryField128x1b, PackedField, cast_base,
	linear_transformation::{
		BytewiseLookupTransformationFactory, InputWrappingTransformationFactory,
		LinearTransformationFactory, OutputWrappingTransformationFactory, Transformation,
	},
};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{bivariate_product_prover, prove_single},
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec, inner_product::inner_product_packed,
	multilinear::eq::eq_ind_partial_eval, tensor_algebra::TensorAlgebra,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use binius_verifier::{
	config::{B1, B128, LOG_WORDS_PER_ELEM},
	protocols::shift::evaluate_words_mle,
};
use itertools::izip;

use crate::{
	fold_word::{fold_row_group, row_fold_tables},
	prove::pack_witness,
};

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
/// * A block must cover a whole number of packed elements, so the chunking below aligns. The
///   exception is a matrix that fits inside one packed element, which is a single block.
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
	assert!(eq_lo.log_len() >= P::LOG_WIDTH || eq_hi.log_len() == 0); // precondition
	assert!(eq_lo.log_len() <= LOG_SPLIT_BLOCK); // precondition

	// One subset-sum table per group of eight rows, built once and reused by every hi-block.
	// A low factor shorter than a full block leaves the trailing weights zero.
	// So every row past its end is weighted zero, whatever bits that row holds:
	// rows past the end of a block read as zero, and a matrix narrower than one packed
	// element leaves lanes that are not rows at all.
	let lo_tables = row_fold_tables::<B128, N_ROW_TABLES>(eq_lo.as_ref());

	// A sub-width low factor still occupies its one packed element, so the block is that element.
	let block_packed_len = 1 << eq_lo.log_len().saturating_sub(P::LOG_WIDTH);

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
					// Rows past the end of the block stay zero, and lanes past the last row
					// carry weight zero, so neither contributes.
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
		// A merge seeded with a partial that already exists never touches a buffer of zeros.
		// An identity would allocate and zero one accumulator per merge, then add all of it.
		.reduce_with(|mut lhs, rhs| {
			for (lhs_i, &rhs_i) in izip!(lhs.as_mut(), rhs.as_ref()) {
				*lhs_i += rhs_i;
			}
			lhs
		})
		// An empty matrix yields no partials at all, and folds to zero.
		.unwrap_or_else(|| FieldBuffer::zeros(log_scalar_bit_width))
}

/// Builds the ring-switching equality indicator directly from the tensor's two factors.
///
/// # Overview
///
/// The indicator is the suffix equality tensor, folded bitwise by the row-batching query.
/// Expanding that tensor and then folding it would read and rewrite a whole `2^n` buffer.
/// This route produces each entry in one pass instead:
///
/// ```text
///     out[hi << n_lo | lo] = fold(eq_lo[lo] * eq_hi[hi])
/// ```
///
/// The tensor expansion already costs one multiply per entry.
/// So the fused product changes no operation count.
/// It removes one full read pass and one full write pass over the `2^n` buffer.
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
/// * A block must cover a whole number of packed elements, so the chunking below aligns. The
///   exception is an output that fits inside one packed element, which is a single block.
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
	assert!(eq_lo.log_len() >= P::LOG_WIDTH || eq_hi.log_len() == 0); // precondition
	assert_eq!(row_batch_query.log_len(), <B128 as ExtensionField<B1>>::LOG_DEGREE); // precondition

	// The bitwise fold that maps a tensor entry to its indicator entry, shared by every block.
	let transform = OutputWrappingTransformationFactory::new(
		InputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory),
	)
	.create(row_batch_query.as_ref());

	let log_len = eq_lo.log_len() + eq_hi.log_len();
	let packed_len = 1usize << log_len.saturating_sub(P::LOG_WIDTH);
	let block_packed_len = 1 << eq_lo.log_len().saturating_sub(P::LOG_WIDTH);

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
			// A sub-width output leaves one short chunk, whose lanes past the last entry are
			// filled with zeros rather than left undefined.
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

/// Expands `point` into the two factors of its equality tensor, low factor first.
///
/// The `2^n` tensor itself is never materialized: both consumers read the factors directly.
/// The low factor spans one row fold chunk, or the whole point when that is shorter.
fn expand_tensor_factors(point: &[B128]) -> (FieldBuffer<B128>, FieldBuffer<B128>) {
	let (point_lo, point_hi) = point.split_at(point.len().min(LOG_SPLIT_BLOCK));
	(eq_ind_partial_eval::<B128>(point_lo), eq_ind_partial_eval::<B128>(point_hi))
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
	let (eq_lo, eq_hi) = tracing::debug_span!("Expand evaluation suffix query")
		.in_scope(|| expand_tensor_factors(eval_point_suffix));

	// Ring-switching partial evaluations (Method of Four Russians)
	let s_hat_v = tracing::debug_span!("Compute ring-switching partial evaluations")
		.in_scope(|| fold_1b_rows_for_b128_split(&packed_witness, &eq_lo, &eq_hi));
	channel.send_many(s_hat_v.as_ref());

	// Basis transpose
	let s_hat_u = TensorAlgebra::<B1, B128>::new(s_hat_v.as_ref().to_vec())
		.transpose()
		.elems;

	// Sample row-batching challenges
	let r_double_prime = channel.sample_many(log_packing);
	let eq_r_double_prime = eq_ind_partial_eval::<B128>(&r_double_prime);

	// GF(2^128) reduction is F2-linear, so it commutes with XOR.
	// Summing 128 wide products then reducing once matches reducing each term first.
	let sumcheck_claim = inner_product_packed::<B128, B128>(
		log_packing,
		s_hat_u.into_iter(),
		eq_r_double_prime.as_ref().iter().copied(),
	);

	// Compute ring-switching equality indicator (transparent poly)
	let rs_eq_ind = tracing::debug_span!("Compute ring-switching equality indicator")
		.in_scope(|| rs_eq_ind_from_factors::<A, P>(alloc, &eq_lo, &eq_hi, &eq_r_double_prime));

	RingSwitchOutput {
		rs_eq_ind,
		sumcheck_claim,
	}
}

/// Proves the public segment's evaluation claim.
///
/// The shift closes over the public segment as a bit matrix, at `r_j` over the bit within a word
/// and the low coordinates of `r_y` over the word index. The verifier holds the segment but not
/// its bits, so the claim is stated here and reduced in two steps:
///
/// 1. a ring-switch onto the segment's packed form, leaving the claim `sum_x P(x) A(x)` against the
///    ring-switching indicator;
/// 2. a sumcheck over the packed segment's own variables, leaving one evaluation of each factor.
///
/// Nothing here is committed, so the verifier finishes on its own: it evaluates the packed
/// segment's multilinear from the words it holds, and the indicator from its succinct formula.
///
/// ## Arguments
///
/// * `alloc` - the allocator the packed segment and the indicator are drawn from
/// * `public_words` - the public segment, unpadded
/// * `r_j` - the bit-index challenges
/// * `r_y` - the word-index challenges, of which the segment spans the low ones
/// * `channel` - the prover channel for sending/sampling
///
/// ## Preconditions
///
/// * `r_y` must have at least as many coordinates as the packed segment spans words
pub fn prove_public_eval<A, P, Channel>(
	alloc: &A,
	public_words: &[Word],
	r_j: &[B128],
	r_y: &[B128],
	channel: &mut Channel,
) where
	A: Allocator,
	P: PackedField<Scalar = B128>,
	Channel: IPProverChannel<B128>,
{
	// The claim is over the packed segment, so it spans whole field elements: a segment shorter
	// than one still spans one, reading the words past its end as zero.
	let log_public_elems = log2_ceil_usize(public_words.len()).saturating_sub(LOG_WORDS_PER_ELEM);
	let r_y_public = &r_y[..log_public_elems + LOG_WORDS_PER_ELEM];

	channel.send_one(evaluate_words_mle::<B128, B128>(public_words, r_j, r_y_public));

	let packed = pack_witness::<P, _>(alloc, log_public_elems, public_words)
		.expect("the element count is derived from the words being packed");
	let RingSwitchOutput {
		rs_eq_ind,
		sumcheck_claim,
	} = prove(alloc, packed.to_ref(), &[r_j, r_y_public].concat(), channel);

	// The reduced claim is the sum of the two multilinears' product over the hypercube, which is
	// what the trace's opening hands to BaseFold. Here it is discharged by the sumcheck alone: the
	// final evaluations need no message, since the verifier computes both itself.
	let prover = bivariate_product_prover(alloc, [packed, rs_eq_ind], sumcheck_claim);
	prove_single(prover, channel);
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

	// The row fold, written straight from its definition:
	//
	//     out[b] = sum_r eq[r] * bit_b(row r)
	//
	// Only the `eq.len()` rows the weights address take part.
	// Lanes past the last row of a sub-width buffer are not rows, so they never appear here.
	fn naive_fold_1b_rows<P: PackedField<Scalar = F>>(mat: &FieldBuffer<P>, eq: &[F]) -> Vec<F> {
		let mut out = vec![F::ZERO; <F as ExtensionField<B1>>::DEGREE];
		for (r, &weight) in eq.iter().enumerate() {
			let row = mat.get(r);
			for (bit, out_b) in iter::zip(ExtensionField::<B1>::iter_bases(&row), &mut out) {
				if bit == B1::ONE {
					*out_b += weight;
				}
			}
		}
		out
	}

	// The split fold must reproduce that definition at every legal split of the point.
	//
	//     split:  fold(mat, eq(lo), eq(hi)) with the tensor never materialized
	//
	// The matrix is random packed elements, so a sub-width buffer holds unrelated data in the
	// lanes past its last row. The fold must weight those zero and reach the same result.
	fn check_split_fold_matches_definition<P: PackedField<Scalar = F>>(log_len: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let mat = random_field_buffer::<P>(&mut rng, log_len);
		let point: Vec<F> = random_scalars(&mut rng, log_len);

		let expected = naive_fold_1b_rows(&mat, eq_ind_partial_eval::<F>(&point).as_ref());

		// The low factor must cover whole packed elements, unless the matrix is one element.
		// The ceiling is the 128-row chunk the row fold consumes.
		for split_at in P::LOG_WIDTH.min(log_len)..=log_len.min(LOG_SPLIT_BLOCK) {
			let (point_lo, point_hi) = point.split_at(split_at);
			let eq_lo = eq_ind_partial_eval::<F>(point_lo);
			let eq_hi = eq_ind_partial_eval::<F>(point_hi);

			let split = fold_1b_rows_for_b128_split(&mat, &eq_lo, &eq_hi);
			assert_eq!(
				split.as_ref(),
				expected.as_slice(),
				"log_len={log_len}, split_at={split_at}"
			);
		}
	}

	#[test]
	fn test_split_fold_matches_definition() {
		// Row counts below, at and above one packed element, for each packing width.
		for (i, log_len) in [0, 1, 2, 6, 7, 8].into_iter().enumerate() {
			let seed = i as u64;
			check_split_fold_matches_definition::<F>(log_len, seed);
			check_split_fold_matches_definition::<PackedBinaryGhash2x128b>(log_len, seed);
			check_split_fold_matches_definition::<PackedBinaryGhash4x128b>(log_len, seed);
		}
	}

	// The indicator must match the verifier's succinct formula for the same polynomial.
	//
	//     out[i] = A(point, i)
	//
	// `eval_rs_eq` evaluates `A` through the tensor algebra, independently of how the prover
	// builds the buffer, so it pins the fused route at every legal split of the point.
	fn check_rs_eq_ind_from_factors<P: PackedField<Scalar = F>>(log_len: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let point: Vec<F> = random_scalars(&mut rng, log_len);
		let row_batching_challenges: Vec<F> =
			random_scalars(&mut rng, <F as ExtensionField<B1>>::LOG_DEGREE);
		let row_batch_query = eq_ind_partial_eval::<F>(&row_batching_challenges);

		// The indicator has no chunk ceiling, so the low factor may span the whole point.
		for split_at in P::LOG_WIDTH.min(log_len)..=log_len {
			let (point_lo, point_hi) = point.split_at(split_at);
			let eq_lo = eq_ind_partial_eval::<F>(point_lo);
			let eq_hi = eq_ind_partial_eval::<F>(point_hi);

			let rs_eq_ind =
				rs_eq_ind_from_factors::<_, P>(&GlobalAllocator, &eq_lo, &eq_hi, &row_batch_query);

			for index in 0..1 << log_len {
				let expected = eval_rs_eq::<F>(
					&point,
					&index_to_hypercube_point::<F>(log_len, index),
					row_batch_query.as_ref(),
				);
				assert_eq!(rs_eq_ind.get(index), expected, "split_at={split_at}, index={index}");
			}

			// A sub-width buffer's trailing lanes are not entries, and the sumcheck provers
			// downstream read them as zero, so the write must leave them zero.
			let trailing = rs_eq_ind.as_ref()[0].iter().skip(1 << log_len);
			assert!(trailing.take(P::WIDTH).all(|lane| lane == F::ZERO), "split_at={split_at}");
		}
	}

	#[test]
	fn test_rs_eq_ind_from_factors() {
		// Entry counts below, at and above one packed element, for each packing width.
		for (i, log_len) in [0, 1, 2, 6, 7, 8].into_iter().enumerate() {
			let seed = i as u64;
			check_rs_eq_ind_from_factors::<F>(log_len, seed);
			check_rs_eq_ind_from_factors::<PackedBinaryGhash2x128b>(log_len, seed);
			check_rs_eq_ind_from_factors::<PackedBinaryGhash4x128b>(log_len, seed);
		}
	}

	#[test]
	fn test_out_of_range_evaluation() {
		let mut rng = StdRng::from_seed([0; 32]);

		// `eval_rs_eq` must agree with the indicator off the hypercube as well:
		//
		//     eval_rs_eq(z, x, q) = sum_i eq(x)[i] * rs_eq_ind[i]
		let n_vars_big_field = 3;

		// setup ring switch eq mle
		let z_vals: Vec<F> = random_scalars(&mut rng, n_vars_big_field);

		let row_batching_challenges: Vec<F> =
			random_scalars(&mut rng, <F as ExtensionField<B1>>::LOG_DEGREE);

		let row_batching_expanded_query: FieldBuffer<F> =
			eq_ind_partial_eval(&row_batching_challenges);

		let (eq_lo, eq_hi) = expand_tensor_factors(&z_vals);
		let rs_eq = rs_eq_ind_from_factors::<_, F>(
			&GlobalAllocator,
			&eq_lo,
			&eq_hi,
			&row_batching_expanded_query,
		);

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
	fn test_row_fold_composes_into_the_claim() {
		let mut rng = StdRng::seed_from_u64(0);

		type P = PackedBinaryGhash2x128b;

		// The prover's row fold and the verifier's partial evaluation compose into the claim:
		//
		//     <eq(point), bits>  ==  evaluate(s_hat_v, point[..log_degree])
		//
		// where `s_hat_v[b] = sum_r eq(point[log_degree..])[r] * bit_b(row r)`.
		// The low coordinates of the point index the bits within a row, the high ones the rows.
		let n = 10;
		let log_degree = <F as ExtensionField<B1>>::LOG_DEGREE;

		// A random B1 matrix of 2^(n + log_degree) bits, and a point over the same index space.
		let bit_matrix = random_field_buffer::<PackedSubfield<P, B1>>(&mut rng, n + log_degree);
		let eval_point: Vec<F> = random_scalars(&mut rng, n + log_degree);
		let (prefix, suffix) = eval_point.split_at(log_degree);

		// Reference: expand the whole point and contract it against every bit.
		let full_tensor = eq_ind_partial_eval::<F>(&eval_point);
		let expected = inner_product_subfield(
			PackedField::iter_slice(bit_matrix.as_ref()),
			PackedField::iter_slice(full_tensor.as_ref()),
		);

		// Prover route: fold the rows against the suffix factors, then evaluate at the prefix.
		// Viewing 128 single-bit scalars as one field element is a reinterpretation, so the rows
		// are the same memory.
		let mat = FieldBuffer::<P>::new(
			n,
			bit_matrix
				.as_ref()
				.iter()
				.map(|&bits_packed| cast_ext::<B1, P>(bits_packed))
				.collect(),
		);
		let (eq_lo, eq_hi) = expand_tensor_factors(suffix);
		let s_hat_v = fold_1b_rows_for_b128_split(&mat, &eq_lo, &eq_hi);

		assert_eq!(evaluate_inplace(s_hat_v, prefix), expected);
	}
}
