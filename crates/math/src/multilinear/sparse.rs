// Copyright 2026 The Binius Developers

//! Bit vectors over the hypercube stored as their set bits, and their multilinear extensions.

use std::{iter, ops::Range};

use binius_field::{Field, field::FieldOps};

use super::eq::eq_ind_partial_eval_scalars;

/// The widest chunk of coordinates expanded into one equality-indicator tensor.
///
/// It bounds every tensor at `2^MAX_CHUNK_WIDTH` elements, however many bits are set.
const MAX_CHUNK_WIDTH: usize = 16;

/// A bit vector of length `2^log_len`, stored as the indices of its set bits.
///
/// Invariant: `indices` is sorted and holds no repeats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseBitVector {
	log_len: usize,
	indices: Vec<u64>,
}

impl SparseBitVector {
	/// Sorts the indices and cancels repeated pairs, since a bit set twice is clear.
	///
	/// # Preconditions
	///
	/// * `log_len` must be less than 64
	/// * every index must be less than `2^log_len`
	pub fn new(log_len: usize, mut indices: Vec<u64>) -> Self {
		assert!(log_len < u64::BITS as usize, "log_len {log_len} must be less than 64");
		assert!(
			indices.iter().all(|&index| index >> log_len == 0),
			"every index must be less than 2^{log_len}"
		);

		indices.sort_unstable();
		// Sorting makes equal indices adjacent, so a stack cancels them in pairs.
		let indices = indices.into_iter().fold(Vec::new(), |mut kept, index| {
			if kept.last() == Some(&index) {
				kept.pop();
			} else {
				kept.push(index);
			}
			kept
		});
		Self { log_len, indices }
	}

	/// Returns the base-2 logarithm of the vector's length.
	pub const fn log_len(&self) -> usize {
		self.log_len
	}

	/// Returns the sorted indices of the set bits.
	pub fn indices(&self) -> &[u64] {
		&self.indices
	}
}

/// A sparse bit vector split into blocks of at most 63 address bits.
///
/// A block's position supplies the high address bits. Each set bit stores only its local `u64`
/// index, so increasing the logical dimension does not widen the per-bit representation.
/// Evaluation shares the low-coordinate tensors across blocks and weights each block's sum by
/// the equality indicator of its high coordinates.
///
/// ```text
/// W(low, high) = Σ_block eq(high, block) · W_block(low)
/// ```
///
/// ```
/// # use binius_field::{Field, Ghash128b as B128};
/// # use binius_math::multilinear::sparse::{PartitionedSparseBitVector, evaluate_partitioned_sparse_b1_multilinear};
/// // The bit at logical address 2^63 is index zero in block one.
/// let bits = PartitionedSparseBitVector::new(64, vec![vec![], vec![0]]);
/// let mut point = vec![B128::ZERO; 64];
/// point[63] = B128::ONE;
/// assert_eq!(evaluate_partitioned_sparse_b1_multilinear(&bits, &point), B128::ONE);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedSparseBitVector {
	log_len: usize,
	blocks: Vec<SparseBitVector>,
}

impl PartitionedSparseBitVector {
	/// The number of low address bits stored in a full block.
	pub const LOG_BLOCK_LEN: usize = 63;

	/// Builds the blocks from their local indices, cancelling repeated pairs within each block.
	///
	/// # Preconditions
	///
	/// * `log_len.saturating_sub(Self::LOG_BLOCK_LEN)` must be less than `usize::BITS`
	/// * `blocks` must contain `2^log_len.saturating_sub(Self::LOG_BLOCK_LEN)` vectors
	/// * each local index must fit `min(log_len, Self::LOG_BLOCK_LEN)` bits
	pub fn new(log_len: usize, blocks: Vec<Vec<u64>>) -> Self {
		let log_blocks = log_len.saturating_sub(Self::LOG_BLOCK_LEN);
		assert!(log_blocks < usize::BITS as usize, "the block count must fit usize");
		assert_eq!(blocks.len(), 1usize << log_blocks, "one vector per block");
		let log_block_len = log_len.min(Self::LOG_BLOCK_LEN);
		Self {
			log_len,
			blocks: blocks
				.into_iter()
				.map(|indices| SparseBitVector::new(log_block_len, indices))
				.collect(),
		}
	}

	/// Returns the logical dimension, including the block-selector coordinates.
	pub const fn log_len(&self) -> usize {
		self.log_len
	}

	/// Returns the blocks in increasing order of their high address bits.
	pub fn blocks(&self) -> &[SparseBitVector] {
		&self.blocks
	}
}

/// Evaluates a partitioned sparse bit vector, sharing the low-coordinate tensors across all blocks.
///
/// # Preconditions
///
/// * `point.len()` must equal `bits.log_len()`
pub fn evaluate_partitioned_sparse_b1_multilinear<E: FieldOps>(
	bits: &PartitionedSparseBitVector,
	point: &[E],
) -> E {
	assert_eq!(point.len(), bits.log_len, "point must have log_len coordinates");
	if bits.blocks.len() == 1 {
		return evaluate_sparse_b1_multilinear(&bits.blocks[0], point);
	}
	let (low, high) = point.split_at(PartitionedSparseBitVector::LOG_BLOCK_LEN);
	let n_set = bits.blocks.iter().map(|block| block.indices.len()).sum();
	let chunks = expand_chunks(n_set, low);
	let weights = eq_ind_partial_eval_scalars(high);
	iter::zip(&bits.blocks, weights)
		.filter(|(block, _)| !block.indices.is_empty())
		.map(|(block, weight)| evaluate_with_chunks(&block.indices, &chunks) * weight)
		.sum()
}

/// Evaluates a partitioned sparse bit vector natively, reducing the wide sum once per nonempty
/// block.
///
/// # Preconditions
///
/// * `point.len()` must equal `bits.log_len()`
pub fn evaluate_partitioned_sparse_b1_multilinear_native<F: Field>(
	bits: &PartitionedSparseBitVector,
	point: &[F],
) -> F {
	assert_eq!(point.len(), bits.log_len, "point must have log_len coordinates");
	if bits.blocks.len() == 1 {
		return evaluate_sparse_b1_multilinear_native(&bits.blocks[0], point);
	}
	let (low, high) = point.split_at(PartitionedSparseBitVector::LOG_BLOCK_LEN);
	let n_set = bits.blocks.iter().map(|block| block.indices.len()).sum();
	let chunks = expand_chunks(n_set, low);
	let weights = eq_ind_partial_eval_scalars(high);
	let wide = iter::zip(&bits.blocks, weights)
		.filter(|(block, _)| !block.indices.is_empty())
		.map(|(block, weight)| {
			F::wide_mul(evaluate_with_chunks_native(&block.indices, &chunks), weight)
		})
		.sum();
	F::reduce(wide)
}

/// Evaluates the multilinear extension of a bit vector at a point of `bits.log_len()` coordinates.
///
/// ```text
/// Σ_{k in set bits} eq(point, k)
/// ```
///
/// The coordinates are split into contiguous chunks, each expanded once into an
/// equality-indicator tensor, and each set bit contributes the product of one lookup per chunk.
/// See [`evaluate_sparse_b1_multilinear_native`] for the faster path over a base field.
///
/// ```
/// # use binius_field::{Field, Ghash128b as B128};
/// # use binius_math::multilinear::sparse::{SparseBitVector, evaluate_sparse_b1_multilinear};
/// // Bits 1 and 2 of a length-4 vector, read back at the vertex of index 2.
/// let bits = SparseBitVector::new(2, vec![2, 1]);
/// let point = [B128::ZERO, B128::ONE];
/// assert_eq!(evaluate_sparse_b1_multilinear(&bits, &point), B128::ONE);
/// ```
///
/// # Preconditions
///
/// * `point.len()` must equal `bits.log_len()`
pub fn evaluate_sparse_b1_multilinear<E: FieldOps>(bits: &SparseBitVector, point: &[E]) -> E {
	assert_eq!(point.len(), bits.log_len, "point must have log_len coordinates");
	let chunks = expand_chunks(bits.indices.len(), point);
	evaluate_with_chunks(&bits.indices, &chunks)
}

/// Evaluates local indices against tensors already expanded for their coordinates.
fn evaluate_with_chunks<E: FieldOps>(indices: &[u64], chunks: &[(Range<usize>, Vec<E>)]) -> E {
	indices
		.iter()
		.map(|&index| {
			chunks
				.iter()
				.map(|(range, tensor)| lookup(index, range, tensor))
				.reduce(|acc, value| acc * value)
				.expect("there is at least one chunk")
		})
		.sum()
}

/// Evaluates the multilinear extension of a bit vector natively in the field `F`.
///
/// Produces the identical result to [`evaluate_sparse_b1_multilinear`], but multiplies each set
/// bit's last lookup unreduced with [`WideMul`](binius_field::arithmetic_traits::WideMul) and
/// reduces the sum once, which the generic path cannot since `E: FieldOps` does not imply it.
///
/// # Preconditions
///
/// * `point.len()` must equal `bits.log_len()`
pub fn evaluate_sparse_b1_multilinear_native<F: Field>(bits: &SparseBitVector, point: &[F]) -> F {
	assert_eq!(point.len(), bits.log_len, "point must have log_len coordinates");
	let chunks = expand_chunks(bits.indices.len(), point);
	evaluate_with_chunks_native(&bits.indices, &chunks)
}

/// The native evaluation against shared tensors, with one reduction of the accumulated wide sum.
fn evaluate_with_chunks_native<F: Field>(indices: &[u64], chunks: &[(Range<usize>, Vec<F>)]) -> F {
	let (last, init) = chunks.split_last().expect("there is at least one chunk");
	let wide = indices
		.iter()
		.map(|&index| {
			let prefix = init
				.iter()
				.map(|(range, tensor)| lookup(index, range, tensor))
				.product();
			F::wide_mul(prefix, lookup(index, &last.0, &last.1))
		})
		.sum();
	F::reduce(wide)
}

/// Splits the point into chunks and expands each into its equality-indicator tensor.
fn expand_chunks<E: FieldOps>(n_set: usize, point: &[E]) -> Vec<(Range<usize>, Vec<E>)> {
	chunk_ranges(n_set, point.len())
		.into_iter()
		.map(|range| {
			let tensor = eq_ind_partial_eval_scalars(&point[range.clone()]);
			(range, tensor)
		})
		.collect()
}

/// Reads the tensor entry a chunk's coordinates select from an index.
fn lookup<E: FieldOps>(index: u64, range: &Range<usize>, tensor: &[E]) -> E {
	tensor[(index >> range.start) as usize & ((1 << range.len()) - 1)].clone()
}

/// Splits `log_len` coordinates into contiguous chunks for `n_set` set bits.
///
/// The chunk count `k` approximately minimizes the field multiplications
///
/// ```text
/// Σ_i 2^{w_i}  +  n_set · (k − 1)
/// ```
///
/// over widths as equal as they can be, none wider than [`MAX_CHUNK_WIDTH`].
/// There is always at least one chunk, of width zero when `log_len` is zero.
fn chunk_ranges(n_set: usize, log_len: usize) -> Vec<Range<usize>> {
	let min_count = log_len.div_ceil(MAX_CHUNK_WIDTH).max(1);
	let count = (min_count..=log_len.max(min_count))
		.min_by_key(|&count| {
			equal_widths(log_len, count)
				.map(|width| 1usize << width)
				.sum::<usize>()
				+ n_set * (count - 1)
		})
		.expect("the range of counts is non-empty");
	equal_widths(log_len, count)
		.scan(0, |start, width| {
			let range = *start..*start + width;
			*start += width;
			Some(range)
		})
		.collect()
}

/// Splits `log_len` into `count` widths that differ by at most one.
fn equal_widths(log_len: usize, count: usize) -> impl Iterator<Item = usize> {
	(0..count).map(move |i| log_len / count + usize::from(i < log_len % count))
}

#[cfg(test)]
mod tests {
	use std::iter;

	use proptest::{prelude::*, test_runner::RngSeed};
	use rand::prelude::*;
	use rstest::rstest;

	use super::*;
	use crate::{
		multilinear::eq::eq_ind,
		test_utils::{B128, index_to_hypercube_point, random_scalars},
	};

	fn random_bits(rng: &mut StdRng, n_set: usize, log_len: usize) -> SparseBitVector {
		let indices = iter::repeat_with(|| rng.random_range(0..1u64 << log_len))
			.take(n_set)
			.collect();
		SparseBitVector::new(log_len, indices)
	}

	#[rstest]
	#[case::empty(0, 8, 4)]
	#[case::zero_vars(1, 0, 1)]
	#[case::one_bit(1, 16, 8)]
	#[case::few_bits(8, 12, 4)]
	#[case::some_bits(64, 12, 3)]
	#[case::many_bits(1024, 12, 2)]
	#[case::dense(4096, 12, 1)]
	fn matches_dense_inner_product(
		#[case] n_set: usize,
		#[case] log_len: usize,
		#[case] n_chunks: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let bits = random_bits(&mut rng, n_set, log_len);
		let point = random_scalars::<B128>(&mut rng, log_len);

		// Fewer set bits favor more, narrower chunks; the cases span one to eight.
		assert_eq!(chunk_ranges(n_set, log_len).len(), n_chunks);

		let tensor = eq_ind_partial_eval_scalars(&point);
		let expected = bits
			.indices()
			.iter()
			.map(|&index| tensor[index as usize])
			.sum::<B128>();
		assert_eq!(evaluate_sparse_b1_multilinear(&bits, &point), expected);
		assert_eq!(evaluate_sparse_b1_multilinear_native(&bits, &point), expected);
	}

	#[test]
	fn wide_point_matches_per_bit_indicator() {
		let mut rng = StdRng::seed_from_u64(0);
		let log_len = 40;
		let bits = random_bits(&mut rng, 3, log_len);
		let point = random_scalars::<B128>(&mut rng, log_len);

		// Too wide for a dense tensor, so every chunk is capped.
		assert!(
			chunk_ranges(3, log_len)
				.iter()
				.all(|range| range.len() <= MAX_CHUNK_WIDTH)
		);

		let expected = bits
			.indices()
			.iter()
			.map(|&index| {
				// The index is wider than `usize` on 32-bit targets, so the vertex is built from
				// `u64`.
				let vertex = (0..log_len)
					.map(|i| {
						if (index >> i) & 1 == 1 {
							B128::ONE
						} else {
							B128::ZERO
						}
					})
					.collect::<Vec<_>>();
				eq_ind(&point, &vertex)
			})
			.sum::<B128>();
		assert_eq!(evaluate_sparse_b1_multilinear(&bits, &point), expected);
		assert_eq!(evaluate_sparse_b1_multilinear_native(&bits, &point), expected);
	}

	#[test]
	fn boolean_point_reads_the_bit() {
		let mut rng = StdRng::seed_from_u64(0);
		let log_len = 6;
		let bits = random_bits(&mut rng, 20, log_len);

		for position in 0..1u64 << log_len {
			let vertex = index_to_hypercube_point::<B128>(log_len, position as usize);
			let bit = if bits.indices().contains(&position) {
				B128::ONE
			} else {
				B128::ZERO
			};
			assert_eq!(evaluate_sparse_b1_multilinear(&bits, &vertex), bit);
			assert_eq!(evaluate_sparse_b1_multilinear_native(&bits, &vertex), bit);
		}
	}

	#[test]
	fn repeated_indices_cancel_in_pairs() {
		let mut rng = StdRng::seed_from_u64(0);
		let log_len = 4;
		let uncancelled = vec![3, 5, 3, 7, 3, 5, 9, 9];
		let bits = SparseBitVector::new(log_len, uncancelled.clone());
		assert_eq!(bits.indices(), &[3, 7]);

		let point = random_scalars::<B128>(&mut rng, log_len);
		let naive = uncancelled
			.iter()
			.map(|&index| eq_ind(&point, &index_to_hypercube_point(log_len, index as usize)))
			.sum::<B128>();
		assert_eq!(evaluate_sparse_b1_multilinear(&bits, &point), naive);
	}

	#[test]
	#[should_panic(expected = "every index must be less than 2^4")]
	fn new_rejects_out_of_range_index() {
		SparseBitVector::new(4, vec![16]);
	}

	proptest! {
		#![proptest_config(ProptestConfig {
			cases: 64,
			rng_seed: RngSeed::Fixed(0),
			..ProptestConfig::default()
		})]

		#[test]
		fn partitioned_evaluations_match_per_bit_indicators(
			addresses in prop::collection::vec(any::<u128>(), 0..32),
			seed: u64,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			for log_len in [0, 6, 40, 63, 64, 65, 73, 77] {
				let log_low = log_len.min(PartitionedSparseBitVector::LOG_BLOCK_LEN);
				let n_blocks = 1usize << log_len.saturating_sub(log_low);
				let point = random_scalars::<B128>(&mut rng, log_len);
				let mut blocks = vec![Vec::new(); n_blocks];
				// Flat logical addresses provide a reference independent of the block layout.
				let indices = addresses
					.iter()
					.map(|&index| index & ((1u128 << log_len) - 1))
					.collect::<Vec<_>>();
				for &index in &indices {
					let block = (index >> log_low) as usize;
					let local = (index & ((1u128 << log_low) - 1)) as u64;
					blocks[block].push(local);
				}
				let expected = indices
					.iter()
					.map(|&index| {
						let vertex = (0..log_len)
							.map(|i| {
								if (index >> i) & 1 == 0 {
									B128::ZERO
								} else {
									B128::ONE
								}
							})
							.collect::<Vec<_>>();
						eq_ind(&point, &vertex)
					})
					.sum::<B128>();
				let bits = PartitionedSparseBitVector::new(log_len, blocks);
				prop_assert!(bits.blocks().iter().all(|block| block.log_len() <= 63));
				prop_assert_eq!(evaluate_partitioned_sparse_b1_multilinear(&bits, &point), expected);
				prop_assert_eq!(evaluate_partitioned_sparse_b1_multilinear_native(&bits, &point), expected);
			}
		}
	}

	#[test]
	fn empty_partitioned_vectors_evaluate_to_zero() {
		let mut rng = StdRng::seed_from_u64(0);
		for log_len in [0usize, 63, 64, 73] {
			let n_blocks =
				1usize << log_len.saturating_sub(PartitionedSparseBitVector::LOG_BLOCK_LEN);
			let bits = PartitionedSparseBitVector::new(log_len, vec![vec![]; n_blocks]);
			let point = random_scalars::<B128>(&mut rng, log_len);
			assert_eq!(evaluate_partitioned_sparse_b1_multilinear(&bits, &point), B128::ZERO);
			assert_eq!(
				evaluate_partitioned_sparse_b1_multilinear_native(&bits, &point),
				B128::ZERO
			);
		}
	}

	#[test]
	fn equal_local_indices_in_different_blocks_do_not_cancel() {
		let bits = PartitionedSparseBitVector::new(65, vec![vec![5], vec![5, 5], vec![], vec![5]]);
		for block in 0..4 {
			let point = (0..63)
				.map(|i| {
					if (5u64 >> i) & 1 == 0 {
						B128::ZERO
					} else {
						B128::ONE
					}
				})
				.chain(index_to_hypercube_point::<B128>(2, block))
				.collect::<Vec<_>>();
			let expected = if block == 0 || block == 3 {
				B128::ONE
			} else {
				B128::ZERO
			};
			assert_eq!(evaluate_partitioned_sparse_b1_multilinear(&bits, &point), expected);
			assert_eq!(evaluate_partitioned_sparse_b1_multilinear_native(&bits, &point), expected);
		}
	}

	#[test]
	#[should_panic(expected = "every index must be less than 2^63")]
	fn local_indices_are_limited_to_63_bits() {
		PartitionedSparseBitVector::new(64, vec![vec![1u64 << 63], vec![]]);
	}

	#[test]
	#[should_panic(expected = "one vector per block")]
	fn partitioned_vector_rejects_missing_blocks() {
		PartitionedSparseBitVector::new(65, vec![vec![]]);
	}
}
