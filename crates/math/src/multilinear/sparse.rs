// Copyright 2026 The Binius Developers

//! Bit vectors over the hypercube stored as their set bits, and their multilinear extensions.

use std::ops::Range;

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
	let chunks = expand_chunks(bits, point);
	bits.indices
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
	let chunks = expand_chunks(bits, point);
	let (last, init) = chunks.split_last().expect("there is at least one chunk");
	let wide = bits
		.indices
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
fn expand_chunks<E: FieldOps>(bits: &SparseBitVector, point: &[E]) -> Vec<(Range<usize>, Vec<E>)> {
	assert_eq!(point.len(), bits.log_len, "point must have log_len coordinates");
	chunk_ranges(bits.indices.len(), bits.log_len)
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
}
