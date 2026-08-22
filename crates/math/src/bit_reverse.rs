// Copyright 2025 Irreducible Inc.

use std::ptr;

use binius_field::{PackedField, transpose_square_blocks};
use binius_utils::{
	checked_arithmetics::checked_log_2,
	rayon::{prelude::*, task_size::min_len_for_bytes},
};
use bytemuck::zeroed_vec;

use crate::field_buffer::FieldSliceMut;

/// Reverses the low `bits` bits of an unsigned integer.
///
/// # Arguments
///
/// * `x` - The value whose bits to reverse
/// * `bits` - The number of low-order bits to reverse
///
/// # Returns
///
/// The value with its low `bits` bits reversed
pub const fn reverse_bits(x: usize, bits: u32) -> usize {
	x.reverse_bits().unbounded_shr(usize::BITS - bits)
}

/// Bytes a cache miss fetches, on every target this runs on.
const CACHE_LINE_BYTES: usize = 64;

/// Base-2 log of the widest tile a permutation instance moves, counted in packed elements.
///
/// # Why this value
///
/// - Every tile-sized loop takes its bound from this parameter.
/// - Each step it may take therefore doubles the number of instances compiled.
/// - Eight elements already reach a cache line for any scalar of eight bytes or wider.
/// - That covers every field permuted here.
/// - A narrower scalar keeps the capped tile rather than paying for further instances.
const MAX_LOG_TILE_PACKED: usize = 3;

/// Applies a bit-reversal permutation to packed field elements in a buffer using parallelization.
///
/// This function permutes the field elements such that element at index `i` is moved to
/// index `reverse_bits(i, log_len)`. The permutation is performed in-place and correctly
/// handles packed field representations.
///
/// # Arguments
///
/// * `buffer` - Mutable slice of packed field elements to permute
pub fn bit_reverse_packed<P: PackedField>(buffer: FieldSliceMut<P>) {
	// A buffer shorter than a square of packed elements leaves the two passes below no room.
	let log_len = buffer.log_len();
	if log_len < 2 * P::LOG_WIDTH {
		return bit_reverse_packed_naive(buffer);
	}

	// Scalars covering one cache line, which is the granularity a permutation is charged at.
	let log_scalar_bytes = size_of::<P::Scalar>().next_power_of_two().ilog2() as usize;
	let log_line = checked_log_2(CACHE_LINE_BYTES).saturating_sub(log_scalar_bytes);

	// Why: a tile under a line wastes bandwidth, and widening one costs sub-block work.
	// Measured, that trade only wins for a packing under half a line.
	// From half a line up the added work costs more than the bandwidth it recovers.
	let log_tile = if P::LOG_WIDTH + 1 < log_line {
		log_line
	} else {
		P::LOG_WIDTH
	};

	// A short buffer takes the widest tile its own length allows.
	// The length check above leaves half the length at or above the packing width.
	// So neither clamp can cut a tile below one packed element.
	let log_tile = log_tile
		.min(P::LOG_WIDTH + MAX_LOG_TILE_PACKED)
		.min(log_len / 2);

	// Why: a constant tile bound is what keeps each pass's gather and scatter unrolled.
	// A run-time bound lowers the moves of a one-element tile to a call.
	// That call is most of the work at that width.
	match log_tile - P::LOG_WIDTH {
		0 => bit_reverse_tiled::<P, 0>(buffer),
		1 => bit_reverse_tiled::<P, 1>(buffer),
		2 => bit_reverse_tiled::<P, 2>(buffer),
		// The clamp above caps the tile at the widest instance, which answers every larger value.
		_ => bit_reverse_tiled::<P, MAX_LOG_TILE_PACKED>(buffer),
	}
}

/// Applies a bit-reversal permutation by moving whole tiles of consecutive scalars.
///
/// # Overview
///
/// A tile is `2^log_tile` consecutive scalars, for `log_tile = P::LOG_WIDTH + LOG_TILE_PACKED`.
/// Both passes below move whole tiles.
/// So the tile width, not the scalar width, is the granularity of every memory access made.
///
/// # Algorithm
///
/// Split the scalar index into three fields, of equal width at both ends:
///
/// ```text
///     i = (h, m, l)        |h| = |l| = log_tile
/// ```
///
/// Reversing every bit of the index maps `(h, m, l)` to `(rev l, rev m, rev h)`.
/// That factors into two passes over disjoint index fields:
///
/// ```text
///     phase 1:  (h, m, l) -> (rev l, m, rev h)      transpose the tiles of one middle index
///     phase 2:  (h, m, l) -> (h, rev m, l)          permute the tiles of one high index
/// ```
///
/// Composing the two reverses every bit.
///
/// # Preconditions
///
/// * `buffer.log_len() >= 2 * (P::LOG_WIDTH + LOG_TILE_PACKED)`
fn bit_reverse_tiled<P: PackedField, const LOG_TILE_PACKED: usize>(mut buffer: FieldSliceMut<P>) {
	// Tile width in scalars, and the middle index field the two ends leave over.
	let log_tile = P::LOG_WIDTH + LOG_TILE_PACKED;
	let log_len = buffer.log_len();
	debug_assert!(log_len >= 2 * log_tile);
	let log_mid = log_len - 2 * log_tile;

	let data = buffer.as_mut();
	// Holding an address rather than the slice is what lets disjoint tasks write one buffer.
	let data_ptr = data.as_mut_ptr() as usize;

	// Phase 1: transpose the square of tiles sitting at each middle index.
	//
	// One iteration reads and writes `2^log_tile` rows of `2^LOG_TILE_PACKED` words each.
	// The byte budget counts single words, so divide it by that factor.
	let min_len = (min_len_for_bytes::<P>() >> (log_tile + LOG_TILE_PACKED + 1)).max(1);
	(0..1 << log_mid)
		.into_par_iter()
		.with_min_len(min_len)
		.for_each_init(
			|| (zeroed_vec::<P>(1 << (log_tile + LOG_TILE_PACKED)), zeroed_vec::<P>(P::WIDTH)),
			|(tile, block), m| {
				// First element of the row at high index `h`, for the middle index of this task.
				// The three index fields hold disjoint bit ranges, so shifting places them.
				let row = |h: usize| (h << (log_mid + LOG_TILE_PACKED)) | (m << LOG_TILE_PACKED);

				// Invariant: rows are visited at their high index reversed, on both passes.
				// A plain transpose over that visiting order is the map this phase needs:
				//
				//     new(h, m, l) = old(rev l, m, rev h)
				//
				// SAFETY:
				// - Every element addressed here carries `m` in the middle field of its index.
				// - No other iteration uses that value, so the tasks write disjoint ranges.
				// - The widest address reached is `2^(log_len - P::LOG_WIDTH) - 1`, the last one.
				// - The address stays live because the buffer outlives this loop.
				unsafe {
					let data = data_ptr as *mut P;

					// Gather the square of tiles into scratch, one contiguous row at a time.
					for j in 0..1 << log_tile {
						let src = data.add(row(reverse_bits(j, log_tile as u32)));
						let dst = tile.as_mut_ptr().add(j << LOG_TILE_PACKED);
						for k in 0..1 << LOG_TILE_PACKED {
							*dst.add(k) = *src.add(k);
						}
					}

					// Transposing in scratch keeps every exchange inside the first cache level.
					transpose_tile::<P, LOG_TILE_PACKED>(tile, block);

					// Scatter the rows back to the places they came from.
					for j in 0..1 << log_tile {
						let src = tile.as_ptr().add(j << LOG_TILE_PACKED);
						let dst = data.add(row(reverse_bits(j, log_tile as u32)));
						for k in 0..1 << LOG_TILE_PACKED {
							*dst.add(k) = *src.add(k);
						}
					}
				}
			},
		);

	// Phase 2: reverse the middle index within each high index, one tile at a time.
	// A high index owns `2^log_mid` consecutive tiles.
	// Different high indices own disjoint runs, so each run permutes on its own.
	data.par_chunks_mut(1 << (log_mid + LOG_TILE_PACKED))
		.for_each(|chunk| {
			bit_reverse_groups::<P, LOG_TILE_PACKED>(chunk);
		});
}

/// Transposes in place the square matrix of scalars a tile buffer holds.
///
/// # Overview
///
/// The matrix is `2^log_n` rows of `2^log_n` scalars in row-major order.
/// One row spans `2^LOG_TILE_PACKED` packed elements.
/// That count is also the number of `P::WIDTH x P::WIDTH` sub-blocks along one side.
///
/// # Algorithm
///
/// The transpose factors over those sub-blocks.
/// A sub-block of the transpose is the transpose of the sub-block mirrored across the diagonal:
///
/// ```text
///     M^T[sub-block (c, r)] = (M[sub-block (r, c)])^T
/// ```
///
/// Two steps therefore cover the whole job:
///
/// - Swap every sub-block with its mirror, which moves whole packed elements.
/// - Transpose each sub-block in place, which is the widest transpose a packing can express.
///
/// # Arguments
///
/// * `tile` - the matrix to transpose, in row-major order
/// * `block` - scratch space for one sub-block
///
/// # Preconditions
///
/// * `tile.len() == 1 << (P::LOG_WIDTH + 2 * LOG_TILE_PACKED)`
/// * `block.len() == P::WIDTH`
fn transpose_tile<P: PackedField, const LOG_TILE_PACKED: usize>(tile: &mut [P], block: &mut [P]) {
	debug_assert_eq!(tile.len(), 1 << (P::LOG_WIDTH + 2 * LOG_TILE_PACKED));
	debug_assert_eq!(block.len(), P::WIDTH);

	// A matrix one sub-block wide is already a single square of lanes.
	// So it needs neither the mirror swaps nor the gather below.
	if LOG_TILE_PACKED == 0 {
		return transpose_square_blocks(P::LOG_WIDTH, tile);
	}

	// Element holding lane row `a` of sub-block `(r, c)`.
	// A sub-block spans `P::WIDTH` matrix rows and takes one element from each, all at column `c`.
	let block_elem =
		|r: usize, a: usize, c: usize| (((r << P::LOG_WIDTH) | a) << LOG_TILE_PACKED) | c;

	// Step 1: swap each sub-block with its mirror across the diagonal.
	for r in 0..1 << LOG_TILE_PACKED {
		for c in r + 1..1 << LOG_TILE_PACKED {
			for a in 0..P::WIDTH {
				tile.swap(block_elem(r, a, c), block_elem(c, a, r));
			}
		}
	}

	// Step 2: transpose each sub-block internally.
	// A packing of one scalar has no lanes to exchange, and the bound folds away per instance.
	if P::LOG_WIDTH > 0 {
		for r in 0..1 << LOG_TILE_PACKED {
			for c in 0..1 << LOG_TILE_PACKED {
				// The elements of one sub-block sit a whole matrix row apart.
				// Gathering them makes the square contiguous, which is what a lane transpose takes.
				for (a, block_i) in block.iter_mut().enumerate() {
					*block_i = tile[block_elem(r, a, c)];
				}
				transpose_square_blocks(P::LOG_WIDTH, block);
				for (a, &block_i) in block.iter().enumerate() {
					tile[block_elem(r, a, c)] = block_i;
				}
			}
		}
	}
}

/// Applies a bit-reversal permutation to packed field elements using a simple algorithm.
///
/// This is a straightforward reference implementation that directly swaps field elements
/// according to the bit-reversal permutation. It serves as a baseline for correctness
/// testing of optimized implementations.
///
/// # Arguments
///
/// * `buffer` - Mutable slice of packed field elements to permute
fn bit_reverse_packed_naive<P: PackedField>(mut buffer: FieldSliceMut<P>) {
	let bits = buffer.log_len() as u32;
	for i in 0..buffer.len() {
		let i_rev = reverse_bits(i, bits);
		if i < i_rev {
			let tmp = buffer.get(i);
			buffer.set(i, buffer.get(i_rev));
			buffer.set(i_rev, tmp);
		}
	}
}

/// Applies a bit-reversal permutation to elements in a slice using parallel iteration.
///
/// This function permutes the elements such that element at index `i` is moved to
/// index `reverse_bits(i, log2(length))`. The permutation is performed in-place
/// by swapping elements in parallel.
///
/// # Arguments
///
/// * `buffer` - Mutable slice of elements to permute
///
/// # Panics
///
/// Panics if the buffer length is not a power of two.
pub fn bit_reverse_indices<T>(buffer: &mut [T]) {
	bit_reverse_groups::<T, 0>(buffer);
}

/// Applies a bit-reversal permutation to groups of `2^LOG_GROUP` consecutive elements.
///
/// # Overview
///
/// Group `i` moves to the group whose index is `i` with its bits reversed.
/// A group of one element permutes single elements.
/// Wider groups permute whole cache lines instead, which is what a strided caller wants.
///
/// # Arguments
///
/// * `buffer` - Mutable slice of elements to permute
///
/// # Panics
///
/// Panics if the group count is not a power of two.
fn bit_reverse_groups<T, const LOG_GROUP: usize>(buffer: &mut [T]) {
	let n_groups = buffer.len() >> LOG_GROUP;
	let bits = checked_log_2(n_groups) as u32;

	// We need to use UnsafeCell-like semantics here to get proper Sync behavior.
	// Creating a raw pointer from the slice inside the closure avoids Sync issues.
	let buffer_ptr = buffer.as_mut_ptr() as usize;

	// Half the iterations swap a pair of groups and half do nothing.
	// So one iteration moves one group of elements on average.
	// The cost is the memory it moves, not the index arithmetic around it.
	let min_len = (min_len_for_bytes::<T>() >> LOG_GROUP).max(1);
	(0..n_groups)
		.into_par_iter()
		.with_min_len(min_len)
		.for_each(|i| {
			let i_rev = reverse_bits(i, bits);
			if i < i_rev {
				// SAFETY: The i < i_rev condition guarantees that:
				// 1. Each (i, i_rev) pair is processed by exactly one thread (the one with i <
				//    i_rev)
				// 2. Since bit-reversal is bijective, no two threads access the same pair
				// 3. The two groups are disjoint runs of `1 << LOG_GROUP` elements
				// 4. Both runs lie in the buffer, since their group indices are below the count
				// 5. No data races can occur
				// 6. buffer_ptr is valid for the lifetime of this closure
				unsafe {
					let ptr = buffer_ptr as *mut T;
					let ptr_i = ptr.add(i << LOG_GROUP);
					let ptr_i_rev = ptr.add(i_rev << LOG_GROUP);
					ptr::swap_nonoverlapping(ptr_i, ptr_i_rev, 1 << LOG_GROUP);
				}
			}
		});
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Field, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b,
	};
	use proptest::prelude::*;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		FieldBuffer,
		test_utils::{random_field_buffer, random_scalars},
	};

	// Packings of one, two and four scalars per element, at a 16-byte scalar.
	// Each drives the tile choice down a different branch, so every property covers all three:
	//
	//     1 scalar  (16 B) -> tile widens to a cache line, sub-block transpose is empty
	//     2 scalars (32 B) -> keeps its own width, tile is one sub-block
	//     4 scalars (64 B) -> keeps its own width, tile is one sub-block
	type P1 = PackedBinaryGhash1x128b;
	type P2 = PackedBinaryGhash2x128b;
	type P4 = PackedBinaryGhash4x128b;

	fn check_equivalence<P: PackedField>(log_d: usize, seed: u64) {
		// Two copies of one random buffer, so each implementation permutes the same input.
		let mut rng = StdRng::seed_from_u64(seed);
		let data_orig = random_field_buffer::<P>(&mut rng, log_d);
		let mut blocked = data_orig.clone();
		let mut naive = data_orig;

		// Invariant: moving whole tiles lands every element where the definition puts it.
		bit_reverse_packed(blocked.to_mut());
		bit_reverse_packed_naive(naive.to_mut());

		assert_eq!(blocked, naive, "mismatch at log_d={log_d}");
	}

	// Lengths chosen to straddle every branch the tile choice can take:
	//
	//     0, 1   -> the length leaves room for no tile at all
	//     3      -> under the square of a four-scalar packing, so its simple path runs
	//     4      -> tile clamped by half the length
	//     7      -> odd length, tile still clamped by half of it
	//     8      -> full tile, exactly one middle index
	//     9, 13  -> odd length with middle indices left over
	//     12     -> several middle indices
	#[rstest::rstest]
	#[case::single_element(0)]
	#[case::one_bit_is_identity(1)]
	#[case::naive_fallback_of_wide_packings(3)]
	#[case::tile_clamped_by_length(4)]
	#[case::odd_length_clamped_tile(7)]
	#[case::one_middle_index(8)]
	#[case::odd_length_full_tile(9)]
	#[case::several_middle_indices(12)]
	#[case::odd_length_several_middle_indices(13)]
	fn test_bit_reverse_packed_equivalence(#[case] log_d: usize) {
		check_equivalence::<P1>(log_d, 0);
		check_equivalence::<P2>(log_d, 0);
		check_equivalence::<P4>(log_d, 0);
	}

	proptest! {
		#[test]
		fn prop_bit_reverse_packed_matches_naive(log_d in 0..14usize, seed: u64) {
			// Sweeps the same equivalence over random lengths and random contents.
			check_equivalence::<P1>(log_d, seed);
			check_equivalence::<P2>(log_d, seed);
			check_equivalence::<P4>(log_d, seed);
		}

		#[test]
		fn prop_bit_reverse_packed_is_an_involution(log_d in 0..14usize, seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let orig = random_field_buffer::<P1>(&mut rng, log_d);

			// Invariant: reversing the bits of an index twice is the identity.
			// So a buffer permuted twice has to come back exactly as it went in.
			let mut twice = orig.clone();
			bit_reverse_packed(twice.to_mut());
			bit_reverse_packed(twice.to_mut());

			prop_assert_eq!(twice, orig);
		}
	}

	fn transpose_reference<F: Field>(log_n: usize, scalars: &[F]) -> Vec<F> {
		// Output row `r` is input column `r`, taking one element from each input row.
		let n = 1 << log_n;
		(0..n)
			.flat_map(|r| (0..n).map(move |c| (r, c)))
			.map(|(r, c)| scalars[(c << log_n) | r])
			.collect()
	}

	fn check_transpose_tile<P: PackedField, const LOG_TILE_PACKED: usize>() {
		// A tile is a square of this many scalars per side.
		let log_n = P::LOG_WIDTH + LOG_TILE_PACKED;
		let mut rng = StdRng::seed_from_u64(log_n as u64);
		let scalars = random_scalars::<P::Scalar>(&mut rng, 1 << (2 * log_n));

		// Pack the square, transpose it in place, then read it back out as scalars.
		let mut tile = FieldBuffer::<P>::from_values(&scalars);
		let mut block = zeroed_vec::<P>(P::WIDTH);
		transpose_tile::<P, LOG_TILE_PACKED>(tile.as_mut(), &mut block);

		// Invariant: exchanging sub-blocks and then lanes equals transposing scalar by scalar.
		let expected = transpose_reference(log_n, &scalars);
		assert_eq!(tile.iter_scalars().collect::<Vec<_>>(), expected, "mismatch at log_n={log_n}");
	}

	#[test]
	fn test_transpose_tile_matches_scalar_transpose() {
		// Fixture state: every tile width the dispatch can select, on every packing.
		// A width of one element takes the single-sub-block return.
		// Wider ones run both the mirror swaps and the per-sub-block lane transpose.
		check_transpose_tile::<P1, 0>();
		check_transpose_tile::<P1, 1>();
		check_transpose_tile::<P1, 2>();
		check_transpose_tile::<P1, 3>();
		check_transpose_tile::<P2, 0>();
		check_transpose_tile::<P2, 1>();
		check_transpose_tile::<P2, 2>();
		check_transpose_tile::<P2, 3>();
		check_transpose_tile::<P4, 0>();
		check_transpose_tile::<P4, 1>();
		check_transpose_tile::<P4, 2>();
		check_transpose_tile::<P4, 3>();
	}

	fn bit_reverse_groups_reference<T: Copy, const LOG_GROUP: usize>(buffer: &[T]) -> Vec<T> {
		// Output group `i` is the input group whose index is `i` with its bits reversed.
		let n_groups = buffer.len() >> LOG_GROUP;
		let bits = checked_log_2(n_groups) as u32;
		(0..n_groups)
			.flat_map(|i| {
				let src = reverse_bits(i, bits) << LOG_GROUP;
				buffer[src..src + (1 << LOG_GROUP)].to_vec()
			})
			.collect()
	}

	fn check_bit_reverse_groups<const LOG_GROUP: usize>() {
		let mut rng = StdRng::seed_from_u64(LOG_GROUP as u64);

		// Group counts from one up to 32, so both the no-op and the split cases run.
		for log_len in LOG_GROUP..LOG_GROUP + 6 {
			let orig = (0..1usize << log_len)
				.map(|_| rng.random::<u64>())
				.collect::<Vec<_>>();

			// Invariant: the parallel swap loop agrees with the definition, group for group.
			let mut permuted = orig.clone();
			bit_reverse_groups::<u64, LOG_GROUP>(&mut permuted);

			assert_eq!(permuted, bit_reverse_groups_reference::<u64, LOG_GROUP>(&orig));
		}
	}

	#[test]
	fn test_bit_reverse_groups_matches_reference() {
		// Fixture state: group widths of 1, 2, 4 and 8 elements.
		// Only the first moves single elements; the rest move runs.
		check_bit_reverse_groups::<0>();
		check_bit_reverse_groups::<1>();
		check_bit_reverse_groups::<2>();
		check_bit_reverse_groups::<3>();
	}

	#[test]
	fn test_bit_reverse_indices_is_the_single_element_group_case() {
		let mut rng = StdRng::seed_from_u64(0);
		let orig = (0..1usize << 10)
			.map(|_| rng.random::<u64>())
			.collect::<Vec<_>>();

		// Invariant: permuting elements one at a time is the width-one group permutation.
		// Fixture state: 1024 elements, so 512 index pairs are candidates for a swap.
		let mut by_indices = orig.clone();
		bit_reverse_indices(&mut by_indices);

		assert_eq!(by_indices, bit_reverse_groups_reference::<u64, 0>(&orig));
	}
}
