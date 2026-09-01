// Copyright 2024-2025 Irreducible Inc.

use std::{
	cmp::{max, min},
	iter,
	ops::Range,
	slice::from_raw_parts_mut,
};

use binius_field::{BinaryField, PackedField};
use binius_utils::rayon::{
	iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
	slice::ParallelSliceMut,
};
use itertools::izip;

use super::{
	AdditiveNTT, DomainContext,
	reference::{NeighborsLastReference, input_check},
};
use crate::field_buffer::FieldSliceMut;

// This value is chosen assuming 128-bit field elements.
//
// Empirically it performs well and is small enough for the buffer to fit comfortably in L1 cache.
const DEFAULT_LOG_BASE_LEN: usize = 10;

/// One of the three linear maps the butterfly network computes.
///
/// All three run the same network over the same twiddles, and cost the same.
/// They differ in the order the layers run and in what one butterfly does:
///
/// ```text
///     forward     layers ascending     u += v * t;  v += u
///     inverse     layers descending    v += u;      u += v * t
///     transpose   layers descending    u += v;      v += u * t
/// ```
///
/// Everything below is written once against this, so all three get the same engine.
trait Butterfly {
	/// Whether the layers run from the earliest to the latest.
	const ASCENDING: bool;

	/// Applies one butterfly under a twiddle.
	fn apply<P: PackedField>(u: &mut P, v: &mut P, twiddle: P);

	/// Applies one butterfly where the twiddle is zero.
	///
	/// One of the two values then keeps what it held, so its cache lines stay clean.
	fn apply_zero<P: PackedField>(u: &mut P, v: &mut P);

	/// Runs the same map through the scalar reference implementation.
	fn reference<P: PackedField>(
		domain_context: &impl DomainContext<Field = P::Scalar>,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	);
}

/// The transformation the additive NTT is named for.
struct Forward;

impl Butterfly for Forward {
	const ASCENDING: bool = true;

	#[inline(always)]
	fn apply<P: PackedField>(u: &mut P, v: &mut P, twiddle: P) {
		*u += *v * twiddle;
		*v += *u;
	}

	#[inline(always)]
	fn apply_zero<P: PackedField>(u: &mut P, v: &mut P) {
		*v += *u;
	}

	fn reference<P: PackedField>(
		domain_context: &impl DomainContext<Field = P::Scalar>,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		NeighborsLastReference { domain_context }.forward_transform(data, skip_early, skip_late);
	}
}

/// The inverse of the forward transformation.
struct Inverse;

impl Butterfly for Inverse {
	const ASCENDING: bool = false;

	#[inline(always)]
	fn apply<P: PackedField>(u: &mut P, v: &mut P, twiddle: P) {
		*v += *u;
		*u += *v * twiddle;
	}

	#[inline(always)]
	fn apply_zero<P: PackedField>(u: &mut P, v: &mut P) {
		*v += *u;
	}

	fn reference<P: PackedField>(
		domain_context: &impl DomainContext<Field = P::Scalar>,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		NeighborsLastReference { domain_context }.inverse_transform(data, skip_early, skip_late);
	}
}

/// The adjoint of the forward transformation.
///
/// The forward butterfly has determinant one, so its inverse and its transpose agree only where
/// the twiddle vanishes.
struct Transpose;

impl Butterfly for Transpose {
	const ASCENDING: bool = false;

	#[inline(always)]
	fn apply<P: PackedField>(u: &mut P, v: &mut P, twiddle: P) {
		*u += *v;
		*v += *u * twiddle;
	}

	#[inline(always)]
	fn apply_zero<P: PackedField>(u: &mut P, v: &mut P) {
		*u += *v;
	}

	fn reference<P: PackedField>(
		domain_context: &impl DomainContext<Field = P::Scalar>,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		NeighborsLastReference { domain_context }.transpose_transform(data, skip_early, skip_late);
	}
}

/// Applies one layer to the block a recursion level owns.
///
/// ## Preconditions
///
/// - `2 * block_size_half == data.len()`
#[inline]
fn one_block<B: Butterfly, P: PackedField>(
	domain_context: &impl DomainContext<Field = P::Scalar>,
	data: &mut [P],
	block_size_half: usize,
	layer: usize,
	block: usize,
) {
	let (block0, block1) = data.split_at_mut(block_size_half);
	if block == 0 {
		// `domain_context.twiddle(layer, 0)` is always zero (see `DomainContext::twiddle`).
		// So the butterfly collapses to a single addition.
		for (u, v) in iter::zip(block0, block1) {
			B::apply_zero(u, v);
		}
	} else {
		let packed_twiddle = P::broadcast(domain_context.twiddle(layer, block));
		for (u, v) in iter::zip(block0, block1) {
			B::apply(u, v, packed_twiddle);
		}
	}
}

/// Runs a **part** of an NTT butterfly network, in depth-first order.
///
/// Concretely, it processes a specific memory block in the butterfly network, which is given by
/// `layer` and `block`. For this memory block, it processes the layers given by `layer_range`.
///
/// For example, suppose `layer=2` and `block=2`.
/// That means we are in an NTT butterfly network in layer 2 (the third layer) and block 2 (the
/// third block in this layer, there are four blocks in total in this layer). `data` contains the
/// data of this block, so it's only a chunk of the total data used in the NTT. Now suppose
/// `layer_range=2..5`. Then we will process the following butterfly blocks:
/// - `layer=2` `block=2`
/// - `layer=3` `block=4`
/// - `layer=3` `block=5`
/// - `layer=4` `block=8`
/// - `layer=4` `block=9`
/// - `layer=4` `block=10`
/// - `layer=4` `block=11`
///
/// (Just in a different order. We listed breadth-first order, we would process them in
/// depth-first order.)
///
/// A descending map visits the same blocks, and applies a level's own layer after recursing rather
/// than before.
///
/// The argument `log_base_len` determines for which `log_d` we call the breadth-first
/// implementation as a base case.
///
/// ## Preconditions
///
/// - `2^(log_d) == data.len() * packing_width`
/// - `data.len() >= 2`
/// - `domain_context` holds all the twiddles up to `layer_range.end` (exclusive)
/// - `layer <= layer_range.start`
fn depth_first<B: Butterfly, P: PackedField>(
	domain_context: &impl DomainContext<Field = P::Scalar>,
	data: &mut [P],
	log_d: usize,
	layer: usize,
	block: usize,
	layer_range: Range<usize>,
	log_base_len: usize,
) {
	// check preconditions
	debug_assert!(P::LOG_WIDTH < log_d);
	debug_assert_eq!(data.len(), 1 << (log_d - P::LOG_WIDTH));
	debug_assert!(layer_range.end <= domain_context.log_domain_size());
	debug_assert!(layer <= layer_range.start);
	debug_assert!(log_base_len > P::LOG_WIDTH);

	let log_n = log_d + layer;
	debug_assert!(layer_range.end <= log_n);

	if layer >= layer_range.end {
		return;
	}

	// if the problem size is small, we just do breadth_first (to get rid of the stack overhead)
	if log_d <= log_base_len {
		breadth_first::<B, P>(domain_context, data, log_d, layer, block, layer_range);
		return;
	}

	let block_size_half = 1 << (log_d - 1 - P::LOG_WIDTH);
	// Every layer this level's descendants own is above its own, so its own is in range whenever
	// the range has started.
	let owns_layer = layer >= layer_range.start;

	// A descendant owns only layers above this one, so the window it is handed starts no lower
	// than the next layer. That is what keeps the precondition above true all the way down.
	let descendant_range = layer_range.start.max(layer + 1)..layer_range.end;

	if B::ASCENDING && owns_layer {
		one_block::<B, P>(domain_context, data, block_size_half, layer, block);
	}

	// then recurse
	depth_first::<B, P>(
		domain_context,
		&mut data[..block_size_half],
		log_d - 1,
		layer + 1,
		block << 1,
		descendant_range.clone(),
		log_base_len,
	);
	depth_first::<B, P>(
		domain_context,
		&mut data[block_size_half..],
		log_d - 1,
		layer + 1,
		(block << 1) + 1,
		descendant_range,
		log_base_len,
	);

	if !B::ASCENDING && owns_layer {
		one_block::<B, P>(domain_context, data, block_size_half, layer, block);
	}
}

/// Applies one layer whose blocks span whole packed elements.
///
/// All butterflies are between values in separate packed elements, and all butterflies within a
/// block share the same twiddle factor.
fn word_layer<B: Butterfly, P: PackedField>(
	domain_context: &impl DomainContext<Field = P::Scalar>,
	data: &mut [P],
	log_n: usize,
	base_layer: usize,
	base_block: usize,
	layer: usize,
) {
	// log_block_size is log2 the number of packed elements forming one block.
	let log_block_size = log_n - P::LOG_WIDTH - layer;
	let log_half_block_size = log_block_size - 1;

	// log2 the number of blocks to process in this layer
	let log_blocks = layer - base_layer;
	let mut layer_twiddles = domain_context
		.iter_twiddles(layer, 0)
		.skip(base_block << log_blocks)
		.take(1 << log_blocks);
	let mut blocks = data.chunks_exact_mut(1 << log_block_size);

	// `domain_context.twiddle(layer, 0)` is always zero (see `DomainContext::twiddle`).
	// `base_block == 0` is the only case where this call's first block is the domain's block 0.
	// Peel it off once per layer instead of branching per element.
	if base_block == 0 {
		layer_twiddles.next();
		if let Some(block) = blocks.next() {
			let (block0, block1) = block.split_at_mut(1 << log_half_block_size);
			for (u, v) in iter::zip(block0, block1) {
				B::apply_zero(u, v);
			}
		}
	}

	for (block, twiddle) in iter::zip(blocks, layer_twiddles) {
		let packed_twiddle = P::broadcast(twiddle);
		let (block0, block1) = block.split_at_mut(1 << log_half_block_size);
		for (u, v) in iter::zip(block0, block1) {
			B::apply(u, v, packed_twiddle);
		}
	}
}

/// Applies one layer whose blocks sit inside a packed element.
///
/// The butterflies operate on elements within packed field elements.
/// We solve this problem by interleaving the packed elements with each other.
fn lane_layer<B: Butterfly, P: PackedField>(
	domain_context: &impl DomainContext<Field = P::Scalar>,
	data: &mut [P],
	log_n: usize,
	log_d: usize,
	base_block: usize,
	layer: usize,
) {
	// log_block_size is log2 the number of single elements forming one block.
	let log_block_size = log_n - layer;
	let log_half_block_size = log_block_size - 1;
	let log_blocks_per_packed = P::LOG_WIDTH - log_block_size;
	let log_half_blocks_per_packed = log_blocks_per_packed + 1;

	// calculate packed_twiddle_offset
	let mut packed_twiddle_offset = P::zero();
	for block in 0..1 << log_blocks_per_packed {
		let twiddle0 = domain_context.twiddle(layer, block);
		let twiddle1 = domain_context.twiddle(layer, (1 << log_blocks_per_packed) | block);

		let block_start = block << log_block_size;
		for j in 0..1 << log_half_block_size {
			packed_twiddle_offset.set(block_start | j, twiddle0);
			packed_twiddle_offset.set(block_start | j | (1 << log_half_block_size), twiddle1);
		}
	}

	// log2 the number of packed element pairs to process in this layer.
	// This call's data is `2^(log_d - P::LOG_WIDTH)` packed elements, hence half that in pairs.
	let log_packed_pairs = log_d - P::LOG_WIDTH - 1;
	let layer_twiddles = domain_context
		.iter_twiddles(layer, log_half_blocks_per_packed)
		.skip(base_block << log_packed_pairs)
		.take(1 << log_packed_pairs);

	let (data_pairs, rest) = data.as_chunks_mut::<2>();
	debug_assert!(
		rest.is_empty(),
		"data_packed length is a power of two; \
			data_packed length is greater than 1 (checked at beginning of method)"
	);
	debug_assert_eq!(data_pairs.len(), 1 << log_packed_pairs);

	for ([packed0, packed1], first_twiddle) in iter::zip(data_pairs, layer_twiddles) {
		let packed_twiddle = P::broadcast(first_twiddle) + packed_twiddle_offset;

		let (mut u, mut v) = (*packed0).interleave(*packed1, log_half_block_size);
		B::apply(&mut u, &mut v, packed_twiddle);
		(*packed0, *packed1) = u.interleave(v, log_half_block_size);
	}
}

/// Same as the depth-first recursion above, but runs in breadth-first order.
///
/// ## Preconditions
///
/// - `P::LOG_WIDTH < log_d`
/// - `2^(log_d) == data.len() * packing_width`
/// - `data.len() >= 2`
/// - `domain_context` holds all the twiddles up to `layer_bound` (exclusive)
/// - `layer <= layer_range.start`
fn breadth_first<B: Butterfly, P: PackedField>(
	domain_context: &impl DomainContext<Field = P::Scalar>,
	data: &mut [P],
	log_d: usize,
	base_layer: usize,
	base_block: usize,
	layer_range: Range<usize>,
) {
	// check preconditions
	debug_assert!(P::LOG_WIDTH < log_d);
	debug_assert_eq!(data.len(), 1 << (log_d - P::LOG_WIDTH));
	debug_assert!(layer_range.end <= domain_context.log_domain_size());
	debug_assert!(base_layer <= layer_range.start);

	let log_n = log_d + base_layer;
	debug_assert!(layer_range.end <= log_n);

	// Layers below the cutoff have blocks of whole packed elements, and layers at or above it have
	// blocks inside one. The two shapes are contiguous, so each direction runs one then the other.
	let packed_cutoff = (log_n - P::LOG_WIDTH).clamp(layer_range.start, layer_range.end);

	if B::ASCENDING {
		for layer in layer_range.start..packed_cutoff {
			word_layer::<B, P>(domain_context, data, log_n, base_layer, base_block, layer);
		}
		for layer in packed_cutoff..layer_range.end {
			lane_layer::<B, P>(domain_context, data, log_n, log_d, base_block, layer);
		}
	} else {
		for layer in (packed_cutoff..layer_range.end).rev() {
			lane_layer::<B, P>(domain_context, data, log_n, log_d, base_block, layer);
		}
		for layer in (layer_range.start..packed_cutoff).rev() {
			word_layer::<B, P>(domain_context, data, log_n, base_layer, base_block, layer);
		}
	}
}

/// Process a layer of the NTT butterfly network in parallel by splitting the work up into
/// `2^log_num_shares` many shares. This will also split up single *blocks* into multiple shares.
///
/// (The latter is the whole purpose of this function. If the number of shares is small enough (and
/// the number of blocks is big enough) so that we don't need to split up blocks, we could just run
/// the depth-first recursion on disjoint chunks.)
///
/// - `2^(log_d) == data.len() * packing_width`
/// - **Important:** `2^log_num_shares * 2 <= data.len()` (every share is working with whole packed
///   elements, so every share needs at least 2 packed elements)
/// - `domain_context` holds the twiddles of `layer`
fn shared_layer<B: Butterfly, P: PackedField>(
	domain_context: &(impl DomainContext<Field = P::Scalar> + Sync),
	data: &mut [P],
	log_d: usize,
	layer: usize,
	log_num_shares: usize,
) {
	// check preconditions
	debug_assert_eq!(data.len() * (1 << P::LOG_WIDTH), 1 << log_d);
	debug_assert!(1 << (log_num_shares + 1) <= data.len());
	debug_assert!(layer < domain_context.log_domain_size());

	let log_num_chunks = log_num_shares + 1;
	let log_d_chunk = log_d - log_num_chunks;
	let data_ptr = data.as_mut_ptr();
	let shift = log_num_shares - layer;
	let tasks: Vec<_> = (0..1 << log_num_shares)
		.map(|k| {
			let (chunk0, chunk1) = with_middle_bit(k, shift);
			let block = chunk0 >> (log_num_chunks - layer);
			assert!(P::LOG_WIDTH <= log_d_chunk);
			let log_chunk_len = log_d_chunk - P::LOG_WIDTH;
			// SAFETY: the two chunks this task lends out are disjoint from every other task's.
			//
			// Inserting one bit at a fixed position is injective, and the inserted bit is what
			// tells the pair apart. So as `k` runs over its whole range the pairs enumerate every
			// chunk index exactly once:
			//
			//     log_num_shares = 2, shift = 1
			//     k = 0 -> (000, 010)   k = 2 -> (100, 110)
			//     k = 1 -> (001, 011)   k = 3 -> (101, 111)
			//
			// Every index is in range, since the widest is one below the chunk count, and the
			// buffer outlives the loop below.
			let chunk0 = unsafe {
				from_raw_parts_mut(data_ptr.add(chunk0 << log_chunk_len), 1 << log_chunk_len)
			};
			let chunk1 = unsafe {
				from_raw_parts_mut(data_ptr.add(chunk1 << log_chunk_len), 1 << log_chunk_len)
			};
			// `domain_context.twiddle(layer, 0)` is always zero (see `DomainContext::twiddle`).
			// `None` signals a task whose butterfly collapses to an add, no multiply.
			let twiddle = (block != 0).then(|| P::broadcast(domain_context.twiddle(layer, block)));
			(chunk0, chunk1, twiddle)
		})
		.collect();

	tasks
		.into_par_iter()
		.for_each(|(chunk0, chunk1, twiddle)| match twiddle {
			Some(twiddle) => {
				for (u, v) in iter::zip(chunk0, chunk1) {
					B::apply(u, v, twiddle);
				}
			}
			None => {
				for (u, v) in iter::zip(chunk0, chunk1) {
					B::apply_zero(u, v);
				}
			}
		});
}

/// Applies two consecutive butterfly layers to four planes held in registers.
///
/// The four planes are the quarters of one super-block, in plane order.
/// `twiddle_0` belongs to the first layer, which pairs planes `(0, 2)` and `(1, 3)`.
/// `twiddle_1_even` and `twiddle_1_odd` belong to the second, which pairs `(0, 1)` and `(2, 3)`.
///
/// Each element is loaded once, takes part in both layers, and is stored once.
///
/// A descending map runs the second layer's pairs first.
fn fused_pair<B: Butterfly, P: PackedField>(
	planes: [&mut [P]; 4],
	twiddle_0: P,
	twiddle_1_even: P,
	twiddle_1_odd: P,
) {
	let [plane_0, plane_1, plane_2, plane_3] = planes;

	for (x_0, x_1, x_2, x_3) in izip!(plane_0, plane_1, plane_2, plane_3) {
		if B::ASCENDING {
			B::apply(x_0, x_2, twiddle_0);
			B::apply(x_1, x_3, twiddle_0);
			B::apply(x_0, x_1, twiddle_1_even);
			B::apply(x_2, x_3, twiddle_1_odd);
		} else {
			B::apply(x_0, x_1, twiddle_1_even);
			B::apply(x_2, x_3, twiddle_1_odd);
			B::apply(x_0, x_2, twiddle_0);
			B::apply(x_1, x_3, twiddle_0);
		}
	}
}

/// Same as the fused pair above, for the super-block whose index is zero.
///
/// There `twiddle_0` and `twiddle_1_even` are both the layer's block-0 twiddle, which is zero.
/// Each of their butterflies collapses to a single addition, leaving one plane untouched.
/// So that plane is never written, and its cache lines stay clean.
fn fused_pair_zero_block<B: Butterfly, P: PackedField>(planes: [&mut [P]; 4], twiddle_1_odd: P) {
	let [plane_0, plane_1, plane_2, plane_3] = planes;

	for (x_0, x_1, x_2, x_3) in izip!(plane_0, plane_1, plane_2, plane_3) {
		if B::ASCENDING {
			B::apply_zero(x_0, x_2);
			B::apply_zero(x_1, x_3);
			B::apply_zero(x_0, x_1);
			B::apply(x_2, x_3, twiddle_1_odd);
		} else {
			B::apply_zero(x_0, x_1);
			B::apply(x_2, x_3, twiddle_1_odd);
			B::apply_zero(x_0, x_2);
			B::apply_zero(x_1, x_3);
		}
	}
}

/// Processes layers `first_layer` and `first_layer + 1` in a single pass over the buffer.
///
/// Index the buffer by `i` in `[0, 2^log_d)`.
/// Layer `l` pairs `i` with `i XOR 2^(log_d - 1 - l)`, under the twiddle of block `i >> (log_d -
/// l)`.
///
/// Writing `a` for `first_layer`, split the index three ways:
///
/// ```text
///     i = h * 2^(log_d - a)  +  m * 2^(log_d - a - 2)  +  offset
///
///     h < 2^a                    super-block index, which the two layers never move
///     m < 4                      plane index, the only bits they do move
///     offset < 2^(log_d - a - 2) position inside a plane, which they never move
/// ```
///
/// The four entries sharing one `(h, offset)` are closed under both layers, and distinct pairs
/// never interact.
/// So a quarter of the buffer's elements can be transformed independently of the rest, which is
/// what lets both layers run without a barrier between them.
///
/// Layer `a` reads the twiddle of block `h`, shared by both of its pairs.
/// Layer `a + 1` reads blocks `2h` and `2h + 1`, one per pair.
///
/// ## Preconditions
///
/// - `2^log_d == data.len() * packing_width`
/// - `first_layer + 2 <= log_num_shares`
/// - `first_layer + 2 + P::LOG_WIDTH < log_d`
/// - `domain_context` holds the twiddles of layers `first_layer` and `first_layer + 1`
fn shared_layer_pair<B: Butterfly, P: PackedField>(
	domain_context: &(impl DomainContext<Field = P::Scalar> + Sync),
	data: &mut [P],
	log_d: usize,
	first_layer: usize,
	log_num_shares: usize,
) {
	// check preconditions
	debug_assert_eq!(data.len() << P::LOG_WIDTH, 1 << log_d);
	debug_assert!(first_layer + 2 <= log_num_shares);
	debug_assert!(first_layer + 2 + P::LOG_WIDTH < log_d);
	debug_assert!(first_layer + 2 <= domain_context.log_domain_size());

	let log_plane_len = log_d - first_layer - 2 - P::LOG_WIDTH;
	let plane_len = 1 << log_plane_len;

	// Cut every plane into equal runs, enough of them to feed each share at least one.
	// A run never drops below one packed element.
	let log_run_len = log_plane_len.saturating_sub(log_num_shares - first_layer);
	let run_len = 1 << log_run_len;

	// One task per (super-block, run) pair, which the borrow checker proves disjoint for us.
	let tasks = data
		.chunks_exact_mut(plane_len << 2)
		.enumerate()
		.flat_map(|(h, super_block)| {
			let twiddle = |layer, block| P::broadcast(domain_context.twiddle(layer, block));
			let twiddles = (
				twiddle(first_layer, h),
				twiddle(first_layer + 1, h << 1),
				twiddle(first_layer + 1, (h << 1) | 1),
			);

			let (halves_0_1, halves_2_3) = super_block.split_at_mut(plane_len << 1);
			let (plane_0, plane_1) = halves_0_1.split_at_mut(plane_len);
			let (plane_2, plane_3) = halves_2_3.split_at_mut(plane_len);

			izip!(
				plane_0.chunks_exact_mut(run_len),
				plane_1.chunks_exact_mut(run_len),
				plane_2.chunks_exact_mut(run_len),
				plane_3.chunks_exact_mut(run_len),
			)
			.map(move |(run_0, run_1, run_2, run_3)| ([run_0, run_1, run_2, run_3], h, twiddles))
		})
		.collect::<Vec<_>>();

	tasks
		.into_par_iter()
		.for_each(|(planes, h, (twiddle_0, twiddle_1_even, twiddle_1_odd))| {
			// `domain_context.twiddle(layer, 0)` is always zero (see `DomainContext::twiddle`).
			// Only super-block 0 reads block 0, and it does so in both of its layers.
			if h == 0 {
				fused_pair_zero_block::<B, P>(planes, twiddle_1_odd);
			} else {
				fused_pair::<B, P>(planes, twiddle_0, twiddle_1_even, twiddle_1_odd);
			}
		});
}

/// Runs the shared layers of the butterfly network, two layers per pass where possible.
///
/// Pairing halves the passes a window of layers costs, so it halves its memory traffic.
/// A layer that cannot be paired -- an odd one out, or a shape whose planes would fall below one
/// packed element -- runs alone, exactly as it did before pairing existed.
///
/// ## Preconditions
///
/// - same as the routines this dispatches to
/// - `layers.end <= log_num_shares`
fn shared_layers<B: Butterfly, P: PackedField>(
	domain_context: &(impl DomainContext<Field = P::Scalar> + Sync),
	data: &mut [P],
	log_d: usize,
	layers: Range<usize>,
	log_num_shares: usize,
) {
	debug_assert!(layers.end <= log_num_shares);

	// A pair needs one more layer to pair with, and planes of at least one packed element.
	// Each direction walks the window from its own end, so a pair is always the two layers a
	// single pass over the buffer is about to visit.
	if B::ASCENDING {
		let mut layer = layers.start;
		while layer < layers.end {
			if layers.end - layer >= 2 && layer + 2 + P::LOG_WIDTH < log_d {
				shared_layer_pair::<B, P>(domain_context, data, log_d, layer, log_num_shares);
				layer += 2;
			} else {
				shared_layer::<B, P>(domain_context, data, log_d, layer, log_num_shares);
				layer += 1;
			}
		}
	} else {
		let mut layer = layers.end;
		while layer > layers.start {
			if layer - layers.start >= 2 && layer + P::LOG_WIDTH < log_d {
				shared_layer_pair::<B, P>(domain_context, data, log_d, layer - 2, log_num_shares);
				layer -= 2;
			} else {
				shared_layer::<B, P>(domain_context, data, log_d, layer - 1, log_num_shares);
				layer -= 1;
			}
		}
	}
}

/// Inserts a bit into `k`. Returns both the version with `0` inserted and `1` inserted.
///
/// The first `shift` bits are preserved, then `0` or `1` is inserted, and then the remaining bits
/// of `k` follow.
///
/// ## Preconditions
///
/// - `shift` must be strictly greater than 0
fn with_middle_bit(k: usize, shift: usize) -> (usize, usize) {
	assert!(shift >= 1);

	// most significant and least significant bits, overlapping in one bit
	let ms = k >> (shift - 1);
	let ls = k & ((1 << shift) - 1);

	let k0 = ls | ((ms & !1) << shift);
	let k1 = ls | ((ms | 1) << shift);

	(k0, k1)
}

#[derive(Debug)]
pub struct NeighborsLastBreadthFirst<DC> {
	/// The domain context from which the twiddles are pulled.
	pub domain_context: DC,
}

impl<F, DC> AdditiveNTT for NeighborsLastBreadthFirst<DC>
where
	F: BinaryField,
	DC: DomainContext<Field = F>,
{
	type Field = F;

	fn forward_transform<P: PackedField<Scalar = F>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Forward, P>(data, skip_early, skip_late);
	}

	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Inverse, P>(data, skip_early, skip_late);
	}

	fn transpose_transform<P: PackedField<Scalar = F>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Transpose, P>(data, skip_early, skip_late);
	}

	fn domain_context(&self) -> &impl DomainContext<Field = F> {
		&self.domain_context
	}
}

impl<DC: DomainContext> NeighborsLastBreadthFirst<DC> {
	/// Runs one of the three maps over the whole buffer.
	fn run<B: Butterfly, P: PackedField<Scalar = DC::Field>>(
		&self,
		mut data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		let log_d = data.log_len();
		// A buffer of at most one packed element has no pair of words to walk.
		if log_d <= P::LOG_WIDTH {
			return B::reference(&self.domain_context, data, skip_early, skip_late);
		}

		input_check(&self.domain_context, log_d, skip_early, skip_late);

		breadth_first::<B, P>(
			&self.domain_context,
			data.as_mut(),
			log_d,
			0,
			0,
			skip_early..(log_d - skip_late),
		);
	}
}

/// A single-threaded implementation of [`AdditiveNTT`].
///
/// The code only makes sure that it's fast for a _large_ data input.
/// For small inputs, it can be comparatively slow!
///
/// The implementation is depth-first, but calls a breadth-first implementation as a base case.
///
/// Note that "neighbors last" refers to the memory layout for the NTT: In the _last_ layer of this
/// NTT algorithm, neighboring elements speak to each other. In the classic FFT that's usually the
/// case for "decimation in frequency".
#[derive(Debug)]
pub struct NeighborsLastSingleThread<DC> {
	/// The domain context from which the twiddles are pulled.
	pub domain_context: DC,
	/// Determines when to switch from depth-first to the breadth-first base case.
	pub log_base_len: usize,
}

impl<DC> NeighborsLastSingleThread<DC> {
	/// Convenience constructor which sets `log_base_len` to a reasonable default.
	pub const fn new(domain_context: DC) -> Self {
		Self {
			domain_context,
			log_base_len: DEFAULT_LOG_BASE_LEN,
		}
	}
}

impl<DC: DomainContext> AdditiveNTT for NeighborsLastSingleThread<DC> {
	type Field = DC::Field;

	fn forward_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Forward, P>(data, skip_early, skip_late);
	}

	fn inverse_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Inverse, P>(data, skip_early, skip_late);
	}

	fn transpose_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Transpose, P>(data, skip_early, skip_late);
	}

	fn domain_context(&self) -> &impl DomainContext<Field = DC::Field> {
		&self.domain_context
	}
}

impl<DC: DomainContext> NeighborsLastSingleThread<DC> {
	/// Runs one of the three maps over the whole buffer.
	fn run<B: Butterfly, P: PackedField<Scalar = DC::Field>>(
		&self,
		mut data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		let log_d = data.log_len();
		// A buffer of at most one packed element has no pair of words to walk.
		if log_d <= P::LOG_WIDTH {
			return B::reference(&self.domain_context, data, skip_early, skip_late);
		}

		input_check(&self.domain_context, log_d, skip_early, skip_late);

		depth_first::<B, P>(
			&self.domain_context,
			data.as_mut(),
			log_d,
			0,
			0,
			skip_early..(log_d - skip_late),
			// Ensures that log_base_len satisfies precondition
			self.log_base_len.max(P::LOG_WIDTH + 1),
		);
	}
}

/// A multi-threaded implementation of [`AdditiveNTT`].
///
/// The code only makes sure that it's fast for a _large_ data input.
/// For small inputs, it can be comparatively slow!
///
/// The implementation is depth-first, but calls a breadth-first implementation as a base case.
///
/// Note that "neighbors last" refers to the memory layout for the NTT: In the _last_ layer of this
/// NTT algorithm, neighboring elements speak to each other. In the classic FFT that's usually the
/// case for "decimation in frequency".
#[derive(Debug)]
pub struct NeighborsLastMultiThread<DC> {
	/// The domain context from which the twiddles are pulled.
	pub domain_context: DC,
	/// Determines when to switch from depth-first to the breadth-first base case.
	pub log_base_len: usize,
	/// The base-2 logarithm of number of equal-sized shares that the problem should be split into.
	/// Each share needs to do the same amount of work. If you have equally powered cores
	/// available, this should be the base-2 logarithm of the number of cores.
	pub log_num_shares: usize,
}

impl<DC> NeighborsLastMultiThread<DC> {
	/// Convenience constructor which sets `log_base_len` to a reasonable default.
	pub const fn new(domain_context: DC, log_num_shares: usize) -> Self {
		Self {
			domain_context,
			log_base_len: DEFAULT_LOG_BASE_LEN,
			log_num_shares,
		}
	}
}

impl<DC: DomainContext + Sync> AdditiveNTT for NeighborsLastMultiThread<DC> {
	type Field = DC::Field;

	fn forward_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Forward, P>(data, skip_early, skip_late);
	}

	fn inverse_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Inverse, P>(data, skip_early, skip_late);
	}

	fn transpose_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		self.run::<Transpose, P>(data, skip_early, skip_late);
	}

	fn domain_context(&self) -> &impl DomainContext<Field = DC::Field> {
		&self.domain_context
	}
}

impl<DC: DomainContext + Sync> NeighborsLastMultiThread<DC> {
	/// Runs one of the three maps over the whole buffer.
	fn run<B: Butterfly, P: PackedField<Scalar = DC::Field>>(
		&self,
		mut data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		let log_d = data.log_len();
		// A buffer of at most one packed element has no pair of words to walk.
		if log_d <= P::LOG_WIDTH {
			return B::reference(&self.domain_context, data, skip_early, skip_late);
		}

		input_check(&self.domain_context, log_d, skip_early, skip_late);

		// Decide on `actual_log_num_shares`, which also determines how many shared rounds we do.
		// By default this would just be `self.log_num_shares`, but we will potentially decrease it
		// in order to make sure that `2^log_num_shares * 2 <= data.len()`. This serves two
		// purposes:
		// - when we do the shared rounds, each thread should have at least 2 packed elements to
		//   work with, see the precondition of the shared-layer pass
		// - when we do the independent rounds, again each share should have `chunk.len() >= 2`
		//   because this is required by the depth-first recursion
		let maximum_log_num_shares = log_d - P::LOG_WIDTH - 1;
		let actual_log_num_shares = min(self.log_num_shares, maximum_log_num_shares);
		let first_independent_layer = actual_log_num_shares;

		let last_layer = log_d - skip_late;
		let shared = skip_early..min(first_independent_layer, last_layer);
		let independent = max(first_independent_layer, skip_early)..last_layer;

		// The early layers hold blocks too wide for one worker, so they are split across all of
		// them; the late layers hold blocks a worker owns outright. Which of the two runs first
		// is the direction the layers run in.
		let run_shared = |data: &mut [P]| {
			shared_layers::<B, P>(
				&self.domain_context,
				data,
				log_d,
				shared.clone(),
				actual_log_num_shares,
			);
		};

		// One might think that we could just call `depth_first` with
		// `layer=independent.start`. However, this would mean that the chunk size (that we
		// split into using `par_chunks_mut`) could be just one packed element, or even less than
		// one packed element.
		let run_independent = |data: &mut [P]| {
			let layer = min(independent.start, maximum_log_num_shares);
			let log_d_chunk = log_d - layer;
			data.par_chunks_exact_mut(1 << (log_d_chunk - P::LOG_WIDTH))
				.enumerate()
				.for_each(|(block, chunk)| {
					depth_first::<B, P>(
						&self.domain_context,
						chunk,
						log_d_chunk,
						layer,
						block,
						independent.clone(),
						self.log_base_len,
					);
				});
		};

		if B::ASCENDING {
			run_shared(data.as_mut());
			run_independent(data.as_mut());
		} else {
			run_independent(data.as_mut());
			run_shared(data.as_mut());
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b};
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		inner_product::inner_product,
		ntt::domain_context::GaoMateerPreExpanded,
		test_utils::{B128, Packed128b, random_field_buffer},
	};

	/// The shared-layer window the multithreaded transform would pick for these parameters.
	///
	/// Returns the window together with the share count after the transform's own clamp.
	/// An empty window means the transform runs no shared layers at all.
	fn shared_window<P: PackedField>(
		log_d: usize,
		skip_early: usize,
		skip_late: usize,
		log_num_shares: usize,
	) -> (Range<usize>, usize) {
		// Every share needs two packed elements, which caps how finely the buffer splits.
		let actual_log_num_shares = min(log_num_shares, log_d - P::LOG_WIDTH - 1);
		let layers = skip_early..min(actual_log_num_shares, log_d - skip_late);
		(layers, actual_log_num_shares)
	}

	/// Asserts the fused dispatcher and the one-pass-per-layer loop agree bit for bit.
	fn check_fused_matches_per_layer<B: Butterfly, P: PackedField<Scalar: BinaryField>>(
		log_d: usize,
		skip_early: usize,
		skip_late: usize,
		log_num_shares: usize,
		seed: u64,
	) {
		let (layers, actual_log_num_shares) =
			shared_window::<P>(log_d, skip_early, skip_late, log_num_shares);
		if layers.is_empty() {
			return;
		}

		let domain_context = GaoMateerPreExpanded::<P::Scalar>::generate(log_d);
		let mut rng = StdRng::seed_from_u64(seed);

		// Same random contents down both paths, so any difference is the fusion's fault.
		let mut fused = random_field_buffer::<P>(&mut rng, log_d);
		let mut per_layer = fused.clone();

		shared_layers::<B, P>(
			&domain_context,
			fused.as_mut(),
			log_d,
			layers.clone(),
			actual_log_num_shares,
		);

		// The window runs from whichever end this map's layers start at, so the unfused loop has
		// to walk it the same way.
		let mut one_at_a_time = |layer| {
			shared_layer::<B, P>(
				&domain_context,
				per_layer.as_mut(),
				log_d,
				layer,
				actual_log_num_shares,
			);
		};
		if B::ASCENDING {
			layers.for_each(&mut one_at_a_time);
		} else {
			layers.rev().for_each(&mut one_at_a_time);
		}

		assert_eq!(fused, per_layer);
	}

	/// Asserts every packed transform matches the scalar reference, for all three maps.
	fn check_transform_matches_reference<P: PackedField<Scalar: BinaryField>>(
		log_d: usize,
		skip_early: usize,
		skip_late: usize,
		log_num_shares: usize,
		seed: u64,
	) {
		let domain_context = GaoMateerPreExpanded::<P::Scalar>::generate(log_d);
		let mut rng = StdRng::seed_from_u64(seed);
		let input = random_field_buffer::<P>(&mut rng, log_d);

		let multi_thread = NeighborsLastMultiThread {
			domain_context: &domain_context,
			log_base_len: 3,
			log_num_shares,
		};
		let single_thread = NeighborsLastSingleThread {
			domain_context: &domain_context,
			log_base_len: 3,
		};
		let breadth_first = NeighborsLastBreadthFirst {
			domain_context: &domain_context,
		};
		let reference = NeighborsLastReference {
			domain_context: &domain_context,
		};

		// The three maps are three layer orders and three butterflies, so a mix-up in either shows
		// up here whichever one it is in. The scalar reference sets the answer for each, and the
		// three packed engines have to reproduce it bit for bit from the same input.
		macro_rules! check_map {
			($map:ident) => {
				let mut expected = input.clone();
				reference.$map(expected.as_mut_view(), skip_early, skip_late);

				let shape = format!(
					"log_d={log_d} skip_early={skip_early} skip_late={skip_late} \
					 log_num_shares={log_num_shares}"
				);

				let mut actual = input.clone();
				multi_thread.$map(actual.as_mut_view(), skip_early, skip_late);
				assert_eq!(actual, expected, "multi thread {} at {shape}", stringify!($map));

				let mut actual = input.clone();
				single_thread.$map(actual.as_mut_view(), skip_early, skip_late);
				assert_eq!(actual, expected, "single thread {} at {shape}", stringify!($map));

				let mut actual = input.clone();
				breadth_first.$map(actual.as_mut_view(), skip_early, skip_late);
				assert_eq!(actual, expected, "breadth first {} at {shape}", stringify!($map));
			};
		}

		check_map!(forward_transform);
		check_map!(inverse_transform);
		check_map!(transpose_transform);
	}

	/// Every window width from one shared layer up to five, at every start offset that fits.
	///
	/// Width 1 has nothing to fuse, widths 2 to 4 fuse in one window, width 5 splits into 4 plus 1.
	/// The start offset is what shifts the super-block count, so it must move too.
	fn sweep_window_widths<P: PackedField<Scalar: BinaryField>>(log_d: usize, seed: u64) {
		for skip_early in 0..4 {
			for width in 1..=5 {
				// Layer indices only exist below the buffer's own depth.
				if skip_early + width > log_d {
					continue;
				}
				// Ending the shared phase at `skip_early + width` is what sets the window width.
				// All three maps fuse the same window, from their own end of it.
				let end = skip_early + width;
				check_fused_matches_per_layer::<Forward, P>(log_d, skip_early, 0, end, seed);
				check_fused_matches_per_layer::<Inverse, P>(log_d, skip_early, 0, end, seed);
				check_fused_matches_per_layer::<Transpose, P>(log_d, skip_early, 0, end, seed);
			}
		}
	}

	#[test]
	fn fused_shared_layers_match_per_layer_over_window_widths() {
		// Fixture: 128-bit scalars at three packing widths.
		//
		//     1x128b -> LOG_WIDTH 0, planes are always whole packed elements
		//     2x128b -> LOG_WIDTH 1
		//     4x128b -> LOG_WIDTH 2, the width that first refuses the widest windows
		for log_d in 6..13 {
			sweep_window_widths::<PackedBinaryGhash1x128b>(log_d, 0);
			sweep_window_widths::<PackedBinaryGhash2x128b>(log_d, 1);
			sweep_window_widths::<PackedBinaryGhash4x128b>(log_d, 2);
		}
	}

	#[test]
	fn fused_shared_layers_match_per_layer_at_the_narrowest_plane() {
		// Invariant: a window is fused only while each plane still holds two packed elements.
		//
		//     log_d = 8, LOG_WIDTH = 2, skip_early = 1, width = 4
		//     plane length = 2^(8 - 1 - 4 - 2) = 2 packed elements, the smallest fusable plane
		check_fused_matches_per_layer::<Forward, PackedBinaryGhash4x128b>(8, 1, 0, 5, 7);

		// One layer deeper the plane would hold a single packed element, so this shape falls back.
		//
		//     log_d = 8, LOG_WIDTH = 2, skip_early = 0, width = 5 -> plane length 2^1
		//     the same start with width 6 would need plane length 2^0, refused
		check_fused_matches_per_layer::<Forward, PackedBinaryGhash4x128b>(8, 0, 0, 5, 8);
	}

	#[test]
	fn multi_thread_transform_matches_the_reference() {
		// Sweep the skips against the share count, so the shared phase starts and ends everywhere.
		for log_d in [6, 9, 12] {
			for skip_early in [0, 1, 2, 4] {
				for skip_late in [0, 1, 3] {
					if skip_early + skip_late > log_d {
						continue;
					}
					for log_num_shares in [0, 1, 2, 3, 5, 1000] {
						check_transform_matches_reference::<PackedBinaryGhash1x128b>(
							log_d,
							skip_early,
							skip_late,
							log_num_shares,
							0,
						);
						check_transform_matches_reference::<PackedBinaryGhash4x128b>(
							log_d,
							skip_early,
							skip_late,
							log_num_shares,
							1,
						);
					}
				}
			}
		}
	}

	proptest! {
		#[test]
		fn prop_fused_shared_layers_match_per_layer(
			log_d in 6..13usize,
			skip_early in 0..5usize,
			skip_late in 0..4usize,
			log_num_shares in 0..8usize,
			seed: u64,
		) {
			// Property: fusing a window of layers is the same linear map as running them one by
			// one, for every shape the multithreaded transform can hand the shared phase.
			prop_assume!(skip_early + skip_late <= log_d);

			check_fused_matches_per_layer::<Forward, PackedBinaryGhash1x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
			check_fused_matches_per_layer::<Forward, PackedBinaryGhash2x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
			check_fused_matches_per_layer::<Forward, PackedBinaryGhash4x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
		}

		#[test]
		fn prop_multi_thread_transform_matches_the_reference(
			log_d in 1..11usize,
			skip_early in 0..5usize,
			skip_late in 0..4usize,
			log_num_shares in 0..8usize,
			seed: u64,
		) {
			// Property: the fused shared phase leaves the transform equal to the reference one,
			// which shares no code with it.
			prop_assume!(skip_early + skip_late <= log_d);

			check_transform_matches_reference::<PackedBinaryGhash1x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
			check_transform_matches_reference::<PackedBinaryGhash4x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
		}
	}

	#[test]
	fn the_inverse_undoes_the_forward_transform() {
		// Property: the two maps are inverse, which no reference implementation takes part in.
		//
		// Every skip split matters, since a skipped layer must be skipped by both.
		for log_d in [1usize, 4, 6, 9, 12] {
			let domain_context = GaoMateerPreExpanded::<B128>::generate(log_d);
			let ntt = NeighborsLastMultiThread::new(&domain_context, 3);
			let mut rng = StdRng::seed_from_u64(log_d as u64);

			for skip_early in 0..=log_d {
				for skip_late in 0..=(log_d - skip_early) {
					let original = random_field_buffer::<Packed128b>(&mut rng, log_d);

					let mut round_trip = original.clone();
					ntt.forward_transform(round_trip.as_mut_view(), skip_early, skip_late);
					ntt.inverse_transform(round_trip.as_mut_view(), skip_early, skip_late);

					assert_eq!(
						round_trip, original,
						"log_d={log_d} skip_early={skip_early} skip_late={skip_late}"
					);
				}
			}
		}
	}

	#[test]
	fn the_transpose_is_the_adjoint_of_the_forward_transform() {
		// Property: the adjoint identity, which fixes the transpose without naming an
		// implementation:
		//
		//     <transpose(a), m> = <a, forward(m)>
		//
		// An error in the layer order, the butterfly, or the twiddle indexing breaks it.
		for log_d in [1usize, 4, 6, 9, 12] {
			let domain_context = GaoMateerPreExpanded::<B128>::generate(log_d);
			let ntt = NeighborsLastMultiThread::new(&domain_context, 3);
			let mut rng = StdRng::seed_from_u64(log_d as u64);

			for skip_early in 0..=log_d {
				for skip_late in 0..=(log_d - skip_early) {
					let a = random_field_buffer::<Packed128b>(&mut rng, log_d);
					let m = random_field_buffer::<Packed128b>(&mut rng, log_d);

					let mut transposed = a.clone();
					ntt.transpose_transform(transposed.as_mut_view(), skip_early, skip_late);

					let mut forward = m.clone();
					ntt.forward_transform(forward.as_mut_view(), skip_early, skip_late);

					assert_eq!(
						inner_product(transposed.iter_scalars(), m.iter_scalars()),
						inner_product(a.iter_scalars(), forward.iter_scalars()),
						"log_d={log_d} skip_early={skip_early} skip_late={skip_late}"
					);
				}
			}
		}
	}

	#[test]
	fn test_with_middle_bit() {
		assert_eq!(with_middle_bit(0b000, 1), (0b0000, 0b0010));
		assert_eq!(with_middle_bit(0b000, 2), (0b0000, 0b0100));
		assert_eq!(with_middle_bit(0b000, 3), (0b0000, 0b1000));

		assert_eq!(with_middle_bit(0b111, 1), (0b1101, 0b1111));
		assert_eq!(with_middle_bit(0b111, 2), (0b1011, 0b1111));
		assert_eq!(with_middle_bit(0b111, 3), (0b0111, 0b1111));

		assert_eq!(with_middle_bit(0b1110110, 2), (0b11101010, 0b11101110));
	}
}
