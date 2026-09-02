// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	array,
	cmp::{max, min},
	iter, mem,
	ops::Range,
	slice::from_raw_parts_mut,
};

use binius_field::{BinaryField, PackedField};
use binius_utils::rayon::{
	iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
	slice::ParallelSliceMut,
};

use super::{
	AdditiveNTT, DomainContext,
	reference::{NeighborsLastReference, input_check},
};
use crate::field_buffer::FieldSliceMut;

// This value is chosen assuming 128-bit field elements.
//
// Empirically it performs well and is small enough for the buffer to fit comfortably in L1 cache.
const DEFAULT_LOG_BASE_LEN: usize = 10;

/// The widest window of butterfly layers one pass over the buffer may cover.
///
/// A wider window divides the traffic of the shared layers by more, up to the point where a
/// super-block's planes and twiddles no longer fit in registers.
const MAX_WINDOW_LAYERS: usize = 4;

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
/// The argument `log_base_len` determines for which `log_d` we call the breadth-first
/// implementation as a base case.
///
/// ## Preconditions
///
/// - `2^(log_d) == data.len() * packing_width`
/// - `data.len() >= 2`
/// - `domain_context` holds all the twiddles up to `layer_range.end` (exclusive)
/// - `layer <= layer_range.start`
fn forward_depth_first<P: PackedField>(
	domain_context: &impl DomainContext<Field = P::Scalar>,
	data: &mut [P],
	log_d: usize,
	layer: usize,
	block: usize,
	mut layer_range: Range<usize>,
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
		forward_breadth_first(domain_context, data, log_d, layer, block, layer_range);
		return;
	}

	let block_size_half = 1 << (log_d - 1 - P::LOG_WIDTH);
	if layer >= layer_range.start {
		// process only one layer of this block
		let (block0, block1) = data.split_at_mut(block_size_half);
		if block == 0 {
			// `domain_context.twiddle(layer, 0)` is always zero (see `DomainContext::twiddle`).
			// So the butterfly collapses to `v += u`, with `u` left unchanged.
			for (u, v) in iter::zip(block0, block1) {
				*v += *u;
			}
		} else {
			let twiddle = domain_context.twiddle(layer, block);
			let packed_twiddle = P::broadcast(twiddle);
			for (u, v) in iter::zip(block0, block1) {
				// perform butterfly
				*u += *v * packed_twiddle;
				*v += *u;
			}
		}

		layer_range.start += 1;
	}

	// then recurse
	forward_depth_first(
		domain_context,
		&mut data[..block_size_half],
		log_d - 1,
		layer + 1,
		block << 1,
		layer_range.clone(),
		log_base_len,
	);
	forward_depth_first(
		domain_context,
		&mut data[block_size_half..],
		log_d - 1,
		layer + 1,
		(block << 1) + 1,
		layer_range,
		log_base_len,
	);
}

/// Same as [`forward_depth_first`], but runs in breadth-first order.
///
/// ## Preconditions
///
/// - `P::LOG_WIDTH < log_d`
/// - `2^(log_d) == data.len() * packing_width`
/// - `data.len() >= 2`
/// - `domain_context` holds all the twiddles up to `layer_bound` (exclusive)
/// - `layer <= layer_range.start`
fn forward_breadth_first<P: PackedField>(
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

	let packed_cutoff = (log_n - P::LOG_WIDTH).clamp(layer_range.start, layer_range.end);

	// In these rounds, layer <= log_n - P::LOG_WIDTH. All butterflies are between values in
	// separate packed elements, and all butterflies within a block share the same twiddle factor.
	for layer in layer_range.start..packed_cutoff {
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
					*v += *u;
				}
			}
		}

		for (block, twiddle) in iter::zip(blocks, layer_twiddles) {
			let packed_twiddle = P::broadcast(twiddle);
			let (block0, block1) = block.split_at_mut(1 << log_half_block_size);
			for (u, v) in iter::zip(block0, block1) {
				// perform butterfly
				*u += *v * packed_twiddle;
				*v += *u;
			}
		}
	}

	// In these rounds, layer > log_n - P::LOG_WIDTH. The butterflies operate on elements within
	// packed field elements. We solve this problem by interleaving the packed elements with each
	// other.
	for layer in packed_cutoff..layer_range.end {
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
			u += v * packed_twiddle;
			v += u;
			(*packed0, *packed1) = u.interleave(v, log_half_block_size);
		}
	}
}

/// Process a layer of the NTT butterfly network in parallel by splitting the work up into
/// `2^log_num_shares` many shares. This will also split up single *blocks* into multiple shares.
///
/// (The latter is the whole purpose of this function. If the number of shares is small enough (and
/// the number of blocks is big enough) so that we don't need to split up blocks, we could just run
/// [`forward_depth_first`] on disjoint chunks.)
///
/// - `2^(log_d) == data.len() * packing_width`
/// - **Important:** `2^log_num_shares * 2 <= data.len()` (every share is working with whole packed
///   elements, so every share needs at least 2 packed elements)
/// - `domain_context` holds the twiddles of `layer`
fn forward_shared_layer<P: PackedField>(
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
					butterfly(u, v, twiddle);
				}
			}
			None => {
				for (u, v) in iter::zip(chunk0, chunk1) {
					*v += *u;
				}
			}
		});
}

/// Applies one butterfly of the network: `u += v * twiddle`, then `v += u`.
#[inline(always)]
fn butterfly<P: PackedField>(u: &mut P, v: &mut P, twiddle: P) {
	*u += *v * twiddle;
	*v += *u;
}

/// Applies a window of consecutive butterfly layers to the planes of one super-block.
///
/// A super-block is the run of planes the window's layers move elements between; a position
/// inside a plane never moves. Each element is loaded once, takes part in every layer of the
/// window, and is stored once, so a window of any width costs one pass over the buffer.
///
/// The twiddles are the window's layers in order, each layer's blocks in order, so layer `l`
/// occupies `2^l` of them.
///
/// `ZERO_BLOCK` marks the super-block that reads block zero in every layer.
/// That block's twiddle is zero, so its butterflies collapse to an add and the first plane is
/// only ever read.
#[inline(always)]
fn fused_window<P: PackedField, const W: usize, const N: usize, const ZERO_BLOCK: bool>(
	planes: [&mut [P]; N],
	twiddles: &[P],
) {
	debug_assert_eq!(N, 1 << W);
	debug_assert_eq!(twiddles.len(), N - 1);

	// Every plane is a run of the same length, which the cursors below step through in lockstep.
	let lens = planes.each_ref().map(|plane| plane.len());
	let len = lens[0];
	debug_assert!(lens.into_iter().all(|plane_len| plane_len == len));

	// One cursor per plane, stepped in lockstep, so the unrolled network reaches each plane
	// without re-deriving it from an index.
	let mut cursors = planes.map(<[P]>::iter_mut);

	for _ in 0..len {
		let slots: [&mut P; N] =
			array::from_fn(|plane| cursors[plane].next().expect("a plane holds `len` elements"));
		let mut x: [P; N] = array::from_fn(|plane| *slots[plane]);

		// Layer `l` pairs planes `2^(W - 1 - l)` apart, in `2^l` blocks of one twiddle each.
		let mut layer_base = 0;
		for layer in 0..W {
			let stride = N >> (layer + 1);
			for block in 0..1 << layer {
				let twiddle = twiddles[layer_base + block];
				let start = block * (stride << 1);
				for offset in start..start + stride {
					if ZERO_BLOCK && block == 0 {
						x[offset + stride] += x[offset];
					} else {
						let u = x[offset] + x[offset + stride] * twiddle;
						x[offset] = u;
						x[offset + stride] += u;
					}
				}
			}
			layer_base += 1 << layer;
		}

		for (slot, value) in iter::zip(slots, x) {
			*slot = value;
		}
	}
}

/// Processes `W` consecutive layers in a single pass over the buffer.
///
/// Index the buffer by `i` in `[0, 2^log_d)`.
/// Layer `l` pairs `i` with `i XOR 2^(log_d - 1 - l)`, under the twiddle of block `i >> (log_d -
/// l)`.
///
/// Writing `a` for `first_layer`, split the index three ways:
///
/// ```text
///     i = h * 2^(log_d - a)  +  m * 2^(log_d - a - W)  +  offset
///
///     h < 2^a                    super-block index, which the window never moves
///     m < 2^W                    plane index, the only bits it does move
///     offset < 2^(log_d - a - W) position inside a plane, which it never moves
/// ```
///
/// The `2^W` entries sharing one `(h, offset)` are closed under every layer of the window, and
/// distinct groups never interact.
/// So a `2^-W` share of the buffer's elements can be transformed independently of the rest, which
/// is what lets the whole window run without a barrier inside it.
///
/// Layer `a + l` reads the blocks `h * 2^l` through `h * 2^l + 2^l - 1`, one per pair it makes.
///
/// ## Preconditions
///
/// - `2^log_d == data.len() * packing_width`
/// - `first_layer + W <= log_num_shares`
/// - `first_layer + W + P::LOG_WIDTH < log_d`
/// - `domain_context` holds the twiddles of layers `first_layer` through `first_layer + W - 1`
fn forward_shared_layer_window<P: PackedField, const W: usize, const N: usize>(
	domain_context: &(impl DomainContext<Field = P::Scalar> + Sync),
	data: &mut [P],
	log_d: usize,
	first_layer: usize,
	log_num_shares: usize,
) {
	// check preconditions
	debug_assert_eq!(N, 1 << W);
	debug_assert_eq!(data.len() << P::LOG_WIDTH, 1 << log_d);
	debug_assert!(first_layer + W <= log_num_shares);
	debug_assert!(first_layer + W + P::LOG_WIDTH < log_d);
	debug_assert!(first_layer + W <= domain_context.log_domain_size());

	let log_plane_len = log_d - first_layer - W - P::LOG_WIDTH;
	let plane_len = 1 << log_plane_len;

	// Cut every plane into equal runs, enough of them to feed each share at least one.
	// A run never drops below one packed element.
	let log_run_len = log_plane_len.saturating_sub(log_num_shares - first_layer);
	let run_len = 1 << log_run_len;
	let n_runs = plane_len >> log_run_len;

	// One task per (super-block, run) pair, which the borrow checker proves disjoint for us.
	let mut tasks = Vec::with_capacity((1 << first_layer) * n_runs);
	for (h, super_block) in data.chunks_exact_mut(plane_len << W).enumerate() {
		// The window's layers in order, each layer's blocks in order.
		let mut twiddles = [P::zero(); N];
		let mut layer_base = 0;
		for layer in 0..W {
			for block in 0..1 << layer {
				twiddles[layer_base + block] =
					P::broadcast(domain_context.twiddle(first_layer + layer, (h << layer) | block));
			}
			layer_base += 1 << layer;
		}

		let mut rest: &mut [P] = super_block;
		let planes: [&mut [P]; N] = array::from_fn(|_| {
			let (plane, tail) = mem::take(&mut rest).split_at_mut(plane_len);
			rest = tail;
			plane
		});

		let mut plane_runs = planes.map(|plane| plane.chunks_exact_mut(run_len));
		for _ in 0..n_runs {
			let runs: [&mut [P]; N] =
				array::from_fn(|plane| plane_runs[plane].next().expect("one run per step"));
			tasks.push((runs, h, twiddles));
		}
	}

	tasks.into_par_iter().for_each(|(runs, h, twiddles)| {
		// `domain_context.twiddle(layer, 0)` is always zero (see `DomainContext::twiddle`).
		// Super-block zero is the only one that reads block zero, and it does so in every layer.
		if h == 0 {
			fused_window::<P, W, N, true>(runs, &twiddles[..N - 1]);
		} else {
			fused_window::<P, W, N, false>(runs, &twiddles[..N - 1]);
		}
	});
}

/// Runs the shared layers of the butterfly network, in as few passes as the window width allows.
///
/// A window of `W` layers costs one pass, so it divides the memory traffic of that many layers by
/// `W`. The width is capped so a super-block's planes and twiddles stay in registers, and it
/// shrinks when too few layers are left or when the planes would fall below one packed element.
///
/// ## Preconditions
///
/// - same as the routines this dispatches to
/// - `layers.end <= log_num_shares`
fn forward_shared_layers<P: PackedField>(
	domain_context: &(impl DomainContext<Field = P::Scalar> + Sync),
	data: &mut [P],
	log_d: usize,
	layers: Range<usize>,
	log_num_shares: usize,
) {
	debug_assert!(layers.end <= log_num_shares);

	let mut layer = layers.start;
	while layer < layers.end {
		// A window needs its layers to exist, and planes of at least one packed element.
		let width = min(
			min(layers.end - layer, MAX_WINDOW_LAYERS),
			(log_d - P::LOG_WIDTH).saturating_sub(layer + 1),
		);
		match width {
			4 => forward_shared_layer_window::<P, 4, 16>(
				domain_context,
				data,
				log_d,
				layer,
				log_num_shares,
			),
			3 => forward_shared_layer_window::<P, 3, 8>(
				domain_context,
				data,
				log_d,
				layer,
				log_num_shares,
			),
			2 => forward_shared_layer_window::<P, 2, 4>(
				domain_context,
				data,
				log_d,
				layer,
				log_num_shares,
			),
			_ => forward_shared_layer(domain_context, data, log_d, layer, log_num_shares),
		}
		layer += max(width, 1);
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
		mut data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		let log_d = data.log_len();
		if log_d <= P::LOG_WIDTH {
			let fallback_ntt = NeighborsLastReference {
				domain_context: &self.domain_context,
			};
			return fallback_ntt.forward_transform(data, skip_early, skip_late);
		}

		input_check(&self.domain_context, log_d, skip_early, skip_late);

		forward_breadth_first(
			self.domain_context(),
			data.as_mut(),
			log_d,
			0,
			0,
			skip_early..(log_d - skip_late),
		);
	}

	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		_data: FieldSliceMut<'_, P>,
		_skip_early: usize,
		_skip_late: usize,
	) {
		todo!()
	}

	fn domain_context(&self) -> &impl DomainContext<Field = F> {
		&self.domain_context
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
		mut data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		let log_d = data.log_len();
		if log_d <= P::LOG_WIDTH {
			let fallback_ntt = NeighborsLastReference {
				domain_context: &self.domain_context,
			};
			return fallback_ntt.forward_transform(data, skip_early, skip_late);
		}

		input_check(&self.domain_context, log_d, skip_early, skip_late);

		forward_depth_first(
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

	fn inverse_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		_data_orig: FieldSliceMut<'_, P>,
		_skip_early: usize,
		_skip_late: usize,
	) {
		unimplemented!()
	}

	fn domain_context(&self) -> &impl DomainContext<Field = DC::Field> {
		&self.domain_context
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
		mut data: FieldSliceMut<'_, P>,
		skip_early: usize,
		skip_late: usize,
	) {
		let log_d = data.log_len();
		if log_d <= P::LOG_WIDTH {
			let fallback_ntt = NeighborsLastReference {
				domain_context: &self.domain_context,
			};
			return fallback_ntt.forward_transform(data, skip_early, skip_late);
		}

		input_check(&self.domain_context, log_d, skip_early, skip_late);

		// Decide on `actual_log_num_shares`, which also determines how many shared rounds we do.
		// By default this would just be `self.log_num_shares`, but we will potentially decrease it
		// in order to make sure that `2^log_num_shares * 2 <= data.len()`. This serves two
		// purposes:
		// - when we do the shared rounds, each thread should have at least 2 packed elements to
		//   work with, see the precondition of [`forward_shared_layer`]
		// - when we do the independent rounds, again each share should have `chunk.len() >= 2`
		//   because this is required by [`forward_depth_first`]
		let maximum_log_num_shares = log_d - P::LOG_WIDTH - 1;
		let actual_log_num_shares = min(self.log_num_shares, maximum_log_num_shares);
		let first_independent_layer = actual_log_num_shares;

		let last_layer = log_d - skip_late;
		let shared_layers = skip_early..min(first_independent_layer, last_layer);
		let independent_layers = max(first_independent_layer, skip_early)..last_layer;

		forward_shared_layers(
			&self.domain_context,
			data.as_mut(),
			log_d,
			shared_layers,
			actual_log_num_shares,
		);

		// One might think that we could just call `forward_depth_first` with
		// `layer=independent_layers.start`. However, this would mean that the chunk size (that we
		// split into using `par_chunks_mut`) could be just one packed element, or even less than
		// one packed element.
		let layer = min(independent_layers.start, maximum_log_num_shares);
		let log_d_chunk = log_d - layer;
		data.as_mut()
			.par_chunks_exact_mut(1 << (log_d_chunk - P::LOG_WIDTH))
			.enumerate()
			.for_each(|(block, chunk)| {
				forward_depth_first(
					&self.domain_context,
					chunk,
					log_d_chunk,
					layer,
					block,
					independent_layers.clone(),
					self.log_base_len,
				);
			});
	}

	fn inverse_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		_data_orig: FieldSliceMut<'_, P>,
		_skip_early: usize,
		_skip_late: usize,
	) {
		unimplemented!()
	}

	fn domain_context(&self) -> &impl DomainContext<Field = DC::Field> {
		&self.domain_context
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{PackedGhash1x128b, PackedGhash2x128b, PackedGhash4x128b};
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{ntt::domain_context::GaoMateerPreExpanded, test_utils::random_field_buffer};

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
	fn check_fused_matches_per_layer<P: PackedField<Scalar: BinaryField>>(
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

		forward_shared_layers(
			&domain_context,
			fused.as_mut(),
			log_d,
			layers.clone(),
			actual_log_num_shares,
		);
		for layer in layers {
			forward_shared_layer(
				&domain_context,
				per_layer.as_mut(),
				log_d,
				layer,
				actual_log_num_shares,
			);
		}

		assert_eq!(fused, per_layer);
	}

	/// Asserts the whole multithreaded transform matches the independent reference transform.
	fn check_transform_matches_reference<P: PackedField<Scalar: BinaryField>>(
		log_d: usize,
		skip_early: usize,
		skip_late: usize,
		log_num_shares: usize,
		seed: u64,
	) {
		let domain_context = GaoMateerPreExpanded::<P::Scalar>::generate(log_d);
		let mut rng = StdRng::seed_from_u64(seed);

		let mut actual = random_field_buffer::<P>(&mut rng, log_d);
		let mut expected = actual.clone();

		let multi_thread = NeighborsLastMultiThread {
			domain_context: &domain_context,
			log_base_len: 3,
			log_num_shares,
		};
		let reference = NeighborsLastReference {
			domain_context: &domain_context,
		};

		multi_thread.forward_transform(actual.as_mut_view(), skip_early, skip_late);
		reference.forward_transform(expected.as_mut_view(), skip_early, skip_late);

		assert_eq!(actual, expected);
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
				check_fused_matches_per_layer::<P>(log_d, skip_early, 0, skip_early + width, seed);
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
			sweep_window_widths::<PackedGhash1x128b>(log_d, 0);
			sweep_window_widths::<PackedGhash2x128b>(log_d, 1);
			sweep_window_widths::<PackedGhash4x128b>(log_d, 2);
		}
	}

	#[test]
	fn fused_shared_layers_match_per_layer_at_the_narrowest_plane() {
		// Invariant: a window is fused only while each plane still holds two packed elements.
		//
		//     log_d = 8, LOG_WIDTH = 2, skip_early = 1, width = 4
		//     plane length = 2^(8 - 1 - 4 - 2) = 2 packed elements, the smallest fusable plane
		check_fused_matches_per_layer::<PackedGhash4x128b>(8, 1, 0, 5, 7);

		// One layer deeper the plane would hold a single packed element, so this shape falls back.
		//
		//     log_d = 8, LOG_WIDTH = 2, skip_early = 0, width = 5 -> plane length 2^1
		//     the same start with width 6 would need plane length 2^0, refused
		check_fused_matches_per_layer::<PackedGhash4x128b>(8, 0, 0, 5, 8);
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
						check_transform_matches_reference::<PackedGhash1x128b>(
							log_d,
							skip_early,
							skip_late,
							log_num_shares,
							0,
						);
						check_transform_matches_reference::<PackedGhash4x128b>(
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

			check_fused_matches_per_layer::<PackedGhash1x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
			check_fused_matches_per_layer::<PackedGhash2x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
			check_fused_matches_per_layer::<PackedGhash4x128b>(
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

			check_transform_matches_reference::<PackedGhash1x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
			check_transform_matches_reference::<PackedGhash4x128b>(
				log_d, skip_early, skip_late, log_num_shares, seed,
			);
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
