// Copyright 2026 The Binius Developers

//! Experimental portable, auto-vectorized Blake3 multi-lane kernel.
//!
//! Two batched entry points share one block-compression core:
//! - Leaf hashing, for any message up to one 1024-byte chunk.
//! - Two-to-one inner-node compression, `blake3(left || right)` over a 64-byte pair.
//!
//! An alternative to driving the `blake3` crate's hand-written SIMD kernel.
//! The bet: LLVM auto-vectorizes plain lane loops into whatever the target has.
//!
//! - Each of the 16 compression-state words is held as `[u32; N]`, one lane per message.
//! - Every step is a fixed-width `0..N` loop of plain scalar `u32` arithmetic.
//! - No intrinsics, no `unsafe`, no per-target code inside the loops.
//!
//! Lanes the vectorizer is expected to fill, per target:
//! - NEON (128-bit) on ARM64 -> 4 lanes per vector.
//! - AVX2 / AVX-512 on x86 -> 8 / 16 lanes per vector.
//! - SVE2 on capable ARM64 -> width-agnostic vectors.
//!
//! Where a hand-written kernel beats the vectorizer, the loops become the fallback:
//! - `load_block_words` hands the message transpose to the parent's `avx512` or `neon` module.
//! - `compress_block` hands the block compression to the parent's `neon` module.
//!
//! Each hands over only on the lane counts that kernel claims; every other count runs the loops.
//!
//! Output is bit-identical to `blake3::hash`, pinned to the reference in tests.
//! Scope: any message up to one 1024-byte chunk, including sub-block and partial-block leaves.

use std::{array, mem::MaybeUninit};

use binius_utils::{
	FixedSizeSerializeBytes, SerializeBytes,
	rayon::{
		iter::{IndexedParallelIterator, ParallelIterator},
		slice::{ParallelSlice, ParallelSliceMut},
	},
};
use blake3::{BLOCK_LEN, CHUNK_LEN, OUT_LEN};
use digest::Output;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use super::avx512;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::neon;
use super::{Blake3Compression, CHUNK_END, CHUNK_START, IV, MSG_SCHEDULE, ROOT};
use crate::{
	parallel_compression::ParallelPseudoCompression,
	parallel_digest::{
		MultiDigest, ParallelDigest, ParallelDigestAdapter, ParallelMultidigestImpl,
	},
};

/// Applies one Blake3 quarter-round across all `N` lanes.
///
/// The state words at positions `a, b, c, d` are mixed with two message words per lane.
/// Every line is an independent `0..N` map, which is what the vectorizer turns into SIMD.
#[inline(always)]
fn quarter_round<const N: usize>(
	v: &mut [[u32; N]; 16],
	a: usize,
	b: usize,
	c: usize,
	d: usize,
	mx: &[u32; N],
	my: &[u32; N],
) {
	// One lane per iteration; lanes are independent, so the loop vectorizes.
	for i in 0..N {
		v[a][i] = v[a][i].wrapping_add(v[b][i]).wrapping_add(mx[i]);
		v[d][i] = (v[d][i] ^ v[a][i]).rotate_right(16);
		v[c][i] = v[c][i].wrapping_add(v[d][i]);
		v[b][i] = (v[b][i] ^ v[c][i]).rotate_right(12);
		v[a][i] = v[a][i].wrapping_add(v[b][i]).wrapping_add(my[i]);
		v[d][i] = (v[d][i] ^ v[a][i]).rotate_right(8);
		v[c][i] = v[c][i].wrapping_add(v[d][i]);
		v[b][i] = (v[b][i] ^ v[c][i]).rotate_right(7);
	}
}

/// Applies round `R`: four column mixes, then four diagonal mixes.
///
/// The two message words each quarter-round folds in come from row `R` of the schedule table.
#[inline(always)]
fn round<const R: usize, const N: usize>(v: &mut [[u32; N]; 16], m: &[[u32; N]; 16]) {
	// The 16 message words this round reads, in slot order.
	let s = MSG_SCHEDULE[R];

	// Columns.
	quarter_round(v, 0, 4, 8, 12, &m[s[0]], &m[s[1]]);
	quarter_round(v, 1, 5, 9, 13, &m[s[2]], &m[s[3]]);
	quarter_round(v, 2, 6, 10, 14, &m[s[4]], &m[s[5]]);
	quarter_round(v, 3, 7, 11, 15, &m[s[6]], &m[s[7]]);
	// Diagonals.
	quarter_round(v, 0, 5, 10, 15, &m[s[8]], &m[s[9]]);
	quarter_round(v, 1, 6, 11, 12, &m[s[10]], &m[s[11]]);
	quarter_round(v, 2, 7, 8, 13, &m[s[12]], &m[s[13]]);
	quarter_round(v, 3, 4, 9, 14, &m[s[14]], &m[s[15]]);
}

/// Loads one 64-byte block per lane into 16 little-endian message words.
///
/// The words arrive one block per lane and are consumed one word per lane, so this is a transpose.
/// Where a vector kernel covers the lane count it moves the square with shuffles instead of loads.
#[inline(always)]
fn load_block_words<const N: usize>(block: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
	if avx512::handles_lanes(N) {
		return avx512::load_block_words(block);
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	if neon::transposes_lanes(N) {
		return neon::load_block_words(block);
	}

	load_block_words_portable(block)
}

/// Loads one 64-byte block per lane into 16 little-endian message words, one word at a time.
///
/// Every target without a hand-written transpose runs this.
///
/// It is also the reference the vector kernels are tested against.
#[inline(always)]
fn load_block_words_portable<const N: usize>(block: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	let mut m = [[0u32; N]; 16];
	for lane in 0..N {
		for (w, slot) in m.iter_mut().enumerate() {
			let off = w * 4;
			slot[lane] = u32::from_le_bytes([
				block[lane][off],
				block[lane][off + 1],
				block[lane][off + 2],
				block[lane][off + 3],
			]);
		}
	}
	m
}

/// Compresses one 64-byte block across all `N` lanes, updating the chaining value in place.
///
/// The counter, block length, and flags are shared by every lane, so they broadcast.
/// Only the input chaining value and the message differ per lane.
#[inline(always)]
fn compress_block<const N: usize>(
	cv: &mut [[u32; N]; 8],
	block: &[[u32; N]; 16],
	counter: u64,
	block_len: u32,
	flags: u32,
) {
	// On aarch64 the hand-written vector kernel holds the same state in registers.
	// Lane counts it does not cover fall through to the lane loops.
	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	if neon::handles_lanes(N) {
		neon::compress_block(cv, block, counter, block_len, flags);
		return;
	}

	compress_block_portable(cv, block, counter, block_len, flags);
}

/// Compresses one 64-byte block across all `N` lanes with plain lane loops.
///
/// Every target without a hand-written kernel runs this, and it is the reference the vector
/// kernels are tested against.
#[inline(always)]
fn compress_block_portable<const N: usize>(
	cv: &mut [[u32; N]; 8],
	block: &[[u32; N]; 16],
	counter: u64,
	block_len: u32,
	flags: u32,
) {
	// Split the 64-bit counter into its two 32-bit words.
	let counter_lo = counter as u32;
	let counter_hi = (counter >> 32) as u32;

	// Initialize the 16-word state: CV, four IV words, counter, block length, flags.
	let mut v: [[u32; N]; 16] = [
		cv[0],
		cv[1],
		cv[2],
		cv[3],
		cv[4],
		cv[5],
		cv[6],
		cv[7],
		[IV[0]; N],
		[IV[1]; N],
		[IV[2]; N],
		[IV[3]; N],
		[counter_lo; N],
		[counter_hi; N],
		[block_len; N],
		[flags; N],
	];

	// Run 7 rounds, each reading the message through its own schedule row.
	// Nothing rewrites the message, so it stays exactly where the caller built it.
	round::<0, N>(&mut v, block);
	round::<1, N>(&mut v, block);
	round::<2, N>(&mut v, block);
	round::<3, N>(&mut v, block);
	round::<4, N>(&mut v, block);
	round::<5, N>(&mut v, block);
	round::<6, N>(&mut v, block);

	// Truncated output: h_i = v_i XOR v_{i+8}, feeding the next block or the final digest.
	for i in 0..8 {
		for lane in 0..N {
			cv[i][lane] = v[i][lane] ^ v[i + 8][lane];
		}
	}
}

/// Broadcasts the eight IV words across `N` lanes to seed a fresh chaining value.
#[inline(always)]
fn broadcast_iv<const N: usize>() -> [[u32; N]; 8] {
	array::from_fn(|w| [IV[w]; N])
}

/// Serializes one lane's eight-word chaining value into its 32-byte little-endian digest.
#[inline(always)]
fn serialize_cv_lane<const N: usize>(cv: &[[u32; N]; 8], lane: usize) -> [u8; OUT_LEN] {
	let mut digest = [0u8; OUT_LEN];
	for (w, chunk) in digest.chunks_exact_mut(4).enumerate() {
		chunk.copy_from_slice(&cv[w][lane].to_le_bytes());
	}
	digest
}

/// Portable multi-lane Blake3 leaf digest over `N` messages, hashed a block at a time.
///
/// One chunk, so the chaining value stays at counter 0 and needs no CV stack.
/// Each message is any length up to `CHUNK_LEN`; all `N` lanes must share that length.
///
/// A block is compressed only once the next block's first byte arrives, so the trailing block
/// is deferred to finalization, where it alone carries `CHUNK_END | ROOT`.
#[derive(Clone)]
pub struct PortableBlake3MultiDigest<const N: usize> {
	/// Running chaining value per lane; seeded from the IV.
	cv: [[u32; N]; 8],
	/// The current block being filled, one 64-byte buffer per lane.
	block: [[u8; BLOCK_LEN]; N],
	/// Bytes buffered in `block` so far, shared across lanes (all lanes share one length).
	block_len: usize,
	/// How many blocks have already been compressed into `cv`.
	blocks_compressed: usize,
}

impl<const N: usize> Default for PortableBlake3MultiDigest<N> {
	fn default() -> Self {
		// Fresh chaining value at the IV, empty block buffer, nothing compressed yet.
		Self {
			cv: broadcast_iv(),
			block: [[0u8; BLOCK_LEN]; N],
			block_len: 0,
			blocks_compressed: 0,
		}
	}
}

impl<const N: usize> PortableBlake3MultiDigest<N> {
	/// Compresses the buffered block as a full, non-final block, then empties the buffer.
	fn compress_full_block(&mut self) {
		// Only the very first block of the chunk carries CHUNK_START.
		let flags = if self.blocks_compressed == 0 {
			CHUNK_START
		} else {
			0
		};
		let m = load_block_words(&self.block);
		compress_block(&mut self.cv, &m, 0, BLOCK_LEN as u32, flags);
		self.blocks_compressed += 1;
		self.block_len = 0;
	}

	/// Compresses the trailing block as the chunk root and writes each lane's digest.
	///
	/// Runs on a copy of the state, so the hasher itself is left untouched for reset.
	fn write_root(&self, out: &mut [MaybeUninit<Output<blake3::Hasher>>; N]) {
		let mut cv = self.cv;
		let mut block = self.block;
		// Zero-pad the trailing block's unused tail, so padding never changes the digest.
		for lane in 0..N {
			block[lane][self.block_len..].fill(0);
		}
		// A single-block message has its only block be both the first and the root block.
		let start = if self.blocks_compressed == 0 {
			CHUNK_START
		} else {
			0
		};
		let m = load_block_words(&block);
		compress_block(&mut cv, &m, 0, self.block_len as u32, start | CHUNK_END | ROOT);

		// Serialize each lane's eight-word chaining value into its 32-byte digest.
		for lane in 0..N {
			out[lane].write(serialize_cv_lane(&cv, lane).into());
		}
	}
}

impl<const N: usize> MultiDigest<N> for PortableBlake3MultiDigest<N> {
	type Digest = blake3::Hasher;

	fn new() -> Self {
		Self::default()
	}

	fn update(&mut self, data: [&[u8]; N]) {
		// Per-lane read cursor into this call's input.
		let mut consumed = [0usize; N];
		loop {
			// Bytes still pending this call; all present lanes share one length, so the max drives.
			let remaining = (0..N)
				.map(|i| data[i].len() - consumed[i])
				.max()
				.unwrap_or(0);
			if remaining == 0 {
				break;
			}
			// A full buffer with more input to come is a non-final block: compress and empty it.
			if self.block_len == BLOCK_LEN {
				self.compress_full_block();
			}
			// Fill the block buffer up to one block from the pending input.
			let take = (BLOCK_LEN - self.block_len).min(remaining);
			for lane in 0..N {
				let avail = data[lane].len() - consumed[lane];
				let n = take.min(avail);
				self.block[lane][self.block_len..self.block_len + n]
					.copy_from_slice(&data[lane][consumed[lane]..consumed[lane] + n]);
				consumed[lane] += n;
			}
			self.block_len += take;
		}
	}

	fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		self.write_root(out);
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		self.write_root(out);
		self.reset();
	}

	fn reset(&mut self) {
		// Reseed the chaining value and forget the block progress; buffer bytes are overwritten
		// on the next update, and the trailing block's tail is zero-padded at finalization.
		self.cv = broadcast_iv();
		self.block_len = 0;
		self.blocks_compressed = 0;
	}

	fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		let mut hasher = Self::new();
		hasher.update(data);
		hasher.finalize_into(out);
	}
}

/// Parallel Blake3 leaf digest backed by the portable auto-vectorized kernel.
///
/// `LANES` is the batch width handed to the vectorizer.
/// Leaf size decides the path:
/// - Up to one 1024-byte chunk (any length): batched through the portable kernel.
/// - Larger (multi-chunk): hashed on its own by the scalar adapter, which walks the tree.
#[derive(Debug, Clone, Default)]
pub struct PortableBlake3ParallelDigest<const LANES: usize>;

impl<const LANES: usize> ParallelDigest for PortableBlake3ParallelDigest<LANES> {
	type Digest = blake3::Hasher;

	fn new() -> Self {
		Self
	}

	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		// Without a fixed leaf length a leaf could exceed one chunk, which the kernel cannot hash.
		// Fall back to the scalar adapter, which handles any length.
		ParallelDigestAdapter::<blake3::Hasher>::new().digest(source, out);
	}

	fn digest_with_const_len<I: IntoIterator<Item: FixedSizeSerializeBytes>>(
		&self,
		n_items_per_input: usize,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		// Every leaf serializes to the same fixed byte length.
		let leaf_len = n_items_per_input * I::Item::BYTE_SIZE;

		if leaf_len <= CHUNK_LEN {
			// One chunk or less, any block structure: batch it through the vectorized kernel.
			ParallelMultidigestImpl::<PortableBlake3MultiDigest<LANES>, LANES>::new()
				.digest(source, out);
		} else {
			// Multi-chunk leaves need the tree; hand them to the scalar adapter.
			ParallelDigestAdapter::<blake3::Hasher>::new().digest(source, out);
		}
	}
}

/// Folds up to `N` node pairs with a single batched Blake3 block compression.
///
/// Each pair is a 64-byte concatenation `left || right` of two 32-byte digests.
/// That 64-byte message is exactly one Blake3 block.
/// A one-block message makes its single block the first, last, and root block at once:
/// - counter   = 0       (a single chunk).
/// - block_len = 64      (a full block).
/// - flags     = CHUNK_START | CHUNK_END | ROOT.
///
/// Folds `out.len()` pairs, which must be at most `N`.
/// `inputs.len()` must be `2 * out.len()`.
/// A partial batch leaves the unused high lanes zero.
/// Their output is never read.
#[inline]
fn compress_node_pairs<const N: usize>(
	inputs: &[Output<blake3::Hasher>],
	out: &mut [MaybeUninit<Output<blake3::Hasher>>],
) {
	// Pack each pair into one 64-byte block: bytes 0..32 = left child, 32..64 = right child.
	let mut blocks = [[0u8; BLOCK_LEN]; N];
	for (lane, block) in blocks.iter_mut().enumerate().take(out.len()) {
		block[..OUT_LEN].copy_from_slice(inputs[2 * lane].as_slice());
		block[OUT_LEN..].copy_from_slice(inputs[2 * lane + 1].as_slice());
	}

	// One block compression seeded from the IV yields each pair's two-to-one digest.
	let m = load_block_words(&blocks);
	let mut cv = broadcast_iv::<N>();
	compress_block(&mut cv, &m, 0, BLOCK_LEN as u32, CHUNK_START | CHUNK_END | ROOT);

	for (lane, slot) in out.iter_mut().enumerate() {
		slot.write(serialize_cv_lane(&cv, lane).into());
	}
}

/// Parallel Blake3 two-to-one compression backed by the portable auto-vectorized kernel.
///
/// The Merkle inner-node counterpart to the leaf digest [`PortableBlake3ParallelDigest`].
/// Every parent folds a pair of 32-byte child digests as `blake3(left || right)`.
/// A batch of `LANES` node pairs is a fixed-length batched Blake3 over 64-byte messages.
/// Each batch runs through the shared block-compression core in one pass.
///
/// Output is bit-identical to the scalar [`Blake3Compression`], pinned to it in tests.
#[derive(Debug, Clone, Default)]
pub struct PortableBlake3ParallelCompression<const LANES: usize> {
	/// The scalar two-to-one function this batched path reproduces, exposed via `compression()`.
	compression: Blake3Compression,
}

impl<const LANES: usize> ParallelPseudoCompression<Output<blake3::Hasher>, 2>
	for PortableBlake3ParallelCompression<LANES>
{
	type Compression = Blake3Compression;

	fn compression(&self) -> &Self::Compression {
		&self.compression
	}

	fn parallel_compress(
		&self,
		inputs: &[Output<blake3::Hasher>],
		out: &mut [MaybeUninit<Output<blake3::Hasher>>],
	) {
		assert_eq!(inputs.len(), 2 * out.len(), "Input length must be 2 * output length");

		// Fold `LANES` pairs per batch.
		// A shorter trailing batch is fine: the kernel only reads its valid lanes.
		inputs
			.par_chunks(2 * LANES)
			.zip(out.par_chunks_mut(LANES))
			.for_each(|(in_batch, out_batch)| compress_node_pairs::<LANES>(in_batch, out_batch));
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_utils::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
	use proptest::prelude::*;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::compress::CompressionFunction;

	/// Folds `pairs` through the `N`-lane portable compression.
	/// Pins every output bit-identical to the scalar [`Blake3Compression`].
	fn check_parallel_compression<const N: usize>(pairs: &[[[u8; OUT_LEN]; 2]]) {
		// Flatten pairs to the `[left_0, right_0, left_1, ...]` layout the compressor reads.
		let inputs: Vec<Output<blake3::Hasher>> = pairs
			.iter()
			.flat_map(|[l, r]| [(*l).into(), (*r).into()])
			.collect();
		let mut out = repeat_with(MaybeUninit::<Output<blake3::Hasher>>::uninit)
			.take(pairs.len())
			.collect::<Vec<_>>();

		PortableBlake3ParallelCompression::<N>::default().parallel_compress(&inputs, &mut out);

		// Invariant: each batched lane equals the scalar two-to-one of the same pair.
		for (slot, [l, r]) in out.into_iter().zip(pairs) {
			let expected = Blake3Compression.compress([(*l).into(), (*r).into()]);
			assert_eq!(unsafe { slot.assume_init() }.as_slice(), expected.as_slice());
		}
	}

	#[test]
	fn test_parallel_compression_boundaries() {
		// Extreme digests exercise all-zero and all-ones message blocks in every left/right slot.
		let zero = [0u8; OUT_LEN];
		let ones = [0xffu8; OUT_LEN];
		check_parallel_compression::<16>(&[[zero, zero], [ones, ones], [zero, ones], [ones, zero]]);

		// Counts straddling the 16-lane batch boundary: empty, single, full, full+1, multi+partial.
		let mut rng = StdRng::seed_from_u64(7);
		for count in [0usize, 1, 15, 16, 17, 33] {
			let pairs: Vec<[[u8; OUT_LEN]; 2]> = (0..count)
				.map(|_| {
					let mut pair = [[0u8; OUT_LEN]; 2];
					rng.fill_bytes(&mut pair[0]);
					rng.fill_bytes(&mut pair[1]);
					pair
				})
				.collect();
			check_parallel_compression::<16>(&pairs);
		}
	}

	proptest! {
		#[test]
		fn parallel_compression_matches_scalar(
			pairs in prop::collection::vec(
				(prop::array::uniform32(any::<u8>()), prop::array::uniform32(any::<u8>())),
				0..40usize,
			),
		) {
			// The batch path is bit-identical to the scalar reference, at every vectorizer width.
			let pairs: Vec<[[u8; OUT_LEN]; 2]> = pairs.into_iter().map(|(l, r)| [l, r]).collect();
			check_parallel_compression::<4>(&pairs);
			check_parallel_compression::<8>(&pairs);
			check_parallel_compression::<16>(&pairs);
		}
	}

	/// Transposes one random block through both loaders and pins the vector words to the byte-wise
	/// words.
	///
	/// # Arguments
	///
	/// * `rng` - source of the random block bytes.
	#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
	fn check_avx512_transpose<const N: usize>(rng: &mut StdRng) {
		use rand::RngExt;

		// Every lane gets its own random bytes.
		// Sharing bytes across lanes would hide a network that reads the wrong row.
		let block: [[u8; BLOCK_LEN]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		// Reference: the byte-wise loader every target without a transpose runs.
		let want = load_block_words_portable(&block);

		// Candidate: the shuffle network, over the same bytes.
		let got = super::avx512::load_block_words(&block);

		// A transpose only permutes words, so the two must agree on all 16 words of all N lanes.
		assert_eq!(got, want, "vector transpose diverged from the byte-wise loader at {N} lanes");
	}

	#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
	#[test]
	fn test_avx512_transpose_matches_portable() {
		let mut rng = StdRng::seed_from_u64(13);

		// A misplaced row shows up only when the bytes differ, so repeat over fresh random blocks.
		for _ in 0..64 {
			check_avx512_transpose::<16>(&mut rng);
		}
	}

	#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
	#[test]
	fn test_avx512_transpose_places_every_word() {
		// A counting block makes each word its own (lane, word) coordinate.
		// Any swapped row or lane then shows up as a wrong coordinate, not just wrong bytes.
		let block: [[u8; BLOCK_LEN]; 16] = array::from_fn(|lane| {
			array::from_fn(|byte| {
				let word = byte / 4;
				// Word `w` of lane `l` is the value `l * 16 + w`, little-endian.
				match byte % 4 {
					0 => (lane * 16 + word) as u8,
					_ => 0,
				}
			})
		});

		let m = super::avx512::load_block_words(&block);

		// Word `w` of lane `l` must land at `m[w][l]`.
		for lane in 0..16 {
			for word in 0..16 {
				assert_eq!(
					m[word][lane],
					(lane * 16 + word) as u32,
					"word {word} of lane {lane} landed wrongly"
				);
			}
		}
	}

	#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
	proptest! {
		#[test]
		fn avx512_transpose_matches_portable_proptest(seed in any::<u64>()) {
			// The fixed cases above pin the layout; this sweeps arbitrary bytes.
			let mut rng = StdRng::seed_from_u64(seed);
			check_avx512_transpose::<16>(&mut rng);
		}
	}

	/// Compresses one block through both cores and pins the vector words to the lane-loop words.
	///
	/// # Arguments
	///
	/// * `rng` - source of the random chaining value and message.
	/// * `counter` - chunk counter to place in the state.
	/// * `block_len` - message byte count to place in the state.
	/// * `flags` - domain-separation flags to place in the state.
	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	fn check_neon_core<const N: usize>(rng: &mut StdRng, counter: u64, block_len: u32, flags: u32) {
		use rand::RngExt;

		// Every lane gets its own random words.
		// Sharing a value across lanes would hide a kernel that reads the wrong vector.
		let cv_in: [[u32; N]; 8] = array::from_fn(|_| array::from_fn(|_| rng.random()));
		let block: [[u32; N]; 16] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		// Reference: the lane loops every target without a vector kernel runs.
		let mut want = cv_in;
		compress_block_portable(&mut want, &block, counter, block_len, flags);

		// Candidate: the vector kernel, started from the same chaining value.
		let mut got = cv_in;
		super::neon::compress_block(&mut got, &block, counter, block_len, flags);

		// Compression is bit-exact, so the two must agree on all 8 words of all N lanes.
		assert_eq!(got, want, "vector kernel diverged from the lane loops at {N} lanes");
	}

	/// Transposes one random block set through both loaders and pins the words together.
	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	fn check_neon_transpose<const N: usize>(rng: &mut StdRng) {
		// Fresh random bytes per lane, so no two lanes hold the same word by accident.
		let mut blocks = [[0u8; BLOCK_LEN]; N];
		for block in blocks.iter_mut() {
			rng.fill_bytes(block);
		}

		// A transpose only permutes words, so the two loaders must agree on all 16 rows.
		assert_eq!(
			super::neon::load_block_words(&blocks),
			load_block_words_portable(&blocks),
			"the shuffle network diverged from the byte-wise loader at {N} lanes"
		);
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	#[test]
	fn test_neon_transpose_matches_portable() {
		// Invariant: a word's output row is its word index, and its output lane is its block index.
		//
		// Fixture: word `w` of lane `l` carries the value `l * 16 + w`, so all 16 * N are distinct.
		//
		//     lane 0 block:  [  0,  1,  2, ...,  15 ]
		//     lane 1 block:  [ 16, 17, 18, ...,  31 ]
		//     ...
		//     output row w:  [  w, 16 + w, 32 + w, ... ]
		//
		// Distinct values are what makes a swapped row or lane show up as a wrong value.
		let mut blocks = [[0u8; BLOCK_LEN]; 16];
		for (lane, block) in blocks.iter_mut().enumerate() {
			for (w, word) in block.chunks_exact_mut(4).enumerate() {
				word.copy_from_slice(&((lane * 16 + w) as u32).to_le_bytes());
			}
		}

		// Every one of the 16 rows must read back its own coordinate, at every lane.
		let m = super::neon::load_block_words(&blocks);
		for (w, row) in m.iter().enumerate() {
			for (lane, got) in row.iter().enumerate() {
				assert_eq!(*got, (lane * 16 + w) as u32, "row {w}, lane {lane}");
			}
		}

		// Random blocks at every width the network claims, from one square up to four.
		let mut rng = StdRng::seed_from_u64(23);
		check_neon_transpose::<4>(&mut rng);
		check_neon_transpose::<8>(&mut rng);
		check_neon_transpose::<12>(&mut rng);
		check_neon_transpose::<16>(&mut rng);
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	proptest! {
		#[test]
		fn neon_transpose_matches_portable_proptest(seed in any::<u64>()) {
			// The fixed block above pins the coordinates.
			// This sweeps arbitrary bytes at every width instead.
			let mut rng = StdRng::seed_from_u64(seed);
			check_neon_transpose::<4>(&mut rng);
			check_neon_transpose::<8>(&mut rng);
			check_neon_transpose::<12>(&mut rng);
			check_neon_transpose::<16>(&mut rng);
		}
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	#[test]
	fn test_neon_core_matches_portable() {
		let mut rng = StdRng::seed_from_u64(11);

		// The last four state words are the ones a kernel is most likely to place wrongly.
		// So cover the combinations a real chunk produces, plus the extremes of each field.
		//
		//     (counter, block_len, flags)
		let cases = [
			// Interior block of a multi-block chunk: no flags, full 64 bytes.
			(0u64, 64u32, 0u32),
			// First block of a chunk.
			(0, 64, CHUNK_START),
			// Last block of a chunk, and the last of the whole tree.
			(0, 64, CHUNK_END | ROOT),
			// A chunk that is one block long, so it carries every flag at once.
			(0, 64, CHUNK_START | CHUNK_END | ROOT),
			// A one-byte message: the block is almost entirely zero padding.
			(0, 1, CHUNK_START | CHUNK_END | ROOT),
			// The empty message, the shortest input Blake3 accepts.
			(0, 0, CHUNK_START | CHUNK_END | ROOT),
			// Both counter halves set, which catches a kernel that drops the high half.
			(u64::MAX, 64, CHUNK_END),
			// A counter whose low half is zero, which catches the two halves being swapped.
			(1 << 32, 63, CHUNK_START),
		];

		// Check every lane count the kernel claims, from one vector group up to four.
		for (counter, block_len, flags) in cases {
			check_neon_core::<4>(&mut rng, counter, block_len, flags);
			check_neon_core::<8>(&mut rng, counter, block_len, flags);
			check_neon_core::<12>(&mut rng, counter, block_len, flags);
			check_neon_core::<16>(&mut rng, counter, block_len, flags);
		}
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	proptest! {
		#[test]
		fn neon_core_matches_portable_proptest(
			seed in any::<u64>(),
			counter in any::<u64>(),
			// A block never carries more than the 64 bytes it holds.
			block_len in 0..=64u32,
			// Blake3 defines flags in the low byte only.
			flags in any::<u8>(),
		) {
			// The fixed cases above pin the boundaries; this sweeps arbitrary state at every width.
			let mut rng = StdRng::seed_from_u64(seed);
			check_neon_core::<4>(&mut rng, counter, block_len, flags as u32);
			check_neon_core::<8>(&mut rng, counter, block_len, flags as u32);
			check_neon_core::<12>(&mut rng, counter, block_len, flags as u32);
			check_neon_core::<16>(&mut rng, counter, block_len, flags as u32);
		}
	}

	/// Runs `N` equal-length messages of `len` bytes through the portable batch and pins each lane
	/// to the scalar reference.
	fn check_portable_batch<const N: usize>(rng: &mut StdRng, len: usize) {
		// Fresh random bytes per lane, so lanes don't share a digest by accident.
		let messages: [Vec<u8>; N] = array::from_fn(|_| {
			let mut m = vec![0u8; len];
			rng.fill_bytes(&mut m);
			m
		});
		let refs: [&[u8]; N] = array::from_fn(|i| messages[i].as_slice());
		let mut out = array::from_fn::<_, N, _>(|_| MaybeUninit::uninit());
		PortableBlake3MultiDigest::<N>::digest(refs, &mut out);

		// Each lane's output must equal the single-message reference hash of that lane.
		for (o, message) in out.iter().zip(messages.iter()) {
			let got = unsafe { o.assume_init_ref() };
			assert_eq!(got.as_slice(), blake3::hash(message).as_bytes(), "len = {len}, N = {N}");
		}
	}

	#[test]
	fn test_portable_lengths_match_reference() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the portable kernel reproduces blake3::hash for any single-chunk length.
		// Lengths cover every block-structure case within one chunk:
		// - 0             : the lone empty block.
		// - 1, 31, 63     : a single sub-block, no full blocks.
		// - 64, 128, 1024 : exact block multiples.
		// - 65, 100, 1000 : leading full blocks plus a partial tail.
		// Three lane widths per length: 4 (NEON), 8, and 16 (the throughput sweet spot).
		for len in [0, 1, 31, 63, 64, 65, 100, 127, 128, 1000, 1024] {
			check_portable_batch::<4>(&mut rng, len);
			check_portable_batch::<8>(&mut rng, len);
			check_portable_batch::<16>(&mut rng, len);
		}
	}

	#[test]
	fn test_portable_chained_update() {
		let mut rng = StdRng::seed_from_u64(2);
		// Four 200-byte messages: three full blocks plus a 8-byte partial tail.
		let messages: [Vec<u8>; 4] = array::from_fn(|_| {
			let mut m = vec![0u8; 200];
			rng.fill_bytes(&mut m);
			m
		});

		// Invariant: a message split across two updates hashes the same as one update of the whole.
		// The 50/150 split lands mid-block, exercising the buffer-fill and deferred-compress paths.
		let mut hasher = PortableBlake3MultiDigest::<4>::new();
		hasher.update(array::from_fn(|i| &messages[i][..50]));
		hasher.update(array::from_fn(|i| &messages[i][50..]));
		let mut out = array::from_fn::<_, 4, _>(|_| MaybeUninit::uninit());
		hasher.finalize_into(&mut out);

		for (o, message) in out.iter().zip(messages.iter()) {
			assert_eq!(unsafe { o.assume_init_ref() }.as_slice(), blake3::hash(message).as_bytes());
		}
	}

	#[test]
	fn test_portable_routing_matches_reference() {
		let mut rng = StdRng::seed_from_u64(3);
		// Build 50 leaves of `leaf_len` bytes each, fed as u8 items (BYTE_SIZE = 1).
		let mut check = |leaf_len: usize| {
			let leaves: Vec<Vec<u8>> = (0..50)
				.map(|_| {
					let mut m = vec![0u8; leaf_len];
					rng.fill_bytes(&mut m);
					m
				})
				.collect();
			let digest = PortableBlake3ParallelDigest::<8>::new();
			let mut results = repeat_with(MaybeUninit::<Output<blake3::Hasher>>::uninit)
				.take(50)
				.collect::<Vec<_>>();
			digest.digest_with_const_len(
				leaf_len,
				leaves.par_iter().map(|leaf| leaf.iter().copied()),
				&mut results,
			);
			for (result, leaf) in results.into_iter().zip(&leaves) {
				let got = unsafe { result.assume_init() };
				assert_eq!(got.as_slice(), blake3::hash(leaf).as_bytes(), "leaf_len {leaf_len}");
			}
		};

		// Invariant: every leaf size reproduces the reference, on the batch or the adapter route.
		// - 0, 1, 63      : sub-block             -> portable batch.
		// - 65, 100, 1000 : partial trailing block -> portable batch.
		// - 64, 1024      : whole blocks           -> portable batch.
		// - 1025, 2048    : multi-chunk (> 1024)   -> scalar adapter.
		for leaf_len in [0, 1, 63, 64, 65, 100, 1000, 1024, 1025, 2048] {
			check(leaf_len);
		}
	}
}
