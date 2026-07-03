// Copyright 2026 The Binius Developers

//! Experimental portable, auto-vectorized Blake3 multi-lane leaf hasher.
//!
//! An alternative to driving the `blake3` crate's hand-written SIMD kernel.
//! The bet: LLVM auto-vectorizes plain lane loops into whatever the target has.
//!
//! - Each of the 16 compression-state words is held as `[u32; N]`, one lane per message.
//! - Every step is a fixed-width `0..N` loop of plain scalar `u32` arithmetic.
//! - No intrinsics, no `unsafe`, no per-target code.
//!
//! Lanes the vectorizer is expected to fill, per target:
//! - NEON (128-bit) on ARM64 -> 4 lanes per vector.
//! - AVX2 / AVX-512 on x86 -> 8 / 16 lanes per vector.
//! - SVE2 on capable ARM64 -> width-agnostic vectors.
//!
//! Output is bit-identical to `blake3::hash`, pinned to the reference in tests.
//! Scope: any message up to one 1024-byte chunk, including sub-block and partial-block leaves.

use std::{array, mem::MaybeUninit};

use binius_utils::{FixedSizeSerializeBytes, SerializeBytes, rayon::iter::IndexedParallelIterator};
use blake3::{BLOCK_LEN, CHUNK_LEN, OUT_LEN};
use digest::Output;

use super::parallel_digest::{
	MultiDigest, ParallelDigest, ParallelDigestAdapter, ParallelMultidigestImpl,
};

/// Blake3 domain-separation flag marking the first block of a chunk.
const CHUNK_START: u32 = 1 << 0;

/// Blake3 domain-separation flag marking the last block of a chunk.
const CHUNK_END: u32 = 1 << 1;

/// Blake3 domain-separation flag marking the last block of the whole tree.
const ROOT: u32 = 1 << 3;

/// Blake3 initial chaining value: the eight IV words, identical to the SHA-256 IV.
const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Blake3 message permutation applied between rounds.
///
/// The single fixed schedule from section 2.2 of the Blake3 spec, Table 2.
const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// The 7-round count of the Blake3 keyed permutation.
const N_ROUNDS: usize = 7;

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

/// Applies one full Blake3 round: four column mixes, then four diagonal mixes.
///
/// Message words are consumed in order `m[0..16]`, two per quarter-round.
#[inline(always)]
fn round<const N: usize>(v: &mut [[u32; N]; 16], m: &[[u32; N]; 16]) {
	// Columns.
	quarter_round(v, 0, 4, 8, 12, &m[0], &m[1]);
	quarter_round(v, 1, 5, 9, 13, &m[2], &m[3]);
	quarter_round(v, 2, 6, 10, 14, &m[4], &m[5]);
	quarter_round(v, 3, 7, 11, 15, &m[6], &m[7]);
	// Diagonals.
	quarter_round(v, 0, 5, 10, 15, &m[8], &m[9]);
	quarter_round(v, 1, 6, 11, 12, &m[10], &m[11]);
	quarter_round(v, 2, 7, 8, 13, &m[12], &m[13]);
	quarter_round(v, 3, 4, 9, 14, &m[14], &m[15]);
}

/// Permutes the message words in place for the next round.
#[inline(always)]
fn permute<const N: usize>(m: &mut [[u32; N]; 16]) {
	// The permutation reads each slot from its source, so build into a fresh array.
	let permuted: [[u32; N]; 16] = array::from_fn(|i| m[MSG_PERMUTATION[i]]);
	*m = permuted;
}

/// Loads one 64-byte block per lane into 16 little-endian message words.
#[inline(always)]
fn load_block_words<const N: usize>(block: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
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

	// Run 7 rounds; permute the message between all but the last.
	let mut m = *block;
	for r in 0..N_ROUNDS {
		round(&mut v, &m);
		if r < N_ROUNDS - 1 {
			permute(&mut m);
		}
	}

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
			let mut digest = [0u8; OUT_LEN];
			for (w, chunk) in digest.chunks_exact_mut(4).enumerate() {
				chunk.copy_from_slice(&cv[w][lane].to_le_bytes());
			}
			out[lane].write(digest.into());
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

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_utils::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;

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
