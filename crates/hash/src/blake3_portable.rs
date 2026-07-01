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
//! Scope: a single chunk of whole 64-byte blocks (`M <= 1024`), like the SIMD leaf digest.

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

/// Hashes `N` single-chunk messages of `M` bytes each, writing one 32-byte digest per lane.
///
/// `M` is a positive multiple of the 64-byte block, no larger than one 1024-byte chunk.
/// Every lane is its own chunk at counter 0.
///
/// The first block flags `CHUNK_START`.
/// The last block flags `CHUNK_END | ROOT`, so its chaining value is the lane's root digest.
fn hash_many_portable<const M: usize, const N: usize>(
	inputs: &[&[u8; M]; N],
	out: &mut [MaybeUninit<Output<blake3::Hasher>>; N],
) {
	// `M` is a whole number of blocks by construction.
	let n_blocks = M / BLOCK_LEN;
	// Every lane's chaining value starts at the IV.
	let mut cv: [[u32; N]; 8] = array::from_fn(|w| [IV[w]; N]);

	for block_idx in 0..n_blocks {
		// Load this block of every lane into 16 little-endian words.
		let base = block_idx * BLOCK_LEN;
		let mut m = [[0u32; N]; 16];
		for lane in 0..N {
			for (w, slot) in m.iter_mut().enumerate() {
				let off = base + w * 4;
				slot[lane] = u32::from_le_bytes([
					inputs[lane][off],
					inputs[lane][off + 1],
					inputs[lane][off + 2],
					inputs[lane][off + 3],
				]);
			}
		}

		// The single chunk is the tree root, so its last block carries CHUNK_END | ROOT.
		let mut flags = 0;
		if block_idx == 0 {
			flags |= CHUNK_START;
		}
		if block_idx == n_blocks - 1 {
			flags |= CHUNK_END | ROOT;
		}
		compress_block(&mut cv, &m, 0, BLOCK_LEN as u32, flags);
	}

	// Serialize each lane's eight-word chaining value into its 32-byte digest.
	for lane in 0..N {
		let mut digest = [0u8; OUT_LEN];
		for (w, chunk) in digest.chunks_exact_mut(4).enumerate() {
			chunk.copy_from_slice(&cv[w][lane].to_le_bytes());
		}
		out[lane].write(digest.into());
	}
}

/// Maps a runtime block count in `1..=16` to a portable batch of compile-time leaf length.
macro_rules! dispatch_portable_blocks {
	($n_blocks:expr, $lanes:ident, $source:expr, $out:expr) => {{
		macro_rules! arm {
			($len:literal) => {
				ParallelMultidigestImpl::<PortableBlake3MultiDigest<$len, $lanes>, $lanes>::new()
					.digest($source, $out)
			};
		}
		match $n_blocks {
			1 => arm!(64),
			2 => arm!(128),
			3 => arm!(192),
			4 => arm!(256),
			5 => arm!(320),
			6 => arm!(384),
			7 => arm!(448),
			8 => arm!(512),
			9 => arm!(576),
			10 => arm!(640),
			11 => arm!(704),
			12 => arm!(768),
			13 => arm!(832),
			14 => arm!(896),
			15 => arm!(960),
			16 => arm!(1024),
			_ => unreachable!("a single chunk has at most CHUNK_LEN / BLOCK_LEN = 16 full blocks"),
		}
	}};
}

/// Portable multi-lane Blake3 digest over `N` messages of `M` bytes each.
///
/// The compression runs in pure Rust `[u32; N]` lane loops, left for LLVM to auto-vectorize.
#[derive(Clone)]
pub struct PortableBlake3MultiDigest<const M: usize, const N: usize> {
	/// One fixed-size `M`-byte buffer per lane, holding that lane's message until finalization.
	buffers: [[u8; M]; N],
	/// Per-lane write cursor: how many bytes each lane's buffer currently holds.
	filled: [usize; N],
}

impl<const M: usize, const N: usize> Default for PortableBlake3MultiDigest<M, N> {
	fn default() -> Self {
		// Every lane starts as a zeroed buffer with an empty cursor.
		Self {
			buffers: array::from_fn(|_| [0u8; M]),
			filled: [0; N],
		}
	}
}

impl<const M: usize, const N: usize> MultiDigest<N> for PortableBlake3MultiDigest<M, N> {
	type Digest = blake3::Hasher;

	fn new() -> Self {
		Self::default()
	}

	fn update(&mut self, data: [&[u8]; N]) {
		// Append each lane's new bytes at that lane's cursor.
		for ((buf, filled), chunk) in self
			.buffers
			.iter_mut()
			.zip(self.filled.iter_mut())
			.zip(data)
		{
			buf[*filled..*filled + chunk.len()].copy_from_slice(chunk);
			*filled += chunk.len();
		}
	}

	fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		// View each lane's buffer as a fixed `M`-byte block-aligned input.
		let inputs: [&[u8; M]; N] = array::from_fn(|i| &self.buffers[i]);
		hash_many_portable(&inputs, out);
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		let inputs: [&[u8; M]; N] = array::from_fn(|i| &self.buffers[i]);
		hash_many_portable(&inputs, out);
		self.reset();
	}

	fn reset(&mut self) {
		// Rewind every lane's cursor; used lanes are refilled to `M` before the next hash.
		self.filled = [0; N];
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
/// Leaf size routing matches the SIMD leaf digest:
/// - A whole number of 64-byte blocks, up to one chunk: batched through the portable kernel.
/// - Anything else: hashed on its own by the scalar adapter.
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
		// Without a fixed leaf length the batch width is unknown; use the scalar adapter.
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
		// The kernel needs a positive whole number of 64-byte blocks within one chunk.
		let n_blocks = leaf_len / BLOCK_LEN;

		if leaf_len.is_multiple_of(BLOCK_LEN) && (1..=CHUNK_LEN / BLOCK_LEN).contains(&n_blocks) {
			// Turn the runtime block count into a compile-time `M`, then batch it.
			dispatch_portable_blocks!(n_blocks, LANES, source, out);
		} else {
			// Sub-block, non-block-multiple, or multi-chunk leaves have no batchable shape.
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

	/// Runs `N` equal-length `M`-byte messages through the portable batch and pins each lane to
	/// the scalar reference.
	fn check_portable_batch<const M: usize, const N: usize>(rng: &mut StdRng) {
		// Fresh random bytes per lane, so lanes don't share a digest by accident.
		let messages: [Vec<u8>; N] = array::from_fn(|_| {
			let mut m = vec![0u8; M];
			rng.fill_bytes(&mut m);
			m
		});
		let refs: [&[u8]; N] = array::from_fn(|i| messages[i].as_slice());
		let mut out = array::from_fn::<_, N, _>(|_| MaybeUninit::uninit());
		PortableBlake3MultiDigest::<M, N>::digest(refs, &mut out);

		// Each lane's output must equal the single-message reference hash of that lane.
		for (o, message) in out.iter().zip(messages.iter()) {
			let got = unsafe { o.assume_init_ref() };
			assert_eq!(got.as_slice(), blake3::hash(message).as_bytes(), "M = {M}, N = {N}");
		}
	}

	#[test]
	fn test_portable_block_multiples_match_reference() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the portable kernel reproduces blake3::hash for whole-block single chunks.
		// Each M is a different block count; each N is a different vectorizer width.
		check_portable_batch::<64, 4>(&mut rng);
		check_portable_batch::<128, 4>(&mut rng);
		check_portable_batch::<192, 4>(&mut rng);
		check_portable_batch::<1024, 4>(&mut rng);
		check_portable_batch::<64, 8>(&mut rng);
		check_portable_batch::<512, 8>(&mut rng);
		check_portable_batch::<64, 16>(&mut rng);
		check_portable_batch::<1024, 16>(&mut rng);
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
		// - 0, 1, 63    : sub-block            -> adapter.
		// - 65, 100     : non-block-multiple   -> adapter.
		// - 2048        : multi-chunk          -> adapter.
		// - 64, 256     : whole blocks         -> portable batch.
		for leaf_len in [0, 1, 63, 64, 65, 100, 256, 2048] {
			check(leaf_len);
		}
	}
}
