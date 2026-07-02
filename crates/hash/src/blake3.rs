// Copyright 2026 The Binius Developers

//! Blake3 hash and compression functions for use in Merkle tree constructions.

use std::{array, mem::MaybeUninit};

use binius_utils::{FixedSizeSerializeBytes, SerializeBytes, rayon::iter::IndexedParallelIterator};
use blake3::{BLOCK_LEN, CHUNK_LEN, IncrementCounter, OUT_LEN, platform::Platform};
use digest::Output;

use super::{
	binary_merkle_tree::HashSuite,
	compress::CompressionFunction,
	parallel_compression::ParallelCompressionAdaptor,
	parallel_digest::{
		MultiDigest, ParallelDigest, ParallelDigestAdapter, ParallelMultidigestImpl,
	},
};

/// A two-to-one compression function that hashes the concatenation of its inputs with Blake3.
#[derive(Debug, Clone, Default)]
pub struct Blake3Compression;

impl CompressionFunction<Output<blake3::Hasher>, 2> for Blake3Compression {
	fn compress(&self, input: [Output<blake3::Hasher>; 2]) -> Output<blake3::Hasher> {
		let mut hasher = blake3::Hasher::new();
		hasher.update(input[0].as_slice());
		hasher.update(input[1].as_slice());
		(*hasher.finalize().as_bytes()).into()
	}
}

/// Blake3 [`HashSuite`]: Blake3 leaves and a Blake3 compression function for inner nodes.
#[derive(Debug, Clone, Default)]
pub struct Blake3HashSuite;

impl HashSuite for Blake3HashSuite {
	type LeafHash = blake3::Hasher;
	type Compression = Blake3Compression;
	type ParLeafHash = Blake3ParallelDigest;
	type ParCompression = ParallelCompressionAdaptor<Blake3Compression>;
}

/// Blake3 domain-separation flag marking the first block of a chunk.
///
/// A flag tags each compression with its role in the tree.
/// This stops a one-block message from colliding with an interior parent node.
///
/// The crate keeps these flag values private, so they are mirrored here.
/// - Values: Table 3, section 2.2 of the [Blake3 spec](https://github.com/BLAKE3-team/BLAKE3-specs).
/// - Rationale: section 7.7.
const CHUNK_START: u8 = 1 << 0;

/// Blake3 domain-separation flag marking the last block of a chunk.
const CHUNK_END: u8 = 1 << 1;

/// Blake3 domain-separation flag marking the last block of the whole tree.
const ROOT: u8 = 1 << 3;

/// Blake3 initial chaining value: the eight IV words, identical to the SHA-256 IV.
///
/// A non-keyed hash starts every chunk from these words.
/// The crate keeps its IV constant private, so the value is mirrored here.
/// - Values: Table 1, section 2.2 of the [Blake3 spec](https://github.com/BLAKE3-team/BLAKE3-specs).
const BLAKE3_IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// How many independent messages are hashed per SIMD batch.
///
/// This matches the widest vector the target supports:
/// - 16 lanes on AVX-512.
/// - 8 lanes on AVX2.
/// - 4 lanes on NEON.
///
/// At this width a batch fills the vector with no leftover lanes.
const SIMD_DEGREE: usize = blake3::platform::MAX_SIMD_DEGREE;

/// Maps a runtime block count in `1..=16` to a batch of compile-time leaf length.
///
/// The batch fixes its per-lane byte length as a const generic, so that length must be a literal.
/// Each block count therefore gets its own arm.
///
/// One chunk holds at most `CHUNK_LEN / BLOCK_LEN = 16` whole blocks, giving sixteen arms.
///
/// `source` is moved and `out` reborrowed inside a single, mutually exclusive match arm.
macro_rules! dispatch_leaf_blocks {
	($n_blocks:expr, $source:expr, $out:expr) => {{
		macro_rules! arm {
			($len:literal) => {
				ParallelMultidigestImpl::<Blake3MultiDigest<$len, SIMD_DEGREE>, SIMD_DEGREE>::new()
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

/// The parallel digest used for Blake3 leaf hashing.
///
/// Leaf size decides the path:
/// - A whole number of 64-byte blocks, up to one 1024-byte chunk: batched through the SIMD kernel.
/// - Anything else: hashed on its own by the scalar adapter.
///
/// The SIMD kernel only accepts messages whose length is an exact multiple of the 64-byte block.
/// So a sub-block, non-block-multiple, or multi-chunk leaf has no batchable shape.
///
/// The scalar adapter still uses SIMD within a single message.
#[derive(Debug, Clone, Default)]
pub struct Blake3ParallelDigest;

impl ParallelDigest for Blake3ParallelDigest {
	type Digest = blake3::Hasher;

	fn new() -> Self {
		Self
	}

	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		// Without a fixed leaf length the SIMD lane width (the compile-time `M`) is unknown.
		// Hash each leaf on its own with the scalar adapter.
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
		// The SIMD kernel needs a leaf that is a positive whole number of 64-byte blocks.
		// One chunk holds at most CHUNK_LEN / BLOCK_LEN = 16 blocks.
		let n_blocks = leaf_len / BLOCK_LEN;

		if leaf_len.is_multiple_of(BLOCK_LEN) && (1..=CHUNK_LEN / BLOCK_LEN).contains(&n_blocks) {
			// Turn the runtime block count into a compile-time `M`, then batch through the kernel.
			dispatch_leaf_blocks!(n_blocks, source, out);
		} else {
			// Sub-block, non-block-multiple, or multi-chunk leaves have no batchable shape.
			ParallelDigestAdapter::<blake3::Hasher>::new().digest(source, out);
		}
	}
}

/// Computes `N` independent Blake3 digests of `M`-byte messages with the crate's SIMD kernel.
///
/// `M` is a positive multiple of the 64-byte block length, no larger than one 1024-byte chunk.
/// This is the shape of a batch of Merkle leaves: each leaf is a single chunk of whole blocks.
///
/// One SIMD pass compresses all `N` lanes together, each its own chunk starting from the IV.
/// It emits every lane's root digest directly:
///
/// ```text
/// lane 0:  [ block | ... | block ]  ─┐
/// lane 1:  [ block | ... | block ]   ├─ one SIMD pass → N root digests
/// ...                                │
/// lane N-1:[ block | ... | block ]  ─┘
/// ```
#[derive(Clone)]
pub struct Blake3MultiDigest<const M: usize, const N: usize> {
	/// One fixed-size `M`-byte buffer per lane, holding that lane's message until finalization.
	///
	/// The hashing interface is streaming, so bytes arrive across successive updates.
	/// A fixed array avoids per-lane allocation, and already matches the kernel's input shape.
	buffers: [[u8; M]; N],
	/// Per-lane write cursor: how many bytes each lane's buffer currently holds.
	filled: [usize; N],
}

impl<const M: usize, const N: usize> Default for Blake3MultiDigest<M, N> {
	fn default() -> Self {
		// Every lane starts as a zeroed buffer with an empty cursor.
		Self {
			buffers: array::from_fn(|_| [0u8; M]),
			filled: [0; N],
		}
	}
}

impl<const M: usize, const N: usize> Blake3MultiDigest<M, N> {
	/// Hashes the `N` `M`-byte lane buffers into `out` with one SIMD pass.
	fn compute(&self, out: &mut [MaybeUninit<Output<blake3::Hasher>>; N]) {
		// Pick the best SIMD backend the CPU supports (AVX-512, AVX2, NEON, ...).
		let platform = Platform::detect();
		// Scratch for one 32-byte chaining value per lane.
		let mut cv_bytes = [[0u8; OUT_LEN]; N];
		// Each lane's buffer is already a fixed `M`-byte block-aligned input.
		let inputs: [&[u8; M]; N] = array::from_fn(|i| &self.buffers[i]);

		// Each lane is its own single chunk at counter 0, so the counter never increments.
		// The last block carries CHUNK_END | ROOT, so this one pass emits each lane's root digest.
		platform.hash_many::<M>(
			&inputs,
			&BLAKE3_IV,
			0,
			IncrementCounter::No,
			0,
			CHUNK_START,
			CHUNK_END | ROOT,
			cv_bytes.as_flattened_mut(),
		);

		// Each lane's chaining value is already its 32-byte root digest.
		for (b, o) in cv_bytes.iter().zip(out.iter_mut()) {
			o.write((*b).into());
		}
	}
}

impl<const M: usize, const N: usize> MultiDigest<N> for Blake3MultiDigest<M, N> {
	type Digest = blake3::Hasher;

	fn new() -> Self {
		// All lanes start with zeroed buffers and empty cursors.
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
		// Hash the buffered messages; the hasher is consumed.
		self.compute(out);
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		// Hash the buffered messages, then rewind so the hasher can be reused.
		self.compute(out);
		self.reset();
	}

	fn reset(&mut self) {
		// Rewind every lane's cursor.
		// The const-len contract refills each used lane to `M`, overwriting any stale bytes.
		self.filled = [0; N];
	}

	fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		// One-shot path: start empty, absorb the data, finalize.
		let mut hasher = Self::new();
		hasher.update(data);
		hasher.finalize_into(out);
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_utils::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
	use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::ParallelDigest;

	/// Runs `N` equal-length `M`-byte messages through the SIMD batch and pins each lane to the
	/// scalar reference.
	fn check_simd_batch<const M: usize, const N: usize>(rng: &mut StdRng) {
		// Fresh random bytes per lane, so lanes don't share a digest by accident.
		let messages: [Vec<u8>; N] = array::from_fn(|_| {
			let mut m = vec![0u8; M];
			rng.fill_bytes(&mut m);
			m
		});
		// Borrow each owned message as a byte slice for the batch input.
		let refs: [&[u8]; N] = array::from_fn(|i| messages[i].as_slice());
		// One uninitialized digest slot per lane.
		let mut out = array::from_fn::<_, N, _>(|_| MaybeUninit::uninit());
		// Run the batch hasher under test.
		Blake3MultiDigest::<M, N>::digest(refs, &mut out);

		// Each lane's output must equal the single-message reference hash of that lane.
		for (o, message) in out.iter().zip(messages.iter()) {
			// digest() initializes every slot, so reading it back is sound.
			let got = unsafe { o.assume_init_ref() };
			assert_eq!(got.as_slice(), blake3::hash(message).as_bytes(), "M = {M}");
		}
	}

	/// Hashes `n_leaves` leaves of `leaf_len` bytes each through the wired parallel digest, then
	/// pins every leaf to the scalar reference.
	fn check_const_len_route(rng: &mut StdRng, leaf_len: usize, n_leaves: usize) {
		// Each leaf is `leaf_len` random bytes, fed as `leaf_len` u8 items (BYTE_SIZE = 1).
		let leaves: Vec<Vec<u8>> = (0..n_leaves)
			.map(|_| {
				let mut m = vec![0u8; leaf_len];
				rng.fill_bytes(&mut m);
				m
			})
			.collect();

		// Drive the wired leaf digest over a parallel iterator of leaves, with a known leaf length.
		let digest = Blake3ParallelDigest::new();
		let mut results = repeat_with(MaybeUninit::<Output<blake3::Hasher>>::uninit)
			.take(n_leaves)
			.collect::<Vec<_>>();
		digest.digest_with_const_len(
			leaf_len,
			leaves.par_iter().map(|leaf| leaf.iter().copied()),
			&mut results,
		);

		// Each leaf digest must equal the reference hash of that leaf's bytes.
		for (result, leaf) in results.into_iter().zip(&leaves) {
			let got = unsafe { result.assume_init() };
			assert_eq!(got.as_slice(), blake3::hash(leaf).as_bytes(), "leaf_len {leaf_len}");
		}
	}

	/// Checks that the compression function matches `blake3::hash` of the concatenated inputs.
	#[test]
	fn test_blake3_compression_matches_reference() {
		let mut rng = StdRng::seed_from_u64(0);
		let left: [u8; 32] = rng.random();
		let right: [u8; 32] = rng.random();

		let compressed = Blake3Compression.compress([left.into(), right.into()]);

		let mut concatenated = [0u8; 64];
		concatenated[..32].copy_from_slice(&left);
		concatenated[32..].copy_from_slice(&right);
		let expected = blake3::hash(&concatenated);

		assert_eq!(compressed.as_slice(), expected.as_bytes());
	}

	#[test]
	fn test_multi_digest_block_multiples_match_reference() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: each lane's digest matches the scalar reference.
		//
		// The SIMD batch handles any whole number of blocks, up to one chunk.
		// Each `M` below exercises a different block count:
		// - 64   : a single block.
		// - 128  : two blocks.
		// - 512  : the mid range.
		// - 1024 : a full chunk, the 16-block maximum.
		//
		// Two lane widths per size:
		// - 4 lanes fill one NEON batch.
		// - 8 lanes force the kernel's internal sub-batching.
		check_simd_batch::<64, 4>(&mut rng);
		check_simd_batch::<64, 8>(&mut rng);
		check_simd_batch::<128, 4>(&mut rng);
		check_simd_batch::<128, 8>(&mut rng);
		check_simd_batch::<512, 4>(&mut rng);
		check_simd_batch::<512, 8>(&mut rng);
		check_simd_batch::<1024, 4>(&mut rng);
		check_simd_batch::<1024, 8>(&mut rng);
	}

	#[test]
	fn test_multi_digest_chained_update() {
		let mut rng = StdRng::seed_from_u64(2);
		// Four 128-byte messages (two blocks each).
		let messages: [Vec<u8>; 4] = array::from_fn(|_| {
			let mut m = vec![0u8; 128];
			rng.fill_bytes(&mut m);
			m
		});

		// Invariant: two updates of a split message hash the same as one update of the whole.
		let mut hasher = Blake3MultiDigest::<128, 4>::new();
		// Feed the first 50 bytes of every lane.
		hasher.update(array::from_fn(|i| &messages[i][..50]));
		// Then feed the remaining 78 bytes of every lane.
		hasher.update(array::from_fn(|i| &messages[i][50..]));
		let mut out = array::from_fn::<_, 4, _>(|_| MaybeUninit::uninit());
		hasher.finalize_into(&mut out);

		// Each lane must equal the reference hash of its full 128-byte message.
		for (o, message) in out.iter().zip(messages.iter()) {
			assert_eq!(unsafe { o.assume_init_ref() }.as_slice(), blake3::hash(message).as_bytes());
		}
	}

	#[test]
	fn test_const_len_routing_matches_reference() {
		let mut rng = StdRng::seed_from_u64(3);

		// Invariant: every leaf size reproduces the scalar reference, whichever path it takes.
		//
		// 50 leaves span several full SIMD batches plus a ragged final batch.
		// Each size targets a specific route:
		// - 0, 1, 63       : sub-block               -> scalar adapter.
		// - 65, 100        : non-block-multiple      -> scalar adapter.
		// - 2048           : multi-chunk (> 1024)    -> scalar adapter.
		// - 64, 128, 1024  : whole blocks, one chunk -> SIMD batch.
		for leaf_len in [0, 1, 63, 64, 65, 100, 128, 1024, 2048] {
			check_const_len_route(&mut rng, leaf_len, 50);
		}
	}

	#[test]
	fn test_parallel_blake3_non_const_len_matches_reference() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_leaves = 50;
		// Each leaf is four u128 values (64 bytes once serialized).
		let leaves: Vec<Vec<u128>> = (0..n_leaves)
			.map(|_| (0..4).map(|_| rng.random::<u128>()).collect())
			.collect();

		// The non-const-len `digest` path routes every leaf through the scalar adapter.
		let digest = Blake3ParallelDigest::new();
		let mut results = repeat_with(MaybeUninit::<Output<blake3::Hasher>>::uninit)
			.take(n_leaves)
			.collect::<Vec<_>>();
		digest.digest(leaves.par_iter().map(|leaf| leaf.iter().copied()), &mut results);

		// Each leaf digest must equal the reference hash of that leaf's serialized bytes.
		for (result, leaf) in results.into_iter().zip(&leaves) {
			// Reproduce the serialization the iterator performs: little-endian, in order.
			let mut bytes = Vec::new();
			for &item in leaf {
				bytes.extend_from_slice(&item.to_le_bytes());
			}
			let expected = blake3::hash(&bytes);
			assert_eq!(unsafe { result.assume_init() }.as_slice(), expected.as_bytes());
		}
	}
}
