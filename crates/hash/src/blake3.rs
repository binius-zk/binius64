// Copyright 2026 The Binius Developers

//! Blake3 hash and compression functions for use in Merkle tree constructions.

use std::{array, mem::MaybeUninit};

use binius_utils::{FixedSizeSerializeBytes, SerializeBytes, rayon::iter::IndexedParallelIterator};
use blake3::{
	BLOCK_LEN, CHUNK_LEN, IncrementCounter, OUT_LEN,
	platform::{Platform, le_bytes_from_words_32, words_from_le_bytes_32},
};
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
/// Each flag tags a compression with its role in the tree.
/// This stops a one-block message from ever colliding with an interior parent node.
/// The crate keeps these flags private, so the spec values are mirrored here.
/// The values are in Table 3, section 2.2 of the [Blake3 spec](https://github.com/BLAKE3-team/BLAKE3-specs).
/// The rationale for these flags is in section 7.7.
const CHUNK_START: u8 = 1 << 0;

/// Blake3 domain-separation flag marking the last block of a chunk.
const CHUNK_END: u8 = 1 << 1;

/// Blake3 domain-separation flag marking the last block of the whole tree.
const ROOT: u8 = 1 << 3;

/// Blake3 initial chaining value: the eight IV words, identical to the SHA-256 IV.
///
/// A non-keyed hash starts every chunk from these words.
/// The crate keeps its IV constant private, so the value is mirrored here.
/// The values are in Table 1, section 2.2 of the [Blake3 spec](https://github.com/BLAKE3-team/BLAKE3-specs).
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

/// A SIMD batch hasher exposed through the parallel-iterator interface.
///
/// The fixed lane count is hidden behind this alias.
type Blake3MultiParallel = ParallelMultidigestImpl<Blake3MultiDigest<SIMD_DEGREE>, SIMD_DEGREE>;

/// The parallel digest used for Blake3 leaf hashing.
///
/// The path is chosen by leaf size:
/// - A leaf of at least one 64-byte block goes through the SIMD batch.
/// - A smaller leaf is hashed on its own by the generic adapter.
///
/// Blake3's only public multi-lane kernel works a whole block at a time.
/// It cannot speed up a sub-block message, where the batch's extra buffering would only add cost.
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
		// Send every leaf through the SIMD batch.
		// The batch picks the fast path or the scalar fallback per group at run time.
		Blake3MultiParallel::new().digest(source, out);
	}

	fn digest_with_const_len<I: IntoIterator<Item: FixedSizeSerializeBytes>>(
		&self,
		n_items_per_input: usize,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		// Every leaf serializes to the same fixed byte length.
		let leaf_len = n_items_per_input * I::Item::BYTE_SIZE;
		// Below one 64-byte block the SIMD kernel gives no speedup.
		// Hash those leaves directly to avoid the batch's buffering overhead.
		if leaf_len < BLOCK_LEN {
			ParallelDigestAdapter::<blake3::Hasher>::new().digest(source, out);
		} else {
			// One block or larger: batch the leaves through the SIMD path.
			self.digest(source, out);
		}
	}
}

/// Computes `N` independent Blake3 digests at once with the official crate's SIMD kernel.
///
/// The kernel compresses one fixed, block-aligned input per lane, all starting from the IV.
/// This fits a message that is a single chunk of at most 1024 bytes, the shape of a Merkle leaf.
///
/// Fast path, taken when all `N` lanes share one length of at most one chunk:
/// - One SIMD call compresses the leading whole 64-byte blocks of every lane.
/// - A scalar compression then finishes each lane's trailing partial or final block.
///
/// Fallback, taken when lane lengths differ or any lane spans more than one chunk:
/// - There is no shared SIMD shape across lanes.
/// - Each lane is hashed on its own by the scalar reference, which still uses SIMD within a
///   message.
#[derive(Clone)]
pub struct Blake3MultiDigest<const N: usize> {
	/// One growable byte buffer per lane, holding that lane's message until finalization.
	///
	/// The hashing interface is streaming, so bytes arrive across calls and cannot be hashed
	/// early. Buffering lets the SIMD batch see each lane's full message at finalization.
	buffers: [Vec<u8>; N],
}

impl<const N: usize> Default for Blake3MultiDigest<N> {
	fn default() -> Self {
		// Every lane starts with an empty, unallocated buffer.
		Self {
			buffers: array::from_fn(|_| Vec::new()),
		}
	}
}

/// SIMD-compresses the first `M` bytes of every lane in one call, where `M` is a whole block count.
///
/// Every lane starts from the IV at chunk counter 0.
/// This is Blake3's chunk-hashing kernel run across the `N` messages, not across one message's
/// chunks.
///
/// # Arguments
///
/// - `flags_start` is applied to each lane's first block.
/// - `flags_end` is applied to each lane's last block.
///
/// # Returns
///
/// Writes the 32-byte chaining value of each lane into `out`.
fn hash_many_blocks<const M: usize, const N: usize>(
	platform: &Platform,
	buffers: &[&[u8]; N],
	flags_start: u8,
	flags_end: u8,
	out: &mut [[u8; OUT_LEN]; N],
) {
	// Reborrow each lane's leading `M` bytes as a fixed-size block-aligned input.
	// The uniform-length precondition guarantees every lane holds at least `M` bytes.
	let inputs: [&[u8; M]; N] = array::from_fn(|i| {
		(&buffers[i][..M])
			.try_into()
			.expect("lane holds at least M bytes")
	});
	// Drive the official kernel: no counter increment, since each lane is its own chunk at counter
	// 0.
	platform.hash_many::<M>(
		&inputs,
		&BLAKE3_IV,
		0,
		IncrementCounter::No,
		0,
		flags_start,
		flags_end,
		out.as_flattened_mut(),
	);
}

/// Dispatches a runtime full-block count in `1..=16` to a fixed-length SIMD call.
///
/// The SIMD kernel takes the per-lane byte length as a const generic.
/// The length must therefore be a compile-time literal, so each block count needs its own arm.
/// A single chunk holds at most 1024 / 64 = 16 whole blocks, giving these sixteen arms.
macro_rules! dispatch_full_blocks {
	($n_full:expr => $f:ident($($args:tt)*)) => {{
		macro_rules! arm { ($m:literal) => { $f::<$m, N>($($args)*) }; }
		match $n_full {
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
			_ => unreachable!("a single chunk has at most CHUNK_LEN/BLOCK_LEN = 16 full blocks"),
		}
	}};
}

impl<const N: usize> Blake3MultiDigest<N> {
	/// Hashes the `N` lane buffers into `out`, choosing the SIMD fast path or the scalar fallback.
	fn compute(buffers: &[Vec<u8>; N], out: &mut [MaybeUninit<Output<blake3::Hasher>>; N]) {
		// View each lane's buffered message as a byte slice.
		let refs: [&[u8]; N] = array::from_fn(|i| buffers[i].as_slice());
		// Take the first lane's length as the candidate shared length.
		let len = refs[0].len();

		// The fast path needs one shared SIMD shape across all lanes:
		// - every lane fits in one chunk of at most 1024 bytes, and
		// - every lane has the same length.
		if len <= CHUNK_LEN && refs.iter().all(|r| r.len() == len) {
			Self::single_chunk_simd(&refs, len, out);
		} else {
			// No shared shape: hash each lane on its own with the scalar reference.
			for (r, o) in refs.iter().zip(out.iter_mut()) {
				// The reference returns a 32-byte root digest.
				// Store it as this lane's output.
				o.write((*blake3::hash(r).as_bytes()).into());
			}
		}
	}

	/// Fast path for `N` equal-length single-chunk messages of length at most one chunk.
	fn single_chunk_simd(
		refs: &[&[u8]; N],
		len: usize,
		out: &mut [MaybeUninit<Output<blake3::Hasher>>; N],
	) {
		// Pick the best SIMD backend the CPU supports (AVX-512, AVX2, NEON, ...).
		let platform = Platform::detect();
		// Split the shared length into whole 64-byte blocks plus a trailing remainder.
		let n_full = len / BLOCK_LEN;
		let rem = len % BLOCK_LEN;
		// Scratch space for one 32-byte chaining value per lane.
		let mut cv_bytes = [[0u8; OUT_LEN]; N];

		// Case 1: the length is a positive multiple of the block size.
		// The chunk is all whole blocks, so the last block carries CHUNK_END | ROOT.
		// One SIMD pass then emits the root digest of every lane directly.
		if rem == 0 && len > 0 {
			dispatch_full_blocks!(n_full =>
				hash_many_blocks(&platform, refs, CHUNK_START, CHUNK_END | ROOT, &mut cv_bytes));
			// Each lane's chaining value is already its 32-byte root digest.
			for (b, o) in cv_bytes.iter().zip(out.iter_mut()) {
				o.write((*b).into());
			}
			return;
		}

		// Case 2: there is a trailing partial block, or the message is empty.
		// Start each lane's chaining value at the IV.
		let mut cvs = [BLAKE3_IV; N];
		// When there are leading whole blocks, compress them across lanes with one SIMD pass.
		// No end flag is set here, since the final block below carries CHUNK_END | ROOT.
		if n_full > 0 {
			dispatch_full_blocks!(n_full =>
				hash_many_blocks(&platform, refs, CHUNK_START, 0, &mut cv_bytes));
			// Reload each lane's running chaining value as words to keep chaining the chunk.
			for (cv, b) in cvs.iter_mut().zip(cv_bytes.iter()) {
				*cv = words_from_le_bytes_32(b);
			}
		}

		// The final block carries CHUNK_END | ROOT.
		// It also carries CHUNK_START when there were no leading blocks, i.e. it is the only block.
		let final_flags = (if n_full == 0 { CHUNK_START } else { 0 }) | CHUNK_END | ROOT;
		// Compress each lane's trailing block on its own, since a sub-block length can't be
		// batched.
		for ((cv, r), o) in cvs.iter_mut().zip(refs.iter()).zip(out.iter_mut()) {
			// Zero-pad the trailing `rem` bytes up to a full 64-byte block.
			let mut block = [0u8; BLOCK_LEN];
			block[..rem].copy_from_slice(&r[n_full * BLOCK_LEN..]);
			// Compress with the true byte count `rem`, so the zero padding does not change the
			// digest.
			platform.compress_in_place(cv, &block, rem as u8, 0, final_flags);
			// After the root compression the chaining value is the lane's 32-byte digest.
			o.write(le_bytes_from_words_32(cv).into());
		}
	}
}

impl<const N: usize> MultiDigest<N> for Blake3MultiDigest<N> {
	type Digest = blake3::Hasher;

	fn new() -> Self {
		// All lanes start with empty buffers.
		Self::default()
	}

	fn update(&mut self, data: [&[u8]; N]) {
		// Append each lane's new bytes to that lane's buffer.
		for (buf, chunk) in self.buffers.iter_mut().zip(data) {
			buf.extend_from_slice(chunk);
		}
	}

	fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		// Hash the buffered messages.
		// The hasher is consumed, so there is nothing to reset.
		Self::compute(&self.buffers, out);
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		// Hash the buffered messages, then clear the buffers so the hasher can be reused.
		Self::compute(&self.buffers, out);
		self.reset();
	}

	fn reset(&mut self) {
		// Drop every lane's buffered bytes while keeping the allocations for reuse.
		for buf in &mut self.buffers {
			buf.clear();
		}
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

	/// Hashes `N` messages through the batch and asserts each lane matches the scalar reference.
	fn check_multi_digest<const N: usize>(messages: &[Vec<u8>; N]) {
		// Borrow each owned message as a byte slice for the batch input.
		let refs: [&[u8]; N] = array::from_fn(|i| messages[i].as_slice());
		// One uninitialized digest slot per lane.
		let mut out = array::from_fn::<_, N, _>(|_| MaybeUninit::uninit());
		// Run the batch hasher under test.
		Blake3MultiDigest::<N>::digest(refs, &mut out);

		// Each lane's output must equal the single-message reference hash of that lane.
		for (o, message) in out.iter().zip(messages.iter()) {
			// digest() initializes every slot, so reading it back is sound.
			let got = unsafe { o.assume_init_ref() };
			assert_eq!(got.as_slice(), blake3::hash(message).as_bytes(), "len {}", message.len());
		}
	}

	/// Builds `N` distinct pseudorandom messages, each `len` bytes long.
	fn equal_len_messages<const N: usize>(rng: &mut StdRng, len: usize) -> [Vec<u8>; N] {
		array::from_fn(|_| {
			// Fresh random bytes per lane, so lanes don't accidentally share a digest.
			let mut m = vec![0u8; len];
			rng.fill_bytes(&mut m);
			m
		})
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
	fn test_multi_digest_equal_lengths_match_reference() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: equal-length lanes take the SIMD fast path.
		// Each lane's digest must match the scalar reference.
		// Each length exercises a different branch of that path:
		// - 0           : the lone empty block.
		// - 1, 31, 63   : a single sub-block (no leading SIMD blocks).
		// - 64,128,1024 : exact block multiples, hashed entirely by the SIMD pass.
		// - 65,127,1000 : a SIMD prefix of whole blocks plus a scalar partial tail.
		for len in [0, 1, 31, 63, 64, 65, 127, 128, 1000, 1024] {
			// Cover two lane widths:
			// - 4 lanes fill one NEON batch.
			// - 8 lanes force the kernel's internal sub-batching.
			check_multi_digest(&equal_len_messages::<4>(&mut rng, len));
			check_multi_digest(&equal_len_messages::<8>(&mut rng, len));
		}
	}

	#[test]
	fn test_multi_digest_fallback_matches_reference() {
		let mut rng = StdRng::seed_from_u64(1);
		// Helper: build four lanes with the given per-lane lengths.
		let mut mixed = |lens: [usize; 4]| -> [Vec<u8>; 4] {
			array::from_fn(|i| {
				let mut m = vec![0u8; lens[i]];
				rng.fill_bytes(&mut m);
				m
			})
		};

		// Invariant: lanes without a shared single-chunk shape each fall back to the scalar
		// reference. Every lane must still match that reference.

		// Unequal lengths, all within one chunk → no shared length → fallback.
		check_multi_digest(&mixed([10, 64, 200, 1000]));
		// At least one lane spans more than one chunk (> 1024 bytes) → fallback.
		check_multi_digest(&mixed([1025, 32, 4096, 0]));
		// Equal length but multi-chunk (2048 bytes), so still the fallback.
		check_multi_digest(&equal_len_messages::<4>(&mut rng, 2048));
	}

	#[test]
	fn test_multi_digest_chained_update() {
		let mut rng = StdRng::seed_from_u64(2);
		// Four 200-byte messages.
		let messages = equal_len_messages::<4>(&mut rng, 200);

		// Invariant: two updates of a split message hash the same as one update of the whole.
		let mut hasher = Blake3MultiDigest::<4>::new();
		// Feed the first 50 bytes of every lane.
		hasher.update(array::from_fn(|i| &messages[i][..50]));
		// Then feed the remaining 150 bytes of every lane.
		hasher.update(array::from_fn(|i| &messages[i][50..]));
		let mut out = array::from_fn::<_, 4, _>(|_| MaybeUninit::uninit());
		hasher.finalize_into(&mut out);

		// Each lane must equal the reference hash of its full 200-byte message.
		for (o, message) in out.iter().zip(messages.iter()) {
			assert_eq!(unsafe { o.assume_init_ref() }.as_slice(), blake3::hash(message).as_bytes());
		}
	}

	#[test]
	fn test_parallel_blake3_matches_serial() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_leaves = 50;
		// Each leaf is four u128 values.
		// A u128 serializes to 16 little-endian bytes, so each leaf is 64 bytes.
		// At 64 bytes a leaf takes the SIMD path.
		let leaves: Vec<Vec<u128>> = (0..n_leaves)
			.map(|_| (0..4).map(|_| rng.random::<u128>()).collect())
			.collect();

		// Drive the wired leaf digest over a parallel iterator of leaves.
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
