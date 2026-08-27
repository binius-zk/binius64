// Copyright 2026 The Binius Developers

//! Portable, auto-vectorized SHA-256 multi-lane kernel.
//!
//! A round reads the state the round before it wrote.
//! One chain therefore stalls on its own latency and leaves most of the machine idle.
//! Every entry point here advances several independent chains at once instead.
//!
//! The lanes are held transposed, one lane per message:
//!
//! ```text
//!     each of the 8 state words    ->  one word per lane
//!     each of the 16 message words ->  one word per lane
//!     every step                   ->  a fixed-width loop over the lanes
//! ```
//!
//! No intrinsics and no unsafe code, so the vectorizer fills whatever width the target has.
//!
//! A hand-written kernel takes over only at the lane counts it claims.
//! Every other count runs these loops, which are also the reference the kernels are tested
//! against.
//!
//! Output is bit-identical to the scalar block function, pinned by tests.
//!
//! Reference: FIPS 180-4, section 6.2.

use std::array;

#[cfg(all(
	target_arch = "x86_64",
	target_feature = "avx512f",
	target_feature = "avx512bw"
))]
use super::avx512;
#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
use super::neon;
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "sha",
	target_feature = "sse2",
	target_feature = "ssse3",
	target_feature = "sse4.1"
))]
use super::sha_ni;
use super::{BLOCK_LEN, DIGEST_LEN, IV, K, SINGLE_BLOCK_MAX_LEN};

/// Batch width the dispatch is tuned for on this target.
///
/// Each width is the point where a batch fills the pipeline it runs on:
///
/// ```text
///     AVX-512         16   the square the shuffle network needs
///     SHA extension    8   where interleaving reaches the unit's throughput
///     crypto extension 4   the widest batch that still fits in registers
///     lane loops       8   what the vectorizer fills with a 256-bit register
/// ```
///
/// Past that point more lanes only add register pressure.
pub const LANES: usize =
	if cfg!(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")) {
		// The shuffle network is a 16x16 square, so no other width has a transpose.
		16
	} else if cfg!(all(target_arch = "x86_64", target_feature = "sha")) {
		8
	} else if cfg!(all(target_arch = "aarch64", target_feature = "sha2")) {
		4
	} else {
		// The width the vectorizer fills with a 256-bit register in the lane loops below.
		8
	};

/// Applies one round to the eight state words, across all lanes.
///
/// FIPS 180-4 section 6.2.2, with the state written as `a` through `h`:
///
/// ```text
///     T1 = h + Sigma1(e) + Ch(e, f, g) + K + W
///     T2 = Sigma0(a) + Maj(a, b, c)
///     (a..h) <- (T1 + T2, a, b, c, d + T1, e, f, g)
/// ```
///
/// Only two words are computed.
/// The other six shift down one slot.
/// Returning them already shifted is what makes that shift cost nothing.
///
/// Every line is an independent map over the lanes, which is what the vectorizer turns into SIMD.
#[inline(always)]
fn round<const N: usize>(state: [[u32; N]; 8], w: &[u32; N], k: u32) -> [[u32; N]; 8] {
	let [a, b, c, d, e, f, g, h] = state;
	let mut next_a = [0u32; N];
	let mut next_e = [0u32; N];

	// One lane per iteration.
	// Lanes are independent, so the loop vectorizes.
	for i in 0..N {
		// Sigma1(x) = ROTR6 ^ ROTR11 ^ ROTR25.
		let sigma1 = e[i].rotate_right(6) ^ e[i].rotate_right(11) ^ e[i].rotate_right(25);
		// Ch(e, f, g) picks f where e is set, g where e is clear.
		let ch = (e[i] & f[i]) ^ (!e[i] & g[i]);
		let t1 = h[i]
			.wrapping_add(sigma1)
			.wrapping_add(ch)
			.wrapping_add(k)
			.wrapping_add(w[i]);
		// Sigma0(x) = ROTR2 ^ ROTR13 ^ ROTR22.
		let sigma0 = a[i].rotate_right(2) ^ a[i].rotate_right(13) ^ a[i].rotate_right(22);
		// Maj(a, b, c) is the bit held by at least two of the three.
		let maj = (a[i] & b[i]) ^ (a[i] & c[i]) ^ (b[i] & c[i]);

		next_e[i] = d[i].wrapping_add(t1);
		next_a[i] = t1.wrapping_add(sigma0.wrapping_add(maj));
	}

	// The six untouched words each move down one slot, which the return order applies:
	//
	//     in  : a  b  c  d  e  f  g  h
	//     out : a' a  b  c  e' e  f  g
	[next_a, a, b, c, next_e, e, f, g]
}

/// Extends the 16-word rolling message window by 16 words, in place, across all lanes.
///
/// FIPS 180-4 section 6.2.2:
///
/// ```text
///     W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
///     sigma0(x) = ROTR7  ^ ROTR18 ^ SHR3
///     sigma1(x) = ROTR17 ^ ROTR19 ^ SHR10
/// ```
///
/// The window holds the last 16 words, oldest at index `t mod 16`, so the four reads sit at
/// offsets 0, 1, 9, and 14 from that slot.
///
/// A straight sweep is correct because a read below the slot being written already holds its
/// new value, and a read above it still holds its old one:
///
/// ```text
///     writing slot j, reading offset 14:
///       j < 2  -> index j+14, untouched this sweep -> the old word, which is W[t-2]
///       j >= 2 -> index j-2,  written at step j-2  -> the new word, which is also W[t-2]
/// ```
#[inline(always)]
fn extend_window<const N: usize>(w: &mut [[u32; N]; 16]) {
	for j in 0..16 {
		// Copy the three reads out before the slot is overwritten.
		let m15 = w[(j + 1) & 15];
		let m7 = w[(j + 9) & 15];
		let m2 = w[(j + 14) & 15];
		for i in 0..N {
			let sigma0 = m15[i].rotate_right(7) ^ m15[i].rotate_right(18) ^ (m15[i] >> 3);
			let sigma1 = m2[i].rotate_right(17) ^ m2[i].rotate_right(19) ^ (m2[i] >> 10);
			w[j][i] = w[j][i]
				.wrapping_add(sigma0)
				.wrapping_add(m7[i])
				.wrapping_add(sigma1);
		}
	}
}

/// Loads one 64-byte block per lane into 16 big-endian message words.
///
/// The words arrive one block per lane and are consumed one word per lane, so this is a transpose.
/// Where a vector kernel covers the lane count it moves the square with shuffles instead of loads.
#[inline(always)]
fn load_block_words<const N: usize>(blocks: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "avx512f",
		target_feature = "avx512bw"
	))]
	if avx512::handles_lanes(N) {
		return avx512::load_block_words(blocks);
	}

	load_block_words_portable(blocks)
}

/// Loads one 64-byte block per lane into 16 big-endian message words, one word at a time.
///
/// Every target without a hand-written transpose runs this, and it is the reference the
/// vector transposes are tested against.
#[inline(always)]
fn load_block_words_portable<const N: usize>(blocks: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	let mut w = [[0u32; N]; 16];
	for lane in 0..N {
		for (t, slot) in w.iter_mut().enumerate() {
			let off = t * 4;
			// FIPS 180-4 section 5.1: a block is 16 big-endian 32-bit words.
			slot[lane] = u32::from_be_bytes([
				blocks[lane][off],
				blocks[lane][off + 1],
				blocks[lane][off + 2],
				blocks[lane][off + 3],
			]);
		}
	}
	w
}

/// Compresses one 64-byte block into each of a batch of independent SHA-256 states, in place.
///
/// The multi-lane analogue of the single-stream block function: each state absorbs its own
/// block through the 64 rounds plus the Davies-Meyer add.
///
/// No padding or length suffix is involved.
/// The caller owns the block contents.
///
/// Each lane ends holding exactly what the scalar block function would leave there,
/// whichever kernel ran.
#[inline]
pub fn compress256_multi<const N: usize>(
	states: &mut [[u32; 8]; N],
	blocks: &[[u8; BLOCK_LEN]; N],
) {
	// Ordered by measured throughput on this target, fastest first.
	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "avx512f",
		target_feature = "avx512bw"
	))]
	if avx512::handles_lanes(N) {
		// SAFETY: the target features above are statically enabled, so the intrinsics exist.
		unsafe { avx512::compress256_multi(states, blocks) };
		return;
	}

	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "sha",
		target_feature = "sse2",
		target_feature = "ssse3",
		target_feature = "sse4.1"
	))]
	if sha_ni::handles_lanes(N) {
		// SAFETY: the target features above are statically enabled, so the intrinsics exist.
		unsafe { sha_ni::compress256_multi(states, blocks) };
		return;
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
	if neon::handles_lanes(N) {
		// SAFETY: the target features above are statically enabled, so the intrinsics exist.
		unsafe { neon::compress256_multi(states, blocks) };
		return;
	}

	compress256_multi_portable(states, blocks);
}

/// Compresses one 64-byte block into each state of a batch, with plain lane loops.
///
/// Every target without a hand-written kernel runs this, and it is the reference the vector
/// kernels are tested against.
#[inline]
pub fn compress256_multi_portable<const N: usize>(
	states: &mut [[u32; 8]; N],
	blocks: &[[u8; BLOCK_LEN]; N],
) {
	let mut w = load_block_words(blocks);

	// Transpose the states into one vector per word, the layout the lane loops read.
	let mut state: [[u32; N]; 8] = array::from_fn(|t| array::from_fn(|lane| states[lane][t]));
	let incoming = state;

	// The first 16 rounds read the message as it arrives.
	// Each later group of 16 extends the whole window, then consumes it.
	for group in 0..4 {
		if group > 0 {
			extend_window(&mut w);
		}
		for t in 16 * group..16 * group + 16 {
			state = round(state, &w[t & 15], K[t]);
		}
	}

	// Davies-Meyer: add the incoming state back into the compressed state.
	for (lane, out) in states.iter_mut().enumerate() {
		for (t, word) in out.iter_mut().enumerate() {
			*word = state[t][lane].wrapping_add(incoming[t][lane]);
		}
	}
}

/// Hashes a batch of equal-length byte inputs into one standard SHA-256 digest each.
///
/// The inputs hash as independent streams, one lane each, sharing a block count.
///
/// # Panics
///
/// Panics if the inputs are not all the same length.
pub fn sha256_multi<const N: usize>(inputs: [&[u8]; N]) -> [[u8; DIGEST_LEN]; N] {
	let len = inputs[0].len();
	assert!(inputs.iter().all(|input| input.len() == len), "the inputs must have equal length");

	let mut states = [IV; N];
	let mut blocks = [[0u8; BLOCK_LEN]; N];

	// Absorb every full 64-byte block of the message.
	for b in 0..len / BLOCK_LEN {
		for (block, input) in blocks.iter_mut().zip(inputs) {
			block.copy_from_slice(&input[b * BLOCK_LEN..(b + 1) * BLOCK_LEN]);
		}
		compress256_multi(&mut states, &blocks);
	}

	// FIPS 180-4 section 5.1.1: the tail is the leftover bytes, a `0x80` terminator, zeros,
	// then the 64-bit big-endian bit length.
	// Room for that suffix decides whether one tail block is enough or two are needed.
	let rem = len % BLOCK_LEN;
	let n_tail = if rem <= SINGLE_BLOCK_MAX_LEN { 1 } else { 2 };
	let mut tails = [[0u8; 2 * BLOCK_LEN]; N];
	for (tail, input) in tails.iter_mut().zip(inputs) {
		tail[..rem].copy_from_slice(&input[len - rem..]);
		tail[rem] = 0x80;
		tail[n_tail * BLOCK_LEN - 8..n_tail * BLOCK_LEN]
			.copy_from_slice(&((len as u64) * 8).to_be_bytes());
	}
	for b in 0..n_tail {
		for (block, tail) in blocks.iter_mut().zip(&tails) {
			block.copy_from_slice(&tail[b * BLOCK_LEN..(b + 1) * BLOCK_LEN]);
		}
		compress256_multi(&mut states, &blocks);
	}

	// FIPS 180-4 section 6.2.2 emits the state most significant byte first.
	states.map(|state| {
		let mut digest = [0u8; DIGEST_LEN];
		for (chunk, word) in digest.chunks_exact_mut(4).zip(state) {
			chunk.copy_from_slice(&word.to_be_bytes());
		}
		digest
	})
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;
	use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
	use sha2::{Digest, Sha256, block_api::compress256};

	use super::*;

	/// Advances a batch of random states through the lane loops and pins each lane to the
	/// reference implementation.
	fn check_portable_against_sha2<const N: usize>(rng: &mut StdRng) {
		// Every lane gets its own random state and block.
		// Sharing a value across lanes would hide a kernel that reads the wrong lane.
		let states: [[u32; 8]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));
		let blocks: [[u8; BLOCK_LEN]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		let mut got = states;
		compress256_multi_portable(&mut got, &blocks);

		// Each lane must equal the single-stream block function on its own state and block.
		for (lane, (got_lane, state)) in got.iter().zip(&states).enumerate() {
			let mut want = *state;
			compress256(&mut want, std::slice::from_ref(&blocks[lane]));
			assert_eq!(*got_lane, want, "lane {lane} of {N} diverged from sha2");
		}
	}

	#[test]
	fn test_portable_matches_sha2() {
		let mut rng = StdRng::seed_from_u64(0);
		// Random states pin the raw-compression contract, not just the fixed IV.
		// Widths cover one lane, every tuned batch width, and the 16-lane square.
		for _ in 0..16 {
			check_portable_against_sha2::<1>(&mut rng);
			check_portable_against_sha2::<4>(&mut rng);
			check_portable_against_sha2::<8>(&mut rng);
			check_portable_against_sha2::<16>(&mut rng);
		}
	}

	#[test]
	fn test_portable_matches_sha2_at_state_extremes() {
		// All-zero and all-ones states and blocks, which random sampling never reaches.
		for state_word in [0u32, u32::MAX] {
			for block_byte in [0u8, 0xff] {
				let states = [[state_word; 8]; 4];
				let blocks = [[block_byte; BLOCK_LEN]; 4];

				let mut got = states;
				compress256_multi_portable(&mut got, &blocks);

				let mut want = states[0];
				compress256(&mut want, &blocks[..1]);
				for (lane, got_lane) in got.iter().enumerate() {
					assert_eq!(*got_lane, want, "lane {lane}, state {state_word:#x}");
				}
			}
		}
	}

	/// Advances a batch of random states through the dispatched kernel and pins it to the loops.
	///
	/// This is the check that covers whichever arch kernel this target compiled in.
	fn check_dispatch_against_portable<const N: usize>(rng: &mut StdRng) {
		let states: [[u32; 8]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));
		let blocks: [[u8; BLOCK_LEN]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		let mut want = states;
		compress256_multi_portable(&mut want, &blocks);

		let mut got = states;
		compress256_multi(&mut got, &blocks);

		assert_eq!(got, want, "the dispatched kernel diverged from the lane loops at {N} lanes");
	}

	#[test]
	fn test_dispatch_matches_portable() {
		let mut rng = StdRng::seed_from_u64(1);
		// Every width a caller can reach: the scalar path, both tuned batch widths, the square.
		for _ in 0..16 {
			check_dispatch_against_portable::<1>(&mut rng);
			check_dispatch_against_portable::<2>(&mut rng);
			check_dispatch_against_portable::<4>(&mut rng);
			check_dispatch_against_portable::<8>(&mut rng);
			check_dispatch_against_portable::<16>(&mut rng);
		}
	}

	proptest! {
		#[test]
		fn dispatch_matches_portable_proptest(seed in any::<u64>()) {
			// The fixed loops above pin the common widths.
			// This sweeps arbitrary states instead.
			let mut rng = StdRng::seed_from_u64(seed);
			check_dispatch_against_portable::<LANES>(&mut rng);
			check_dispatch_against_portable::<16>(&mut rng);
		}
	}

	/// Hashes a batch of equal-length messages and pins each lane to the reference digest.
	fn check_sha256_multi<const N: usize>(rng: &mut StdRng, len: usize) {
		// Distinct bytes per lane, so no two lanes coincide by accident.
		let messages: [Vec<u8>; N] = array::from_fn(|_| {
			let mut m = vec![0u8; len];
			rng.fill_bytes(&mut m);
			m
		});
		let refs: [&[u8]; N] = array::from_fn(|i| messages[i].as_slice());

		let got = sha256_multi(refs);
		for (lane, (got_lane, message)) in got.iter().zip(&messages).enumerate() {
			let want: [u8; DIGEST_LEN] = <Sha256 as Digest>::digest(message).into();
			assert_eq!(*got_lane, want, "len {len}, lane {lane} of {N}");
		}
	}

	#[test]
	fn test_sha256_multi_matches_sha2_at_padding_boundaries() {
		let mut rng = StdRng::seed_from_u64(2);
		// Lengths straddle every case the padding rule distinguishes:
		//
		//     0            : the lone padding block, no message bytes.
		//     1, 54, 55    : leftover still leaves room for the length suffix -> one tail block.
		//     56, 63       : leftover crowds out the suffix                   -> two tail blocks.
		//     64, 128      : exact block multiples, so the leftover is empty.
		//     65, 119, 120 : full blocks plus a leftover on each side of 55.
		for len in [0, 1, 54, 55, 56, 63, 64, 65, 119, 120, 128, 256] {
			check_sha256_multi::<1>(&mut rng, len);
			check_sha256_multi::<4>(&mut rng, len);
			check_sha256_multi::<LANES>(&mut rng, len);
		}
	}

	proptest! {
		#[test]
		fn sha256_multi_matches_sha2_proptest(seed in any::<u64>(), len in 0..300usize) {
			let mut rng = StdRng::seed_from_u64(seed);
			check_sha256_multi::<LANES>(&mut rng, len);
		}
	}

	#[test]
	#[should_panic(expected = "the inputs must have equal length")]
	fn test_sha256_multi_rejects_unequal_lengths() {
		// One block count is shared across the batch, so unequal lengths cannot be hashed.
		sha256_multi::<2>([&[0u8; 8], &[0u8; 9]]);
	}

	/// Transposes one random block set through both loaders and pins the words together.
	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "avx512f",
		target_feature = "avx512bw"
	))]
	fn check_avx512_transpose<const N: usize>(rng: &mut StdRng) {
		// Fresh random bytes per lane, so a network reading the wrong row shows up.
		let blocks: [[u8; BLOCK_LEN]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		// A transpose only permutes words, so the two loaders must agree on all 16 rows.
		assert_eq!(
			avx512::load_block_words(&blocks),
			load_block_words_portable(&blocks),
			"the shuffle network diverged from the byte-wise loader at {N} lanes"
		);
	}

	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "avx512f",
		target_feature = "avx512bw"
	))]
	#[test]
	fn test_avx512_transpose_places_every_word() {
		// Invariant: a word's output row is its word index, and its output lane is its block index.
		//
		// Fixture: word `w` of lane `l` carries the value `l * 16 + w`, so all 256 are distinct.
		//
		//     lane 0 block:  [  0,  1,  2, ...,  15 ]
		//     lane 1 block:  [ 16, 17, 18, ...,  31 ]
		//     ...
		//     output row w:  [  w, 16 + w, 32 + w, ... ]
		//
		// Distinct values are what makes a swapped row or lane show up as a wrong value.
		// The bytes are written big-endian, since that is the order the loader reads.
		let mut blocks = [[0u8; BLOCK_LEN]; 16];
		for (lane, block) in blocks.iter_mut().enumerate() {
			for (w, word) in block.chunks_exact_mut(4).enumerate() {
				word.copy_from_slice(&((lane * 16 + w) as u32).to_be_bytes());
			}
		}

		let m = avx512::load_block_words(&blocks);
		for (w, row) in m.iter().enumerate() {
			for (lane, got) in row.iter().enumerate() {
				assert_eq!(*got, (lane * 16 + w) as u32, "row {w}, lane {lane}");
			}
		}

		// Random blocks, where a misplaced row shows up only because the bytes differ.
		let mut rng = StdRng::seed_from_u64(3);
		for _ in 0..64 {
			check_avx512_transpose::<16>(&mut rng);
		}
	}

	/// Compresses one block set through the AVX-512 kernel and pins it to the lane loops.
	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "avx512f",
		target_feature = "avx512bw"
	))]
	fn check_avx512_core(rng: &mut StdRng) {
		let states: [[u32; 8]; 16] = array::from_fn(|_| array::from_fn(|_| rng.random()));
		let blocks: [[u8; BLOCK_LEN]; 16] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		let mut want = states;
		compress256_multi_portable(&mut want, &blocks);

		let mut got = states;
		// SAFETY: the module is compiled in only when `avx512f` and `avx512bw` are enabled.
		unsafe { avx512::compress256_multi(&mut got, &blocks) };

		assert_eq!(got, want, "the AVX-512 kernel diverged from the lane loops");
	}

	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "avx512f",
		target_feature = "avx512bw"
	))]
	#[test]
	fn test_avx512_core_matches_portable() {
		// The kernel is reachable through the dispatch only on a target without `sha`,
		// so it is exercised here directly, whichever extensions this target has.
		let mut rng = StdRng::seed_from_u64(4);
		for _ in 0..64 {
			check_avx512_core(&mut rng);
		}
	}

	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "avx512f",
		target_feature = "avx512bw"
	))]
	proptest! {
		#[test]
		fn avx512_core_matches_portable_proptest(seed in any::<u64>()) {
			let mut rng = StdRng::seed_from_u64(seed);
			check_avx512_core(&mut rng);
			check_avx512_transpose::<16>(&mut rng);
		}
	}

	/// Compresses one block set through the SHA-extension kernel and pins it to the lane loops.
	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "sha",
		target_feature = "sse2",
		target_feature = "ssse3",
		target_feature = "sse4.1"
	))]
	fn check_sha_ni_core<const N: usize>(rng: &mut StdRng) {
		let states: [[u32; 8]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));
		let blocks: [[u8; BLOCK_LEN]; N] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		let mut want = states;
		compress256_multi_portable(&mut want, &blocks);

		let mut got = states;
		// SAFETY: the module is compiled in only when the four target features are enabled.
		unsafe { sha_ni::compress256_multi(&mut got, &blocks) };

		assert_eq!(got, want, "the SHA-extension kernel diverged from the lane loops at {N} lanes");
	}

	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "sha",
		target_feature = "sse2",
		target_feature = "ssse3",
		target_feature = "sse4.1"
	))]
	#[test]
	fn test_sha_ni_core_matches_portable() {
		let mut rng = StdRng::seed_from_u64(5);
		// The kernel claims every lane count, so cover the scalar case up through the square.
		for _ in 0..32 {
			check_sha_ni_core::<1>(&mut rng);
			check_sha_ni_core::<2>(&mut rng);
			check_sha_ni_core::<4>(&mut rng);
			check_sha_ni_core::<8>(&mut rng);
			check_sha_ni_core::<16>(&mut rng);
		}
	}

	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "sha",
		target_feature = "sse2",
		target_feature = "ssse3",
		target_feature = "sse4.1"
	))]
	proptest! {
		#[test]
		fn sha_ni_core_matches_portable_proptest(seed in any::<u64>()) {
			let mut rng = StdRng::seed_from_u64(seed);
			check_sha_ni_core::<8>(&mut rng);
		}
	}

	/// Compresses one block set through the crypto-extension kernel and pins it to the lane loops.
	#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
	fn check_neon_core(rng: &mut StdRng) {
		let states: [[u32; 8]; 4] = array::from_fn(|_| array::from_fn(|_| rng.random()));
		let blocks: [[u8; BLOCK_LEN]; 4] = array::from_fn(|_| array::from_fn(|_| rng.random()));

		let mut want = states;
		compress256_multi_portable(&mut want, &blocks);

		let mut got = states;
		// SAFETY: the module is compiled in only when `sha2` is enabled.
		unsafe { neon::compress256_multi(&mut got, &blocks) };

		assert_eq!(got, want, "the crypto-extension kernel diverged from the lane loops");
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
	#[test]
	fn test_neon_core_matches_portable() {
		let mut rng = StdRng::seed_from_u64(6);
		for _ in 0..64 {
			check_neon_core(&mut rng);
		}
	}

	#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
	proptest! {
		#[test]
		fn neon_core_matches_portable_proptest(seed in any::<u64>()) {
			let mut rng = StdRng::seed_from_u64(seed);
			check_neon_core(&mut rng);
		}
	}
}
