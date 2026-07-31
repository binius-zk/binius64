// Copyright 2026 The Binius Developers

//! AArch64 NEON block compression for the multi-lane Blake3 kernel.

use std::arch::aarch64::{
	uint32x4_t, vaddq_u32, vdupq_n_u32, veorq_u32, vld1q_u32, vreinterpretq_u16_u32,
	vreinterpretq_u32_u16, vrev32q_u16, vshlq_n_u32, vsriq_n_u32, vst1q_u32,
};

use super::{IV, MSG_PERMUTATION, N_ROUNDS};

/// Widest interleave this module implements, counted in four-lane groups.
///
/// # Why this value
///
/// - A group holds 16 state vectors plus 16 message vectors, against 32 architectural registers.
/// - Interleaving groups hides the latency of each one's add, xor, rotate chain.
/// - Past four groups the spills cost more than the added parallelism returns.
///
/// Throughput hashing 256-byte leaves on an Apple M1 Pro:
///
/// ```text
///      4 lanes (1 group):  1.16 GiB/s
///      8 lanes (2 groups): 1.77 GiB/s
///     12 lanes (3 groups): 1.95 GiB/s   <- peak
///     16 lanes (4 groups): 1.93 GiB/s
///     20 lanes (5 groups): 1.53 GiB/s
/// ```
const MAX_GROUPS: usize = 4;

/// Rotates every 32-bit lane right by 16 bits.
///
/// A 16-bit rotate of a 32-bit word is exactly a swap of its two halfwords.
/// That is a single reverse instruction, cheaper than the shift-insert pair the other amounts need.
#[inline(always)]
fn rotr16(x: uint32x4_t) -> uint32x4_t {
	// SAFETY: this module is only reachable on aarch64 with `neon` statically enabled.
	unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
}

/// Rotates every 32-bit lane right by `R` bits, where `L` is the complementary left shift.
///
/// # Arguments
///
/// * `R` - bits to rotate right, in `0 < R < 32`.
/// * `L` - the value `32 - R`, supplied separately because a const parameter cannot be used in
///   arithmetic at an intrinsic's immediate operand.
///
/// # Performance
///
/// Shift-right-and-insert merges the wrapped bits into the shifted value in one instruction.
/// A rotate therefore costs two instructions rather than the shift, shift, or of three.
#[inline(always)]
fn rotr<const R: i32, const L: i32>(x: uint32x4_t) -> uint32x4_t {
	// A rotate only reproduces every input bit exactly once when the two amounts sum to the width.
	debug_assert_eq!(R + L, 32, "invariant: the two shift amounts must complete a rotate");
	// SAFETY: this module is only reachable on aarch64 with `neon` statically enabled.
	unsafe { vsriq_n_u32::<R>(vshlq_n_u32::<L>(x), x) }
}

/// Rotates every 32-bit lane right by 12 bits, the first of Blake3's two odd rotation amounts.
#[inline(always)]
fn rotr12(x: uint32x4_t) -> uint32x4_t {
	rotr::<12, 20>(x)
}

/// Rotates every 32-bit lane right by 8 bits.
#[inline(always)]
fn rotr8(x: uint32x4_t) -> uint32x4_t {
	rotr::<8, 24>(x)
}

/// Rotates every 32-bit lane right by 7 bits.
#[inline(always)]
fn rotr7(x: uint32x4_t) -> uint32x4_t {
	rotr::<7, 25>(x)
}

/// Applies one Blake3 quarter-round to every four-lane group.
///
/// # Arguments
///
/// * `v` - the 16-word working state, one array of 16 vectors per group.
/// * `m` - the permuted message schedule, laid out the same way.
/// * `a`, `b`, `c`, `d` - the four state positions this quarter-round mixes.
/// * `x`, `y` - the two message positions folded in, one per half-round.
///
/// # Algorithm
///
/// The mixing function from section 2.2 of the Blake3 spec, run twice over the same four words:
///
/// ```text
///     a += b + m_x;   d = rotr_16(d ^ a);   c += d;   b = rotr_12(b ^ c)
///     a += b + m_y;   d = rotr_8 (d ^ a);   c += d;   b = rotr_7 (b ^ c)
/// ```
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn quarter_round<const S: usize>(
	v: &mut [[uint32x4_t; 16]; S],
	m: &[[uint32x4_t; 16]; S],
	a: usize,
	b: usize,
	c: usize,
	d: usize,
	x: usize,
	y: usize,
) {
	// SAFETY: this module is only reachable on aarch64 with `neon` statically enabled.
	unsafe {
		// Why: each step reads what the step before it wrote, so a per-group loop stalls.
		//
		//     grouped by step:  a0 a1 a2 | d0 d1 d2 | c0 c1 c2 | ...   <- independent, issues
		// freely     grouped by group: a0 d0 c0 | a1 d1 c1 | a2 d2 c2 | ...   <- serial within a
		// group

		// First half-round: fold in the message word at the first position.
		for s in 0..S {
			v[s][a] = vaddq_u32(vaddq_u32(v[s][a], v[s][b]), m[s][x]);
		}
		for s in 0..S {
			v[s][d] = rotr16(veorq_u32(v[s][d], v[s][a]));
		}
		for s in 0..S {
			v[s][c] = vaddq_u32(v[s][c], v[s][d]);
		}
		for s in 0..S {
			v[s][b] = rotr12(veorq_u32(v[s][b], v[s][c]));
		}

		// Second half-round: same shape, the second message word, the other two rotation amounts.
		for s in 0..S {
			v[s][a] = vaddq_u32(vaddq_u32(v[s][a], v[s][b]), m[s][y]);
		}
		for s in 0..S {
			v[s][d] = rotr8(veorq_u32(v[s][d], v[s][a]));
		}
		for s in 0..S {
			v[s][c] = vaddq_u32(v[s][c], v[s][d]);
		}
		for s in 0..S {
			v[s][b] = rotr7(veorq_u32(v[s][b], v[s][c]));
		}
	}
}

/// Applies one full Blake3 round to every four-lane group.
///
/// # Algorithm
///
/// A round mixes the 16 state words as a 4x4 matrix, first down its columns, then along its
/// diagonals:
///
/// ```text
///     columns:    (0,4,8,12)  (1,5,9,13)  (2,6,10,14)  (3,7,11,15)
///     diagonals:  (0,5,10,15) (1,6,11,12) (2,7,8,13)   (3,4,9,14)
/// ```
///
/// Every state word is touched exactly twice, so after one round each word depends on all others.
#[inline(always)]
fn round<const S: usize>(v: &mut [[uint32x4_t; 16]; S], m: &[[uint32x4_t; 16]; S]) {
	// Column step: the four disjoint columns of the 4x4 state matrix.
	quarter_round(v, m, 0, 4, 8, 12, 0, 1);
	quarter_round(v, m, 1, 5, 9, 13, 2, 3);
	quarter_round(v, m, 2, 6, 10, 14, 4, 5);
	quarter_round(v, m, 3, 7, 11, 15, 6, 7);

	// Diagonal step: the four disjoint diagonals, which is what couples the columns together.
	quarter_round(v, m, 0, 5, 10, 15, 8, 9);
	quarter_round(v, m, 1, 6, 11, 12, 10, 11);
	quarter_round(v, m, 2, 7, 8, 13, 12, 13);
	quarter_round(v, m, 3, 4, 9, 14, 14, 15);
}

/// Compresses one 64-byte block into each of `4 * S` chaining values.
///
/// # Memory layout
///
/// Both buffers are word-major, one row per compression word, `stride` words between rows.
/// A row holds the same word of every lane, so `S` adjacent vectors cover all the lanes:
///
/// ```text
///     row 0:  [ lane_0 lane_1 lane_2 lane_3 | lane_4 ... ]   word 0 of every lane
///     row 1:  [ lane_0 lane_1 lane_2 lane_3 | lane_4 ... ]   word 1 of every lane
///               \_________ group 0 ______/   \__ group 1 ...
/// ```
///
/// # Arguments
///
/// * `cv` - the running chaining value, read as 8 rows and overwritten with the block's output.
/// * `block` - the message, 16 rows of little-endian words.
/// * `stride` - words between consecutive rows in both buffers.
/// * `counter` - the chunk counter, shared by every lane.
/// * `block_len` - bytes of this block that are message rather than zero padding.
/// * `flags` - the domain-separation flags for this block.
///
/// # Safety
///
/// * `cv` must be valid for reads and writes of 8 rows of `4 * S` words, `stride` words apart.
/// * `block` must be valid for reads of 16 rows of `4 * S` words, `stride` words apart.
/// * `stride` must be at least `4 * S`.
#[inline(always)]
unsafe fn compress_groups<const S: usize>(
	cv: *mut u32,
	block: *const u32,
	stride: usize,
	counter: u64,
	block_len: u32,
	flags: u32,
) {
	unsafe {
		// Phase 1: load the message schedule, one row at a time.
		//
		// Rows are strided, but the `S` vectors within a row are contiguous.
		let mut m = [[vdupq_n_u32(0); 16]; S];
		for w in 0..16 {
			for s in 0..S {
				m[s][w] = vld1q_u32(block.add(w * stride + s * 4));
			}
		}

		// Phase 2: build the 16-word working state.
		//
		//     words  0..8  : the incoming chaining value, which differs per lane
		//     words  8..12 : the first four initialization vector words
		//     word  12, 13 : the chunk counter, low half then high half
		//     word  14     : the message length of this block
		//     word  15     : the domain-separation flags
		let mut v = [[vdupq_n_u32(0); 16]; S];
		for w in 0..8 {
			for s in 0..S {
				v[s][w] = vld1q_u32(cv.add(w * stride + s * 4));
			}
		}
		for s in 0..S {
			for w in 0..4 {
				v[s][8 + w] = vdupq_n_u32(IV[w]);
			}
			// The last four words are the same for every lane, so they broadcast.
			v[s][12] = vdupq_n_u32(counter as u32);
			v[s][13] = vdupq_n_u32((counter >> 32) as u32);
			v[s][14] = vdupq_n_u32(block_len);
			v[s][15] = vdupq_n_u32(flags);
		}

		// Phase 3: seven rounds, with the message schedule permuted between consecutive rounds.
		for r in 0..N_ROUNDS {
			round(&mut v, &m);

			// The last round is followed by no further round, so its permutation is dead work.
			if r < N_ROUNDS - 1 {
				for s in 0..S {
					// Each slot reads from its source in the old schedule, so copy before writing.
					let prev = m[s];
					for w in 0..16 {
						m[s][w] = prev[MSG_PERMUTATION[w]];
					}
				}
			}
		}

		// Phase 4: truncated output, folding the two halves of the final state together.
		//
		//     h_w = v_w XOR v_{w+8}   for w in 0..8
		//
		// This becomes the chaining value of the next block, or the digest if this block was last.
		for w in 0..8 {
			for s in 0..S {
				vst1q_u32(cv.add(w * stride + s * 4), veorq_u32(v[s][w], v[s][8 + w]));
			}
		}
	}
}

/// Reports whether this module has a kernel for the given lane count.
///
/// A lane count qualifies when it splits into whole four-lane vectors and yields few enough
/// groups to stay near the register file.
#[inline(always)]
pub const fn handles_lanes(n: usize) -> bool {
	n > 0 && n.is_multiple_of(4) && n / 4 <= MAX_GROUPS
}

/// Compresses one 64-byte block across all `N` lanes, updating the chaining value in place.
///
/// Produces the same words as the portable lane-loop core, for every input.
///
/// # Panics
///
/// Panics if no kernel covers `N`.
#[inline(always)]
pub fn compress_block<const N: usize>(
	cv: &mut [[u32; N]; 8],
	block: &[[u32; N]; 16],
	counter: u64,
	block_len: u32,
	flags: u32,
) {
	assert!(handles_lanes(N), "precondition: the lane count must have a kernel");

	// Both buffers are arrays of rows of exactly `N` words, so consecutive rows sit `N` apart.
	let cv_ptr = cv.as_mut_ptr().cast::<u32>();
	let block_ptr = block.as_ptr().cast::<u32>();

	// SAFETY:
	// - The chaining value is 8 rows of `N` words, `N` words apart.
	// - The message is 16 rows of `N` words, `N` words apart.
	// - The check above pins the lane count to four times one of the group counts below.
	unsafe {
		match N / 4 {
			1 => compress_groups::<1>(cv_ptr, block_ptr, N, counter, block_len, flags),
			2 => compress_groups::<2>(cv_ptr, block_ptr, N, counter, block_len, flags),
			3 => compress_groups::<3>(cv_ptr, block_ptr, N, counter, block_len, flags),
			_ => compress_groups::<4>(cv_ptr, block_ptr, N, counter, block_len, flags),
		}
	}
}
