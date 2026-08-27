// Copyright 2026 The Binius Developers

//! AArch64 NEON block compression for the multi-lane Blake3 kernel.

use std::arch::aarch64::{
	uint32x4_t, vaddq_u32, vdupq_n_u32, veorq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u16_u32,
	vreinterpretq_u32_u8, vreinterpretq_u32_u16, vreinterpretq_u32_u64, vreinterpretq_u64_u32,
	vrev32q_u16, vshlq_n_u32, vsriq_n_u32, vst1q_u32, vtrn1q_u32, vtrn1q_u64, vtrn2q_u32,
	vtrn2q_u64,
};

use blake3::BLOCK_LEN;

use super::{IV, MSG_SCHEDULE};

/// Widest interleave this module implements, counted in four-lane groups.
///
/// # Why this value
///
/// - A group holds 16 state vectors plus 16 message vectors, against 32 architectural registers.
/// - Interleaving groups hides the latency of each one's add, xor, rotate chain.
/// - Past four groups the spills cost more than the added parallelism returns.
///
/// Throughput hashing 256-byte leaves on an Apple M1 Pro, single-threaded:
///
/// ```text
///      4 lanes (1 group):  1.02 GiB/s
///      8 lanes (2 groups): 1.41 GiB/s
///     12 lanes (3 groups): 1.45 GiB/s
///     16 lanes (4 groups): 1.47 GiB/s   <- peak
/// ```
///
/// A wider batch has no kernel and falls to the lane loops, which reach 1.30 GiB/s at 20 lanes.
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

/// Applies round `R` to every four-lane group.
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
///
/// Message positions come from row `R` of the schedule table.
/// The schedule advances by changing which register a round reads, never by moving one.
#[inline(always)]
fn round<const R: usize, const S: usize>(v: &mut [[uint32x4_t; 16]; S], m: &[[uint32x4_t; 16]; S]) {
	// The 16 message positions this round reads, in slot order.
	let s = MSG_SCHEDULE[R];

	// Column step: the four disjoint columns of the 4x4 state matrix.
	quarter_round(v, m, 0, 4, 8, 12, s[0], s[1]);
	quarter_round(v, m, 1, 5, 9, 13, s[2], s[3]);
	quarter_round(v, m, 2, 6, 10, 14, s[4], s[5]);
	quarter_round(v, m, 3, 7, 11, 15, s[6], s[7]);

	// Diagonal step: the four disjoint diagonals, which is what couples the columns together.
	quarter_round(v, m, 0, 5, 10, 15, s[8], s[9]);
	quarter_round(v, m, 1, 6, 11, 12, s[10], s[11]);
	quarter_round(v, m, 2, 7, 8, 13, s[12], s[13]);
	quarter_round(v, m, 3, 4, 9, 14, s[14], s[15]);
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

		// Phase 3: seven rounds, each reading the message through its own schedule row.
		// The 16 message vectors stay in the registers Phase 1 loaded them into.
		round::<0, S>(&mut v, &m);
		round::<1, S>(&mut v, &m);
		round::<2, S>(&mut v, &m);
		round::<3, S>(&mut v, &m);
		round::<4, S>(&mut v, &m);
		round::<5, S>(&mut v, &m);
		round::<6, S>(&mut v, &m);

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

/// Transposes a 4x4 square of 32-bit words held in four vector registers.
///
/// # Memory layout
///
/// Each input row is four consecutive words of one lane:
///
/// ```text
///     rows[0]:  [ a_00, a_01, a_02, a_03 ]    lane 0
///     rows[1]:  [ a_10, a_11, a_12, a_13 ]    lane 1
///     rows[2]:  [ a_20, a_21, a_22, a_23 ]    lane 2
///     rows[3]:  [ a_30, a_31, a_32, a_33 ]    lane 3
/// ```
///
/// Each output row is one word of all four lanes:
///
/// ```text
///     out[0]:   [ a_00, a_10, a_20, a_30 ]    word 0
///     out[1]:   [ a_01, a_11, a_21, a_31 ]    word 1
///     out[2]:   [ a_02, a_12, a_22, a_32 ]    word 2
///     out[3]:   [ a_03, a_13, a_23, a_33 ]    word 3
/// ```
///
/// # Algorithm
///
/// A two-stage butterfly, each stage swapping elements twice as far apart as the one before:
///
/// ```text
///     stage 1:  TRN1 / TRN2 over 32-bit lanes    swaps elements 1 apart
///     stage 2:  TRN1 / TRN2 over 64-bit lanes    swaps elements 2 apart
/// ```
///
/// Both stages stay in registers, so the square never touches memory.
#[inline(always)]
fn transpose4x4(rows: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
	// SAFETY: this module is only reachable on aarch64 with `neon` statically enabled.
	unsafe {
		// Stage 1: the first form keeps the even-indexed words of a row pair, the second the odd.
		let a = vtrn1q_u32(rows[0], rows[1]); // [ a_00, a_10, a_02, a_12 ]
		let b = vtrn2q_u32(rows[0], rows[1]); // [ a_01, a_11, a_03, a_13 ]
		let c = vtrn1q_u32(rows[2], rows[3]); // [ a_20, a_30, a_22, a_32 ]
		let d = vtrn2q_u32(rows[2], rows[3]); // [ a_21, a_31, a_23, a_33 ]

		// Stage 2: the same interleave one level up pairs the 64-bit halves into finished rows.
		let (a, b) = (vreinterpretq_u64_u32(a), vreinterpretq_u64_u32(b));
		let (c, d) = (vreinterpretq_u64_u32(c), vreinterpretq_u64_u32(d));
		[
			vreinterpretq_u32_u64(vtrn1q_u64(a, c)),
			vreinterpretq_u32_u64(vtrn1q_u64(b, d)),
			vreinterpretq_u32_u64(vtrn2q_u64(a, c)),
			vreinterpretq_u32_u64(vtrn2q_u64(b, d)),
		]
	}
}

/// Reports whether this module can transpose the message for the given lane count.
///
/// The square the butterfly moves is four lanes wide, so the lane count must fill it.
#[inline(always)]
pub const fn transposes_lanes(n: usize) -> bool {
	n > 0 && n.is_multiple_of(4)
}

/// Loads one 64-byte block per lane and transposes it into 16 rows of `N` lanes.
///
/// Produces the same words as the byte-wise loader, for every input.
///
/// # Algorithm
///
/// The kernel wants the message as 16 words of `N` lanes.
/// The input arrives as `N` blocks of 16 words, which is that square transposed.
///
/// The square splits into a grid of independent 4x4 blocks, four lanes by four words:
///
/// ```text
///     lanes 0..4, words 0..4    lanes 0..4, words 4..8    ...
///     lanes 4..8, words 0..4    lanes 4..8, words 4..8    ...
///     ...
/// ```
///
/// # Performance
///
/// Each 4x4 block costs 4 vector loads, 8 shuffles, and 4 vector stores.
/// The byte-wise loader instead moves each of the `16 * N` words on its own.
///
/// # Panics
///
/// Panics if the lane count does not fill the square.
#[inline(always)]
pub fn load_block_words<const N: usize>(block: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	assert!(transposes_lanes(N), "precondition: the lane count must fill the 4x4 square");

	let mut m = [[0u32; N]; 16];
	// SAFETY:
	// - Each load reads 16 bytes at `w * 16` inside a 64-byte block, so `w < 4` stays in bounds.
	// - Byte pointers carry no alignment requirement beyond the block's own.
	// - Each store writes 4 words at lane `g * 4` of a row of `N` words, and `g * 4 + 4 <= N`.
	unsafe {
		for g in 0..N / 4 {
			for w in 0..4 {
				// Gather four lanes of the same four words, one lane per vector.
				// A raw 16-byte load is already four little-endian words on this target.
				let rows = std::array::from_fn(|lane| {
					vreinterpretq_u32_u8(vld1q_u8(block[g * 4 + lane].as_ptr().add(w * 16)))
				});

				// Scatter them back as four words of the same four lanes, one word per vector.
				for (j, col) in transpose4x4(rows).into_iter().enumerate() {
					vst1q_u32(m[w * 4 + j].as_mut_ptr().add(g * 4), col);
				}
			}
		}
	}
	m
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
