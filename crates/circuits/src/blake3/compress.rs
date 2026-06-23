// Copyright 2025 Irreducible Inc.
//! BLAKE3 compression primitive.
//!
//! A BLAKE3 block is 64 bytes (16 × 32-bit words). The compression function mixes an
//! 8-word chaining value with a 16-word message block, a 64-bit block counter, a byte
//! count, and a flags word, producing an updated 8-word chaining value.
//!
//! The structure mirrors the [reference implementation] from the BLAKE3 crate.
//!
//! [reference implementation]: https://github.com/BLAKE3-team/BLAKE3/blob/master/src/portable.rs

use std::array;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Hint, Wire};

use super::{IV, MSG_SCHEDULE};

/// BLAKE3 compression function.
///
/// # Arguments
///
/// - `cv`: 8 chaining-value words (32-bit each, stored in the low 32 bits of each wire).
/// - `block`: 16 message words (32-bit each, low 32 bits of each wire, little-endian).
/// - `counter`: the 64-bit block counter. Low 32 bits are `t_low`, high 32 are `t_high`. The wire
///   may carry either a genuinely-64-bit counter (multi-chunk) or a 32-bit value with zero high
///   half (single-chunk).
/// - `block_len`: byte count for this block, 0..=64. 32-bit value in low 32 bits.
/// - `flags`: domain-separation flags. 32-bit value in low 32 bits.
///
/// # Returns
///
/// The updated 8-word chaining value.
pub fn blake3_compress(
	builder: &CircuitBuilder,
	cv: [Wire; 8],
	block: [Wire; 16],
	counter: Wire,
	block_len: Wire,
	flags: Wire,
) -> [Wire; 8] {
	// Split the counter into 32-bit halves.
	let mask_lo32 = builder.add_constant(Word(0xFFFF_FFFF));
	let t_low = builder.band(counter, mask_lo32);
	let t_high = builder.shr(counter, 32);

	let v: [Wire; 16] = [
		cv[0],
		cv[1],
		cv[2],
		cv[3],
		cv[4],
		cv[5],
		cv[6],
		cv[7],
		builder.add_constant(Word(IV[0] as u64)),
		builder.add_constant(Word(IV[1] as u64)),
		builder.add_constant(Word(IV[2] as u64)),
		builder.add_constant(Word(IV[3] as u64)),
		t_low,
		t_high,
		block_len,
		flags,
	];

	compress_core(builder, v, block)
}

/// BLAKE3 compression function running two independent compressions in parallel.
///
/// Each 64-bit input wire packs two 32-bit lanes: bits `[0:32]` hold the lane-0 word,
/// bits `[32:64]` hold the lane-1 word. This matches the lane layout expected by the
/// parallel-halves [`iadd_32`](CircuitBuilder::iadd_32) and
/// [`rotr32`](CircuitBuilder::rotr32) gates, so the 7-round core runs both
/// compressions at the gate cost of a single one.
///
/// The 64-bit block counter is split by the caller into low and high 32-bit halves:
/// `counter_lo` packs each lane's `t_low`, `counter_hi` packs each lane's `t_high`.
///
/// # Arguments
///
/// All wires follow the packing convention above.
///
/// - `cv`: 8 chaining-value words (per lane).
/// - `block`: 16 message words (per lane).
/// - `counter_lo`: low 32 bits of each lane's block counter.
/// - `counter_hi`: high 32 bits of each lane's block counter.
/// - `block_len`: byte count (0..=64) per lane.
/// - `flags`: domain-separation flags per lane.
///
/// # Returns
///
/// The updated 8-word chaining value, with each word packing both lanes.
pub fn blake3_compress_2x(
	builder: &CircuitBuilder,
	cv: [Wire; 8],
	block: [Wire; 16],
	counter_lo: Wire,
	counter_hi: Wire,
	block_len: Wire,
	flags: Wire,
) -> [Wire; 8] {
	// IV constants replicated into both 32-bit halves.
	let iv_2x = |i: usize| {
		let w = IV[i] as u64;
		builder.add_constant(Word(w | (w << 32)))
	};

	let v: [Wire; 16] = [
		cv[0],
		cv[1],
		cv[2],
		cv[3],
		cv[4],
		cv[5],
		cv[6],
		cv[7],
		iv_2x(0),
		iv_2x(1),
		iv_2x(2),
		iv_2x(3),
		counter_lo,
		counter_hi,
		block_len,
		flags,
	];

	compress_core(builder, v, block)
}

/// Shared body: 7 rounds of mixing followed by feed-forward.
///
/// Lane-agnostic: `g()` uses parallel-halves `iadd_32` / `rotr32` and bit-parallel
/// `bxor`, so the same core advances one or two independent compressions depending
/// on how the caller packed `v` and `block`.
fn compress_core(builder: &CircuitBuilder, mut v: [Wire; 16], block: [Wire; 16]) -> [Wire; 8] {
	for i in 0..7 {
		round(builder, &mut v, &block, i);
	}
	array::from_fn(|i| builder.bxor(v[i], v[i + 8]))
}

/// BLAKE3 G mixing function.
#[allow(clippy::too_many_arguments)]
fn g(
	builder: &CircuitBuilder,
	v: &mut [Wire; 16],
	a: usize,
	b: usize,
	c: usize,
	d: usize,
	x: Wire,
	y: Wire,
) {
	v[a] = builder.iadd_32(builder.iadd_32(v[a], v[b]), x);
	v[d] = builder.rotr32(builder.bxor(v[d], v[a]), 16);
	v[c] = builder.iadd_32(v[c], v[d]);
	v[b] = builder.rotr32(builder.bxor(v[b], v[c]), 12);
	v[a] = builder.iadd_32(builder.iadd_32(v[a], v[b]), y);
	v[d] = builder.rotr32(builder.bxor(v[d], v[a]), 8);
	v[c] = builder.iadd_32(v[c], v[d]);
	v[b] = builder.rotr32(builder.bxor(v[b], v[c]), 7);
}

/// One BLAKE3 round: four column G's followed by four diagonal G's.
fn round(builder: &CircuitBuilder, state: &mut [Wire; 16], msg: &[Wire; 16], round: usize) {
	let schedule = MSG_SCHEDULE[round];

	// Mix the columns.
	g(builder, state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
	g(builder, state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
	g(builder, state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
	g(builder, state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

	// Mix the diagonals.
	g(builder, state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
	g(builder, state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
	g(builder, state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
	g(builder, state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

/// Single BLAKE3 compression with the 7 rounds split across two 2x lanes.
///
/// Computes `compress(cv, block, …)` at the gate cost of **4 rounds** rather than 7, by
/// running the first and second halves in the two parallel lanes of [`blake3_compress_2x`]:
///
/// - **Lane 1 (high 32 bits)** — runs rounds 0–3 starting from the initial state.
/// - **Lane 0 (low 32 bits)** — runs rounds 4–6 (+ one padding round) starting from the
///   intermediate state after round 3, supplied by a `Blake3HalfCompressHint`.
///
/// The two lanes use different message schedules per step, so the block words are
/// re-packed at each of the 4 steps to route `MSG_SCHEDULE[lane1_round][k]` into the
/// high half and `MSG_SCHEDULE[lane0_round][k]` into the low half of each message wire.
///
/// After step 2 (round 6 for lane 0) the lane-0 half of the state is XOR'd with the
/// lane-0 half of `v[8..16]` to produce the 8-word output chaining value. Step 3 runs
/// the remaining round 3 for lane 1 (plus a padding round for lane 0) so that lane 1's
/// final state can be asserted equal to the hint, binding it to the correct intermediate
/// value.
///
/// # Arguments
///
/// Same as [`blake3_compress`]; all wires carry 32-bit values in their low 32 bits.
///
/// # Returns
///
/// The 8-word output chaining value (low 32 bits of each wire).
pub fn blake3_compress_opt(
	builder: &CircuitBuilder,
	cv: [Wire; 8],
	block: [Wire; 16],
	counter: Wire,
	block_len: Wire,
	flags: Wire,
) -> [Wire; 8] {
	// Reconstruct the initial 16-word state (same layout as blake3_compress).
	let mask_lo32 = builder.add_constant(Word(0xFFFF_FFFF));
	let t_low = builder.band(counter, mask_lo32);
	let t_high = builder.shr(counter, 32);

	let iv = |i: usize| builder.add_constant(Word(IV[i] as u64));
	let v_init: [Wire; 16] = [
		cv[0],
		cv[1],
		cv[2],
		cv[3],
		cv[4],
		cv[5],
		cv[6],
		cv[7],
		iv(0),
		iv(1),
		iv(2),
		iv(3),
		t_low,
		t_high,
		block_len,
		flags,
	];

	// Hint: precompute the 16-word intermediate state after 4 rounds.
	// Each output word is packed as (v_mid word) | (v_init word << 32).
	let mut hint_inputs = Vec::with_capacity(27);
	hint_inputs.extend_from_slice(&cv);
	hint_inputs.extend_from_slice(&block);
	hint_inputs.push(counter);
	hint_inputs.push(block_len);
	hint_inputs.push(flags);
	let hint = builder.call_hint(Blake3HalfCompressHint, &[], &hint_inputs);

	// Extract v_mid (low 32 bits of each hint word) for lane 0's starting state.
	let v_mid: [Wire; 16] = array::from_fn(|i| builder.shr(builder.shl(hint[i], 32), 32));

	// Pack merged state: lane 0 (low) = v_mid, lane 1 (high) = v_init.
	let pack = |lo: Wire, hi: Wire| builder.bor(lo, builder.shl(hi, 32));
	let mut v: [Wire; 16] = array::from_fn(|i| pack(v_mid[i], v_init[i]));

	// Run 3 steps: lane 0 advances rounds 4, 5, 6; lane 1 advances rounds 0, 1, 2.
	// Each step uses a step-specific block packing so each lane sees its own schedule.
	for step in 0..3usize {
		round_opt_step(builder, &mut v, &block, 4 + step, step);
	}

	// After step 2 (rounds 6 / 2): lane 0 has completed all 7 real rounds.
	// Apply feed-forward to the lane-0 (low) half only.
	let out: [Wire; 8] = array::from_fn(|i| builder.band(builder.bxor(v[i], v[i + 8]), mask_lo32));

	// Step 3: lane 0 runs a padding round (MSG_SCHEDULE[0]) so the step count is even;
	// lane 1 runs round 3 to complete its half for the binding check.
	round_opt_step(builder, &mut v, &block, 0, 3);

	// Bind the hint: lane 1's state after round 3 (high half of v) must equal v_mid.
	for i in 0..16 {
		let lane1_state = builder.shr(v[i], 32);
		builder.assert_eq("blake3_compress_opt.vmid", lane1_state, v_mid[i]);
	}

	out
}

/// One step of [`blake3_compress_opt`]: applies one round of mixing with per-lane
/// message schedules. Lane 0 (low 32 bits) uses `MSG_SCHEDULE[round_lo]` and lane 1
/// (high 32 bits) uses `MSG_SCHEDULE[round_hi]`. The block words are re-packed so each
/// lane's G calls receive the correct schedule-permuted words.
fn round_opt_step(
	builder: &CircuitBuilder,
	state: &mut [Wire; 16],
	block: &[Wire; 16],
	round_lo: usize,
	round_hi: usize,
) {
	let s_lo = MSG_SCHEDULE[round_lo];
	let s_hi = MSG_SCHEDULE[round_hi];

	// For each G-input position k, pack: low = block[s_lo[k]], high = block[s_hi[k]].
	let msg: [Wire; 16] =
		array::from_fn(|k| builder.bor(block[s_lo[k]], builder.shl(block[s_hi[k]], 32)));

	g(builder, state, 0, 4, 8, 12, msg[0], msg[1]);
	g(builder, state, 1, 5, 9, 13, msg[2], msg[3]);
	g(builder, state, 2, 6, 10, 14, msg[4], msg[5]);
	g(builder, state, 3, 7, 11, 15, msg[6], msg[7]);

	g(builder, state, 0, 5, 10, 15, msg[8], msg[9]);
	g(builder, state, 1, 6, 11, 12, msg[10], msg[11]);
	g(builder, state, 2, 7, 8, 13, msg[12], msg[13]);
	g(builder, state, 3, 4, 9, 14, msg[14], msg[15]);
}

/// Hint computing the 16-word BLAKE3 state after 4 rounds.
///
/// Input layout (27 words, value in the low 32 bits of each): `cv[0..8]`, `block[0..16]`,
/// `counter` (full 64 bits), `block_len`, `flags`. Output: 16 packed words where the
/// low 32 bits hold the intermediate-state word after 4 rounds and the high 32 bits hold
/// the corresponding initial-state word.
struct Blake3HalfCompressHint;

impl Hint for Blake3HalfCompressHint {
	const NAME: &'static str = "binius.blake3_half_compress";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(27, 16)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let cv: [u32; 8] = array::from_fn(|i| inputs[i].as_u64() as u32);
		let block: [u32; 16] = array::from_fn(|i| inputs[8 + i].as_u64() as u32);
		let counter = inputs[24].as_u64();
		let block_len = inputs[25].as_u64() as u32;
		let flags = inputs[26].as_u64() as u32;

		let v_init: [u32; 16] = [
			cv[0],
			cv[1],
			cv[2],
			cv[3],
			cv[4],
			cv[5],
			cv[6],
			cv[7],
			IV[0],
			IV[1],
			IV[2],
			IV[3],
			counter as u32,
			(counter >> 32) as u32,
			block_len,
			flags,
		];

		let v_mid = ref_half_compress(&cv, &block, counter, block_len, flags);

		for (i, slot) in outputs.iter_mut().enumerate() {
			*slot = Word(v_mid[i] as u64 | ((v_init[i] as u64) << 32));
		}
	}
}

// --- Pure-Rust reference implementations ----------------------------------------

fn ref_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
	v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
	v[d] = (v[d] ^ v[a]).rotate_right(16);
	v[c] = v[c].wrapping_add(v[d]);
	v[b] = (v[b] ^ v[c]).rotate_right(12);
	v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
	v[d] = (v[d] ^ v[a]).rotate_right(8);
	v[c] = v[c].wrapping_add(v[d]);
	v[b] = (v[b] ^ v[c]).rotate_right(7);
}

fn ref_round(state: &mut [u32; 16], msg: &[u32; 16], round: usize) {
	let schedule = MSG_SCHEDULE[round];

	ref_g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
	ref_g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
	ref_g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
	ref_g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

	ref_g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
	ref_g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
	ref_g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
	ref_g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

fn ref_half_compress(
	cv: &[u32; 8],
	block: &[u32; 16],
	counter: u64,
	block_len: u32,
	flags: u32,
) -> [u32; 16] {
	let mut v: [u32; 16] = [
		cv[0],
		cv[1],
		cv[2],
		cv[3],
		cv[4],
		cv[5],
		cv[6],
		cv[7],
		IV[0],
		IV[1],
		IV[2],
		IV[3],
		counter as u32,
		(counter >> 32) as u32,
		block_len,
		flags,
	];
	for i in 0..4 {
		ref_round(&mut v, block, i);
	}
	v
}

#[cfg(test)]
mod tests {
	use binius_frontend::CircuitBuilder;

	use super::*;

	// --- Pure-Rust reference implementation of BLAKE3 compression ---------------------

	fn ref_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
		v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
		v[d] = (v[d] ^ v[a]).rotate_right(16);
		v[c] = v[c].wrapping_add(v[d]);
		v[b] = (v[b] ^ v[c]).rotate_right(12);
		v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
		v[d] = (v[d] ^ v[a]).rotate_right(8);
		v[c] = v[c].wrapping_add(v[d]);
		v[b] = (v[b] ^ v[c]).rotate_right(7);
	}

	fn ref_round(state: &mut [u32; 16], msg: &[u32; 16], round: usize) {
		let schedule = MSG_SCHEDULE[round];

		ref_g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
		ref_g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
		ref_g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
		ref_g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

		ref_g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
		ref_g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
		ref_g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
		ref_g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
	}

	fn ref_compress(
		cv: &[u32; 8],
		block: &[u32; 16],
		counter: u64,
		block_len: u32,
		flags: u32,
	) -> [u32; 8] {
		let mut v = [
			cv[0],
			cv[1],
			cv[2],
			cv[3],
			cv[4],
			cv[5],
			cv[6],
			cv[7],
			IV[0],
			IV[1],
			IV[2],
			IV[3],
			counter as u32,
			(counter >> 32) as u32,
			block_len,
			flags,
		];
		for i in 0..7 {
			ref_round(&mut v, block, i);
		}
		std::array::from_fn(|i| v[i] ^ v[i + 8])
	}

	// --- Circuit-level tests --------------------------------------------------------

	/// Build a circuit that computes `blake3_compress` on witness inputs, populate the
	/// witness with the given values, and return the evaluated 8-word output.
	fn run_compress(
		cv: [u32; 8],
		block: [u32; 16],
		counter: u64,
		block_len: u32,
		flags: u32,
	) -> [u32; 8] {
		let builder = CircuitBuilder::new();
		let cv_wires: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());
		let block_wires: [Wire; 16] = std::array::from_fn(|_| builder.add_witness());
		let counter_w = builder.add_witness();
		let block_len_w = builder.add_witness();
		let flags_w = builder.add_witness();

		let out = blake3_compress(&builder, cv_wires, block_wires, counter_w, block_len_w, flags_w);
		let out_inout: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("out_match", out[i], out_inout[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[cv_wires[i]] = Word(cv[i] as u64);
		}
		for i in 0..16 {
			w[block_wires[i]] = Word(block[i] as u64);
		}
		w[counter_w] = Word(counter);
		w[block_len_w] = Word(block_len as u64);
		w[flags_w] = Word(flags as u64);

		let expected = ref_compress(&cv, &block, counter, block_len, flags);
		for i in 0..8 {
			w[out_inout[i]] = Word(expected[i] as u64);
		}
		circuit.populate_wire_witness(&mut w).unwrap();
		std::array::from_fn(|i| w[out_inout[i]].0 as u32)
	}

	#[test]
	fn zero_block_chunk_start_end_root() {
		let cv = IV;
		let block = [0u32; 16];
		let flags = super::super::CHUNK_START | super::super::CHUNK_END | super::super::ROOT;
		let actual = run_compress(cv, block, 0, 0, flags);
		let expected = ref_compress(&cv, &block, 0, 0, flags);
		assert_eq!(actual, expected);
	}

	#[test]
	fn all_ones_block() {
		let cv = IV;
		let block = [0xFFFF_FFFFu32; 16];
		let actual = run_compress(cv, block, 0, 64, 0);
		let expected = ref_compress(&cv, &block, 0, 64, 0);
		assert_eq!(actual, expected);
	}

	#[test]
	fn nonzero_counter_splits_correctly() {
		let cv = IV;
		let block = std::array::from_fn(|i| i as u32 * 0x0101_0101);
		let counter: u64 = 0x0123_4567_89AB_CDEF;
		let actual = run_compress(cv, block, counter, 64, super::super::CHUNK_END);
		let expected = ref_compress(&cv, &block, counter, 64, super::super::CHUNK_END);
		assert_eq!(actual, expected);
	}

	#[test]
	fn nontrivial_cv() {
		let cv = [
			0xDEAD_BEEF,
			0xCAFE_BABE,
			0x1234_5678,
			0x9ABC_DEF0,
			0x0BAD_F00D,
			0xFEED_FACE,
			0x0123_4567,
			0x89AB_CDEF,
		];
		let block = std::array::from_fn(|i| (i as u32).wrapping_mul(0xDEAD_BEEFu32));
		let actual = run_compress(cv, block, 42, 32, super::super::CHUNK_START);
		let expected = ref_compress(&cv, &block, 42, 32, super::super::CHUNK_START);
		assert_eq!(actual, expected);
	}

	// --- 2× SIMD tests -------------------------------------------------------------

	fn pack2x(lo: u32, hi: u32) -> u64 {
		(lo as u64) | ((hi as u64) << 32)
	}

	fn unpack2x(w: u64) -> (u32, u32) {
		(w as u32, (w >> 32) as u32)
	}

	/// Run `blake3_compress_2x` with two independent per-lane inputs and return the
	/// two per-lane 8-word outputs.
	fn run_compress_2x(
		cv: [[u32; 8]; 2],
		block: [[u32; 16]; 2],
		counter: [u64; 2],
		block_len: [u32; 2],
		flags: [u32; 2],
	) -> [[u32; 8]; 2] {
		let builder = CircuitBuilder::new();
		let cv_wires: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());
		let block_wires: [Wire; 16] = std::array::from_fn(|_| builder.add_witness());
		let counter_lo_w = builder.add_witness();
		let counter_hi_w = builder.add_witness();
		let block_len_w = builder.add_witness();
		let flags_w = builder.add_witness();

		let out = blake3_compress_2x(
			&builder,
			cv_wires,
			block_wires,
			counter_lo_w,
			counter_hi_w,
			block_len_w,
			flags_w,
		);
		let out_inout: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("out_match_2x", out[i], out_inout[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[cv_wires[i]] = Word(pack2x(cv[0][i], cv[1][i]));
		}
		for i in 0..16 {
			w[block_wires[i]] = Word(pack2x(block[0][i], block[1][i]));
		}
		w[counter_lo_w] = Word(pack2x(counter[0] as u32, counter[1] as u32));
		w[counter_hi_w] = Word(pack2x((counter[0] >> 32) as u32, (counter[1] >> 32) as u32));
		w[block_len_w] = Word(pack2x(block_len[0], block_len[1]));
		w[flags_w] = Word(pack2x(flags[0], flags[1]));

		let exp0 = ref_compress(&cv[0], &block[0], counter[0], block_len[0], flags[0]);
		let exp1 = ref_compress(&cv[1], &block[1], counter[1], block_len[1], flags[1]);
		for i in 0..8 {
			w[out_inout[i]] = Word(pack2x(exp0[i], exp1[i]));
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		let mut actual = [[0u32; 8]; 2];
		for i in 0..8 {
			let (lo, hi) = unpack2x(w[out_inout[i]].0);
			actual[0][i] = lo;
			actual[1][i] = hi;
		}
		actual
	}

	#[test]
	fn compress_2x_identical_lanes() {
		let cv = IV;
		let block = [0u32; 16];
		let flags = super::super::CHUNK_START | super::super::CHUNK_END | super::super::ROOT;
		let actual = run_compress_2x([cv, cv], [block, block], [0, 0], [0, 0], [flags, flags]);
		let expected = ref_compress(&cv, &block, 0, 0, flags);
		assert_eq!(actual[0], expected);
		assert_eq!(actual[1], expected);
	}

	#[test]
	fn compress_2x_distinct_lanes() {
		let cv0 = IV;
		let cv1 = [
			0xDEAD_BEEF,
			0xCAFE_BABE,
			0x1234_5678,
			0x9ABC_DEF0,
			0x0BAD_F00D,
			0xFEED_FACE,
			0x0123_4567,
			0x89AB_CDEF,
		];
		let block0: [u32; 16] = std::array::from_fn(|i| i as u32 * 0x0101_0101);
		let block1: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0xDEAD_BEEFu32));

		let actual = run_compress_2x(
			[cv0, cv1],
			[block0, block1],
			[0, 42],
			[64, 32],
			[super::super::CHUNK_END, super::super::CHUNK_START],
		);
		let exp0 = ref_compress(&cv0, &block0, 0, 64, super::super::CHUNK_END);
		let exp1 = ref_compress(&cv1, &block1, 42, 32, super::super::CHUNK_START);
		assert_eq!(actual[0], exp0);
		assert_eq!(actual[1], exp1);
	}

	#[test]
	fn compress_2x_counter_across_32bit_boundary() {
		let cv = IV;
		let block: [u32; 16] = std::array::from_fn(|i| i as u32);
		let counter0: u64 = 0x0123_4567_89AB_CDEF;
		let counter1: u64 = 0;
		let actual = run_compress_2x(
			[cv, cv],
			[block, block],
			[counter0, counter1],
			[64, 64],
			[
				super::super::CHUNK_START | super::super::ROOT,
				super::super::CHUNK_END,
			],
		);
		let exp0 =
			ref_compress(&cv, &block, counter0, 64, super::super::CHUNK_START | super::super::ROOT);
		let exp1 = ref_compress(&cv, &block, counter1, 64, super::super::CHUNK_END);
		assert_eq!(actual[0], exp0);
		assert_eq!(actual[1], exp1);
	}

	// --- Optimized single compress tests -------------------------------------------

	fn run_compress_opt(
		cv: [u32; 8],
		block: [u32; 16],
		counter: u64,
		block_len: u32,
		flags: u32,
	) -> [u32; 8] {
		let builder = CircuitBuilder::new();
		let cv_wires: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());
		let block_wires: [Wire; 16] = std::array::from_fn(|_| builder.add_witness());
		let counter_w = builder.add_witness();
		let block_len_w = builder.add_witness();
		let flags_w = builder.add_witness();

		let out =
			blake3_compress_opt(&builder, cv_wires, block_wires, counter_w, block_len_w, flags_w);
		let out_inout: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("out_match_opt", out[i], out_inout[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[cv_wires[i]] = Word(cv[i] as u64);
		}
		for i in 0..16 {
			w[block_wires[i]] = Word(block[i] as u64);
		}
		w[counter_w] = Word(counter);
		w[block_len_w] = Word(block_len as u64);
		w[flags_w] = Word(flags as u64);

		let expected = ref_compress(&cv, &block, counter, block_len, flags);
		for i in 0..8 {
			w[out_inout[i]] = Word(expected[i] as u64);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		std::array::from_fn(|i| w[out_inout[i]].as_u64() as u32)
	}

	#[test]
	fn compress_opt_zero_block() {
		let cv = IV;
		let block = [0u32; 16];
		let actual = run_compress_opt(cv, block, 0, 64, super::super::CHUNK_START);
		let expected = ref_compress(&cv, &block, 0, 64, super::super::CHUNK_START);
		assert_eq!(actual, expected);
	}

	#[test]
	fn compress_opt_nontrivial() {
		let cv = [
			0xDEAD_BEEFu32,
			0xCAFE_BABE,
			0x1234_5678,
			0x9ABC_DEF0,
			0x0BAD_F00D,
			0xFEED_FACE,
			0x0123_4567,
			0x89AB_CDEF,
		];
		let block: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0x0101_0101));
		let actual =
			run_compress_opt(cv, block, 0, 64, super::super::CHUNK_END | super::super::ROOT);
		let expected =
			ref_compress(&cv, &block, 0, 64, super::super::CHUNK_END | super::super::ROOT);
		assert_eq!(actual, expected);
	}

	#[test]
	fn compress_opt_counter_across_32bit_boundary() {
		let cv = IV;
		let block: [u32; 16] = std::array::from_fn(|i| i as u32);
		let counter: u64 = 0x0123_4567_89AB_CDEF;
		let actual = run_compress_opt(cv, block, counter, 64, super::super::CHUNK_START);
		let expected = ref_compress(&cv, &block, counter, 64, super::super::CHUNK_START);
		assert_eq!(actual, expected);
	}
}
