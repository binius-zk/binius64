// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! BLAKE3 compression primitive.
//!
//! A BLAKE3 block is 64 bytes (16 × 32-bit words). The compression function mixes an
//! 8-word chaining value with a 16-word message block, a 64-bit block counter, a byte
//! count, and a flags word, producing an updated 8-word chaining value.
//!
//! The structure mirrors the [reference implementation] from the BLAKE3 crate.
//!
//! [reference implementation]: https://github.com/BLAKE3-team/BLAKE3/blob/master/src/portable.rs

use std::{array, iter};

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Hint, Wire};

use super::{IV, MSG_SCHEDULE};
use crate::util::clear_high_bits;

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

/// Two sequential BLAKE3 compressions evaluated as the two lanes of [`blake3_compress_2x`].
///
/// Computes `C2 = compress(C1, block2, …)` where `C1 = compress(cv, block1, …)` — the output
/// chaining value of the first compression is the input chaining value of the second. Both
/// compressions share the single 7-round core of [`blake3_compress_2x`]: the first runs in the
/// high lane (bits `[32:64]`), the second in the low lane (bits `[0:32]`).
///
/// The data dependency — the second compression needs the first's *output* as its input — is
/// resolved with a `Blake3CompressHint` that precomputes `C1`'s output chaining value. That
/// value is fed into the low lane of the merged input chaining value (the second compression's
/// input) and constrained word-for-word against the first compression's in-circuit output (the
/// high lane of the result), so the hint cannot lie.
///
/// # Arguments
///
/// All wires carry 32-bit values in their low 32 bits, matching [`blake3_compress`].
///
/// - `cv`: input chaining value for the first compression (8 words).
/// - `blocks`: the two message blocks (`blocks[0]` for C1, `blocks[1]` for C2), 16 words each.
/// - `counter`: the 64-bit block counter, shared by both compressions. Sequential chaining only
///   happens within a single BLAKE3 chunk, where every block carries the chunk counter unchanged.
/// - `block_lens`: per-compression block lengths.
/// - `flags`: per-compression flags.
///
/// # Returns
///
/// The two output chaining values packed into 8 wires: the second compression's output in the
/// low 32 bits of each wire, the first compression's output in the high 32 bits.
pub fn blake3_compress_2x_seq(
	builder: &CircuitBuilder,
	cv: [Wire; 8],
	blocks: [[Wire; 16]; 2],
	counter: Wire,
	block_lens: [Wire; 2],
	flags: [Wire; 2],
) -> [Wire; 8] {
	// The hint returns the *merged* input chaining value directly: each word packs the first
	// compression's output in the low 32 bits (the second compression's input lane) and the first
	// compression's input `cv` word in the high 32 bits (the first compression's input lane). Both
	// halves are constrained below, so the hint itself is untrusted.
	let mut hint_inputs = Vec::with_capacity(27);
	hint_inputs.extend_from_slice(&cv);
	hint_inputs.extend_from_slice(&blocks[0]);
	hint_inputs.push(counter);
	hint_inputs.push(block_lens[0]);
	hint_inputs.push(flags[0]);
	let merged_cv_vec = builder.call_hint(Blake3CompressHint, &[], &hint_inputs);
	let merged_cv: [Wire; 8] = array::from_fn(|i| merged_cv_vec[i]);

	// Pack two lane values into one wire: low 32 bits = lane 0 (C2), high 32 bits = lane 1 (C1).
	// `shl` clears the high operand's upper bits; the low operand is cleared explicitly at each
	// call site, since block/scalar inputs are not guaranteed to be zero-extended to 64 bits.
	let pack = |lo: Wire, hi: Wire| builder.bxor(lo, builder.shl(hi, 32));
	let clear = |w: Wire| clear_high_bits(builder, w, 32);

	// Bind the hint's claimed first-compression input (high 32 bits of each merged CV word) to the
	// genuine input `cv` (low 32 bits), so the high lane provably compresses the real `cv`.
	for (merged, cv_word) in iter::zip(merged_cv, cv) {
		builder.assert_eq("blake3_compress_2x_seq.cv_in", builder.shr(merged, 32), clear(cv_word));
	}

	let merged_block: [Wire; 16] = array::from_fn(|i| pack(clear(blocks[1][i]), blocks[0][i]));

	// Both compressions share the same block counter. Sequential chaining (C2 takes C1's output
	// as its input chaining value) only occurs within a single BLAKE3 chunk, and every block in a
	// chunk carries the chunk counter unchanged.
	let counter_lo = clear(counter);
	let counter_hi = builder.shr(counter, 32);
	let merged_counter_lo = pack(counter_lo, counter_lo);
	let merged_counter_hi = pack(counter_hi, counter_hi);
	let merged_block_len = pack(clear(block_lens[1]), block_lens[0]);
	let merged_flags = pack(clear(flags[1]), flags[0]);

	let out = blake3_compress_2x(
		builder,
		merged_cv,
		merged_block,
		merged_counter_lo,
		merged_counter_hi,
		merged_block_len,
		merged_flags,
	);

	// Bind the hint: the first compression's in-circuit output (high lane of `out`) must equal the
	// chaining value the hint fed into the second compression's input (low 32 bits of `merged_cv`).
	for (merged, out_word) in iter::zip(merged_cv, out) {
		builder.assert_eq(
			"blake3_compress_2x_seq.c1_out",
			clear(merged),
			builder.shr(out_word, 32),
		);
	}

	out
}

/// Custom hint computing the merged input chaining value for [`blake3_compress_2x_seq`].
///
/// Runs the first compression off-circuit and packs its result so the output can be fed directly
/// as the two-lane input chaining value: the low 32 bits seed the second compression's input lane
/// with the first compression's output, the high 32 bits carry the first compression's input `cv`
/// word. Both halves are re-derived in-circuit and constrained, so the hint only needs to produce
/// the honest result.
///
/// Input layout (27 words, value in the low 32 bits of each): `cv[0..8]`, `block[0..16]`,
/// `counter` (full 64 bits), `block_len`, `flags`. Output: 8 packed words where the low 32 bits
/// hold the compression output chaining value and the high 32 bits hold the corresponding `cv`
/// input word.
struct Blake3CompressHint;

impl Hint for Blake3CompressHint {
	const NAME: &'static str = "binius.blake3_compress";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(27, 8)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let cv: [u32; 8] = array::from_fn(|i| inputs[i].as_u64() as u32);
		let block: [u32; 16] = array::from_fn(|i| inputs[8 + i].as_u64() as u32);
		let counter = inputs[24].as_u64();
		let block_len = inputs[25].as_u64() as u32;
		let flags = inputs[26].as_u64() as u32;

		let out = ref_compress(&cv, &block, counter, block_len, flags);
		for (i, slot) in outputs.iter_mut().enumerate() {
			*slot = Word(out[i] as u64 | ((cv[i] as u64) << 32));
		}
	}
}

// --- Pure-Rust reference implementation of BLAKE3 compression ------------------------
//
// Shared by [`Blake3CompressHint`] (prover-side witness generation) and the tests.

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

/// Pure-Rust BLAKE3 compression, matching the in-circuit compression exactly.
///
/// - Exposed for callers that use a raw 2-to-1 compression as a tweakable hash.
/// - It reproduces the same value off-circuit for witness generation.
pub fn ref_compress(
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
	array::from_fn(|i| v[i] ^ v[i + 8])
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_frontend::CircuitBuilder;

	use super::*;

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
		let cv_wires: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let block_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let counter_w = builder.add_witness();
		let block_len_w = builder.add_witness();
		let flags_w = builder.add_witness();

		let out = blake3_compress(&builder, cv_wires, block_wires, counter_w, block_len_w, flags_w);
		let out_inout: [Wire; 8] = array::from_fn(|_| builder.add_inout());
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
		array::from_fn(|i| w[out_inout[i]].0 as u32)
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
		let block = array::from_fn(|i| i as u32 * 0x0101_0101);
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
		let block = array::from_fn(|i| (i as u32).wrapping_mul(0xDEAD_BEEFu32));
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
		let cv_wires: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let block_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
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
		let out_inout: [Wire; 8] = array::from_fn(|_| builder.add_inout());
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
		let block0: [u32; 16] = array::from_fn(|i| i as u32 * 0x0101_0101);
		let block1: [u32; 16] = array::from_fn(|i| (i as u32).wrapping_mul(0xDEAD_BEEFu32));

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
		let block: [u32; 16] = array::from_fn(|i| i as u32);
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

	// --- 2× sequential tests -------------------------------------------------------

	/// Run `blake3_compress_2x_seq` and return `(c2_out, c1_out)`: the second and first
	/// compression outputs, unpacked from the low and high lanes of the packed result.
	#[allow(clippy::too_many_arguments)]
	fn run_compress_2x_seq(
		cv: [u32; 8],
		block1: [u32; 16],
		block2: [u32; 16],
		counter: u64,
		block_len1: u32,
		flags1: u32,
		block_len2: u32,
		flags2: u32,
	) -> ([u32; 8], [u32; 8]) {
		let builder = CircuitBuilder::new();
		let cv_wires: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let block1_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let block2_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let counter_w = builder.add_witness();
		let block_len1_w = builder.add_witness();
		let flags1_w = builder.add_witness();
		let block_len2_w = builder.add_witness();
		let flags2_w = builder.add_witness();

		let out = blake3_compress_2x_seq(
			&builder,
			cv_wires,
			[block1_wires, block2_wires],
			counter_w,
			[block_len1_w, block_len2_w],
			[flags1_w, flags2_w],
		);
		let out_inout: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("out_match_2x_seq", out[i], out_inout[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[cv_wires[i]] = Word(cv[i] as u64);
		}
		for i in 0..16 {
			w[block1_wires[i]] = Word(block1[i] as u64);
			w[block2_wires[i]] = Word(block2[i] as u64);
		}
		w[counter_w] = Word(counter);
		w[block_len1_w] = Word(block_len1 as u64);
		w[flags1_w] = Word(flags1 as u64);
		w[block_len2_w] = Word(block_len2 as u64);
		w[flags2_w] = Word(flags2 as u64);

		// Expected: the first compression feeds the second; both share the same counter.
		let c1 = ref_compress(&cv, &block1, counter, block_len1, flags1);
		let c2 = ref_compress(&c1, &block2, counter, block_len2, flags2);
		for i in 0..8 {
			w[out_inout[i]] = Word(pack2x(c2[i], c1[i]));
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		let mut c2_out = [0u32; 8];
		let mut c1_out = [0u32; 8];
		for i in 0..8 {
			let (lo, hi) = unpack2x(w[out_inout[i]].0);
			c2_out[i] = lo;
			c1_out[i] = hi;
		}
		(c2_out, c1_out)
	}

	#[test]
	fn compress_2x_seq_chains_two_blocks() {
		let cv = IV;
		let block1 = [0u32; 16];
		let block2: [u32; 16] = array::from_fn(|i| i as u32);
		let (c2, c1) = run_compress_2x_seq(
			cv,
			block1,
			block2,
			0,
			64,
			super::super::CHUNK_START,
			64,
			super::super::CHUNK_END | super::super::ROOT,
		);
		let exp_c1 = ref_compress(&cv, &block1, 0, 64, super::super::CHUNK_START);
		let exp_c2 =
			ref_compress(&exp_c1, &block2, 0, 64, super::super::CHUNK_END | super::super::ROOT);
		assert_eq!(c1, exp_c1);
		assert_eq!(c2, exp_c2);
	}

	#[test]
	fn compress_2x_seq_distinct_params() {
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
		let block1: [u32; 16] = array::from_fn(|i| (i as u32).wrapping_mul(0x0101_0101));
		let block2: [u32; 16] = array::from_fn(|i| (i as u32).wrapping_mul(0xDEAD_BEEFu32));
		// Distinct block lengths / flags per compression exercise the lane packing of every
		// parameter. The counter has a nonzero high half so both 32-bit halves are packed into
		// both lanes.
		let counter: u64 = 0x0000_0001_FFFF_FFFF;
		let (c2, c1) = run_compress_2x_seq(
			cv,
			block1,
			block2,
			counter,
			64,
			super::super::CHUNK_START,
			40,
			super::super::CHUNK_END,
		);
		let exp_c1 = ref_compress(&cv, &block1, counter, 64, super::super::CHUNK_START);
		let exp_c2 = ref_compress(&exp_c1, &block2, counter, 40, super::super::CHUNK_END);
		assert_eq!(c1, exp_c1);
		assert_eq!(c2, exp_c2);
	}
}
