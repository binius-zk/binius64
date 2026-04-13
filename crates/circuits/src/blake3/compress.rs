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
use binius_frontend::{CircuitBuilder, Wire};

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

	// Initialize the 16-word working state.
	let mut v: [Wire; 16] = [
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

	// Seven rounds of mixing.
	for i in 0..7 {
		round(builder, &mut v, &block, i);
	}

	// Feed-forward: new_cv[i] = v[i] ^ v[i+8].
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
}
