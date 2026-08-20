// Copyright 2026 The Binius Developers
//! BLAKE2s compression primitive.
//!
//! A BLAKE2s block is 64 bytes: 16 32-bit words.
//!
//! The compression function mixes an 8-word chained state with a 16-word message block, a
//! 64-bit byte counter split into two 32-bit halves, and a finalization flag.
//!
//! It produces an updated 8-word state.
//!
//! Every gate here is lane-agnostic.
//!
//! A 32-bit add, a 32-bit rotate, and a bitwise exclusive-or all act independently on each
//! 32-bit half of a 64-bit wire.
//!
//! That lets one mixing core serve three shapes:
//!
//! - One compression, packed into the low 32 bits of each wire.
//! - Two independent compressions, one per 32-bit lane.
//! - Two sequential compressions, where the second's input state is the first's output.
//!
//! The sequential pair is packed into the same two lanes through a hint that breaks the
//! circular dependency between them.

use std::{array, iter};

use binius_core::word::Word;
use binius_frontend::{ChipGadget, CircuitBuilder, Hint, Wire};

use super::constants::{IV, SIGMA};
use crate::util::clear_high_bits;

/// BLAKE2s G mixing function.
///
/// Every operation here is a parallel-halves gate or a bitwise one.
///
/// A parallel-halves gate acts independently on each 32-bit half of a 64-bit wire.
///
/// So the same code mixes either one compression alone, or two compressions packed side by
/// side.
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
	// Mix the first message word into a, then rotate d by 16 bits.
	v[a] = builder.iadd_32(builder.iadd_32(v[a], v[b]), x);
	v[d] = builder.rotr32(builder.bxor(v[d], v[a]), 16);
	// Fold d back into c, then rotate b by 12 bits.
	v[c] = builder.iadd_32(v[c], v[d]);
	v[b] = builder.rotr32(builder.bxor(v[b], v[c]), 12);
	// Mix the second message word into a, then rotate d by 8 bits.
	v[a] = builder.iadd_32(builder.iadd_32(v[a], v[b]), y);
	v[d] = builder.rotr32(builder.bxor(v[d], v[a]), 8);
	// Fold d back into c again, then rotate b by 7 bits.
	v[c] = builder.iadd_32(v[c], v[d]);
	v[b] = builder.rotr32(builder.bxor(v[b], v[c]), 7);
}

/// One mixing round.
///
/// Four column mixes, followed by four diagonal mixes, using the round's own message-word
/// schedule.
fn round(builder: &CircuitBuilder, v: &mut [Wire; 16], m: &[Wire; 16], round_idx: usize) {
	let s = &SIGMA[round_idx];
	// Mix the four columns: (0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15).
	g(builder, v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
	g(builder, v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
	g(builder, v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
	g(builder, v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
	// Mix the four diagonals: (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14).
	g(builder, v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
	g(builder, v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
	g(builder, v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
	g(builder, v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
}

/// The compression's initial 16-word working vector.
///
/// The chained state fills the low half.
///
/// The IV fills the high half.
///
/// The byte counter and the finalization flag are then folded into the last four words.
///
/// The `iv` argument carries one lane's worth of IV words for a single compression, or the
/// same words replicated into both lanes for two packed compressions.
///
/// That replication is the only difference between those two shapes.
fn init_v(
	builder: &CircuitBuilder,
	h: [Wire; 8],
	iv: [Wire; 8],
	t_lo: Wire,
	t_hi: Wire,
	last: Wire,
) -> [Wire; 16] {
	let mut v: [Wire; 16] = array::from_fn(|i| if i < 8 { h[i] } else { iv[i - 8] });
	// Mix the low 32 bits of the byte counter into word 12.
	v[12] = builder.bxor(v[12], t_lo);
	// Mix the high 32 bits of the byte counter into word 13.
	v[13] = builder.bxor(v[13], t_hi);
	// Mix the finalization flag into word 14: an all-ones flag flips every bit of that word.
	v[14] = builder.bxor(v[14], last);
	v
}

/// Runs the ten mixing rounds, then folds the working vector's two halves back into the state.
///
/// Shared by every shape this module offers.
///
/// The state and the working vector are single-lane for one compression, or packed two per
/// wire for a pair.
///
/// This code never needs to know which.
fn compress_core(
	builder: &CircuitBuilder,
	h: [Wire; 8],
	mut v: [Wire; 16],
	m: [Wire; 16],
) -> [Wire; 8] {
	for round_idx in 0..10 {
		round(builder, &mut v, &m, round_idx);
	}
	// Fold the vector's two halves back into the chained state.
	array::from_fn(|i| builder.bxor(h[i], builder.bxor(v[i], v[i + 8])))
}

/// BLAKE2s compression function.
///
/// # Arguments
/// * `builder` - Circuit builder.
/// * `h` - The 8-word chained state, one 32-bit value per wire.
/// * `m` - The 16-word message block, one 32-bit value per wire.
/// * `t_lo` - Low 32 bits of the byte counter.
/// * `t_hi` - High 32 bits of the byte counter.
/// * `last` - The finalization flag.
///
/// All-ones for the final block, zero otherwise.
///
/// # Preconditions
/// The high 32 bits of every input wire must be empty.
///
/// Ensuring this is the caller's responsibility.
///
/// Violating it leaves the gadget's behavior undefined, and unsafe to rely on.
///
/// # Returns
/// The updated 8-word state.
///
/// Every wire's high 32 bits are empty.
pub fn blake2s_compress(
	builder: &CircuitBuilder,
	h: [Wire; 8],
	m: [Wire; 16],
	t_lo: Wire,
	t_hi: Wire,
	last: Wire,
) -> [Wire; 8] {
	// The IV constant occupies only the low 32 bits of each wire, satisfying the precondition
	// this compression relies on.
	let iv: [Wire; 8] = array::from_fn(|i| builder.add_constant(Word(IV[i] as u64)));
	let v = init_v(builder, h, iv, t_lo, t_hi, last);
	compress_core(builder, h, v, m)
}

/// BLAKE2s compression function running two independent compressions in parallel.
///
/// Each 64-bit input wire packs two 32-bit lanes.
///
/// Bits 0 through 31 hold lane 0's word.
///
/// Bits 32 through 63 hold lane 1's word.
///
/// Every gate the mixing core uses already acts independently on each half.
///
/// So the ten mixing rounds compute both compressions for the gate cost of a single one.
///
/// # Arguments
/// Every wire packs two lanes, as described above.
///
/// * `h` - The 8-word chained state.
/// * `m` - The 16-word message block.
/// * `t_lo` - Low 32 bits of the byte counter.
/// * `t_hi` - High 32 bits of the byte counter.
/// * `last` - The finalization flag.
///
/// # Returns
/// The updated 8-word state.
///
/// Each wire packs both lanes' results.
///
/// # Chips
/// This gadget can be registered as a chip.
///
/// Doing so turns every paired compression under it into a chip call, including the ones a
/// sequential pairing and the fixed-length hasher reach.
pub fn blake2s_compress_2x(
	builder: &CircuitBuilder,
	h: [Wire; 8],
	m: [Wire; 16],
	t_lo: Wire,
	t_hi: Wire,
	last: Wire,
) -> [Wire; 8] {
	let inputs: Vec<Wire> = h.into_iter().chain(m).chain([t_lo, t_hi, last]).collect();
	let outputs = builder.build_gadget(Blake2sCompress2x, &[], &inputs);
	array::from_fn(|i| outputs[i])
}

/// The two-lane compression above, in a form a circuit can register as a chip.
///
/// Its interface is the flat 27 input words: the 8-word state, the 16-word message block, the
/// counter's low half, the counter's high half, and the finalization flag.
///
/// The 8 output words pack two lanes the same way the inputs do.
pub struct Blake2sCompress2x;

impl Hint for Blake2sCompress2x {
	const NAME: &'static str = "binius.blake2s_compress_2x";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(27, 8)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		// Every input word packs one lane per 32-bit half, and the two halves never interact.
		//
		// Every add, rotate, and mix this core uses is either 32-bit or bitwise.
		//
		// So computing one lane is exactly a plain compression of that lane's own words.
		let compress_lane = |i: usize| {
			let lane = |word: Word| (word.as_u64() >> (32 * i)) as u32;
			let h: [u32; 8] = array::from_fn(|j| lane(inputs[j]));
			let m: [u32; 16] = array::from_fn(|j| lane(inputs[8 + j]));
			ref_compress(h, m, lane(inputs[24]), lane(inputs[25]), lane(inputs[26]))
		};

		// Compute both lanes, then repack them into one word per output.
		let (lane_0, lane_1) = (compress_lane(0), compress_lane(1));
		for (slot, (low, high)) in iter::zip(outputs, iter::zip(lane_0, lane_1)) {
			*slot = Word(low as u64 | ((high as u64) << 32));
		}
	}
}

impl ChipGadget for Blake2sCompress2x {
	fn build(&self, builder: &CircuitBuilder, _dimensions: &[usize], inputs: &[Wire]) -> Vec<Wire> {
		// Split the flat input list back into the state and the message block, then run the
		// shared gate path a direct call would use.
		let h: [Wire; 8] = array::from_fn(|i| inputs[i]);
		let m: [Wire; 16] = array::from_fn(|i| inputs[8 + i]);
		compress_2x_gates(builder, h, m, inputs[24], inputs[25], inputs[26]).to_vec()
	}
}

/// The two-lane compression, expressed directly in gates.
///
/// Used both by a direct call and by the chip's gate path, so the two stay identical by
/// construction.
fn compress_2x_gates(
	builder: &CircuitBuilder,
	h: [Wire; 8],
	m: [Wire; 16],
	t_lo: Wire,
	t_hi: Wire,
	last: Wire,
) -> [Wire; 8] {
	// Replicate the IV into both 32-bit halves, so each lane mixes in its own copy.
	let iv_2x: [Wire; 8] = array::from_fn(|i| {
		let w = IV[i] as u64;
		builder.add_constant(Word(w | (w << 32)))
	});
	let v = init_v(builder, h, iv_2x, t_lo, t_hi, last);
	compress_core(builder, h, v, m)
}

/// Two sequential BLAKE2s block compressions, evaluated in one parallel core.
///
/// The second block's compression takes the first block's output state as its own input
/// state.
///
/// Both compressions run as the two 32-bit lanes of a single parallel compression.
///
/// ```text
///     high lane [32:64]:  S1 = compress(input state, first block)
///     low  lane [0:32] :  S2 = compress(S1,          second block)
/// ```
///
/// So two chained blocks cost one compression, instead of two.
///
/// The two lanes run at the same time.
///
/// Yet the low lane needs the high lane's *output* as its own input, before that output
/// exists.
///
/// A hint breaks this circular dependency by computing that output off-circuit first.
///
/// The hinted value seeds the low lane's input.
///
/// It is then constrained two ways, so it cannot lie:
///
/// - Its high half must equal the real input state.
/// - Its low half must equal what the first compression actually produces in-circuit.
///
/// # Arguments
/// * `builder` - Circuit builder.
/// * `h` - The 8-word input state for the first compression, one 32-bit value per wire.
/// * `blocks` - The two 16-word message blocks.
///
/// The first feeds the first compression, the second feeds the second.
///
/// * `t_los` - Each compression's low 32 bits of its byte counter.
/// * `t_his` - Each compression's high 32 bits of its byte counter.
/// * `lasts` - Each compression's finalization flag.
///
/// # Preconditions
/// Every input wire holds a valid 32-bit value in its low 32 bits.
///
/// High halves need not be empty:
///
/// - `h`'s high half is discarded by the shift that lifts it into the high lane.
/// - The first block's words, and the first compression's counter and flag, are likewise only ever
///   shifted, never read directly.
/// - The second block's words, and the second compression's counter and flag, are masked before
///   use.
///
/// # Returns
/// 8 wires, each packing both output states.
///
/// - Low 32 bits: the second compression's output.
/// - High 32 bits: the first compression's output.
pub fn blake2s_compress_2x_seq(
	builder: &CircuitBuilder,
	h: [Wire; 8],
	blocks: [[Wire; 16]; 2],
	t_los: [Wire; 2],
	t_his: [Wire; 2],
	lasts: [Wire; 2],
) -> [Wire; 8] {
	// The hint returns the merged state directly, one word at a time:
	//
	//     low 32 bits : first compression's output  = second compression's input
	//     high 32 bits: first compression's input state word
	//
	// Both halves are re-derived and constrained below, so the hint itself need not be
	// trusted.
	let mut hint_inputs = Vec::with_capacity(27);
	hint_inputs.extend_from_slice(&h);
	hint_inputs.extend_from_slice(&blocks[0]);
	hint_inputs.push(t_los[0]);
	hint_inputs.push(t_his[0]);
	hint_inputs.push(lasts[0]);
	let merged_vec = builder.call_hint(Blake2sCompressHint, &[], &hint_inputs);
	let merged: [Wire; 8] = array::from_fn(|i| merged_vec[i]);

	// Pack a lane pair into one wire: the low 32 bits hold lane 0, the high 32 bits hold
	// lane 1.
	//
	// Shifting left by 32 already clears the shifted operand's own high bits.
	//
	// The operand placed in the low half is cleared explicitly, since nothing here guarantees
	// its high bits start empty.
	let pack = |lo: Wire, hi: Wire| builder.bxor(lo, builder.shl(hi, 32));
	let clear = |w: Wire| clear_high_bits(builder, w, 32);

	// Merge each pair of per-compression values into its own two-lane wire: the second
	// compression's value in the low lane, the first's in the high lane.
	let merged_block: [Wire; 16] = array::from_fn(|i| pack(clear(blocks[1][i]), blocks[0][i]));
	let merged_t_lo = pack(clear(t_los[1]), t_los[0]);
	let merged_t_hi = pack(clear(t_his[1]), t_his[0]);
	let merged_last = pack(clear(lasts[1]), lasts[0]);

	let out =
		blake2s_compress_2x(builder, merged, merged_block, merged_t_lo, merged_t_hi, merged_last);

	// Bind the hinted state, one 64-bit equality per word.
	//
	// A single equality pins both halves at once, since they never overlap.
	//
	//     hinted word:  [ high lane = input state | low lane = S1 output ]
	//                          must equal                must equal
	//                     input state << 32       ^     result >> 32
	//
	// Together these leave the hint no freedom:
	//
	// - The high lane provably compresses the caller's own input state.
	// - The low lane provably chains from what the first compression really produced.
	//
	// Each side is one shift of an already-committed word, so nothing needs masking:
	//
	// - Shifting up discards the input's own high bits.
	// - Shifting down discards the result's low bits.
	for (m, (s, o)) in iter::zip(merged, iter::zip(h, out)) {
		let expected = builder.bxor(builder.shl(s, 32), builder.shr(o, 32));
		builder.assert_eq("blake2s_compress_2x_seq.merged_state", m, expected);
	}

	out
}

/// Precomputes the merged input state for the sequential two-lane compression above.
///
/// It runs the first compression off-circuit, then packs each output word to seed both
/// lanes at once.
///
/// - Low 32 bits: the first compression's output, which is the second compression's input.
/// - High 32 bits: the first compression's input state word.
///
/// Both halves are re-derived and constrained in-circuit, so this hint only needs to be
/// honest, not trusted.
///
/// # Input layout
/// 27 words, with the value in the low 32 bits of each.
///
/// - Words 0 through 7: the input state.
/// - Words 8 through 23: the first message block.
/// - Word 24: the first compression's low counter half.
/// - Word 25: the first compression's high counter half.
/// - Word 26: the first compression's finalization flag.
struct Blake2sCompressHint;

impl Hint for Blake2sCompressHint {
	const NAME: &'static str = "binius.blake2s_compress";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(27, 8)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		// Read the first compression's inputs out of the flat 27-word layout.
		let h: [u32; 8] = array::from_fn(|i| inputs[i].as_u64() as u32);
		let m: [u32; 16] = array::from_fn(|i| inputs[8 + i].as_u64() as u32);
		let t_lo = inputs[24].as_u64() as u32;
		let t_hi = inputs[25].as_u64() as u32;
		let last = inputs[26].as_u64() as u32;

		// Pack each output word with the compression's result in the low half and the
		// original input state word in the high half.
		let out = ref_compress(h, m, t_lo, t_hi, last);
		for (i, slot) in outputs.iter_mut().enumerate() {
			*slot = Word(out[i] as u64 | ((h[i] as u64) << 32));
		}
	}
}

/// Pure-Rust BLAKE2s compression of a single 64-byte block.
///
/// Matches the in-circuit compression exactly, per RFC 7693 Section 3.2.
///
/// Used for prover-side witness generation, and as the test reference.
///
/// # Arguments
/// * `h` - The 8-word input state.
/// * `m` - The 16-word message block.
/// * `t_lo` - Low 32 bits of the byte counter.
/// * `t_hi` - High 32 bits of the byte counter.
/// * `last` - The finalization flag.
///
/// All-ones for the final block, zero otherwise.
///
/// # Returns
/// The updated 8-word state.
pub fn ref_compress(h: [u32; 8], m: [u32; 16], t_lo: u32, t_hi: u32, last: u32) -> [u32; 8] {
	const fn ref_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
		// Mix the first message word into a, then rotate d by 16 bits.
		v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
		v[d] = (v[d] ^ v[a]).rotate_right(16);
		// Fold d back into c, then rotate b by 12 bits.
		v[c] = v[c].wrapping_add(v[d]);
		v[b] = (v[b] ^ v[c]).rotate_right(12);
		// Mix the second message word into a, then rotate d by 8 bits.
		v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
		v[d] = (v[d] ^ v[a]).rotate_right(8);
		// Fold d back into c again, then rotate b by 7 bits.
		v[c] = v[c].wrapping_add(v[d]);
		v[b] = (v[b] ^ v[c]).rotate_right(7);
	}

	let mut v = [0u32; 16];
	// The chained state fills the low half of the working vector, the IV the high half.
	v[..8].copy_from_slice(&h);
	v[8..].copy_from_slice(&IV);
	// Mix the byte counter and the finalization flag into the last four words.
	v[12] ^= t_lo;
	v[13] ^= t_hi;
	v[14] ^= last;

	// Ten rounds: four column mixes, then four diagonal mixes, per round.
	for s in &SIGMA {
		ref_g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
		ref_g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
		ref_g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
		ref_g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
		ref_g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
		ref_g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
		ref_g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
		ref_g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
	}

	// Fold the vector's two halves back into the chained state.
	array::from_fn(|i| h[i] ^ v[i] ^ v[i + 8])
}

#[cfg(test)]
mod tests {
	use binius_frontend::CircuitBuilder;
	use hex_literal::hex;
	use proptest::prelude::*;

	use super::*;

	// Circuit-level tests.

	// Builds a circuit around the single-lane compression, populates the witness with the
	// given values, and returns the evaluated 8-word output.
	//
	// Every input is fed in with an empty high half, satisfying the gadget's precondition.
	fn run_compress(h: [u32; 8], m: [u32; 16], t_lo: u32, t_hi: u32, last: u32) -> [u32; 8] {
		let builder = CircuitBuilder::new();
		let h_wires: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let m_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let t_lo_w = builder.add_witness();
		let t_hi_w = builder.add_witness();
		let last_w = builder.add_witness();

		// Wire the gadget under test, and pin its output to a public value the witness fills
		// with the reference result: a disagreement then surfaces as a failure to populate.
		let out = blake2s_compress(&builder, h_wires, m_wires, t_lo_w, t_hi_w, last_w);
		let out_inout: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("out_match", out[i], out_inout[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[h_wires[i]] = Word(h[i] as u64);
		}
		for i in 0..16 {
			w[m_wires[i]] = Word(m[i] as u64);
		}
		w[t_lo_w] = Word(t_lo as u64);
		w[t_hi_w] = Word(t_hi as u64);
		w[last_w] = Word(last as u64);

		let expected = ref_compress(h, m, t_lo, t_hi, last);
		for i in 0..8 {
			w[out_inout[i]] = Word(expected[i] as u64);
		}
		circuit.populate_wire_witness(&mut w).unwrap();
		array::from_fn(|i| w[out_inout[i]].0 as u32)
	}

	#[test]
	fn rfc7693_appendix_b_trace_matches_spec() {
		// Trace from RFC 7693 Appendix B: the unkeyed BLAKE2s-256 compression of "abc".
		//
		// A 3-byte message is one block.
		//
		// So the whole hash is a single compression, run from the parameter-mixed IV.
		//
		// This pins the compression against the specification text itself.
		//
		// It therefore holds even if the reference crate and this circuit were wrong
		// together.

		// h[0] carries the parameter block: unkeyed (key length 0), 32-byte digest.
		let h: [u32; 8] = [
			IV[0] ^ 0x0101_0020,
			IV[1],
			IV[2],
			IV[3],
			IV[4],
			IV[5],
			IV[6],
			IV[7],
		];
		// "abc" packed little-endian into the first message word; the rest of the block is
		// zero.
		let mut m = [0u32; 16];
		m[0] = 0x0063_6261;
		// A single block carries the whole 3-byte length, and the finalization flag.
		let (t_lo, t_hi, last) = (3u32, 0u32, 0xFFFF_FFFFu32);

		let expected: [u32; 8] = [
			0x8C5E_8C50,
			0xE214_7C32,
			0xA32B_A7E1,
			0x2F45_EB4E,
			0x208B_4537,
			0x293A_D69E,
			0x4C9B_994D,
			0x8259_6786,
		];
		assert_eq!(ref_compress(h, m, t_lo, t_hi, last), expected);
		assert_eq!(run_compress(h, m, t_lo, t_hi, last), expected);

		// The published digest is these words read little-endian, byte for byte.
		let digest_bytes: Vec<u8> = expected.iter().flat_map(|w| w.to_le_bytes()).collect();
		assert_eq!(
			digest_bytes,
			hex!("508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982")
		);
	}

	// 2x SIMD tests.

	fn pack2x(lo: u32, hi: u32) -> u64 {
		(lo as u64) | ((hi as u64) << 32)
	}

	fn unpack2x(w: u64) -> (u32, u32) {
		(w as u32, (w >> 32) as u32)
	}

	// Runs the two-lane compression with two independent per-lane inputs, and returns the two
	// per-lane 8-word outputs.
	fn run_compress_2x(
		h: [[u32; 8]; 2],
		m: [[u32; 16]; 2],
		t_lo: [u32; 2],
		t_hi: [u32; 2],
		last: [u32; 2],
	) -> [[u32; 8]; 2] {
		let builder = CircuitBuilder::new();
		let h_wires: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let m_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let t_lo_w = builder.add_witness();
		let t_hi_w = builder.add_witness();
		let last_w = builder.add_witness();

		let out = blake2s_compress_2x(&builder, h_wires, m_wires, t_lo_w, t_hi_w, last_w);
		let out_inout: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("out_match_2x", out[i], out_inout[i]);
		}

		// Pack each pair of per-lane values into its two-lane wire before populating.
		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[h_wires[i]] = Word(pack2x(h[0][i], h[1][i]));
		}
		for i in 0..16 {
			w[m_wires[i]] = Word(pack2x(m[0][i], m[1][i]));
		}
		w[t_lo_w] = Word(pack2x(t_lo[0], t_lo[1]));
		w[t_hi_w] = Word(pack2x(t_hi[0], t_hi[1]));
		w[last_w] = Word(pack2x(last[0], last[1]));

		let exp0 = ref_compress(h[0], m[0], t_lo[0], t_hi[0], last[0]);
		let exp1 = ref_compress(h[1], m[1], t_lo[1], t_hi[1], last[1]);
		for i in 0..8 {
			w[out_inout[i]] = Word(pack2x(exp0[i], exp1[i]));
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		// Unpack the two lanes of the evaluated result back into separate per-lane arrays.
		let mut actual = [[0u32; 8]; 2];
		for i in 0..8 {
			let (lo, hi) = unpack2x(w[out_inout[i]].0);
			actual[0][i] = lo;
			actual[1][i] = hi;
		}
		actual
	}

	#[test]
	fn compress_2x_distinct_lanes() {
		// Lane 0 reruns the RFC trace.
		//
		// Lane 1 runs a completely different chained-state and message-block pair.
		//
		// This confirms the lanes are independent: no bits cross the 32-bit boundary.
		let h0: [u32; 8] = [
			IV[0] ^ 0x0101_0020,
			IV[1],
			IV[2],
			IV[3],
			IV[4],
			IV[5],
			IV[6],
			IV[7],
		];
		let mut m0 = [0u32; 16];
		m0[0] = 0x0063_6261;

		let h1: [u32; 8] = [
			0xDEAD_BEEF,
			0xCAFE_BABE,
			0x1234_5678,
			0x9ABC_DEF0,
			0x0BAD_F00D,
			0xFEED_FACE,
			0x0123_4567,
			0x89AB_CDEF,
		];
		let m1: [u32; 16] = array::from_fn(|i| (i as u32).wrapping_mul(0x0101_0101));

		let actual = run_compress_2x([h0, h1], [m0, m1], [3, 64], [0, 0], [0xFFFF_FFFF, 0]);
		assert_eq!(actual[0], ref_compress(h0, m0, 3, 0, 0xFFFF_FFFF));
		assert_eq!(actual[1], ref_compress(h1, m1, 64, 0, 0));
	}

	#[test]
	fn compress_2x_lane_independence() {
		// Lane 0 is the RFC trace.
		//
		// Lane 1 is an all-zero block, compressed from the same parameter-mixed IV.
		//
		// Each lane must match its own reference.
		//
		// That proves the zero lane does not perturb the "abc" lane, and vice versa.
		let h: [u32; 8] = [
			IV[0] ^ 0x0101_0020,
			IV[1],
			IV[2],
			IV[3],
			IV[4],
			IV[5],
			IV[6],
			IV[7],
		];
		let mut m = [0u32; 16];
		m[0] = 0x0063_6261;
		let actual =
			run_compress_2x([h, h], [m, [0u32; 16]], [3, 0], [0, 0], [0xFFFF_FFFF, 0xFFFF_FFFF]);
		assert_eq!(actual[0], ref_compress(h, m, 3, 0, 0xFFFF_FFFF));
		assert_eq!(actual[1], ref_compress(h, [0u32; 16], 0, 0, 0xFFFF_FFFF));
	}

	// 2x sequential tests.

	// Runs the sequential two-lane compression, and returns the second and first compression
	// outputs, unpacked from the low and high lanes of the packed result.
	#[allow(clippy::too_many_arguments)]
	fn run_compress_2x_seq(
		h: [u32; 8],
		m1: [u32; 16],
		m2: [u32; 16],
		t_lo1: u32,
		t_hi1: u32,
		last1: u32,
		t_lo2: u32,
		t_hi2: u32,
		last2: u32,
	) -> ([u32; 8], [u32; 8]) {
		let builder = CircuitBuilder::new();
		let h_wires: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let m1_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let m2_wires: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let t_lo1_w = builder.add_witness();
		let t_hi1_w = builder.add_witness();
		let last1_w = builder.add_witness();
		let t_lo2_w = builder.add_witness();
		let t_hi2_w = builder.add_witness();
		let last2_w = builder.add_witness();

		let out = blake2s_compress_2x_seq(
			&builder,
			h_wires,
			[m1_wires, m2_wires],
			[t_lo1_w, t_lo2_w],
			[t_hi1_w, t_hi2_w],
			[last1_w, last2_w],
		);
		let out_inout: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq("out_match_2x_seq", out[i], out_inout[i]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[h_wires[i]] = Word(h[i] as u64);
		}
		for i in 0..16 {
			w[m1_wires[i]] = Word(m1[i] as u64);
			w[m2_wires[i]] = Word(m2[i] as u64);
		}
		w[t_lo1_w] = Word(t_lo1 as u64);
		w[t_hi1_w] = Word(t_hi1 as u64);
		w[last1_w] = Word(last1 as u64);
		w[t_lo2_w] = Word(t_lo2 as u64);
		w[t_hi2_w] = Word(t_hi2 as u64);
		w[last2_w] = Word(last2 as u64);

		// The expected result chains two plain compressions: the first compression's output
		// becomes the second compression's input state.
		let s1 = ref_compress(h, m1, t_lo1, t_hi1, last1);
		let s2 = ref_compress(s1, m2, t_lo2, t_hi2, last2);
		for i in 0..8 {
			w[out_inout[i]] = Word(pack2x(s2[i], s1[i]));
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		let mut s2_out = [0u32; 8];
		let mut s1_out = [0u32; 8];
		for i in 0..8 {
			let (lo, hi) = unpack2x(w[out_inout[i]].0);
			s2_out[i] = lo;
			s1_out[i] = hi;
		}
		(s2_out, s1_out)
	}

	#[test]
	fn compress_2x_seq_chains_two_blocks() {
		// A two-block message: the first block is not final, the second is.
		let h: [u32; 8] = [
			IV[0] ^ 0x0101_0020,
			IV[1],
			IV[2],
			IV[3],
			IV[4],
			IV[5],
			IV[6],
			IV[7],
		];
		let m1 = [0u32; 16];
		let m2: [u32; 16] = array::from_fn(|i| i as u32);
		let (s2, s1) = run_compress_2x_seq(h, m1, m2, 64, 0, 0, 100, 0, 0xFFFF_FFFF);
		let exp_s1 = ref_compress(h, m1, 64, 0, 0);
		let exp_s2 = ref_compress(exp_s1, m2, 100, 0, 0xFFFF_FFFF);
		assert_eq!(s1, exp_s1);
		assert_eq!(s2, exp_s2);
	}

	#[test]
	fn compress_2x_seq_distinct_params() {
		// Invariant: the hinted first output is bound twice.
		//
		// Once as the second compression's input state.
		//
		// Once against the first compression's own in-circuit output.
		//
		// Fixture: a non-IV starting state and two unrelated blocks exercise the full lane
		// packing.
		let h: [u32; 8] = [
			0xDEAD_BEEF,
			0xCAFE_BABE,
			0x1234_5678,
			0x9ABC_DEF0,
			0x0BAD_F00D,
			0xFEED_FACE,
			0x0123_4567,
			0x89AB_CDEF,
		];
		let m1: [u32; 16] = array::from_fn(|i| (i as u32).wrapping_mul(0xDEAD_BEEF));
		let m2: [u32; 16] = array::from_fn(|i| (i as u32).wrapping_mul(0x0101_0101));
		let (s2, s1) = run_compress_2x_seq(h, m1, m2, 64, 0, 0, 40, 0, 0xFFFF_FFFF);
		let exp_s1 = ref_compress(h, m1, 64, 0, 0);
		let exp_s2 = ref_compress(exp_s1, m2, 40, 0, 0xFFFF_FFFF);
		assert_eq!(s1, exp_s1);
		assert_eq!(s2, exp_s2);
	}

	// Runs the two-lane compression's own gate path directly, over its flat packed-word
	// interface, bypassing the chip machinery.
	fn run_compress_2x_words(inputs: [u64; 27]) -> [u64; 8] {
		let builder = CircuitBuilder::new();
		let wires: [Wire; 27] = array::from_fn(|_| builder.add_witness());
		let out = compress_2x_gates(
			&builder,
			array::from_fn(|i| wires[i]),
			array::from_fn(|i| wires[8 + i]),
			wires[24],
			wires[25],
			wires[26],
		);
		for wire in out {
			builder.mark_inout(wire);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for (wire, word) in iter::zip(wires, inputs) {
			w[wire] = Word(word);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		array::from_fn(|i| w[out[i]].as_u64())
	}

	// A 32-bit word, weighted towards the values that stress carry propagation.
	//
	// One case in four is drawn from the boundary set rather than uniformly.
	//
	// - Zero and one exercise the shortest carry chains.
	// - All-ones makes every position carry.
	// - The lone top bit and the all-ones-below-it value straddle the lane boundary.
	fn word32() -> impl Strategy<Value = u32> {
		prop_oneof![
			3 => any::<u32>(),
			1 => prop_oneof![Just(0), Just(1), Just(u32::MAX), Just(1 << 31), Just(u32::MAX >> 1)],
		]
	}

	fn h8() -> impl Strategy<Value = [u32; 8]> {
		prop::array::uniform8(word32())
	}

	fn block16() -> impl Strategy<Value = [u32; 16]> {
		prop::array::uniform16(word32())
	}

	proptest! {
		// Every case compiles and evaluates a whole compression circuit.
		//
		// So the sample stays small, and the boundary weighting above carries the coverage.
		#![proptest_config(ProptestConfig::with_cases(16))]

		#[test]
		fn compress_matches_reference(
			h in h8(), m in block16(), t_lo in any::<u32>(), t_hi in any::<u32>(), last in any::<u32>(),
		) {
			prop_assert_eq!(run_compress(h, m, t_lo, t_hi, last), ref_compress(h, m, t_lo, t_hi, last));
		}

		#[test]
		fn compress_2x_lanes_are_independent(
			h0 in h8(), h1 in h8(), m0 in block16(), m1 in block16(),
			t0 in any::<u32>(), t1 in any::<u32>(), l0 in any::<u32>(), l1 in any::<u32>(),
		) {
			// Invariant: the two lanes share one core, but must not leak into each other.
			//
			// Every parameter differs per lane, so any carry or rotate crossing bit 32 shows
			// up.
			let actual = run_compress_2x([h0, h1], [m0, m1], [t0, t1], [0, 0], [l0, l1]);
			prop_assert_eq!(actual[0], ref_compress(h0, m0, t0, 0, l0));
			prop_assert_eq!(actual[1], ref_compress(h1, m1, t1, 0, l1));
		}

		#[test]
		fn compress_2x_hint_matches_its_gates(words in prop::collection::vec(any::<u64>(), 27)) {
			// Invariant: the chip's off-circuit computation and its in-circuit gate path
			// must agree on every word a circuit can reach them with.
			//
			// A random 64-bit word is a valid lane pair here, so this covers the whole
			// interface.
			let inputs: [u64; 27] = array::from_fn(|i| words[i]);

			let mut hinted = [Word::ZERO; 8];
			Blake2sCompress2x.execute(&[], &inputs.map(Word), &mut hinted);

			prop_assert_eq!(hinted.map(|word| word.as_u64()), run_compress_2x_words(inputs));
		}

		#[test]
		fn compress_2x_seq_matches_two_chained_references(
			h in h8(), m1 in block16(), m2 in block16(),
			t1 in any::<u32>(), l1 in any::<u32>(), t2 in any::<u32>(), l2 in any::<u32>(),
		) {
			// Invariant: the two lanes run one after the other, not side by side.
			//
			// The second compression's input state is the first one's output.
			//
			//     h --block 1--> S1 --block 2--> S2
			//
			// The first output arrives through a hint, so this is what pins that hint honest.
			let (s2, s1) = run_compress_2x_seq(h, m1, m2, t1, 0, l1, t2, 0, l2);
			let exp_s1 = ref_compress(h, m1, t1, 0, l1);
			prop_assert_eq!(s1, exp_s1);
			prop_assert_eq!(s2, ref_compress(exp_s1, m2, t2, 0, l2));
		}
	}
}
