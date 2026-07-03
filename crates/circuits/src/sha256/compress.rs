// Copyright 2025 Irreducible Inc.
use std::{array, iter};

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Hint, Wire, WitnessFiller};

use crate::util::clear_high_bits;

const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const K: [u32; 64] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The internal state of SHA-256.
///
/// The state size is 256 bits. For efficiency reasons it's packed in 8 x 32-bit words, and not
/// 4 x 64-bit words.
///
/// The elements are referred to as a–h or H0–H7.
#[derive(Clone)]
pub struct State(pub [Wire; 8]);

impl State {
	pub const fn new(wires: [Wire; 8]) -> Self {
		State(wires)
	}

	pub fn public(builder: &CircuitBuilder) -> Self {
		State(std::array::from_fn(|_| builder.add_inout()))
	}

	pub fn private(builder: &CircuitBuilder) -> Self {
		State(std::array::from_fn(|_| builder.add_witness()))
	}

	pub fn iv(builder: &CircuitBuilder) -> Self {
		State(std::array::from_fn(|i| builder.add_constant(Word(IV[i] as u64))))
	}

	/// Packs the state into 4 x 64-bit words.
	pub fn pack_4x64b(&self, builder: &CircuitBuilder) -> [Wire; 4] {
		fn pack_pair(b: &CircuitBuilder, hi: Wire, lo: Wire) -> Wire {
			b.bxor(lo, b.shl(hi, 32))
		}

		[
			pack_pair(builder, self.0[0], self.0[1]),
			pack_pair(builder, self.0[2], self.0[3]),
			pack_pair(builder, self.0[4], self.0[5]),
			pack_pair(builder, self.0[6], self.0[7]),
		]
	}
}

/// SHA-256 compression function.
///
/// Runs the message schedule and 64 rounds over `state_in` for a single 512-bit block, then adds
/// the round output back into `state_in`, returning the updated state.
///
/// # Arguments
///
/// - `state_in`: the 8-word input state (each word in the low 32 bits of a wire).
/// - `m`: 16 message words for this block, each a 32-bit big-endian word in the low 32 bits of a
///   wire.
///
/// It is a PRECONDITION that the high halves of the `m` wires be empty, i.e. `m[i] & 0xffffffff ==
/// m[i]` must hold for each wire. It is the caller's responsibility to ensure this; otherwise the
/// gadget's behavior is undefined / insecure.
///
/// # Returns
///
/// The updated 8-word state.
pub fn sha256_compress(builder: &CircuitBuilder, state_in: State, m: [Wire; 16]) -> State {
	// Round constants live in the low 32 bits; the high 32 bits stay zero, matching the
	// single-lane convention that every wire's high half is empty.
	let k: [Wire; 64] = std::array::from_fn(|t| builder.add_constant(Word(K[t] as u64)));
	compress_inner(builder, state_in, m, &k)
}

/// SHA-256 compression function running two independent compressions in parallel.
///
/// Each 64-bit input wire packs two 32-bit lanes: bits `[0:32]` hold the lane-0 word, bits
/// `[32:64]` hold the lane-1 word. SHA-256's mixing is built entirely from the parallel-halves
/// gates ([`iadd_32`](CircuitBuilder::iadd_32), [`rotr32`](CircuitBuilder::rotr32),
/// [`srl32`](CircuitBuilder::srl32)) plus lane-agnostic `bxor`/`band`, so the schedule and all 64
/// rounds run both compressions at the gate cost of a single one. The only lane-specific detail is
/// the round constants, which are replicated into both halves here.
///
/// # Arguments
///
/// - `state_in`: the 8-word input state, each wire packing both lanes' words.
/// - `m`: 16 message words for this block, each wire packing both lanes' words.
///
/// It is a PRECONDITION that each 32-bit lane be a valid 32-bit value (no spillover across the
/// lane boundary), i.e. the input words fit in their respective halves. It is the caller's
/// responsibility to ensure this; otherwise the gadget's behavior is undefined / insecure.
///
/// # Returns
///
/// The updated 8-word state, with each wire packing both lanes' results.
pub fn sha256_compress_2x(builder: &CircuitBuilder, state_in: State, m: [Wire; 16]) -> State {
	// Round constants replicated into both 32-bit halves, so each lane adds the same K[t].
	let k: [Wire; 64] = std::array::from_fn(|t| {
		let kt = K[t] as u64;
		builder.add_constant(Word(kt | (kt << 32)))
	});
	compress_inner(builder, state_in, m, &k)
}

/// Two *sequential* SHA-256 block compressions evaluated in one parallel core.
///
/// The second block's compression takes the first block's output state as its input state.
/// Both run as the two 32-bit lanes of a single parallel compression:
///
/// ```text
///     high lane [32:64]:  S1 = compress(input state, first block)
///     low  lane [0:32] :  S2 = compress(S1,          second block)
/// ```
///
/// So two chained blocks cost one compression instead of two.
///
/// The two lanes run concurrently, yet the low lane needs the high lane's *output* as its input.
/// A hint breaks this dependency by precomputing `S1` off-circuit.
/// The hinted `S1` seeds the low lane's input and is constrained two ways, so it cannot lie:
///
/// - its high half must equal the real input state (the first compression's input).
/// - its low half must equal the first compression's in-circuit output.
///
/// # Arguments
///
/// - `state_in`: 8-word input state for the first compression, value in the low 32 bits of each.
/// - `blocks`: two 16-word message blocks; `blocks[0]` feeds the first, `blocks[1]` the second.
///
/// # Preconditions
///
/// - Every input wire holds a valid 32-bit value in its low 32 bits.
/// - High halves need not be empty; they are cleared internally where it matters.
///
/// # Returns
///
/// 8 wires, each packing both output states:
///
/// - low 32 bits: the second compression's output.
/// - high 32 bits: the first compression's output.
pub fn sha256_compress_2x_seq(
	builder: &CircuitBuilder,
	state_in: State,
	blocks: [[Wire; 16]; 2],
) -> State {
	// The hint returns the merged input state directly, packed per word as:
	// - low 32 bits : first compression's output  = second compression's input (low lane)
	// - high 32 bits: first compression's input state word (high lane)
	//
	// Both halves are re-derived and constrained below, so the hint is untrusted.
	let mut hint_inputs = Vec::with_capacity(24);
	hint_inputs.extend_from_slice(&state_in.0);
	hint_inputs.extend_from_slice(&blocks[0]);
	let merged_vec = builder.call_hint(Sha256CompressHint, &[], &hint_inputs);
	let merged: [Wire; 8] = array::from_fn(|i| merged_vec[i]);

	// Pack two lanes into one wire: low 32 bits = lane 0 (S2), high 32 bits = lane 1 (S1).
	// `shl` already clears the high operand's upper bits.
	// The low operand is cleared explicitly, since inputs are not guaranteed zero-extended.
	let pack = |lo: Wire, hi: Wire| builder.bxor(lo, builder.shl(hi, 32));
	let clear = |w: Wire| clear_high_bits(builder, w, 32);

	// Bind the high half of each merged word to the real input state.
	// The high half is the first compression's claimed input.
	// This proves the high lane compresses the genuine state.
	for (m, s) in iter::zip(merged, state_in.0) {
		builder.assert_eq("sha256_compress_2x_seq.state_in", builder.shr(m, 32), clear(s));
	}

	// Merged block: low lane = second block, high lane = first block.
	let merged_block: [Wire; 16] = array::from_fn(|i| pack(clear(blocks[1][i]), blocks[0][i]));

	let out = sha256_compress_2x(builder, State::new(merged), merged_block);

	// Bind the hint's honesty by equating the two derivations of the first output:
	// - first compression's in-circuit output = high lane of the result
	// - value the hint fed as the second input = low half of each merged word
	for (m, o) in iter::zip(merged, out.0) {
		builder.assert_eq("sha256_compress_2x_seq.s1_out", clear(m), builder.shr(o, 32));
	}

	out
}

/// Precomputes the merged input state for the sequential two-lane compression.
///
/// Runs the first compression off-circuit and packs each output word to seed both lanes at once:
///
/// - low 32 bits: the first compression's output = the second compression's input.
/// - high 32 bits: the first compression's input state word.
///
/// Both halves are re-derived and constrained in-circuit, so the hint only needs to be honest.
///
/// Input layout, 24 words with the value in the low 32 bits of each:
///
/// - words `0..8`: the input state.
/// - words `8..24`: the first message block.
struct Sha256CompressHint;

impl Hint for Sha256CompressHint {
	const NAME: &'static str = "binius.sha256_compress";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(24, 8)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let state_in: [u32; 8] = array::from_fn(|i| inputs[i].as_u64() as u32);
		let block: [u32; 16] = array::from_fn(|i| inputs[8 + i].as_u64() as u32);

		let out = ref_compress(state_in, block);
		for (i, slot) in outputs.iter_mut().enumerate() {
			*slot = Word(out[i] as u64 | ((state_in[i] as u64) << 32));
		}
	}
}

/// Pure-Rust SHA-256 compression of a single 512-bit block.
///
/// Matches the in-circuit compression exactly.
/// Used for prover-side witness generation and as the test reference.
///
/// # Arguments
///
/// - `state_in`: the 8-word input state, one 32-bit word per entry.
/// - `m`: the 16-word message block, one 32-bit word per entry.
///
/// # Returns
///
/// The updated 8-word state.
pub fn ref_compress(state_in: [u32; 8], m: [u32; 16]) -> [u32; 8] {
	let mut w = [0u32; 64];
	w[..16].copy_from_slice(&m);
	for t in 16..64 {
		let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
		let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
		w[t] = w[t - 16]
			.wrapping_add(s0)
			.wrapping_add(w[t - 7])
			.wrapping_add(s1);
	}

	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state_in;
	for t in 0..64 {
		let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
		let ch = (e & f) ^ ((!e) & g);
		let t1 = h
			.wrapping_add(big_s1)
			.wrapping_add(ch)
			.wrapping_add(K[t])
			.wrapping_add(w[t]);
		let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
		let maj = (a & b) ^ (a & c) ^ (b & c);
		let t2 = big_s0.wrapping_add(maj);
		h = g;
		g = f;
		f = e;
		e = d.wrapping_add(t1);
		d = c;
		c = b;
		b = a;
		a = t1.wrapping_add(t2);
	}

	[
		state_in[0].wrapping_add(a),
		state_in[1].wrapping_add(b),
		state_in[2].wrapping_add(c),
		state_in[3].wrapping_add(d),
		state_in[4].wrapping_add(e),
		state_in[5].wrapping_add(f),
		state_in[6].wrapping_add(g),
		state_in[7].wrapping_add(h),
	]
}

/// Shared core of [`sha256_compress`] and [`sha256_compress_2x`]: runs the message schedule and 64
/// rounds over `state_in`, then adds the round output back into `state_in`.
///
/// `k` supplies the 64 round constants as wires, letting the caller decide whether they occupy the
/// low 32 bits only (single-lane) or both halves (two-lane). Every other operation is a
/// parallel-halves or lane-agnostic gate, so the same code serves one or two compressions.
fn compress_inner(
	builder: &CircuitBuilder,
	state_in: State,
	m: [Wire; 16],
	k: &[Wire; 64],
) -> State {
	// ---- message-schedule ----
	// W[0..15] = block_words
	// for t = 16 .. 63:
	//     s0   = σ0(W[t-15])
	//     s1   = σ1(W[t-2])
	//     (p, _)  = Add32(W[t-16], s0)
	//     (q, _)  = Add32(p, W[t-7])
	//     (W[t],_) = Add32(q, s1)

	let mut w: Vec<Wire> = Vec::with_capacity(64);
	// W[0..15] = block_words
	w.extend_from_slice(&m);

	// W[16..63] computed from previous W values
	for t in 16..64 {
		let s0 = small_sigma_0(builder, w[t - 15]);
		let s1 = small_sigma_1(builder, w[t - 2]);
		let p = builder.iadd_32(w[t - 16], s0);
		let q = builder.iadd_32(p, w[t - 7]);
		w.push(builder.iadd_32(q, s1));
	}

	let w: &[Wire; 64] = (&*w).try_into().unwrap();
	let mut state = state_in.clone();
	for t in 0..64 {
		state = round(builder, k[t], w[t], state);
	}

	// Add the compressed chunk to the current hash value
	State([
		builder.iadd_32(state_in.0[0], state.0[0]),
		builder.iadd_32(state_in.0[1], state.0[1]),
		builder.iadd_32(state_in.0[2], state.0[2]),
		builder.iadd_32(state_in.0[3], state.0[3]),
		builder.iadd_32(state_in.0[4], state.0[4]),
		builder.iadd_32(state_in.0[5], state.0[5]),
		builder.iadd_32(state_in.0[6], state.0[6]),
		builder.iadd_32(state_in.0[7], state.0[7]),
	])
}

/// Populates the 16 message-block wires of a [`sha256_compress`] block from its 64 message bytes.
///
/// The bytes are packed big-endian into 16 32-bit words, one per wire, with the high 32 bits left
/// zero — matching the precondition on `m` documented in [`sha256_compress`].
pub fn populate_message_block(w: &mut WitnessFiller, m: &[Wire; 16], bytes: [u8; 64]) {
	for (wire, chunk) in m.iter().zip(bytes.chunks_exact(4)) {
		let word = u32::from_be_bytes(chunk.try_into().unwrap());
		w[*wire] = Word(word as u64);
	}
}

fn round(builder: &CircuitBuilder, k_t: Wire, w_t: Wire, state: State) -> State {
	let State([a, b, c, d, e, f, g, h]) = state;

	let big_sigma_e = big_sigma_1(builder, e);
	let ch_efg = ch(builder, e, f, g);
	let t1a = builder.iadd_32(h, big_sigma_e);
	let t1b = builder.iadd_32(t1a, ch_efg);
	let t1c = builder.iadd_32(t1b, k_t);
	let t1 = builder.iadd_32(t1c, w_t);

	let big_sigma_a = big_sigma_0(builder, a);
	let maj_abc = maj(builder, a, b, c);
	let t2 = builder.iadd_32(big_sigma_a, maj_abc);

	let h = g;
	let g = f;
	let f = e;
	let e = builder.iadd_32(d, t1);
	let d = c;
	let c = b;
	let b = a;
	let a = builder.iadd_32(t1, t2);

	State([a, b, c, d, e, f, g, h])
}

/// Ch(x, y, z) = (x AND y) XOR (NOT y AND z)
///             = z XOR (x AND (y XOR z))
fn ch(builder: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
	builder.bxor(z, builder.band(x, builder.bxor(y, z)))
}

/// Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
///              = (x XOR z) AND (y XOR z) XOR z.
fn maj(builder: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
	builder.bxor(builder.band(builder.bxor(x, z), builder.bxor(y, z)), z)
}

/// Σ0(a)       = XOR( XOR( ROTR(x,  2), ROTR(x, 13) ), ROTR(x, 22) )
fn big_sigma_0(b: &CircuitBuilder, x: Wire) -> Wire {
	let r1 = b.rotr32(x, 2);
	let r2 = b.rotr32(x, 13);
	let r3 = b.rotr32(x, 22);
	let x1 = b.bxor(r1, r2);
	b.bxor(x1, r3)
}

/// Σ1(e)       = XOR( XOR( ROTR(x,  6), ROTR(x, 11) ), ROTR(x, 25) )
fn big_sigma_1(b: &CircuitBuilder, x: Wire) -> Wire {
	let r1 = b.rotr32(x, 6);
	let r2 = b.rotr32(x, 11);
	let r3 = b.rotr32(x, 25);
	let x1 = b.bxor(r1, r2);
	b.bxor(x1, r3)
}

/// σ0(x)       = XOR( XOR( ROTR(x,  7), ROTR(x, 18) ), SHR(x,  3) )
fn small_sigma_0(b: &CircuitBuilder, x: Wire) -> Wire {
	let r1 = b.rotr32(x, 7);
	let r2 = b.rotr32(x, 18);
	let s1 = b.srl32(x, 3);
	let x1 = b.bxor(r1, r2);
	b.bxor(x1, s1)
}

/// σ1(x)       = XOR( XOR( ROTR(x, 17), ROTR(x, 19) ), SHR(x, 10) )
fn small_sigma_1(b: &CircuitBuilder, x: Wire) -> Wire {
	let r1 = b.rotr32(x, 17);
	let r2 = b.rotr32(x, 19);
	let s1 = b.srl32(x, 10);
	let x1 = b.bxor(r1, r2);
	b.bxor(x1, s1)
}

#[cfg(test)]
mod tests {
	use binius_core::{verify::verify_constraints, word::Word};
	use binius_frontend::{CircuitBuilder, Wire};

	use super::{
		IV, State, populate_message_block, ref_compress, sha256_compress, sha256_compress_2x,
		sha256_compress_2x_seq,
	};

	/// A test circuit that proves a knowledge of preimage for a given state vector S in
	///
	///     compress512(preimage) = S
	///
	/// without revealing the preimage, only S.
	#[test]
	fn proof_preimage() {
		// Use the test-vector for SHA256 single block message: "abc".
		let mut preimage: [u8; 64] = [0; 64];
		preimage[0..3].copy_from_slice(b"abc");
		preimage[3] = 0x80;
		preimage[63] = 0x18;

		#[rustfmt::skip]
		let expected_state: [u32; 8] = [
			0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
			0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
		];

		let circuit = CircuitBuilder::new();
		let state = State::iv(&circuit);
		let input: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());
		let output: [Wire; 8] = std::array::from_fn(|_| circuit.add_inout());
		let state_out = sha256_compress(&circuit, state, input);

		// Mask to only low 32-bit.
		let mask32 = circuit.add_constant(Word::MASK_32);
		for (i, (actual_x, expected_x)) in state_out.0.iter().zip(output).enumerate() {
			circuit.assert_eq(
				format!("preimage_eq[{i}]"),
				circuit.band(*actual_x, mask32),
				expected_x,
			);
		}

		let circuit = circuit.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		// Populate the input message for the compression function.
		populate_message_block(&mut w, &input, preimage);

		for (i, &output) in output.iter().enumerate() {
			w[output] = Word(expected_state[i] as u64);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	#[test]
	fn sha256_chain() {
		// Tests multiple SHA-256 compress512 invocations where the outputs are linked to the inputs
		// of the following compression function.
		const N: usize = 3;
		let circuit = CircuitBuilder::new();

		let mut m_vec = Vec::with_capacity(N);

		// First, declare the initial state.
		let mut state = State::iv(&circuit);
		for i in 0..N {
			// Create a new subcircuit builder. This is not necessary but can improve readability
			// and diagnostics.
			let sha256_builder = circuit.subcircuit(format!("sha256[{i}]"));

			// Build a new instance of the sha256 verification subcircuit, passing the inputs `m` to
			// it. For the first compression `m` is public but everything else if private.
			let m: [Wire; 16] = if i == 0 {
				std::array::from_fn(|_| sha256_builder.add_inout())
			} else {
				std::array::from_fn(|_| sha256_builder.add_witness())
			};
			state = sha256_compress(&sha256_builder, state, m);

			m_vec.push(m);
		}

		let circuit = circuit.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		for m in &m_vec {
			populate_message_block(&mut w, m, [0; 64]);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	#[test]
	fn sha256_parallel() {
		// Test multiple SHA-256 compressions in parallel (no chaining)
		const N: usize = 3;
		let circuit = CircuitBuilder::new();

		let mut m_vec = Vec::with_capacity(N);

		for i in 0..N {
			// Create a new subcircuit builder
			let sha256_builder = circuit.subcircuit(format!("sha256[{i}]"));

			// Each SHA-256 instance gets its own IV and input (all committed)
			let state = State::iv(&sha256_builder);
			let m: [Wire; 16] = std::array::from_fn(|_| sha256_builder.add_inout());
			sha256_compress(&sha256_builder, state, m);

			m_vec.push(m);
		}

		let circuit = circuit.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		for m in &m_vec {
			populate_message_block(&mut w, m, [0; 64]);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	fn pack2x(lo: u32, hi: u32) -> u64 {
		(lo as u64) | ((hi as u64) << 32)
	}

	/// Runs `sha256_compress_2x` with lane 0 = `(state0, m0)` and lane 1 = `(state1, m1)`, and
	/// asserts each output lane against [`ref_compress`].
	fn run_2x(state0: [u32; 8], m0: [u32; 16], state1: [u32; 8], m1: [u32; 16]) {
		let exp0 = ref_compress(state0, m0);
		let exp1 = ref_compress(state1, m1);

		let circuit = CircuitBuilder::new();
		let state_in: [Wire; 8] = std::array::from_fn(|_| circuit.add_witness());
		let m: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());
		let out = sha256_compress_2x(&circuit, State::new(state_in), m);
		let out_inout: [Wire; 8] = std::array::from_fn(|_| circuit.add_inout());
		for i in 0..8 {
			circuit.assert_eq(format!("out[{i}]"), out.0[i], out_inout[i]);
		}

		let circuit = circuit.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();
		for i in 0..8 {
			w[state_in[i]] = Word(pack2x(state0[i], state1[i]));
		}
		for i in 0..16 {
			w[m[i]] = Word(pack2x(m0[i], m1[i]));
		}
		for i in 0..8 {
			w[out_inout[i]] = Word(pack2x(exp0[i], exp1[i]));
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	/// The single-block padding of the message "abc": `"abc" || 0x80 || 0*` with the 24-bit length
	/// in the final word.
	fn abc_block() -> [u32; 16] {
		let mut m = [0u32; 16];
		m[0] = 0x6162_6380;
		m[15] = 0x0000_0018;
		m
	}

	/// Known SHA-256 digest of "abc" (also the compression output for the single ABC block).
	const ABC_DIGEST: [u32; 8] = [
		0xba78_16bf,
		0x8f01_cfea,
		0x4141_40de,
		0x5dae_2223,
		0xb003_61a3,
		0x9617_7a9c,
		0xb410_ff61,
		0xf200_15ad,
	];

	#[test]
	fn compress_2x_distinct_lanes() {
		// Anchor the reference: lane 0 compresses the "abc" block from the IV, which must yield the
		// known SHA-256 digest of "abc".
		assert_eq!(ref_compress(IV, abc_block()), ABC_DIGEST);

		// Lane 1 runs a completely different (state, message) pair to confirm the lanes are
		// independent — no bits cross the 32-bit boundary.
		let state1: [u32; 8] = [
			0xdead_beef,
			0xcafe_babe,
			0x1234_5678,
			0x9abc_def0,
			0x0bad_f00d,
			0xfeed_face,
			0x0123_4567,
			0x89ab_cdef,
		];
		let m1: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0x0101_0101));

		run_2x(IV, abc_block(), state1, m1);
	}

	#[test]
	fn compress_2x_lane_independence() {
		// Lane 0 = "abc", lane 1 = an all-zero block from the IV. Each lane must match its own
		// reference, proving the zero lane does not perturb the "abc" lane and vice versa.
		run_2x(IV, abc_block(), IV, [0; 16]);
	}

	/// Runs the sequential two-block compression and checks it against the scalar reference.
	///
	/// The reference chains two single-block compressions:
	///
	/// ```text
	///     S1 = compress(state_in, block1)   → high lane
	///     S2 = compress(S1,       block2)   → low lane
	/// ```
	fn run_2x_seq(state_in: [u32; 8], block1: [u32; 16], block2: [u32; 16]) {
		// Scalar reference: the second compression consumes the first's output as its input state.
		let s1 = ref_compress(state_in, block1);
		let s2 = ref_compress(s1, block2);

		// Fresh witness wires for the input state and both message blocks.
		let circuit = CircuitBuilder::new();
		let state_wires: [Wire; 8] = std::array::from_fn(|_| circuit.add_witness());
		let block1_wires: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());
		let block2_wires: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());

		// Gadget under test: one parallel core evaluating both sequential compressions.
		let out =
			sha256_compress_2x_seq(&circuit, State::new(state_wires), [block1_wires, block2_wires]);

		// Pin the packed output to public wires so dead-code elimination keeps the computation.
		let out_inout: [Wire; 8] = std::array::from_fn(|_| circuit.add_inout());
		for i in 0..8 {
			circuit.assert_eq(format!("out[{i}]"), out.0[i], out_inout[i]);
		}

		let circuit = circuit.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		// Each 32-bit input goes in the low half of its wire; the high half stays empty.
		for i in 0..8 {
			w[state_wires[i]] = Word(state_in[i] as u64);
		}
		for i in 0..16 {
			w[block1_wires[i]] = Word(block1[i] as u64);
			w[block2_wires[i]] = Word(block2[i] as u64);
		}

		// Expected packing per word: low 32 bits = second output, high 32 bits = first output.
		for i in 0..8 {
			w[out_inout[i]] = Word(pack2x(s2[i], s1[i]));
		}

		// Witness generation runs the hint; constraint checking then verifies every gate.
		circuit.populate_wire_witness(&mut w).unwrap();
		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	/// Packs 64 message bytes into 16 big-endian 32-bit schedule words.
	fn pack_block_be(bytes: &[u8; 64]) -> [u32; 16] {
		std::array::from_fn(|i| u32::from_be_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
	}

	#[test]
	fn compress_2x_seq_matches_rfc_two_block_kat() {
		// Known-answer test: RFC 6234 SHA-256 TEST2_1.
		// A 56-byte message spans two blocks after padding, so the digest is a chained compression.
		// Chaining both blocks from the IV must reproduce the published digest.
		let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

		// Padded layout over 128 bytes = two blocks:
		//     bytes 0..56   : the message.
		//     byte  56      : the 0x80 delimiter.
		//     bytes 57..120 : zero fill.
		//     bytes 120..128: the 64-bit message bit length (56 * 8 = 448).
		let mut padded = [0u8; 128];
		padded[..56].copy_from_slice(msg);
		padded[56] = 0x80;
		padded[120..128].copy_from_slice(&(56u64 * 8).to_be_bytes());
		let block0 = pack_block_be(padded[0..64].try_into().unwrap());
		let block1 = pack_block_be(padded[64..128].try_into().unwrap());

		// Published digest (RFC 6234, SHA-256 TEST2_1).
		let expected: [u32; 8] = [
			0x248d_6a61,
			0xd206_38b8,
			0xe5c0_2693,
			0x0c3e_6039,
			0xa33c_e459,
			0x64ff_2167,
			0xf6ec_edd4,
			0x19db_06c1,
		];

		// The second compression's output is the message digest; anchor it to the RFC vector.
		let s1 = ref_compress(IV, block0);
		let s2 = ref_compress(s1, block1);
		assert_eq!(s2, expected);

		// Run the same two-block chain in-circuit and cross-check both lanes.
		run_2x_seq(IV, block0, block1);
	}

	#[test]
	fn compress_2x_seq_distinct_params() {
		// Invariant: the hinted first output is bound twice.
		//     once as the second compression's input state.
		//     once against the first compression's in-circuit output.
		//
		// Fixture: a non-IV starting state and two unrelated blocks exercise the full lane packing.
		let state_in: [u32; 8] = [
			0xdead_beef,
			0xcafe_babe,
			0x1234_5678,
			0x9abc_def0,
			0x0bad_f00d,
			0xfeed_face,
			0x0123_4567,
			0x89ab_cdef,
		];
		let block1: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0xdead_beef));
		let block2: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0x0101_0101));
		run_2x_seq(state_in, block1, block2);
	}
}
