// Copyright 2026 The Binius Developers

// Fixtures shaped like the primitives they name, not cryptographically faithful.
// Each bench binary uses part of this module.
#![allow(dead_code)]

use std::env;

use binius_frontend::{CircuitBuilder, Wire};

/// Number of 64-bit lanes in the permutation state.
pub const STATE_LANES: usize = 25;

/// Number of rounds one permutation applies.
pub const PERMUTATION_ROUNDS: usize = 24;

/// Number of 32-bit words one compression consumes.
pub const BLOCK_WORDS: usize = 16;

/// Number of 32-bit words one compression carries between blocks.
pub const CHAINING_WORDS: usize = 8;

/// Number of rounds one compression applies.
pub const COMPRESSION_ROUNDS: usize = 64;

/// Per-lane rotation offsets, in lane order.
#[rustfmt::skip]
const RHO_OFFSETS: [u32; STATE_LANES] = [
	 0,  1, 62, 28, 27,
	36, 44,  6, 55, 20,
	 3, 10, 43, 25, 39,
	41, 45, 15, 21,  8,
	18,  2, 61, 56, 14,
];

/// Per-round constants folded into lane zero.
const ROUND_CONSTANTS: [u64; PERMUTATION_ROUNDS] = [
	0x0000_0000_0000_0001,
	0x0000_0000_0000_8082,
	0x8000_0000_0000_808a,
	0x8000_0000_8000_8000,
	0x0000_0000_0000_808b,
	0x0000_0000_8000_0001,
	0x8000_0000_8000_8081,
	0x8000_0000_0000_8009,
	0x0000_0000_0000_008a,
	0x0000_0000_0000_0088,
	0x0000_0000_8000_8009,
	0x0000_0000_8000_000a,
	0x0000_0000_8000_808b,
	0x8000_0000_0000_008b,
	0x8000_0000_0000_8089,
	0x8000_0000_0000_8003,
	0x8000_0000_0000_8002,
	0x8000_0000_0000_0080,
	0x0000_0000_0000_800a,
	0x8000_0000_8000_000a,
	0x8000_0000_8000_8081,
	0x8000_0000_0000_8080,
	0x0000_0000_8000_0001,
	0x8000_0000_8000_8008,
];

/// Per-round additive constants of the compression.
const COMPRESSION_CONSTANTS: [u32; COMPRESSION_ROUNDS] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Extends the state by one permutation, shaped like Keccak-f1600.
pub fn permutation(b: &CircuitBuilder, state: &mut [Wire; STATE_LANES]) {
	for &round_constant in &ROUND_CONSTANTS {
		// Column parities, then the correction each column absorbs.
		let mut column = [state[0]; 5];
		for (x, slot) in column.iter_mut().enumerate() {
			*slot = b.bxor_multi(&[
				state[x],
				state[x + 5],
				state[x + 10],
				state[x + 15],
				state[x + 20],
			]);
		}
		let mut correction = [state[0]; 5];
		for (x, slot) in correction.iter_mut().enumerate() {
			*slot = b.bxor(column[(x + 4) % 5], b.rotl(column[(x + 1) % 5], 1));
		}
		for y in 0..5 {
			for x in 0..5 {
				state[x + 5 * y] = b.bxor(state[x + 5 * y], correction[x]);
			}
		}

		// Rotate each lane, then move it to its permuted position.
		let mut rotated = [state[0]; STATE_LANES];
		for y in 0..5 {
			for x in 0..5 {
				let src = x + 5 * y;
				let dst = y + 5 * ((2 * x + 3 * y) % 5);
				rotated[dst] = b.rotl(state[src], RHO_OFFSETS[src]);
			}
		}

		// The only non-linear step.
		for y in 0..5 {
			for x in 0..5 {
				let hi = rotated[(x + 2) % 5 + 5 * y];
				let masked = b.band(b.bnot(rotated[(x + 1) % 5 + 5 * y]), hi);
				state[x + 5 * y] = b.bxor(rotated[x + 5 * y], masked);
			}
		}

		// Break the symmetry between rounds.
		state[0] = b.bxor(state[0], b.add_constant_64(round_constant));
	}
}

/// Extends the chaining value by one compression, shaped like a SHA-256 block.
pub fn compression(
	b: &CircuitBuilder,
	chaining: &mut [Wire; CHAINING_WORDS],
	block: &[Wire; BLOCK_WORDS],
) {
	// The block's words, then one derived word per remaining round.
	let mut schedule = Vec::with_capacity(COMPRESSION_ROUNDS);
	schedule.extend_from_slice(block);
	for i in BLOCK_WORDS..COMPRESSION_ROUNDS {
		let near = b.bxor_multi(&[
			b.rotr32(schedule[i - 15], 7),
			b.rotr32(schedule[i - 15], 18),
			b.srl32(schedule[i - 15], 3),
		]);
		let far = b.bxor_multi(&[
			b.rotr32(schedule[i - 2], 17),
			b.rotr32(schedule[i - 2], 19),
			b.srl32(schedule[i - 2], 10),
		]);
		let sum = b.iadd_32(b.iadd_32(schedule[i - 16], near), b.iadd_32(schedule[i - 7], far));
		schedule.push(sum);
	}

	// Eight working words, shifted by one position each round.
	let mut w = *chaining;
	for i in 0..COMPRESSION_ROUNDS {
		let sigma_1 = b.bxor_multi(&[b.rotr32(w[4], 6), b.rotr32(w[4], 11), b.rotr32(w[4], 25)]);
		let choose = b.bxor(b.band(w[4], w[5]), b.band(b.bnot(w[4]), w[6]));
		// Both 32-bit halves run their own lane.
		let round_constant = b.add_constant_64(u64::from(COMPRESSION_CONSTANTS[i]) * 0x1_0000_0001);
		let t1 = b.iadd_32(
			b.iadd_32(b.iadd_32(w[7], sigma_1), b.iadd_32(choose, round_constant)),
			schedule[i],
		);

		let sigma_0 = b.bxor_multi(&[b.rotr32(w[0], 2), b.rotr32(w[0], 13), b.rotr32(w[0], 22)]);
		let majority = b.bxor_multi(&[b.band(w[0], w[1]), b.band(w[0], w[2]), b.band(w[1], w[2])]);
		let t2 = b.iadd_32(sigma_0, majority);

		w = [
			b.iadd_32(t1, t2),
			w[0],
			w[1],
			w[2],
			b.iadd_32(w[3], t1),
			w[4],
			w[5],
			w[6],
		];
	}

	for (slot, word) in chaining.iter_mut().zip(w) {
		*slot = b.iadd_32(*slot, word);
	}
}

/// Reads a `usize` environment variable, returning `None` when unset or not a number.
pub fn env_usize(key: &str) -> Option<usize> {
	env::var(key).ok().and_then(|s| s.parse().ok())
}
