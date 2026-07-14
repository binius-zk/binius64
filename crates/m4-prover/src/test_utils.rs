// Copyright 2026 The Binius Developers

//! A CRC-64/GO-ISO circuit and reference implementation for building shift-heavy test witnesses.
//!
//! It is shared by the shift-reduction and constraint-reduction tests.
//! Both need a circuit whose AND-constraint operands carry real shifts.

use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};

use crate::ValueTable;

/// The CRC-64/GO-ISO generator polynomial, in reflected form.
///
/// The polynomial is `x^64 + x^4 + x^3 + x + 1`, normal form `0x1b`.
/// Input and output are reflected, so it enters the register bit-reversed.
const POLY_REFLECTED: u64 = 0xd800000000000000;
/// The register is preset to all ones before absorbing the message.
const INIT: u64 = 0xffffffffffffffff;
/// The final register is XORed with all ones before being returned.
const XOR_OUT: u64 = 0xffffffffffffffff;

/// The number of 64-bit input words the CRC circuit consumes.
pub const N_INPUT_WORDS: usize = 4;

/// Computes CRC-64/GO-ISO over `words`, absorbing bits least-significant-first.
///
/// Each input word contributes its 64 bits in order from bit 0 up to bit 63.
/// The words are absorbed in index order.
///
/// This is the reflected bitwise algorithm. For every message bit:
/// - combine the register's low bit with the message bit;
/// - shift the register right by one;
/// - conditionally mix in the polynomial.
///
/// The `Circuit` counterpart mirrors this loop gate for gate, so the two agree bit for bit.
pub fn crc64_iso_reference(words: &[u64; N_INPUT_WORDS]) -> u64 {
	let mut crc = INIT;
	for &word in words {
		for i in 0..64 {
			let bit = (word >> i) & 1;
			let mix = (crc ^ bit) & 1;
			crc >>= 1;
			if mix != 0 {
				crc ^= POLY_REFLECTED;
			}
		}
	}
	crc ^ XOR_OUT
}

/// A circuit computing CRC-64/GO-ISO over four private witness words.
///
/// The four inputs are ordinary witness wires, not public inout wires.
/// So the whole computation lives in the private witness.
///
/// The output wire is force-committed.
/// Without an assertion or public output reading it, dead-code elimination would prune the CRC.
pub struct Crc64Circuit {
	pub circuit: Circuit,
	pub input: [Wire; N_INPUT_WORDS],
	pub output: Wire,
}

/// Builds the CRC-64/GO-ISO circuit, mirroring [`crc64_iso_reference`] gate for gate.
pub fn crc64_circuit() -> Crc64Circuit {
	let builder = CircuitBuilder::new();

	// The four message words are private witnesses supplied by the prover.
	let input = std::array::from_fn(|_| builder.add_witness());

	// The register starts at the all-ones preset and the polynomial is a constant.
	let mut crc = builder.add_constant_64(INIT);
	let poly = builder.add_constant_64(POLY_REFLECTED);

	for word in input {
		for i in 0..64 {
			// Isolate message bit `i` into the low bit; the higher bits are junk we discard.
			let bit = if i == 0 { word } else { builder.shr(word, i) };

			// The low bit that decides whether the polynomial is mixed in this step.
			let mixed = builder.bxor(crc, bit);

			// Broadcast that low bit across the whole word: all ones iff it is set, else zero.
			// Shifting it up to bit 63 then arithmetic-shifting back fills every bit from it.
			let to_msb = builder.shl(mixed, 63);
			let mask = builder.sar(to_msb, 63);
			let poly_term = builder.band(mask, poly);

			// Advance the register: shift right by one, then conditionally mix the polynomial.
			let shifted = builder.shr(crc, 1);
			crc = builder.bxor(shifted, poly_term);
		}
	}

	// Apply the final output XOR to produce the CRC value.
	let output = builder.bxor(crc, builder.add_constant_64(XOR_OUT));

	// Pin the output so the constraint compiler keeps the CRC computation alive.
	builder.force_commit(output);

	Crc64Circuit {
		circuit: builder.build(),
		input,
		output,
	}
}

/// Populates a wire-major batch table with one instance per input tuple.
///
/// The instance count is the number of tuples, which must be a power of two.
/// Each instance's four message words are the corresponding tuple.
/// Circuit evaluation derives the rest.
/// The circuit has no inout wires, so it is admissible in the wire-major table.
pub fn populate_crc64_witness(c: &Crc64Circuit, inputs: &[[u64; N_INPUT_WORDS]]) -> ValueTable {
	let log_instances = inputs.len().ilog2() as usize;
	ValueTable::populate(&c.circuit, log_instances, |i, filler| {
		for (wire, &w) in c.input.iter().zip(&inputs[i]) {
			filler[*wire] = Word(w);
		}
	})
	.unwrap()
}
