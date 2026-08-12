// Copyright 2026 The Binius Developers

//! A CRC-64/GO-ISO circuit and reference implementation for building shift-heavy test witnesses.
//!
//! It is shared by the shift-reduction and constraint-reduction tests.
//! Both need a circuit whose AND-constraint operands carry real shifts.

use binius_core::{ValueTable, word::Word};
use binius_frontend::{Circuit, CircuitBuilder, Wire};

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

/// A circuit computing CRC-64/GO-ISO over four message words.
///
/// The message words are inout wires and the CRC is promoted to one, so both are the circuit's
/// public interface and the private witness holds only the register's intermediate states.
pub struct Crc64Circuit {
	pub circuit: Circuit,
	pub input: [Wire; N_INPUT_WORDS],
	pub output: Wire,
}

/// Builds the CRC-64/GO-ISO circuit, mirroring [`crc64_iso_reference`] gate for gate.
pub fn crc64_circuit() -> Crc64Circuit {
	let builder = CircuitBuilder::new();

	// The four message words are public inputs.
	let input = std::array::from_fn(|_| builder.add_inout());

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

	// Apply the final output XOR to produce the CRC value, and promote it to a public output.
	// That promotion is what keeps the CRC computation alive under dead-code elimination.
	let output = builder.bxor(crc, builder.add_constant_64(XOR_OUT));
	builder.mark_inout(output);

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
/// Circuit evaluation derives the rest, including the public CRC.
pub fn populate_crc64_witness(c: &Crc64Circuit, inputs: &[[u64; N_INPUT_WORDS]]) -> ValueTable {
	let log_instances = inputs.len().ilog2() as usize;
	c.circuit
		.populate_batch(log_instances, |i, filler| {
			for (wire, &w) in c.input.iter().zip(&inputs[i]) {
				filler[*wire] = Word(w);
			}
		})
		.unwrap()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn circuit_matches_reference() {
		let c = crc64_circuit();

		// A handful of fixed inputs, checked against the standalone reference implementation.
		let cases: [[u64; N_INPUT_WORDS]; 3] = [
			[0, 0, 0, 0],
			[1, 2, 3, 4],
			[
				0x0123456789abcdef,
				0xfedcba9876543210,
				0xdeadbeefcafebabe,
				0x00ff00ff00ff00ff,
			],
		];

		for words in cases {
			let mut filler = c.circuit.new_witness_filler();
			for (wire, &w) in c.input.iter().zip(&words) {
				filler[*wire] = Word(w);
			}
			c.circuit.populate_wire_witness(&mut filler).unwrap();

			assert_eq!(filler[c.output], Word(crc64_iso_reference(&words)));
		}
	}
}
