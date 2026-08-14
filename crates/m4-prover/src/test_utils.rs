// Copyright 2026 The Binius Developers

//! A CRC-64/GO-ISO circuit and reference implementation for building shift-heavy test witnesses.
//!
//! It is shared by the shift-reduction and constraint-reduction tests.
//! Both need a circuit whose AND-constraint operands carry real shifts.

use binius_compute::GlobalAllocator;
use binius_core::{ValueTable, word::Word};
use binius_frontend::{Circuit, CircuitBuilder, CircuitM4, Wire, hints::Hint};

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
	words.iter().fold(INIT, |crc, &word| mix_word(crc, word)) ^ XOR_OUT
}

/// Mixes one message word into the register, absorbing its bits from bit 0 up.
///
/// This is one iteration of [`crc64_iso_reference`]'s fold, which the chip fixture below makes a
/// chip of its own.
pub fn mix_word(mut crc: u64, word: u64) -> u64 {
	for i in 0..64 {
		let bit = (word >> i) & 1;
		let mix = (crc ^ bit) & 1;
		crc >>= 1;
		if mix != 0 {
			crc ^= POLY_REFLECTED;
		}
	}
	crc
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
		.populate_batch(&GlobalAllocator, log_instances, |i, filler| {
			for (wire, &w) in c.input.iter().zip(&inputs[i]) {
				filler[*wire] = Word(w);
			}
		})
		.unwrap()
}

/// Mixes one message word into the register in gates, mirroring [`mix_word`] round for round.
fn mix_word_gates(builder: &CircuitBuilder, mut crc: Wire, word: Wire) -> Wire {
	let poly = builder.add_constant_64(POLY_REFLECTED);
	for i in 0..64 {
		// Isolate message bit `i` into the low bit; the higher bits are junk we discard.
		let bit = if i == 0 { word } else { builder.shr(word, i) };

		// The low bit that decides whether the polynomial is mixed in this round.
		let mixed = builder.bxor(crc, bit);

		// Broadcast that low bit across the whole word: all ones iff it is set, else zero.
		let to_msb = builder.shl(mixed, 63);
		let mask = builder.sar(to_msb, 63);
		let poly_term = builder.band(mask, poly);

		// Advance the register: shift right by one, then conditionally mix the polynomial.
		crc = builder.bxor(builder.shr(crc, 1), poly_term);
	}
	crc
}

/// The register state one call produces, which main takes on trust and the call constrains.
struct MixWord;

impl Hint for MixWord {
	const NAME: &'static str = "crc64_iso_mix_word";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(2, 1)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		outputs[0] = Word(mix_word(inputs[0].as_u64(), inputs[1].as_u64()));
	}
}

/// A chip-accelerated CRC-64/GO-ISO circuit over message words.
pub struct Crc64ChipCircuit {
	pub circuit: CircuitM4,
	pub input: Vec<Wire>,
	pub output: Wire,
}

/// Builds a composite CRC-64/GO-ISO circuit over `n_words` message words: one chip, called once
/// per word.
///
/// Mixing one word into the register is the chip. Main learns each register state from a hint
/// rather than computing it, so its own constraints say nothing about the CRC — the chip calls are
/// what tie the states together. That makes main much smaller than the chip it calls, which is
/// what a composite fixture wants: the two sub-systems commit oracles of different sizes.
pub fn crc64_chip_circuit(n_words: usize) -> Crc64ChipCircuit {
	let chip = {
		let builder = CircuitBuilder::new();
		let crc_in = builder.add_inout();
		let word = builder.add_inout();
		// `crc_out` is promoted rather than declared, so the chip states the relation in the word
		// it already computes rather than paying for a third inout and an assertion.
		builder.mark_inout(mix_word_gates(&builder, crc_in, word));
		builder.build_m4()
	};

	let builder = CircuitBuilder::new();
	let chip = builder.add_chip(chip);

	let input = (0..n_words)
		.map(|_| builder.add_inout())
		.collect::<Vec<_>>();

	let mut crc = builder.add_constant_64(INIT);
	for &word in &input {
		let next = builder.call_hint(MixWord, &[], &[crc, word])[0];
		builder.call_chip(chip, &[crc, word, next]);
		crc = next;
	}

	let output = builder.bxor(crc, builder.add_constant_64(XOR_OUT));
	builder.mark_inout(output);

	Crc64ChipCircuit {
		circuit: builder.build_m4(),
		input,
		output,
	}
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
