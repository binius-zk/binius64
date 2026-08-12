// Copyright 2026 The Binius Developers

//! A chip-accelerated CRC-64/GO-ISO circuit, exercising the chip path end to end.
//!
//! Mixing one message word into the register is a chip; the main circuit calls it once per word.
//! Main learns each register state from a hint rather than computing it, so its own constraints
//! say nothing about the CRC at all — the chip calls are what tie the states together.
//!
//! The plain circuit this mirrors is the CRC-64 fixture in this crate's `test_utils`.

use std::iter;

use binius_core::Word;
use binius_frontend::{CircuitBuilder, CircuitM4, Wire, hints::Hint};
use binius_m4_prover::WitnessM4;

/// The CRC-64/GO-ISO generator polynomial, in reflected form.
const POLY_REFLECTED: u64 = 0xd800_0000_0000_0000;
/// The register is preset to all ones before absorbing the message.
const INIT: u64 = 0xffff_ffff_ffff_ffff;
/// The final register is XORed with all ones before being returned.
const XOR_OUT: u64 = 0xffff_ffff_ffff_ffff;

/// Mixes one message word into the register, absorbing its bits from bit 0 up.
fn mix_word(mut crc: u64, word: u64) -> u64 {
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

/// Computes CRC-64/GO-ISO over `words`, absorbing them in index order.
fn crc64_iso_reference(words: &[u64]) -> u64 {
	words.iter().fold(INIT, |crc, &word| mix_word(crc, word)) ^ XOR_OUT
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

/// The chip, as a system of its own: inout words `(crc_in, word, crc_out)`.
///
/// `crc_out` is promoted rather than declared, so the chip states the relation in the words it
/// already computes rather than paying for a fourth and an assertion.
fn mix_chip() -> CircuitM4 {
	let builder = CircuitBuilder::new();
	let crc_in = builder.add_inout();
	let word = builder.add_inout();
	builder.mark_inout(mix_word_gates(&builder, crc_in, word));
	builder.build_m4()
}

/// A chip-accelerated CRC-64/GO-ISO circuit over message words.
struct Crc64ChipCircuit {
	circuit: CircuitM4,
	input: Vec<Wire>,
	output: Wire,
}

/// Builds the circuit over `n_words` message words: one chip, called once per word.
fn crc64_chip_circuit(n_words: usize) -> Crc64ChipCircuit {
	let builder = CircuitBuilder::new();
	let chip = builder.add_chip(mix_chip());

	let input = (0..n_words)
		.map(|_| builder.add_inout())
		.collect::<Vec<_>>();

	// Each round hands the register state to the chip and takes the next one back from the hint.
	// Nothing in this circuit computes the mixing, so the call is the only thing tying the two.
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

#[test]
fn chip_accelerated_crc64_matches_the_reference() {
	let words = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 0, u64::MAX];
	let c = crc64_chip_circuit(words.len());
	c.circuit.validate().unwrap();
	let cs = c.circuit.to_constraint_system();
	cs.validate().unwrap();

	let witness = WitnessM4::generate(&c.circuit, |filler| {
		for (&wire, &word) in iter::zip(&c.input, &words) {
			filler[wire] = Word(word);
		}
	})
	.unwrap();

	// The hints carried the whole computation, and they carried it correctly.
	let main = &c.circuit.main.circuit;
	assert_eq!(witness.main[main.witness_index(c.output)].as_u64(), crc64_iso_reference(&words));

	// Everything else is the verifier's: main's constraints, every chip instance's, and the calls
	// each instance serves. The calls are what hold the hints to the mixing — a hint returning the
	// wrong state leaves its instance's recomputed word disagreeing with the call.
	witness.verify(&cs).unwrap();
}
