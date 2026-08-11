// Copyright 2026 The Binius Developers

//! The recursion pipeline, end to end, over a CRC-64 inner circuit.
//!
//! ```text
//!   build crc64  ->  prove it  ->  transcript
//!        |                              |
//!        |  verifier run symbolically   |  verifier run for real
//!        v                              v
//!   recursive circuit  <----------  its witness  ->  validated
//! ```
//!
//! What this does *not* show is that the recursive circuit rejects anything. The skeleton leaves
//! the Fiat-Shamir state and the Merkle openings unconstrained, so the values that would pin them
//! are supplied by the replay rather than derived. What it does show is that the shapes line up:
//! one verifier runs over both channels, reaching the same operations in the same order, and the
//! circuit the first produced is satisfied by the witness the second filled.

use binius_core::{constraint_system::ValueVec, word::Word};
use binius_field::arch::OptimalPackedB128;
use binius_frontend::{Circuit, CircuitBuilder, CircuitStat, Wire};
use binius_hash::StdHashSuite;
use binius_prover::Prover;
use binius_recursion::{Binius64BuilderChannel, WitnessFillerChannel};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{Verifier, config::StdChallenger};

const LOG_INV_RATE: usize = 1;
const N_INPUT_WORDS: usize = 4;

// CRC-64/GO-ISO, reflected.
const POLY_REFLECTED: u64 = 0xd800000000000000;
const INIT: u64 = 0xffffffffffffffff;
const XOR_OUT: u64 = 0xffffffffffffffff;

fn crc64_reference(words: &[u64; N_INPUT_WORDS]) -> u64 {
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

struct Crc64 {
	circuit: Circuit,
	input: [Wire; N_INPUT_WORDS],
	output: Wire,
}

/// Builds a CRC-64 circuit whose message and digest are its public interface.
fn crc64_circuit() -> Crc64 {
	let builder = CircuitBuilder::new();
	let input: [Wire; N_INPUT_WORDS] = std::array::from_fn(|_| builder.add_inout());

	let mut crc = builder.add_constant_64(INIT);
	let poly = builder.add_constant_64(POLY_REFLECTED);
	for word in input {
		for i in 0..64 {
			let bit = if i == 0 { word } else { builder.shr(word, i) };
			let mixed = builder.bxor(crc, bit);
			// Broadcast the low bit across the word: all ones iff it is set.
			let to_msb = builder.shl(mixed, 63);
			let mask = builder.sar(to_msb, 63);
			let poly_term = builder.band(mask, poly);
			let shifted = builder.shr(crc, 1);
			crc = builder.bxor(shifted, poly_term);
		}
	}
	let xor_out = builder.add_constant_64(XOR_OUT);
	let output = builder.bxor(crc, xor_out);
	builder.mark_inout(output);

	Crc64 {
		circuit: builder.build(),
		input,
		output,
	}
}

/// Fills the CRC-64 witness for one message.
fn crc64_witness(crc64: &Crc64, message: &[u64; N_INPUT_WORDS]) -> ValueVec {
	let mut filler = crc64.circuit.new_witness_filler();
	for (wire, &word) in crc64.input.iter().zip(message) {
		filler[*wire] = Word::from_u64(word);
	}
	crc64.circuit.populate_wire_witness(&mut filler).unwrap();
	assert_eq!(filler[crc64.output], Word::from_u64(crc64_reference(message)));
	filler.into_value_vec()
}

#[test]
fn recursive_circuit_is_satisfied_by_a_real_proof() {
	// --- the inner circuit and its proof -------------------------------------------------------
	let crc64 = crc64_circuit();
	let message = [0x0123456789abcdef, 0xfedcba9876543210, 1, u64::MAX];
	let witness = crc64_witness(&crc64, &message);

	let verifier =
		Verifier::<StdHashSuite>::setup(crc64.circuit.constraint_system().clone(), LOG_INV_RATE)
			.unwrap();
	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(verifier.clone()).unwrap();

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	prover.prove(&witness, &mut prover_transcript).unwrap();
	let proof = prover_transcript.finalize();

	// The proof verifies natively, so anything the recursion trips over is the recursion's fault.
	{
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.clone());
		verifier.verify(witness.inout(), &mut transcript).unwrap();
		transcript.finalize().unwrap();
	}

	// --- the recursive circuit -----------------------------------------------------------------
	// The verifier runs over the builder channel, wrapped in the same BaseFold layer the transcript
	// channel gets, and what it records becomes the circuit.
	let recorded = {
		let builder_channel = Binius64BuilderChannel::new();
		let mut channel = verifier.iop_compiler().create_channel(builder_channel);
		verifier
			.iop_verifier()
			.verify(witness.inout(), &mut channel)
			.expect("the symbolic run records rather than checks, so it cannot fail");
		let builder_channel = channel.finish().unwrap();
		builder_channel.build()
	};

	let stat = CircuitStat::collect(&recorded.circuit);
	println!(
		"recursive circuit: {} gates, {} AND, {} BMUL, {} ZERO, {} recorded inputs",
		stat.n_gates,
		stat.n_and_constraints,
		stat.n_bmul_constraints,
		stat.n_zero_constraints,
		recorded.inputs.len(),
	);
	assert!(stat.n_bmul_constraints > 0, "the verifier's field arithmetic should be recorded");

	// --- its witness ---------------------------------------------------------------------------
	// The same verifier runs again over the real transcript, and every value the circuit cannot
	// derive is written into the wire the build recorded for it.
	let mut filler = recorded.circuit.new_witness_filler();
	{
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let filler_channel = WitnessFillerChannel::<_, StdChallenger, StdHashSuite>::new(
			&mut transcript,
			&mut filler,
			recorded.inputs.clone(),
		);
		let mut channel = verifier.iop_compiler().create_channel(filler_channel);
		verifier
			.iop_verifier()
			.verify(witness.inout(), &mut channel)
			.unwrap();
		channel.finish().unwrap().finish();
	}

	recorded
		.circuit
		.populate_wire_witness(&mut filler)
		.expect("the recorded circuit is satisfied by the replayed witness");
}
