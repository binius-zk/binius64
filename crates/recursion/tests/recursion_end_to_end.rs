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
//! One verifier runs over both channels and reaches the same operations in the same order.
//! The circuit the first produced is satisfied by the witness the second filled.
//!
//! The Merkle commitments are checked in-circuit, so tampering with their advice is rejected here.
//! The Fiat-Shamir state is not, so a challenge is still whatever the replay supplies.

use std::iter;

use binius_core::{constraint_system::ValueVec, word::Word};
use binius_field::arch::OptimalPackedB128;
use binius_frontend::{
	Circuit, CircuitBuilder, CircuitStat, MAX_ASSERTION_FAILURES, PopulateError, Wire,
	WitnessFiller,
};
use binius_hash::StdHashSuite;
use binius_ip::channel::WordIPVerifierChannel;
use binius_prover::Prover;
use binius_recursion::{Binius64BuilderChannel, Recorded, WitnessFillerChannel};
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

/// One CRC-64 statement, proved once, for the recursion pipeline to consume.
struct Proved {
	verifier: Verifier<StdHashSuite>,
	witness: ValueVec,
	proof: Vec<u8>,
}

/// Proves one CRC-64 statement, and checks it verifies natively.
///
/// A proof that fails here would make every recursion failure below ambiguous.
fn prove_crc64() -> Proved {
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

	let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.clone());
	verifier.verify(witness.inout(), &mut transcript).unwrap();
	transcript.finalize().unwrap();

	Proved {
		verifier,
		witness,
		proof,
	}
}

/// The recursive circuit, and the public wires the inner statement was bound to.
struct Recording {
	recorded: Recorded,
	/// One inout wire per inner statement word, each constrained to equal what the verifier read.
	public: Vec<Wire>,
}

/// Records the verifier run as a circuit, with the whole inner statement made public.
///
/// The builder channel is wrapped in the same BaseFold layer the transcript channel gets.
/// What the verifier does over it becomes the circuit.
fn record(proved: &Proved) -> Recording {
	let builder_channel = Binius64BuilderChannel::new();
	let mut channel = proved
		.verifier
		.iop_compiler()
		.create_channel(builder_channel);
	// The statement is observed by the caller, and what comes back is what the verifier reads.
	// On this channel those are wires, so the recorded circuit takes the statement as input
	// rather than baking it in.
	let statement = channel.observe_words(proved.witness.inout());
	proved
		.verifier
		.iop_verifier()
		.verify(&statement, &mut channel)
		.expect("the symbolic run records rather than checks, so it cannot fail")
		.check_symbolic(&mut channel)
		.expect("the symbolic run records rather than checks, so it cannot fail");

	// Binding every word is this caller's choice, not the channel's.
	// One that exposed only part of its inner statement would bind only that part.
	let mut builder_channel = channel.finish().unwrap();
	let public = builder_channel.bind_public(statement);
	Recording {
		recorded: builder_channel.build(),
		public,
	}
}

/// Replays the verifier over the real transcript, filling every wire the build recorded.
fn replay(proved: &Proved, recorded: &Recorded, filler: &mut WitnessFiller) {
	let mut transcript = VerifierTranscript::new(StdChallenger::default(), proved.proof.clone());
	let filler_channel = WitnessFillerChannel::<_, StdChallenger, StdHashSuite>::new(
		&mut transcript,
		filler,
		recorded.inputs.clone(),
	);
	let mut channel = proved
		.verifier
		.iop_compiler()
		.create_channel(filler_channel);
	let inout = channel.observe_words(proved.witness.inout());
	proved
		.verifier
		.iop_verifier()
		.verify(&inout, &mut channel)
		.unwrap()
		.check_symbolic(&mut channel)
		.unwrap();
	channel.finish().unwrap().finish();
}

#[test]
fn recursive_circuit_is_satisfied_by_a_real_proof() {
	let proved = prove_crc64();
	let Recording { recorded, public } = record(&proved);
	let witness = &proved.witness;

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

	// The statement still reaches the verifier as wires the replay fills, one per inout word.
	assert_eq!(
		recorded
			.inputs
			.iter()
			.filter(|input| input.kind == "observe_words")
			.count(),
		witness.inout().len()
	);
	// One public wire was bound beside each of them.
	assert_eq!(public.len(), witness.inout().len());

	// The decommitted layer is on wires now, which is what the opened leaves are matched against.
	assert!(
		recorded
			.inputs
			.iter()
			.any(|input| input.kind == "merkle_layer"),
		"the query phase must record the layer it decommits to"
	);

	// --- its witness ---------------------------------------------------------------------------
	// Whoever checks the outer proof supplies the public half of each binding.
	let mut filler = recorded.circuit.new_witness_filler();
	for (&wire, &word) in iter::zip(&public, witness.inout()) {
		filler[wire] = word;
	}

	// The same verifier runs again over the real transcript, and every value the circuit cannot
	// derive is written into the wire the build recorded for it.
	replay(&proved, &recorded, &mut filler);

	recorded
		.circuit
		.populate_wire_witness(&mut filler)
		.expect("the recorded circuit is satisfied by the replayed witness");

	// The point of the change: the circuit's public interface *is* the inner statement.
	// An outer proof can pin what was verified instead of trusting the filler.
	assert_eq!(
		filler.into_value_vec().inout(),
		witness.inout(),
		"the recursive circuit's public values must be the statement it verifies"
	);
}

/// Flips one bit of the first recorded wire of `kind`, and returns why the circuit then fails.
///
/// The wire is tampered after an honest replay, so only that bit differs from a good witness.
fn reject_tampered(kind: &'static str) -> Vec<String> {
	let proved = prove_crc64();
	let Recording { recorded, public } = record(&proved);
	let mut filler = recorded.circuit.new_witness_filler();
	for (&wire, &word) in iter::zip(&public, proved.witness.inout()) {
		filler[wire] = word;
	}
	replay(&proved, &recorded, &mut filler);

	let input = recorded
		.inputs
		.iter()
		.find(|input| input.kind == kind)
		.unwrap_or_else(|| panic!("the verifier must record at least one {kind} wire"));
	filler[input.wire] = Word(filler[input.wire].0 ^ 1);

	let error = recorded
		.circuit
		.populate_wire_witness(&mut filler)
		.expect_err("a tampered witness must leave the circuit unsatisfied");

	// `..` is forced: `PopulateError` is non-exhaustive. Both of its fields are checked here.
	let PopulateError {
		failures, total, ..
	} = error;
	assert!(total > 0, "an unsatisfied circuit must report a failing assertion");
	assert_eq!(failures.len(), total.min(MAX_ASSERTION_FAILURES));
	for failure in &failures {
		assert!(!failure.detail.is_empty(), "a failure must carry a diagnostic");
	}
	failures.into_iter().map(|failure| failure.path).collect()
}

#[test]
fn a_tampered_layer_digest_leaves_the_circuit_unsatisfied() {
	// Invariant: the decommitted layer is folded to the root, so its digests are not free.
	//
	// Fixture state: one honest proof, replayed in full, with one layer digest then flipped.
	//
	//     before:  layer  -> fold -> the root the commitment carries
	//     after:   layer' -> fold -> a digest that root does not match
	//
	// A layer digest reaches no arithmetic anywhere in the verifier, so the fold is the only thing
	// that reads it. That is what makes this isolate the binding under test.
	let paths = reject_tampered("merkle_layer");
	assert!(
		paths.iter().any(|path| path.contains("verify_layer")),
		"a corrupted layer digest must fail the fold to the root: {paths:?}"
	);
}

#[test]
fn a_tampered_committed_vector_leaves_the_circuit_unsatisfied() {
	// Invariant: a vector sent in the clear is bound by the tree rebuilt over it.
	//
	// Fixture state: one honest proof, replayed in full, with one committed value then flipped.
	//
	//     before:  data  -> rebuild -> the root the commitment carries
	//     after:   data' -> rebuild -> a different root
	let paths = reject_tampered("committed_vector");
	assert!(
		paths.iter().any(|path| path.contains("verify_vector")),
		"a corrupted committed value must fail the rebuilt tree: {paths:?}"
	);
}

#[test]
fn a_public_input_that_disagrees_with_the_statement_is_rejected() {
	// Invariant: binding is a constraint, so a disagreeing public input is rejected.
	//
	// Fixture state: one honest proof, replayed in full, with one public word then flipped.
	//
	//     before:  public word == the word the replay filled
	//     after:   public word != it, and the equality the binding emitted breaks
	let proved = prove_crc64();
	let Recording { recorded, public } = record(&proved);
	let mut filler = recorded.circuit.new_witness_filler();
	for (&wire, &word) in iter::zip(&public, proved.witness.inout()) {
		filler[wire] = word;
	}
	replay(&proved, &recorded, &mut filler);

	// Mutation: flip one bit of the first public word, leaving the replay's own wire alone.
	filler[public[0]] = Word(filler[public[0]].0 ^ 1);

	let error = recorded
		.circuit
		.populate_wire_witness(&mut filler)
		.expect_err("a public input that disagrees must leave the circuit unsatisfied");

	// `..` is forced: `PopulateError` is non-exhaustive. Both of its fields are checked here.
	let PopulateError {
		failures, total, ..
	} = error;
	assert_eq!(total, 1, "only the binding may fail");
	assert_eq!(failures.len(), 1);
	assert!(
		failures[0].path.contains("bind_public"),
		"the failure must come from the binding: {}",
		failures[0].path
	);
	assert!(!failures[0].detail.is_empty(), "a failure must carry a diagnostic");
}
