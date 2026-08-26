// Copyright 2026 The Binius Developers

//! The recursion pipeline, end to end, over a CRC-64 inner circuit.
//!
//! ```text
//!   build crc64  ->  prove it  ->  transcript
//!        |                              |
//!        |  verifier run symbolically   |  verifier run for real
//!        v                              v
//!   recursive circuit  <----------  its witness  ->  proved  ->  verified
//! ```
//!
//! One verifier runs over both channels and reaches the same operations in the same order.
//! The circuit the first produced is satisfied by the witness the second filled, and that
//! circuit is then proved and verified like any other.
//!
//! Everything the verifier reads is derived in-circuit or bound to something that is: the
//! Fiat-Shamir state, the Merkle commitments, the query indices, the proof of work. What a
//! replay still supplies is the proof itself and the statement, which is why tampering with
//! either is rejected here.

use binius_core::{constraint_system::ValueVec, word::Word};
use binius_field::arch::OptimalPackedB128;
use binius_frontend::{
	Circuit, CircuitBuilder, CircuitStat, MAX_ASSERTION_FAILURES, PopulateError, Wire,
};
use binius_hash::StdHashSuite;
use binius_prover::Prover;
use binius_recursion::{Error, RecursiveCircuit};
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

/// One statement, proved once, for the recursion pipeline to consume.
struct Proved {
	verifier: Verifier<StdHashSuite>,
	witness: ValueVec,
	proof: Vec<u8>,
}

/// Proves one statement of `circuit`, and checks it verifies natively.
///
/// A proof that fails here would make every recursion failure below ambiguous.
fn prove(circuit: &Circuit, witness: ValueVec) -> Proved {
	let verifier =
		Verifier::<StdHashSuite>::setup(circuit.constraint_system().clone(), LOG_INV_RATE).unwrap();
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

/// Proves one CRC-64 statement.
fn prove_crc64() -> Proved {
	let crc64 = crc64_circuit();
	let message = [0x0123456789abcdef, 0xfedcba9876543210, 1, u64::MAX];
	let witness = crc64_witness(&crc64, &message);
	prove(&crc64.circuit, witness)
}

/// A circuit's constraint rows, which is what one level's cost is compared on.
const fn rows(stat: &CircuitStat) -> usize {
	stat.n_and_constraints + stat.n_bmul_constraints + stat.n_zero_constraints
}

#[test]
fn recursive_circuit_is_satisfied_by_a_real_proof() {
	let proved = prove_crc64();
	let recursive = RecursiveCircuit::build(proved.verifier.clone()).unwrap();

	let stat = CircuitStat::collect(recursive.circuit());
	println!(
		"recursive circuit: {} gates, {} AND, {} BMUL, {} ZERO, {} recorded inputs",
		stat.n_gates,
		stat.n_and_constraints,
		stat.n_bmul_constraints,
		stat.n_zero_constraints,
		recursive.recorded().inputs.len(),
	);
	assert!(stat.n_bmul_constraints > 0, "the verifier's field arithmetic should be recorded");

	// The statement reaches the verifier as wires the replay fills, one per inout word.
	let inout = proved.witness.inout();
	assert_eq!(
		recursive
			.recorded()
			.inputs
			.iter()
			.filter(|input| input.kind == "observe_words")
			.count(),
		inout.len()
	);
	// One public wire was bound beside each of them.
	assert_eq!(recursive.statement().len(), inout.len());

	// The decommitted layer is on wires, which is what the opened leaves are matched against.
	assert!(
		recursive
			.recorded()
			.inputs
			.iter()
			.any(|input| input.kind == "merkle_layer"),
		"the query phase must record the layer it decommits to"
	);

	let witness = recursive.witness(inout, proved.proof).unwrap();

	// The point of the binding: the circuit's public interface *is* the inner statement.
	// An outer proof can pin what was verified instead of trusting the filler.
	assert_eq!(
		witness.inout(),
		inout,
		"the recursive circuit's public values must be the statement it verifies"
	);
}

#[test]
fn the_recursive_circuit_proves_and_verifies() {
	// Invariant: the recursive circuit is an ordinary circuit, so it proves like any other.
	//
	// This is the step that makes the pipeline recursion rather than verification-in-circuit:
	// what comes out is a proof, of the same kind that went in, carrying the same statement.
	let inner = prove_crc64();
	let recursive = RecursiveCircuit::build(inner.verifier.clone()).unwrap();
	let witness = recursive
		.witness(inner.witness.inout(), inner.proof.clone())
		.unwrap();

	// Proving verifies natively too, so a wrong witness fails here rather than silently.
	let outer = prove(recursive.circuit(), witness);

	println!("inner proof {} bytes -> outer proof {} bytes", inner.proof.len(), outer.proof.len());

	// The outer statement is the inner one, so whoever checks the outer proof learns what was
	// verified without ever seeing the inner proof.
	assert_eq!(
		outer.witness.inout(),
		inner.witness.inout(),
		"the outer proof must carry the inner statement"
	);
}

#[test]
#[ignore = "builds a 50M-gate circuit; run explicitly to re-measure per-level growth"]
fn one_more_level_costs_far_more_than_the_level_below() {
	// Invariant: discharging the wiring claim in-circuit is proportional to the inner system.
	//
	// A level's cost is dominated by a term proportional to the rows of the level beneath it, so
	// the tower diverges: `rows(n+1) / rows(n)` sits far above one and does not fall with depth.
	// That ratio is the whole reason the claim has to be deferred instead.
	//
	//     depth 1: verifies the CRC-64 proof
	//     depth 2: verifies depth 1's proof
	let inner = prove_crc64();

	let depth1 = RecursiveCircuit::build(inner.verifier.clone()).unwrap();
	let witness1 = depth1
		.witness(inner.witness.inout(), inner.proof.clone())
		.unwrap();
	let rows1 = rows(&CircuitStat::collect(depth1.circuit()));

	// Depth 2 reads only depth 1's *shape*, so depth 1 need not be proved to measure this.
	let verifier1 =
		Verifier::<StdHashSuite>::setup(depth1.circuit().constraint_system().clone(), LOG_INV_RATE)
			.unwrap();
	let depth2 = RecursiveCircuit::build(verifier1).unwrap();
	let rows2 = rows(&CircuitStat::collect(depth2.circuit()));

	let growth = rows2 as f64 / rows1 as f64;
	println!("rows: depth1 {rows1} -> depth2 {rows2} = {growth:.2}x");

	// A closing tower would sit at or below one. Pinning a floor documents that it does not, and
	// turns the day this drops into a test failure worth reading.
	assert!(growth > 5.0, "inline discharge is expected to diverge, measured {growth:.2}x");
	assert_eq!(witness1.inout(), inner.witness.inout());
}

/// Flips one bit of the first recorded wire of `kind`, and returns why the circuit then fails.
///
/// The wire is tampered after an honest replay, so only that bit differs from a good witness.
fn reject_tampered(kind: &'static str) -> Vec<String> {
	let proved = prove_crc64();
	let recursive = RecursiveCircuit::build(proved.verifier.clone()).unwrap();
	let mut filler = recursive
		.fill(proved.witness.inout(), proved.proof)
		.unwrap();

	let input = recursive
		.recorded()
		.inputs
		.iter()
		.find(|input| input.kind == kind)
		.unwrap_or_else(|| panic!("the verifier must record at least one {kind} wire"));
	filler[input.wire] = Word(filler[input.wire].0 ^ 1);

	let error = recursive
		.circuit()
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
	let recursive = RecursiveCircuit::build(proved.verifier.clone()).unwrap();
	let mut filler = recursive
		.fill(proved.witness.inout(), proved.proof)
		.unwrap();

	// Mutation: flip one bit of the first public word, leaving the replay's own wire alone.
	let bound = recursive.statement()[0];
	filler[bound] = Word(filler[bound].0 ^ 1);

	let error = recursive
		.circuit()
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

#[test]
fn a_statement_of_the_wrong_length_is_rejected() {
	// Invariant: the statement's length is the inner system's, and nothing else is accepted.
	//
	// A short statement would otherwise bind only a prefix, leaving the rest of the public
	// interface to whoever filled the witness.
	let proved = prove_crc64();
	let recursive = RecursiveCircuit::build(proved.verifier.clone()).unwrap();
	let inout = proved.witness.inout();

	let error = recursive
		.witness(&inout[..inout.len() - 1], proved.proof)
		.expect_err("a statement one word short must be rejected");

	assert!(
		matches!(
			error,
			Error::StatementLength { expected, actual }
				if expected == inout.len() && actual == inout.len() - 1
		),
		"expected a length mismatch, got {error}"
	);
}

#[test]
fn a_malformed_proof_is_rejected_rather_than_crashing() {
	// Invariant: how much a replay reads is fixed by the shape, never by what it reads.
	//
	// So a truncated tape runs out of bytes rather than out of recorded wires, and the transcript
	// reports that as an error. A shape-dependent read would instead walk off the wire cursor and
	// take the process down, which is why this is worth pinning.
	let proved = prove_crc64();
	let recursive = RecursiveCircuit::build(proved.verifier.clone()).unwrap();
	let inout = proved.witness.inout();

	// Half a tape, then no tape at all.
	for proof in [proved.proof[..proved.proof.len() / 2].to_vec(), Vec::new()] {
		let len = proof.len();
		let error = recursive
			.witness(inout, proof)
			.expect_err("a malformed proof must be rejected");
		// Reaching here at all is the invariant: an error came back, nothing unwound.
		// The variant still has to name the tape, not the statement, or the wrong check fired.
		assert!(
			!matches!(error, Error::StatementLength { .. } | Error::UnsupportedPcs { .. }),
			"a {len}-byte tape must fail on the tape, got {error}"
		);
	}
}
