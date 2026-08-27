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
use binius_recursion::{Discharge, Error, RecursiveCircuit};
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
#[ignore = "builds two multi-million-row circuits; run explicitly to re-measure per-level growth"]
fn deferring_shrinks_a_level_by_the_cost_of_the_level_below() {
	// Invariant: the wiring claim is the only cost that tracks the inner system's size.
	//
	// So the two discharges diverge at different rates, and the gap between them at depth 2 *is*
	// the term deferral removes. One level below, that term is small; one level up it dominates,
	// which is why the comparison has to be made here and not at depth 1.
	//
	//     depth 1: verifies the CRC-64 proof
	//     depth 2: verifies depth 1's proof
	let inner = prove_crc64();

	let mut report = Vec::new();
	for discharge in [Discharge::InCircuit, Discharge::Deferred] {
		let depth1 = RecursiveCircuit::build_with(inner.verifier.clone(), discharge).unwrap();
		let rows1 = rows(&CircuitStat::collect(depth1.circuit()));

		// Depth 2 reads only depth 1's *shape*, so depth 1 need not be proved to measure this.
		let verifier1 = Verifier::<StdHashSuite>::setup(
			depth1.circuit().constraint_system().clone(),
			LOG_INV_RATE,
		)
		.unwrap();
		let depth2 = RecursiveCircuit::build_with(verifier1, discharge).unwrap();
		let rows2 = rows(&CircuitStat::collect(depth2.circuit()));

		println!(
			"{discharge:?}: depth1 {rows1} -> depth2 {rows2} = {:.2}x",
			rows2 as f64 / rows1 as f64
		);
		report.push((rows1, rows2));
	}

	let [(inline1, inline2), (defer1, defer2)] = report[..] else {
		unreachable!()
	};

	// Deferring costs a little at depth 1, where the claim it exports is nearly free either way.
	assert!(defer1 < inline1, "deferring cannot add rows");

	// It pays at depth 2, where the claim it dropped was proportional to depth 1's whole size.
	// This gap is the entire reason the seam exists.
	assert!(
		defer2 < inline2,
		"deferring must remove the term proportional to the level below: {defer2} vs {inline2}"
	);
	let saved = inline2 - defer2;
	println!(
		"deferral saved {saved} rows at depth 2, {:.1}% of the inline level",
		100.0 * saved as f64 / inline2 as f64
	);

	// Both still diverge. The replay itself is sublinear but not yet below one at these sizes, so
	// a closing tower needs the merge node's own parameters, not just this seam.
	assert!(defer2 > defer1, "the replay term still grows: closure is the merge node's checkpoint");
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

#[test]
fn deferring_the_wiring_claim_removes_its_cost_and_keeps_it_checkable() {
	// Invariant: discharging the wiring claim is the one cost proportional to the inner system.
	//
	// Deferring it exports the claim on public wires instead. So the BMUL column drops by the
	// evaluation's whole cost, and the statement grows only by the claim, which is logarithmic in
	// the inner system.
	//
	//     in-circuit: assert_zero(eval(inputs) - claimed)   -> constraints over every inner row
	//     deferred:   inputs, claimed -> public wires       -> a check the holder owes
	let inner = prove_crc64();

	let inline =
		RecursiveCircuit::build_with(inner.verifier.clone(), Discharge::InCircuit).unwrap();
	let deferred =
		RecursiveCircuit::build_with(inner.verifier.clone(), Discharge::Deferred).unwrap();

	let a = CircuitStat::collect(inline.circuit());
	let b = CircuitStat::collect(deferred.circuit());
	let claim_elems = deferred.deferred_wires().unwrap().len() / 2;
	println!(
		"in-circuit {} BMUL / {} rows  ->  deferred {} BMUL / {} rows, claim {} elements",
		a.n_bmul_constraints,
		rows(&a),
		b.n_bmul_constraints,
		rows(&b),
		claim_elems,
	);

	// The evaluation is pure field arithmetic, so deferring can only take BMUL away.
	assert!(
		b.n_bmul_constraints < a.n_bmul_constraints,
		"deferring must drop the evaluation's BMUL cost"
	);
	// And the claim it exports has to stay small, or deferral trades one cost for another.
	assert!(
		claim_elems < 512,
		"the exported claim must be logarithmic in the inner system, got {claim_elems} elements"
	);

	// The deferred circuit still proves and verifies, and its statement carries the claim.
	let witness = deferred
		.witness(inner.witness.inout(), inner.proof.clone())
		.unwrap();
	let outer_inout = witness.inout().to_vec();
	assert_eq!(outer_inout.len(), deferred.statement().len() + 2 * claim_elems);
	prove(deferred.circuit(), witness);

	// The exported claim holds against the inner constraint system, which is the check the
	// circuit skipped. This is the whole obligation deferral creates.
	deferred
		.check_deferred(&outer_inout)
		.expect("an honest proof must export a claim that holds");
}

#[test]
fn a_tampered_deferred_claim_is_rejected() {
	// Invariant: the deferred check is a real evaluation, not a formality.
	//
	// Fixture state: one honest proof, its exported claim read off the outer statement, with one
	// word of that claim then flipped.
	//
	//     before:  eval(inputs) == claimed
	//     after:   one coordinate moved, so the two sides disagree
	//
	// If this passed, deferral would be exporting an unchecked constraint and calling it checked.
	let inner = prove_crc64();
	let deferred =
		RecursiveCircuit::build_with(inner.verifier.clone(), Discharge::Deferred).unwrap();
	let witness = deferred
		.witness(inner.witness.inout(), inner.proof.clone())
		.unwrap();
	let honest = witness.inout().to_vec();

	// Every word of the claim is load-bearing: the inputs are the point it is evaluated at, and
	// the last element is the value itself. Moving any one of them must break the equality.
	for offset in [0, 1, 2 * (deferred.deferred_wires().unwrap().len() / 2) - 1] {
		let mut tampered = honest.clone();
		let word = &mut tampered[deferred.statement().len() + offset];
		*word = Word(word.0 ^ 1);
		deferred
			.check_deferred(&tampered)
			.expect_err("a moved claim word must fail the deferred check");
	}
}

#[test]
fn an_in_circuit_discharge_leaves_nothing_deferred() {
	// Invariant: the two discharges are exclusive, so asking one for the other's work is an error
	// rather than a silent pass. A silent pass is how a caller ends up believing a claim was
	// settled twice when it was settled once.
	let inner = prove_crc64();
	let inline = RecursiveCircuit::build_with(inner.verifier, Discharge::InCircuit).unwrap();

	assert!(inline.deferred_wires().is_none());
	assert!(matches!(
		inline
			.check_deferred(&[])
			.expect_err("there is no claim to check"),
		Error::NothingDeferred
	));
}

#[test]
fn a_deferred_claim_that_disagrees_with_the_circuit_is_rejected() {
	// Invariant: the exported claim is bound, not merely reported.
	//
	// `check_deferred` reads the claim off the outer statement, so the statement has to be the
	// claim the circuit actually derived. The binding is what forces that, and this is the test
	// that it is a constraint rather than a convention.
	//
	//     before:  public claim word == the word the circuit derived
	//     after:   public claim word != it, and the equality the binding emitted breaks
	//
	// Without this, a prover could publish a claim that holds while verifying a proof that raised
	// a different one — and `check_deferred` would happily confirm the published one.
	let inner = prove_crc64();
	let deferred =
		RecursiveCircuit::build_with(inner.verifier.clone(), Discharge::Deferred).unwrap();
	let mut filler = deferred
		.fill(inner.witness.inout(), inner.proof.clone())
		.unwrap();

	// Mutation: flip one bit of the first claim word, leaving every derived wire alone.
	let bound = deferred.deferred_wires().unwrap()[0];
	filler[bound] = Word(filler[bound].0 ^ 1);

	let error = deferred
		.circuit()
		.populate_wire_witness(&mut filler)
		.expect_err("a claim word that disagrees must leave the circuit unsatisfied");

	// `..` is forced: `PopulateError` is non-exhaustive. Both of its fields are checked here.
	let PopulateError {
		failures, total, ..
	} = error;
	assert_eq!(total, 1, "only the binding may fail");
	assert!(
		failures[0].path.contains("bind_public_elem"),
		"the failure must come from the claim binding: {}",
		failures[0].path
	);
	assert!(!failures[0].detail.is_empty(), "a failure must carry a diagnostic");
}

#[test]
fn verify_outer_settles_the_deferred_claim() {
	// Invariant: the one-call path checks strictly more than verifying the proof does.
	//
	// The outer proof verifies on its own even when the claim it exported is false, because the
	// circuit never constrained the claim to hold — only to be reported honestly. So a statement
	// carrying a broken claim passes `Verifier::verify` and must still fail `verify_outer`.
	let inner = prove_crc64();
	let deferred =
		RecursiveCircuit::build_with(inner.verifier.clone(), Discharge::Deferred).unwrap();
	let witness = deferred
		.witness(inner.witness.inout(), inner.proof.clone())
		.unwrap();
	let honest = witness.inout().to_vec();
	let outer = prove(deferred.circuit(), witness);

	deferred
		.verify_outer(&outer.verifier, &honest, outer.proof.clone())
		.expect("an honest outer proof must verify and its claim must hold");

	// The same proof against a statement whose claim was moved: the transcript no longer matches
	// the statement, so this fails at the proof. Either rejection is correct; what matters is that
	// no path accepts it.
	let mut tampered = honest;
	let at = deferred.statement().len();
	tampered[at] = Word(tampered[at].0 ^ 1);
	deferred
		.verify_outer(&outer.verifier, &tampered, outer.proof)
		.expect_err("a moved claim must not be accepted");
}
