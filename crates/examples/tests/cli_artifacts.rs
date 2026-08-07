// Copyright 2026 The Binius Developers
//! End-to-end check of the artifact files that carry a proving job between hosts.
//!
//! ```text
//!     circuit  -> cs.bin, inout.bin, non_public.bin
//!     prover   -> proof.bin
//!     verifier -> accept or reject
//! ```
//!
//! Each step below runs the real binary.
//! So the files cross a process boundary, exactly as they would between two hosts.

use std::{
	fs,
	path::{Path, PathBuf},
	process::Command,
};

use binius_core::{
	constraint_system::{ValueVec, ValuesRef},
	word::Word,
};
use binius_frontend::{Circuit, CircuitBuilder};
use binius_utils::serialization::SerializeBytes;

// Where each artifact of one proving job lives on disk.
struct Artifacts {
	// Circuit shape: constraints, constants and section sizes.
	cs: PathBuf,
	// Public inout values: this instance's inputs and outputs. The verifier reads this one.
	inout: PathBuf,
	// Non-public segment: witness and internal values. The prover alone reads this one.
	non_public: PathBuf,
	// Proof transcript, written by the prover and read by the verifier.
	proof: PathBuf,
}

// A scratch directory per test, keyed by process id so concurrent runs cannot collide.
fn scratch_dir(name: &str) -> PathBuf {
	let dir =
		std::env::temp_dir().join(format!("binius-cli-artifacts-{}-{name}", std::process::id()));
	fs::create_dir_all(&dir).unwrap();
	dir
}

fn write_serialized<T: SerializeBytes>(value: &T, path: &Path) {
	let mut buf = Vec::new();
	value.serialize(&mut buf).unwrap();
	fs::write(path, &buf).unwrap();
}

// Compiles a one-gate circuit and fills its value vector.
//
// The circuit proves knowledge of a value that masks to a public output:
//
//     secret AND 0xFF00 == output
//     0x1234 AND 0xFF00 == 0x1200
//
// One hidden wire and one public wire is the smallest shape with both segments non-empty.
// That is the least the artifact files have to carry.
fn build_circuit() -> (Circuit, ValueVec) {
	let builder = CircuitBuilder::new();
	let mask = builder.add_constant_64(0xFF00);
	let secret = builder.add_witness();
	let output = builder.add_inout();
	let masked = builder.band(secret, mask);
	builder.assert_eq("masked_result", masked, output);
	let circuit = builder.build();

	// Only the two endpoints are assigned; the gate output is derived from them.
	let mut filler = circuit.new_witness_filler();
	filler[secret] = Word(0x1234);
	filler[output] = Word(0x1200);
	circuit.populate_wire_witness(&mut filler).unwrap();
	let values = filler.into_value_vec();

	(circuit, values)
}

// Writes the three files a prover needs.
fn write_artifacts(dir: &Path) -> Artifacts {
	let (circuit, values) = build_circuit();

	let artifacts = Artifacts {
		cs: dir.join("cs.bin"),
		inout: dir.join("inout.bin"),
		non_public: dir.join("non_public.bin"),
		proof: dir.join("proof.bin"),
	};

	let cs = circuit.constraint_system();
	write_serialized(cs, &artifacts.cs);

	// Each run of words carries its own version tag, so each becomes a file of its own. The mask
	// constant is in neither of them: it rides along in the constraint system.
	write_serialized(&ValuesRef::new(cs.inout_values(values.public())), &artifacts.inout);
	write_serialized(&ValuesRef::new(values.non_public()), &artifacts.non_public);

	artifacts
}

// Runs the prover binary over all three input files, producing the proof.
fn run_prover(artifacts: &Artifacts) -> std::process::Output {
	Command::new(env!("CARGO_BIN_EXE_prover"))
		.arg("--cs-path")
		.arg(&artifacts.cs)
		.arg("--pub-witness-path")
		.arg(&artifacts.inout)
		.arg("--non-pub-data-path")
		.arg(&artifacts.non_public)
		.arg("--proof-path")
		.arg(&artifacts.proof)
		.output()
		.unwrap()
}

// Runs the verifier binary, which never sees the non-public segment.
fn run_verifier(artifacts: &Artifacts) -> std::process::Output {
	Command::new(env!("CARGO_BIN_EXE_verifier"))
		.arg("--cs-path")
		.arg(&artifacts.cs)
		.arg("--pub-witness-path")
		.arg(&artifacts.inout)
		.arg("--proof-path")
		.arg(&artifacts.proof)
		.output()
		.unwrap()
}

#[test]
fn saved_artifacts_prove_and_verify_across_processes() {
	// Invariant: the three files are a complete handoff, with nothing passed in memory.
	//
	// Fixture state: artifacts for the masked-AND circuit, freshly written to a scratch dir.
	let dir = scratch_dir("round-trip");
	let artifacts = write_artifacts(&dir);

	// One process reads all three files and writes a proof.
	let out = run_prover(&artifacts);
	assert!(out.status.success(), "prover failed: {}", String::from_utf8_lossy(&out.stderr));
	assert!(artifacts.proof.is_file(), "prover wrote no proof");

	// Another reads the shape, the inout values and that proof, and accepts.
	let out = run_verifier(&artifacts);
	assert!(out.status.success(), "verifier failed: {}", String::from_utf8_lossy(&out.stderr));

	fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn verifier_rejects_a_public_segment_from_another_format_version() {
	// Invariant: a values file written to an unknown layout stops the verifier.
	//
	// Fixture state: valid artifacts and a valid proof.
	// So the version tag is the only thing left between the verifier and acceptance.
	let dir = scratch_dir("bad-version");
	let artifacts = write_artifacts(&dir);

	let out = run_prover(&artifacts);
	assert!(out.status.success(), "prover failed: {}", String::from_utf8_lossy(&out.stderr));

	// Mutation: raise the leading version tag by one, leaving every word intact.
	//
	//     inout.bin:  [ 2 | n | word_0 .. ]
	//     expected:     1
	//     -> reject before a single word is read
	let mut bytes = fs::read(&artifacts.inout).unwrap();
	bytes[0] = bytes[0].wrapping_add(1);
	fs::write(&artifacts.inout, &bytes).unwrap();

	let out = run_verifier(&artifacts);
	assert!(!out.status.success(), "verifier accepted an unknown layout");

	// The failure names the version tag, not a downstream symptom of the wrong words.
	let stderr = String::from_utf8_lossy(&out.stderr);
	assert!(stderr.contains("ValuesData::version"), "expected a version error, got: {stderr}");

	fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn verifier_rejects_a_whole_public_segment_in_place_of_the_inout_values() {
	// Invariant: the file carries the inout values alone, and a longer run is not silently taken
	// as one. This is what keeps the constants a property of the circuit rather than the instance.
	//
	// Fixture state: valid artifacts and a valid proof, with the inout file then replaced by the
	// whole public segment — constants, padding and all.
	//
	//     inout.bin was:  [ output ]
	//     inout.bin now:  [ 0xFF00 | pad | output | pad ]
	let dir = scratch_dir("whole-segment");
	let artifacts = write_artifacts(&dir);

	let out = run_prover(&artifacts);
	assert!(out.status.success(), "prover failed: {}", String::from_utf8_lossy(&out.stderr));

	let (_circuit, values) = build_circuit();
	write_serialized(&ValuesRef::new(values.public()), &artifacts.inout);

	let out = run_verifier(&artifacts);
	assert!(!out.status.success(), "verifier accepted a whole public segment");

	// The failure names the length, so the mistake is legible without reading the bytes.
	let stderr = String::from_utf8_lossy(&out.stderr);
	assert!(stderr.contains("inout values must be"), "expected a length error, got: {stderr}");

	fs::remove_dir_all(&dir).unwrap();
}
