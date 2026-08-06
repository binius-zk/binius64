// Copyright 2026 The Binius Developers
//! End-to-end check of the artifact files that carry a proving job between hosts.
//!
//! ```text
//!     circuit  -> cs.bin, public.bin, non_public.bin
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

use binius_core::{constraint_system::ValuesRef, word::Word};
use binius_frontend::CircuitBuilder;
use binius_utils::serialization::SerializeBytes;

// Where each artifact of one proving job lives on disk.
struct Artifacts {
	// Circuit shape: constraints and section sizes.
	cs: PathBuf,
	// Public segment: constants, inputs and outputs. The verifier reads this one.
	public: PathBuf,
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

// Compiles a one-gate circuit and writes the three files a prover needs.
//
// The circuit proves knowledge of a value that masks to a public output:
//
//     secret AND 0xFF00 == output
//     0x1234 AND 0xFF00 == 0x1200
//
// One hidden wire and one public wire is the smallest shape with both segments non-empty.
// That is the least the artifact files have to carry.
fn write_artifacts(dir: &Path) -> Artifacts {
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

	let artifacts = Artifacts {
		cs: dir.join("cs.bin"),
		public: dir.join("public.bin"),
		non_public: dir.join("non_public.bin"),
		proof: dir.join("proof.bin"),
	};

	write_serialized(circuit.constraint_system(), &artifacts.cs);

	// Each segment carries its own version tag, so each becomes a file of its own.
	write_serialized(&ValuesRef::new(values.public()), &artifacts.public);
	write_serialized(&ValuesRef::new(values.non_public()), &artifacts.non_public);

	artifacts
}

// Runs the prover binary over all three input files, producing the proof.
fn run_prover(artifacts: &Artifacts) -> std::process::Output {
	Command::new(env!("CARGO_BIN_EXE_prover"))
		.arg("--cs-path")
		.arg(&artifacts.cs)
		.arg("--pub-witness-path")
		.arg(&artifacts.public)
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
		.arg(&artifacts.public)
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

	// Another reads the shape, the public segment and that proof, and accepts.
	let out = run_verifier(&artifacts);
	assert!(out.status.success(), "verifier failed: {}", String::from_utf8_lossy(&out.stderr));

	fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn verifier_rejects_a_public_segment_from_another_format_version() {
	// Invariant: a segment file written to an unknown layout stops the verifier.
	//
	// Fixture state: valid artifacts and a valid proof.
	// So the version tag is the only thing left between the verifier and acceptance.
	let dir = scratch_dir("bad-version");
	let artifacts = write_artifacts(&dir);

	let out = run_prover(&artifacts);
	assert!(out.status.success(), "prover failed: {}", String::from_utf8_lossy(&out.stderr));

	// Mutation: raise the leading version tag by one, leaving every word intact.
	//
	//     public.bin:  [ 2 | n | word_0 .. ]
	//     expected:      1
	//     -> reject before a single word is read
	let mut bytes = fs::read(&artifacts.public).unwrap();
	bytes[0] = bytes[0].wrapping_add(1);
	fs::write(&artifacts.public, &bytes).unwrap();

	let out = run_verifier(&artifacts);
	assert!(!out.status.success(), "verifier accepted an unknown layout");

	// The failure names the version tag, not a downstream symptom of the wrong words.
	let stderr = String::from_utf8_lossy(&out.stderr);
	assert!(stderr.contains("ValuesData::version"), "expected a version error, got: {stderr}");

	fs::remove_dir_all(&dir).unwrap();
}
