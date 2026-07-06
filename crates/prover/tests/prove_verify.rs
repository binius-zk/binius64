// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_circuits::sha256::{State, populate_message_block, sha256_compress};
use binius_core::{
	constraint_system::{ConstraintSystem, ValueVec},
	word::Word,
};
use binius_field::arch::OptimalPackedB128;
use binius_frontend::{CircuitBuilder, Wire};
use binius_hash::StdHashSuite;
use binius_prover::{Prover, zk_config::ZKProver};
use binius_transcript::ProverTranscript;
use binius_utils::{DeserializeBytes, SerializeBytes};
use binius_verifier::{Verifier, config::StdChallenger, zk_config::ZKVerifier};
use rand::{SeedableRng, rngs::StdRng};

fn prove_verify(cs: ConstraintSystem, witness: ValueVec) {
	const LOG_INV_RATE: usize = 1;

	let verifier = Verifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();

	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(verifier.clone()).unwrap();

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	prover
		.prove(witness.clone(), &mut prover_transcript)
		.unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	verifier
		.verify(witness.public(), &mut verifier_transcript)
		.unwrap();
	verifier_transcript.finalize().unwrap();
}

fn prove_verify_zk(cs: ConstraintSystem, witness: ValueVec) {
	const LOG_INV_RATE: usize = 1;

	let zk_verifier = ZKVerifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();

	let zk_prover =
		ZKProver::<OptimalPackedB128, StdHashSuite>::setup(zk_verifier.clone()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	zk_prover
		.prove(witness.clone(), &mut rng, &mut prover_transcript)
		.unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	zk_verifier
		.verify(witness.public(), &mut verifier_transcript)
		.unwrap();
	verifier_transcript.finalize().unwrap();
}

fn prove_verify_zk_serialized(cs: ConstraintSystem, witness: ValueVec) {
	const LOG_INV_RATE: usize = 1;

	let zk_verifier = ZKVerifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();
	let zk_prover =
		ZKProver::<OptimalPackedB128, StdHashSuite>::setup(zk_verifier.clone()).unwrap();

	// Round-trip both through serialization, mimicking save-to-disk / reload-in-a-fresh-process.
	// The reloaded prover (which reuses the deserialized KeyCollection and recomputes the cheaper
	// derived state) must produce a proof the reloaded verifier accepts.
	let mut prover_bytes = Vec::new();
	zk_prover.serialize(&mut prover_bytes).unwrap();
	let zk_prover =
		ZKProver::<OptimalPackedB128, StdHashSuite>::deserialize(prover_bytes.as_slice()).unwrap();

	let mut verifier_bytes = Vec::new();
	zk_verifier.serialize(&mut verifier_bytes).unwrap();
	let zk_verifier = ZKVerifier::<StdHashSuite>::deserialize(verifier_bytes.as_slice()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	zk_prover
		.prove(witness.clone(), &mut rng, &mut prover_transcript)
		.unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	zk_verifier
		.verify(witness.public(), &mut verifier_transcript)
		.unwrap();
	verifier_transcript.finalize().unwrap();
}

fn sha256_preimage_circuit() -> (ConstraintSystem, ValueVec) {
	// Use the test-vector for SHA256 single block message: "abc".
	let mut preimage: [u8; 64] = [0; 64];
	preimage[0..3].copy_from_slice(b"abc");
	preimage[3] = 0x80;
	preimage[63] = 0x18;

	#[rustfmt::skip]
	let expected_state: [u32; 8] = [
		0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
		0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
	];

	let circuit = CircuitBuilder::new();
	let state = State::iv(&circuit);
	let input: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());
	let output: [Wire; 8] = std::array::from_fn(|_| circuit.add_inout());
	let state_out = sha256_compress(&circuit, state, input);

	// Mask to only low 32-bit.
	let mask32 = circuit.add_constant(Word::MASK_32);
	for (actual_x, expected_x) in state_out.0.iter().zip(output) {
		circuit.assert_eq("eq", circuit.band(*actual_x, mask32), expected_x);
	}

	let circuit = circuit.build();
	let mut w = circuit.new_witness_filler();

	// Populate the input message for the compression function.
	populate_message_block(&mut w, &input, preimage);

	for (i, &output) in output.iter().enumerate() {
		w[output] = Word(expected_state[i] as u64);
	}
	circuit.populate_wire_witness(&mut w).unwrap();

	(circuit.constraint_system().clone(), w.into_value_vec())
}

#[test]
fn test_prove_verify_sha256_preimage() {
	let (cs, witness) = sha256_preimage_circuit();
	prove_verify(cs, witness);
}

/// A SHA-256 circuit uses only AND constraints, so its constraint system has zero MUL
/// constraints. This locks in that the prover and verifier skip the IntMul reduction entirely
/// (rather than padding up to a dummy MUL constraint); see `IOPProver::prove` /
/// `IOPVerifier::verify`.
#[test]
fn test_prove_verify_zero_mul_constraints() {
	let (cs, witness) = sha256_preimage_circuit();
	assert_eq!(cs.n_mul_constraints(), 0, "SHA-256 circuit should have no MUL constraints");
	prove_verify(cs.clone(), witness.clone());
	prove_verify_zk(cs, witness);
}

#[test]
fn test_zk_prove_verify_sha256_preimage() {
	let (cs, witness) = sha256_preimage_circuit();
	prove_verify_zk(cs, witness);
}

#[test]
fn test_zk_prove_verify_serialized() {
	let (cs, witness) = sha256_preimage_circuit();
	prove_verify_zk_serialized(cs, witness);
}

/// Produces a ZK signature-of-knowledge proof over `sign_message`, then verifies it against
/// `verify_message`. Returns whether verification (including transcript finalization) succeeded.
///
/// Signatures of knowledge are only supported by the ZK prover/verifier.
fn sign_verify(
	cs: ConstraintSystem,
	witness: ValueVec,
	sign_message: Option<&[u8]>,
	verify_message: Option<&[u8]>,
) -> bool {
	const LOG_INV_RATE: usize = 1;

	let zk_verifier = ZKVerifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();
	let zk_prover =
		ZKProver::<OptimalPackedB128, StdHashSuite>::setup(zk_verifier.clone()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	match sign_message {
		Some(message) => zk_prover
			.prove_sig(witness.clone(), message, &mut rng, &mut prover_transcript)
			.unwrap(),
		None => zk_prover
			.prove(witness.clone(), &mut rng, &mut prover_transcript)
			.unwrap(),
	}

	let mut verifier_transcript = prover_transcript.into_verifier();
	let verify_ok = match verify_message {
		Some(message) => zk_verifier
			.verify_sig(witness.public(), message, &mut verifier_transcript)
			.is_ok(),
		None => zk_verifier
			.verify(witness.public(), &mut verifier_transcript)
			.is_ok(),
	};
	verify_ok && verifier_transcript.finalize().is_ok()
}

#[test]
fn test_signature_of_knowledge_roundtrip() {
	let (cs, witness) = sha256_preimage_circuit();
	// Signing and verifying with the same message succeeds.
	assert!(sign_verify(cs, witness, Some(b"hello world"), Some(b"hello world")));
}

#[test]
fn test_signature_of_knowledge_wrong_message_fails() {
	let (cs, witness) = sha256_preimage_circuit();
	// A proof signed over one message must not verify against a different message.
	assert!(!sign_verify(cs, witness, Some(b"hello world"), Some(b"goodbye world")));
}

#[test]
fn test_signature_of_knowledge_missing_message_fails() {
	let (cs, witness) = sha256_preimage_circuit();
	// A signature of knowledge must not verify as a plain proof of knowledge (no message).
	assert!(!sign_verify(cs, witness, Some(b"hello world"), None));
}

#[test]
fn test_plain_proof_rejects_message() {
	let (cs, witness) = sha256_preimage_circuit();
	// A plain proof of knowledge must not verify when a message is supplied.
	assert!(!sign_verify(cs, witness, None, Some(b"hello world")));
}
