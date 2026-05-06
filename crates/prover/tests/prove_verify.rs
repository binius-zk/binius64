// Copyright 2025 Irreducible Inc.

use binius_circuits::sha256::{Compress, State};
use binius_core::{
	constraint_system::{ConstraintSystem, ValueVec},
	word::Word,
};
use binius_field::arch::OptimalPackedB128;
use binius_frontend::{CircuitBuilder, Wire};
use binius_hash::StdHashSuite;
use binius_prover::{Prover, zk_config::ZKProver};
use binius_transcript::ProverTranscript;
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
	let compress = Compress::new(&circuit, state, input);

	// Mask to only low 32-bit.
	let mask32 = circuit.add_constant(Word::MASK_32);
	for (actual_x, expected_x) in compress.state_out.0.iter().zip(output) {
		circuit.assert_eq("eq", circuit.band(*actual_x, mask32), expected_x);
	}

	let circuit = circuit.build();
	let mut w = circuit.new_witness_filler();

	// Populate the input message for the compression function.
	compress.populate_m(&mut w, preimage);

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

#[test]
fn test_zk_prove_verify_sha256_preimage() {
	let (cs, witness) = sha256_preimage_circuit();
	prove_verify_zk(cs, witness);
}
