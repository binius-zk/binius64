// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{BinaryField128bGhash as B128, Field, Random, arch::OptimalPackedB128};
use binius_hash::StdHashSuite;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, InstanceGenerator, WitnessGenerator},
	circuits::powers,
	compiler::compile,
	constraint_system::Witness,
};
use binius_spartan_prover::Prover;
use binius_spartan_verifier::{Verifier, config::StdChallenger};
use binius_transcript::ProverTranscript;
use rand::{SeedableRng, rngs::StdRng};

// Build a power7 circuit: assert that x^7 = y
fn power7_circuit<Builder: CircuitBuilder>(
	builder: &mut Builder,
	x_wire: Builder::Wire,
	y_wire: Builder::Wire,
) {
	let powers_vec = powers(builder, x_wire, 7);
	let x7 = powers_vec[6]; // x^7 is the 7th element (0-indexed)
	builder.assert_eq(x7, y_wire);
}

#[test]
fn test_power7_circuit_prover_verifier() {
	// Build the constraint system
	let mut constraint_builder = ConstraintBuilder::new();
	let x_wire = constraint_builder.alloc_inout();
	let y_wire = constraint_builder.alloc_inout();
	power7_circuit(&mut constraint_builder, x_wire, y_wire);
	let (cs, layout) = compile(constraint_builder);

	// x and y are inout, so the whole x^7 power chain is public-derivable: the derived elision
	// leaves no private wires and a single mul constraint (the folded assert_eq).
	assert_eq!(cs.n_private(), 0);
	assert_eq!(cs.mul_constraints().len(), 1);

	// Setup prover and verifier
	let log_inv_rate = 1;
	let verifier =
		Verifier::<_, StdHashSuite>::setup(cs, log_inv_rate).expect("verifier setup failed");
	let prover =
		Prover::<OptimalPackedB128, StdHashSuite>::setup(&verifier).expect("prover setup failed");

	let cs = verifier.constraint_system();
	let layout = layout.with_blinding(cs.blinding_info());

	// Choose test values: x = random, y = x^7
	let mut rng = StdRng::seed_from_u64(0);
	let x_val = B128::random(&mut rng);
	let y_val = x_val * x_val * x_val * x_val * x_val * x_val * x_val; // x^7

	// Generate witness
	let mut witness_gen = WitnessGenerator::new(&layout);
	let x_assigned = witness_gen.write_inout(x_wire, x_val);
	let y_assigned = witness_gen.write_inout(y_wire, y_val);
	power7_circuit(&mut witness_gen, x_assigned, y_assigned);
	let witness = witness_gen.build().expect("failed to build witness");

	// Validate witness satisfies constraints
	cs.validate(&witness);

	// The verifier reconstructs the public vector (constants + inout + derived) itself by
	// re-running the circuit with an InstanceGenerator — it never trusts the prover's derived
	// values. For x^7 with x, y inout, the whole power chain is derived, so this exercises the
	// derived path and must match the witness's public segment.
	let mut instance_gen = InstanceGenerator::new(&layout);
	let x_instance = instance_gen.write_inout(x_wire, x_val);
	let y_instance = instance_gen.write_inout(y_wire, y_val);
	power7_circuit(&mut instance_gen, x_instance, y_instance);
	let public = instance_gen.build();
	assert_eq!(public, witness.public(), "InstanceGenerator public must match witness public");

	// Generate proof
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	prover
		.prove(&witness, &mut rng, &mut prover_transcript)
		.expect("prove failed");

	// Verify proof
	let mut verifier_transcript = prover_transcript.into_verifier();
	verifier
		.verify(&public, &mut verifier_transcript)
		.expect("verify failed");
	verifier_transcript.finalize().expect("finalize failed");
}

/// Soundness: a prover that tampers a derived value in its witness cannot pass verification,
/// because the verifier independently recomputes the public segment via `InstanceGenerator` and
/// never trusts the prover-supplied derived value.
#[test]
fn test_tampered_derived_value_fails_verification() {
	let mut constraint_builder = ConstraintBuilder::new();
	let x_wire = constraint_builder.alloc_inout();
	let y_wire = constraint_builder.alloc_inout();
	power7_circuit(&mut constraint_builder, x_wire, y_wire);
	let (cs, layout) = compile(constraint_builder);

	let log_inv_rate = 1;
	let verifier =
		Verifier::<_, StdHashSuite>::setup(cs, log_inv_rate).expect("verifier setup failed");
	let prover =
		Prover::<OptimalPackedB128, StdHashSuite>::setup(&verifier).expect("prover setup failed");

	let cs = verifier.constraint_system();
	let layout = layout.with_blinding(cs.blinding_info());

	let mut rng = StdRng::seed_from_u64(0);
	let x_val = B128::random(&mut rng);
	let y_val = x_val * x_val * x_val * x_val * x_val * x_val * x_val; // x^7

	// Honest witness.
	let mut witness_gen = WitnessGenerator::new(&layout);
	let x_assigned = witness_gen.write_inout(x_wire, x_val);
	let y_assigned = witness_gen.write_inout(y_wire, y_val);
	power7_circuit(&mut witness_gen, x_assigned, y_assigned);
	let witness = witness_gen.build().expect("failed to build witness");

	// The correct public segment, as the verifier recomputes it.
	let mut instance_gen = InstanceGenerator::new(&layout);
	let x_instance = instance_gen.write_inout(x_wire, x_val);
	let y_instance = instance_gen.write_inout(y_wire, y_val);
	power7_circuit(&mut instance_gen, x_instance, y_instance);
	let correct_public = instance_gen.build();

	// A malicious prover corrupts a derived value in its witness. Derived wires occupy the public
	// segment right after the constants and inout values.
	let derived_slot = layout.n_constants() + layout.n_inout();
	let mut tampered_public = witness.public().to_vec();
	tampered_public[derived_slot] += B128::ONE;
	let tampered_witness =
		Witness::new(tampered_public, witness.precommit().to_vec(), witness.private().to_vec());

	// The prover observes its tampered public into the transcript while proving.
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	prover
		.prove(&tampered_witness, &mut rng, &mut prover_transcript)
		.expect("prove failed");

	// The verifier verifies against the recomputed (correct) public and must reject the proof.
	let mut verifier_transcript = prover_transcript.into_verifier();
	assert!(
		verifier
			.verify(&correct_public, &mut verifier_transcript)
			.is_err(),
		"verification must fail when the prover tampers a derived value"
	);
}
