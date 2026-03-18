// Copyright 2026 The Binius Developers

use binius_field::BinaryField128bGhash as B128;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder},
	circuits::powers,
	compiler::compile,
};
use binius_spartan_verifier::{
	Verifier,
	config::{StdCompression, StdDigest},
};

// Build a power7 circuit: assert that x^7 = y
fn power7_circuit<Builder: CircuitBuilder>(
	builder: &mut Builder,
	x_wire: Builder::Wire,
	y_wire: Builder::Wire,
) {
	let powers_vec = powers(builder, x_wire, 7);
	let x7 = powers_vec[6];
	builder.assert_eq(x7, y_wire);
}

#[test]
fn test_power7_ip_proof_size() {
	// Build the constraint system
	let mut constraint_builder = ConstraintBuilder::new();
	let x_wire = constraint_builder.alloc_inout();
	let y_wire = constraint_builder.alloc_inout();
	power7_circuit(&mut constraint_builder, x_wire, y_wire);
	let (cs, _layout) = compile(constraint_builder);

	// Setup verifier
	let log_inv_rate = 1;
	let compression = StdCompression::default();
	let verifier = Verifier::<_, StdDigest, _>::setup(cs, log_inv_rate, compression)
		.expect("verifier setup failed");

	let cs = verifier.constraint_system();

	// Create size tracking channel; proof_size is written by the channel and readable after
	// finish() consumes it.
	let mut proof_size = 0;
	let channel = verifier
		.iop_compiler()
		.create_size_tracking_channel(&mut proof_size);

	// Run verify_iop with dummy public inputs (SizeTrackingChannel ignores values)
	let public = vec![B128::default(); 1 << cs.log_public()];
	verifier
		.verify_iop(&public, channel)
		.expect("verify_iop with size tracking channel should succeed");
	println!("IP-layer proof size: {proof_size} bytes");

	// Hardcoded expected value to detect proof size regressions.
	// This only measures IP-layer bytes (sumcheck rounds, oracle commitments, evaluations).
	// FRI decommitment size tracking is not yet implemented.
	assert_eq!(proof_size, 272, "IP-layer proof size regression");
}
