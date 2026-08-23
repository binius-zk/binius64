// Copyright 2026 The Binius Developers

use binius_field::Ghash128b as B128;
use binius_hash::StdHashSuite;
use binius_iop::{
	channel::{IOPVerifierChannel, size_tracking::SizeTrackingChannel},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder},
	circuits::powers,
	compiler::compile,
};
use binius_spartan_verifier::Verifier;

// Build a power circuit: assert that x^n = y
fn power_circuit<Builder: CircuitBuilder>(
	builder: &mut Builder,
	x_wire: Builder::Wire,
	y_wire: Builder::Wire,
	n: usize,
) {
	let powers_vec = powers(builder, x_wire, n);
	let xn = powers_vec[n - 1];
	builder.assert_eq(xn, y_wire);
}

#[test]
fn test_ip_proof_size() {
	// Build the constraint system
	let mut constraint_builder = ConstraintBuilder::new();
	let x_wire = constraint_builder.alloc_inout();
	let y_wire = constraint_builder.alloc_inout();
	power_circuit(&mut constraint_builder, x_wire, y_wire, 7);
	let (cs, _layout) = compile(constraint_builder);

	// Setup verifier
	let log_inv_rate = 3;
	let verifier =
		Verifier::<_, StdHashSuite>::setup(cs, log_inv_rate).expect("verifier setup failed");

	let cs = verifier.constraint_system();

	// The size tracker is the Merkle channel, with the real reduction running on top.
	// So every commitment, branch and leaf the opening asks for is counted as it happens.
	let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
	let mut channel = verifier
		.iop_compiler()
		.create_channel(SizeTrackingChannel::new(&merkle_scheme));
	let public = vec![B128::default(); 1 << cs.log_public()];
	let public_elems = channel.observe_many(&public);
	let precommit_oracle = channel
		.recv_oracle(cs.log_precommit() as usize, true)
		.expect("recv_oracle on size-tracking channel should succeed");
	verifier
		.iop_verifier()
		.verify(precommit_oracle, &public_elems, &mut channel)
		.expect("verify with size tracking channel should succeed");
	let proof_size = channel
		.finish()
		.expect("the opening should verify against all-zero values")
		.proof_size();

	// Hardcoded expected value to detect proof size regressions.
	//
	// This is the whole proof, not an estimate of it: every byte is counted by the reduction that
	// would send it. `size_tracking_matches_real_proof_bytes` in the prover crate pins the count
	// against an honest transcript.
	//
	// The power chain x^2..x^7 is public-derivable (x and y are inout), so those wires are
	// `Derived` and emit no mul constraints — only `assert_eq(x^7, y)` survives.
	//
	// This circuit commits three oracles; FRI opens each against its own commitment.
	assert_eq!(proof_size, 73936, "proof size regression");
}
