// Copyright 2026 The Binius Developers

use binius_field::BinaryField128bGhash as B128;
use binius_hash::StdHashSuite;
use binius_iop::{
	channel::IOPVerifierChannel, merkle_tree::BinaryMerkleTreeScheme,
	size_tracking_channel::SizeTrackingChannel,
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

	// Create size tracking channel and run verify with dummy public inputs
	// (SizeTrackingChannel ignores values). The Merkle scheme matches the one the verifier
	// estimates proof sizes with at setup.
	let iop_compiler = verifier.iop_compiler();
	let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
	let mut channel = SizeTrackingChannel::new(
		iop_compiler.oracle_specs().to_vec(),
		std::slice::from_ref(iop_compiler.fri_params()),
		&merkle_scheme,
	);
	let public = vec![B128::default(); 1 << cs.log_public()];
	let public_elems = channel.observe_many(&public);
	// SizeTrackingChannel::Oracle = (), but we still bind to exercise the real call pattern.
	#[allow(clippy::let_unit_value)]
	let precommit_oracle = channel
		.recv_oracle(cs.log_precommit() as usize, true)
		.expect("recv_oracle on size-tracking channel should succeed");
	verifier
		.iop_verifier()
		.verify(precommit_oracle, public_elems, &mut channel)
		.expect("verify with size tracking channel should succeed");
	let proof_size = channel.proof_size();

	// Hardcoded expected value to detect proof size regressions.
	// This measures IP-layer bytes (sumcheck rounds, oracle commitments, evaluations) plus the
	// single combined FRI proof opening all oracles together. It is a slight underestimate because
	// it does not account for some smaller BaseFold components (e.g. sumcheck coefficients within
	// BaseFold, blinding elements for ZK).
	//
	// The power chain x^2..x^7 is public-derivable (x and y are inout), so those wires are now
	// `Derived` and emit no mul constraints — only `assert_eq(x^7, y)` survives — shrinking the
	// proof relative to the pre-derived-wire baseline of 46848.
	assert_eq!(proof_size, 46784, "proof size regression");
}
