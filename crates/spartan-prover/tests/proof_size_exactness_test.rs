// Copyright 2026 The Binius Developers

use binius_field::{BinaryField128bGhash as B128, PackedField, Random, arch::OptimalPackedB128};
use binius_hash::StdHashSuite;
use binius_iop::{
	channel::{IOPVerifierChannel, size_tracking::SizeTrackingChannel},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
	circuits::powers,
	compiler::compile,
};
use binius_spartan_prover::Prover;
use binius_spartan_verifier::{Verifier, config::StdChallenger};
use binius_transcript::ProverTranscript;
use rand::{SeedableRng, rngs::StdRng};

fn power_circuit<Builder: CircuitBuilder>(
	builder: &mut Builder,
	x_wire: Builder::Wire,
	y_wire: Builder::Wire,
	n: usize,
) {
	let powers_vec = powers(builder, x_wire, n);
	builder.assert_eq(powers_vec[n - 1], y_wire);
}

// Invariant: the counted size is the size an honest prover sends.
//
//     real     prove into a transcript, take its byte length
//     model    run the same verifier over the size-tracking channel
//
// A transcript holds only sent bytes; observed values are absorbed without being written.
// Rates vary because the rate moves the FRI arities, and with them the digest-to-value split.
#[test]
fn size_tracking_matches_real_proof_bytes() {
	for log_inv_rate in [1, 2, 3] {
		// Build the circuit and its satisfying assignment: y = x^7.
		let mut constraint_builder = ConstraintBuilder::new();
		let x_wire = constraint_builder.alloc_inout();
		let y_wire = constraint_builder.alloc_inout();
		power_circuit(&mut constraint_builder, x_wire, y_wire, 7);
		let (cs, layout) = compile(constraint_builder);

		let verifier =
			Verifier::<_, StdHashSuite>::setup(cs, log_inv_rate).expect("verifier setup failed");
		let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(&verifier)
			.expect("prover setup failed");

		let cs = verifier.constraint_system();
		let layout = layout.with_blinding(*cs.blinding_info());

		let mut rng = StdRng::seed_from_u64(0);
		let x_val = B128::random(&mut rng);
		let y_val = x_val.pow(7);

		let mut witness_gen = WitnessGenerator::new(&layout);
		let x_assigned = witness_gen.write_inout(x_wire, x_val);
		let y_assigned = witness_gen.write_inout(y_wire, y_val);
		power_circuit(&mut witness_gen, x_assigned, y_assigned, 7);
		let witness = witness_gen.build().expect("failed to build witness");

		// The real path: the byte length of an honest proof.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover
			.prove(&witness, &mut rng, &mut prover_transcript)
			.expect("prove failed");
		let real_size = prover_transcript.finalize().len();

		// The model path: the same verifier, driven over the size-tracking channel.
		let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let mut channel = verifier
			.iop_compiler()
			.create_channel(SizeTrackingChannel::new(&merkle_scheme));
		let public = vec![B128::default(); 1 << cs.log_public()];
		let public_elems = channel.observe_many(&public);
		let precommit_oracle = channel
			.recv_oracle(cs.log_precommit() as usize, true)
			.expect("recv_oracle should succeed");
		verifier
			.iop_verifier()
			.verify(precommit_oracle, &public_elems, &mut channel)
			.expect("verify over the size-tracking channel should succeed");
		let modelled_size = channel
			.finish()
			.expect("the opening should verify against all-zero values")
			.proof_size();

		assert_eq!(modelled_size, real_size, "log_inv_rate={log_inv_rate}");
	}
}
