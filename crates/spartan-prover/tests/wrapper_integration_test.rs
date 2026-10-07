// Copyright 2026 The Binius Developers

use std::sync::Arc;

use binius_compute::GlobalAllocator;
use binius_field::{Field, Ghash128b as B128, Random, arch::OptimalPackedB128};
use binius_hash::StdHashSuite;
use binius_iop::{
	basefold::compiler::BaseFoldVerifierCompiler,
	channel::{
		IOPVerifierChannel,
		merge::MergeVerifierChannel,
		oracle_setup::{DummyElem, OracleSetupChannel},
	},
	fri::{self, MinProofSizeStrategy},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::{
	basefold::compiler::BaseFoldProverCompiler, channel::merge::MergeProverChannel,
};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly};
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
	circuits::powers,
	compiler::compile,
	constraint_system::BlindingInfo,
};
use binius_spartan_prover::{
	IOPProver,
	wrapper::{ReplayChannel, ZKWrappedProverChannel},
};
use binius_spartan_verifier::{
	IOPVerifier, SECURITY_BITS,
	config::StdChallenger,
	constraint_system::ConstraintSystemPadded,
	wrapper::{IronSpartanBuilderChannel, ZKWrappedVerifierChannel},
};
use binius_transcript::ProverTranscript;
use rand::{SeedableRng, rngs::StdRng};

/// Build a power7 circuit: assert that x^7 = y
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
fn test_zk_wrapped_prove_verify() {
	// === Step 1: Build the inner constraint system (the pow7 circuit) ===
	let mut inner_builder = ConstraintBuilder::new();
	let x_wire = inner_builder.alloc_inout();
	let y_wire = inner_builder.alloc_inout();
	power7_circuit(&mut inner_builder, x_wire, y_wire);
	let (inner_cs, inner_layout) = compile(inner_builder);

	// === Step 2: Setup inner IOP verifier and IOP prover ===
	let inner_cs = ConstraintSystemPadded::new(
		inner_cs,
		BlindingInfo {
			n_dummy_wires: 0,
			n_dummy_constraints: 0,
		},
	);
	let inner_layout = inner_layout.with_blinding(*inner_cs.blinding_info());

	let inner_iop_verifier = IOPVerifier::new(inner_cs.clone());
	let inner_iop_prover = IOPProver::new(inner_cs.clone());

	// === Step 3: Symbolically execute verify to build the outer constraint system ===
	let inner_public_size = 1 << inner_cs.log_public();

	let mut builder_channel = IronSpartanBuilderChannel::new();
	let dummy_public = vec![B128::ZERO; inner_public_size];
	let dummy_public_elems = builder_channel.observe_many(&dummy_public);
	// IronSpartanBuilderChannel::Oracle = () and recv_oracle is a no-op, so pass () directly.
	inner_iop_verifier
		.verify((), &dummy_public_elems, &mut builder_channel)
		.expect("symbolic verify failed");
	let outer_builder = builder_channel.finish();
	let (outer_cs, outer_layout) = compile(outer_builder);

	// === Step 4: Build outer padded constraint system ===
	let log_inv_rate = 1;
	let n_test_queries = fri::calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
	let blinding_info = BlindingInfo::for_fri_queries(n_test_queries);
	let outer_cs = ConstraintSystemPadded::new(outer_cs, blinding_info);
	let outer_layout = Arc::new(outer_layout.with_blinding(*outer_cs.blinding_info()));

	// === Step 5: Make combined proof compiler (inner + outer oracle schedule) ===
	let outer_iop_verifier = IOPVerifier::new(outer_cs.clone());
	let outer_iop_prover = IOPProver::new(outer_cs);

	let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();

	// Transcript layout: outer precommit oracle first (committed at wrapper construction),
	// then all inner oracles, then the remaining outer oracles (private, mask). Replaying that
	// sequence against one setup channel records where each round ends.
	let inner_log_precommit = inner_cs.log_precommit() as usize;
	let combined_schedule = {
		let mut channel = OracleSetupChannel::new(true);
		for log_precommit in [
			outer_iop_verifier.constraint_system().log_precommit() as usize,
			inner_log_precommit,
		] {
			<OracleSetupChannel as IOPVerifierChannel<B128>>::recv_oracle(
				&mut channel,
				log_precommit,
				true,
			)
			.unwrap();
		}
		let inner_public = vec![DummyElem::<B128>::default(); inner_public_size];
		let _ = inner_iop_verifier.verify((), &inner_public, &mut channel);
		let outer_public = vec![
			DummyElem::<B128>::default();
			1 << outer_iop_verifier.constraint_system().log_public()
		];
		let _ = outer_iop_verifier.verify((), &outer_public, &mut channel);
		channel.into_oracle_schedule()
	};

	let zk_basefold_compiler = BaseFoldVerifierCompiler::new(
		&merkle_scheme,
		combined_schedule.merged_specs(),
		log_inv_rate,
		n_test_queries,
		&MinProofSizeStrategy,
	);

	let domain_context = GaoMateerOnTheFly::generate(zk_basefold_compiler.max_log_domain_size());
	let ntt = NeighborsLastSingleThread::new(domain_context);
	let zk_basefold_prover: BaseFoldProverCompiler<OptimalPackedB128, _> =
		BaseFoldProverCompiler::from_verifier_compiler(&zk_basefold_compiler, ntt);

	// === Step 6: Generate inner witness ===
	let mut rng = StdRng::seed_from_u64(0);
	let x_val = B128::random(&mut rng);
	let y_val = x_val.pow([7]);

	let mut witness_gen = WitnessGenerator::new(&inner_layout);
	let x_assigned = witness_gen.write_inout(x_wire, x_val);
	let y_assigned = witness_gen.write_inout(y_wire, y_val);
	power7_circuit(&mut witness_gen, x_assigned, y_assigned);
	let inner_witness = witness_gen.build().expect("failed to build inner witness");

	inner_cs.validate(&inner_witness);

	let public = inner_witness.public().to_vec();

	// === Step 7: Prove with ZKWrappedProverChannel ===
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());

	// Observe inner public input on the transcript (Fiat-Shamir).
	prover_transcript.observe().write_slice(&public);

	let basefold_channel = zk_basefold_prover
		.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
			&mut prover_transcript,
			&mut rng,
			GlobalAllocator,
		);
	let mut wrapped_prover_channel = ZKWrappedProverChannel::new(
		MergeProverChannel::new(basefold_channel, &combined_schedule, GlobalAllocator),
		&outer_iop_prover,
		Arc::clone(&outer_layout),
		&GlobalAllocator,
		&mut rng,
		{
			let inner_iop_verifier = &inner_iop_verifier;
			let public = &public;
			move |replay_channel: &mut ReplayChannel<B128>| {
				let inner_public_elems = replay_channel.observe_many(public);
				// ReplayChannel::Oracle = () and recv_oracle is a no-op, so pass ().
				inner_iop_verifier
					.verify((), &inner_public_elems, replay_channel)
					.expect("replay verification should not fail");
			}
		},
	);

	// Observe public input through the wrapped channel.
	wrapped_prover_channel.observe_many(&public);

	// Commit the inner precommit oracle on the wrapped channel, then run the inner proof.
	let (inner_precommit_oracle, inner_precommit_packed) = inner_iop_prover
		.commit_precommit::<OptimalPackedB128, _, _>(
			&inner_witness,
			&mut rng,
			&mut wrapped_prover_channel,
			&GlobalAllocator,
		);
	inner_iop_prover
		.prove::<OptimalPackedB128, _, _>(
			&inner_witness,
			inner_precommit_oracle,
			inner_precommit_packed,
			&mut rng,
			&mut wrapped_prover_channel,
			&GlobalAllocator,
		)
		.expect("inner prove failed");

	// Finish runs the outer proof.
	wrapped_prover_channel
		.finish(rng)
		.expect("outer prove failed");

	// === Step 8: Verify with ZKWrappedVerifierChannel ===
	let mut verifier_transcript = prover_transcript.into_verifier();

	// Verifier observes the public input on the transcript (Fiat-Shamir).
	verifier_transcript.observe().write_slice(&public);

	let verifier_channel = zk_basefold_compiler
		.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(&mut verifier_transcript);
	let mut wrapped_verifier_channel = ZKWrappedVerifierChannel::new(
		MergeVerifierChannel::new(verifier_channel, &combined_schedule),
		&outer_iop_verifier,
		Arc::clone(&outer_layout),
	)
	.expect("ZKWrappedVerifierChannel::new should succeed");

	// Observe public input through the wrapped channel.
	let inner_public_elems = wrapped_verifier_channel.observe_many(&public);

	// Run the inner IOP verify through the wrapped channel.
	let inner_precommit_oracle = wrapped_verifier_channel
		.recv_oracle(inner_log_precommit, true)
		.unwrap();
	inner_iop_verifier
		.verify(inner_precommit_oracle, &inner_public_elems, &mut wrapped_verifier_channel)
		.expect("inner IOP verify failed");

	// Finish verifies the outer proof.
	wrapped_verifier_channel
		.finish()
		.expect("outer IOP verify failed");
}
