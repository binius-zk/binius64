// Copyright 2026 The Binius Developers

use binius_field::{BinaryField128bGhash as B128, Field, Random, arch::OptimalPackedB128};
use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
use binius_iop::{
	basefold_compiler::BaseFoldZKVerifierCompiler, merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::{
	basefold_compiler::BaseFoldZKProverCompiler, merkle_tree::prover::BinaryMerkleTreeProver,
};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly};
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
	circuits::powers,
	compiler::compile,
	constraint_system::BlindingInfo,
};
use binius_spartan_prover::IOPProver;
use binius_spartan_verifier::{
	IOPVerifier, SECURITY_BITS, config::StdChallenger, constraint_system::ConstraintSystemPadded,
};
use binius_spartan_wrapper::{IronSpartanBuilderChannel, ZKWrappedVerifierChannel};
use binius_spartan_wrapper_prover::ZKWrappedProverChannel;
use binius_transcript::ProverTranscript;
use binius_verifier::fri::{self, MinProofSizeStrategy};
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
	let inner_layout = inner_layout.with_blinding(inner_cs.blinding_info().clone());

	let inner_iop_verifier = IOPVerifier::new(inner_cs.clone());
	let inner_iop_prover = IOPProver::new(inner_cs.clone());

	// === Step 3: Symbolically execute verify to build the outer constraint system ===
	let inner_public_size = 1 << inner_cs.log_public();

	let mut builder_channel = IronSpartanBuilderChannel::new(ConstraintBuilder::new());
	let dummy_public = vec![B128::ZERO; inner_public_size];
	let dummy_public_elems = builder_channel.observe_many(&dummy_public);
	inner_iop_verifier
		.verify(dummy_public_elems, &mut builder_channel)
		.expect("symbolic verify failed");
	let outer_builder = builder_channel.finish();
	let (outer_cs, outer_layout) = compile(outer_builder);

	// === Step 4: Build outer padded constraint system ===
	let log_inv_rate = 1;
	let n_test_queries = fri::calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
	let blinding_info = BlindingInfo {
		n_dummy_wires: n_test_queries,
		n_dummy_constraints: 2,
	};
	let outer_cs = ConstraintSystemPadded::new(outer_cs, blinding_info);
	let outer_layout = outer_layout.with_blinding(outer_cs.blinding_info().clone());

	// === Step 5: Make combined proof compiler (inner + outer oracle specs) ===
	let outer_iop_verifier = IOPVerifier::new(outer_cs.clone());
	let outer_iop_prover = IOPProver::new(outer_cs);

	let compression = StdCompression::default();
	let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdDigest, _>::new(compression.clone());
	let zk_basefold_compiler = BaseFoldZKVerifierCompiler::new(
		merkle_scheme,
		[
			inner_iop_verifier.oracle_specs(),
			outer_iop_verifier.oracle_specs(),
		]
		.concat(),
		log_inv_rate,
		n_test_queries,
		&MinProofSizeStrategy,
	);

	let subspace = zk_basefold_compiler.max_subspace();
	let domain_context = GenericOnTheFly::generate_from_subspace(subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);
	let merkle_prover = BinaryMerkleTreeProver::<_, StdDigest, _>::new(
		ParallelCompressionAdaptor::new(compression.clone()),
	);
	let zk_basefold_prover: BaseFoldZKProverCompiler<OptimalPackedB128, _, _> =
		BaseFoldZKProverCompiler::from_verifier_compiler(&zk_basefold_compiler, ntt, merkle_prover);

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

	let public = &inner_witness[..inner_public_size];

	// === Step 7: Prove with ZKWrappedProverChannel ===
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());

	// Observe inner public input on the transcript (Fiat-Shamir).
	prover_transcript.observe().write_slice(public);

	let basefold_channel = zk_basefold_prover.create_channel(&mut prover_transcript, &mut rng);
	let mut wrapped_prover_channel = ZKWrappedProverChannel::new(
		basefold_channel,
		&outer_iop_prover,
		&inner_iop_verifier,
		&outer_layout,
	);

	// Observe public input through the wrapped channel.
	(&mut wrapped_prover_channel).observe_many(public);

	// Run the inner proof through the wrapped channel.
	inner_iop_prover
		.prove::<B128, OptimalPackedB128, _>(&inner_witness, &mut rng, &mut wrapped_prover_channel)
		.expect("inner prove failed");

	// Finish runs the outer proof.
	wrapped_prover_channel
		.finish(rng)
		.expect("outer prove failed");

	// === Step 8: Verify with ZKWrappedVerifierChannel ===
	let mut verifier_transcript = prover_transcript.into_verifier();

	// Verifier observes the public input on the transcript (Fiat-Shamir).
	verifier_transcript.observe().write_slice(public);

	let verifier_channel = zk_basefold_compiler.create_channel(&mut verifier_transcript);
	let mut wrapped_verifier_channel =
		ZKWrappedVerifierChannel::new(verifier_channel, &outer_iop_verifier);

	// Observe public input through the wrapped channel.
	let inner_public_elems = wrapped_verifier_channel.observe_many(public);

	// Run the inner IOP verify through the wrapped channel.
	inner_iop_verifier
		.verify(inner_public_elems, &mut wrapped_verifier_channel)
		.expect("inner IOP verify failed");

	// Finish verifies the outer proof.
	wrapped_verifier_channel
		.finish()
		.expect("outer IOP verify failed");
}
