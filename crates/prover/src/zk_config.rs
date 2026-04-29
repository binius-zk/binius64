// Copyright 2026 The Binius Developers

//! Zero-knowledge proving configuration for Binius64 constraint systems.
//!
//! This module provides [`ZKProver`], which wraps the Binius64 IOP prover with a
//! Spartan-based zero-knowledge wrapper. The prover counterpart to
//! [`binius_verifier::zk_config::ZKVerifier`].

use binius_core::{constraint_system::ValueVec, word::Word};
use binius_field::{
	BinaryField128bGhash as B128, PackedExtension, PackedField, UnderlierWithBitOps, WithUnderlier,
};
use binius_iop_prover::basefold_compiler::BaseFoldZKProverCompiler;
use binius_math::ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded};
use binius_spartan_frontend::{
	circuit_builder::ConstraintBuilder, compiler::compile, constraint_system::WitnessLayout,
};
use binius_spartan_prover::wrapper::ZKWrappedProverChannel;
use binius_spartan_verifier::{constraint_system::ConstraintSystemPadded, wrapper::ReplayChannel};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use binius_verifier::{IOPVerifier, zk_config::ZKVerifier};
use digest::{Digest, FixedOutputReset, Output, block_api::BlockSizeUser};
use rand::CryptoRng;

use crate::{
	IOPProver,
	hash::{ParallelDigest, parallel_compression::ParallelPseudoCompression},
	merkle_tree::prover::BinaryMerkleTreeProver,
	protocols::shift::build_key_collection,
};

type ProverNTT<F> = NeighborsLastMultiThread<GenericPreExpanded<F>>;
type ProverMerkleProver<F, ParallelMerkleHasher, ParallelMerkleCompress> =
	BinaryMerkleTreeProver<F, ParallelMerkleHasher, ParallelMerkleCompress>;

/// Zero-knowledge prover for Binius64 constraint systems.
///
/// Wraps the Binius64 IOP prover with a Spartan-based ZK wrapper. Call [`Self::setup`] with
/// a [`ZKVerifier`], then [`Self::prove`] with witness data and a proof transcript.
pub struct ZKProver<P, ParallelMerkleCompress, ParallelMerkleHasher>
where
	P: PackedField<Scalar = B128>,
	ParallelMerkleHasher: ParallelDigest,
	ParallelMerkleHasher::Digest: Digest + BlockSizeUser + FixedOutputReset,
	ParallelMerkleCompress: ParallelPseudoCompression<Output<ParallelMerkleHasher::Digest>, 2>,
{
	inner_iop_prover: IOPProver,
	inner_iop_verifier: IOPVerifier,
	outer_iop_prover: binius_spartan_prover::IOPProver<B128>,
	outer_layout: WitnessLayout<B128>,
	#[allow(clippy::type_complexity)]
	basefold_compiler: BaseFoldZKProverCompiler<
		P,
		ProverNTT<B128>,
		ProverMerkleProver<B128, ParallelMerkleHasher, ParallelMerkleCompress>,
	>,
}

impl<P, MerkleHash, ParallelMerkleCompress, ParallelMerkleHasher>
	ZKProver<P, ParallelMerkleCompress, ParallelMerkleHasher>
where
	P: PackedField<Scalar = B128>
		+ PackedExtension<B128>
		+ PackedExtension<binius_verifier::config::B1>
		+ WithUnderlier<Underlier: UnderlierWithBitOps>,
	MerkleHash: Digest + BlockSizeUser + FixedOutputReset,
	ParallelMerkleHasher: ParallelDigest<Digest = MerkleHash>,
	ParallelMerkleCompress: ParallelPseudoCompression<Output<MerkleHash>, 2>,
	Output<MerkleHash>: SerializeBytes,
{
	/// Constructs a ZK prover from a [`ZKVerifier`].
	pub fn setup(
		zk_verifier: ZKVerifier<MerkleHash, ParallelMerkleCompress::Compression>,
		compression: ParallelMerkleCompress,
	) -> Result<Self, Error> {
		// Build the inner IOPProver.
		let inner_iop_verifier = zk_verifier.inner_iop_verifier().clone();
		let key_collection = build_key_collection(inner_iop_verifier.constraint_system());
		let inner_iop_prover = IOPProver::new(inner_iop_verifier.clone(), key_collection);

		// Re-derive the outer constraint system and layout via symbolic execution.
		let dummy_public_words =
			vec![Word::from_u64(0); 1 << inner_iop_verifier.log_public_words()];
		let mut builder_channel = binius_spartan_verifier::wrapper::IronSpartanBuilderChannel::new(
			ConstraintBuilder::new(),
		);
		inner_iop_verifier
			.verify(&dummy_public_words, &mut builder_channel)
			.expect("symbolic verify should not fail");
		let outer_builder = builder_channel.finish();
		let (outer_cs, outer_layout) = compile(outer_builder);

		// Pad the outer constraint system with the same blinding as the verifier.
		let outer_cs = ConstraintSystemPadded::new(
			outer_cs,
			zk_verifier
				.outer_iop_verifier()
				.constraint_system()
				.blinding_info()
				.clone(),
		);
		let outer_layout = outer_layout.with_blinding(outer_cs.blinding_info().clone());

		let outer_iop_prover = binius_spartan_prover::IOPProver::new(outer_cs);

		// Build the BaseFoldZK prover compiler from the verifier compiler.
		let subspace = zk_verifier.basefold_compiler().max_subspace();
		let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);
		let merkle_prover = BinaryMerkleTreeProver::<_, ParallelMerkleHasher, _>::new(compression);
		let basefold_compiler = BaseFoldZKProverCompiler::from_verifier_compiler(
			zk_verifier.basefold_compiler(),
			ntt,
			merkle_prover,
		);

		Ok(Self {
			inner_iop_prover,
			inner_iop_verifier,
			outer_iop_prover,
			outer_layout,
			basefold_compiler,
		})
	}

	/// Returns a reference to the inner IOP prover.
	pub fn inner_iop_prover(&self) -> &IOPProver {
		&self.inner_iop_prover
	}

	/// Returns a reference to the KeyCollection.
	pub fn key_collection(&self) -> &crate::protocols::shift::KeyCollection {
		self.inner_iop_prover.key_collection()
	}

	/// Generates a ZK proof for a witness.
	pub fn prove<Challenger_: Challenger>(
		&self,
		witness: ValueVec,
		mut rng: impl CryptoRng,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		// Clone public words before moving witness into prove().
		let public_words = witness.public().to_vec();

		// Create BaseFoldZK prover channel and wrap with outer prover.
		let basefold_channel = self.basefold_compiler.create_channel(transcript, &mut rng);
		let mut wrapped_channel = ZKWrappedProverChannel::new(
			basefold_channel,
			&self.outer_iop_prover,
			&self.outer_layout,
			&mut rng,
			{
				let inner_iop_verifier = &self.inner_iop_verifier;
				move |replay_channel: &mut ReplayChannel<'_, B128>| {
					inner_iop_verifier
						.verify(&public_words, replay_channel)
						.expect("replay verification should not fail");
				}
			},
		);

		// Run the inner IOP proof through the wrapped channel.
		self.inner_iop_prover
			.prove::<P, _>(witness, &mut wrapped_channel)?;

		// Finish runs the outer spartan proof.
		wrapped_channel.finish(rng)?;

		Ok(())
	}
}

/// Error type for ZK proving.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("inner proving error: {0}")]
	InnerProving(#[from] crate::error::Error),
	#[error("outer proving error: {0}")]
	OuterProving(#[from] binius_spartan_prover::Error),
}
