// Copyright 2026 The Binius Developers

//! Zero-knowledge verification configuration for Binius64 constraint systems.
//!
//! This module provides [`ZKVerifier`], which wraps the Binius64 IOP verifier with a
//! Spartan-based zero-knowledge wrapper. The wrapper transforms the non-ZK Binius64
//! verification into a ZK proof by:
//!
//! 1. Symbolically executing the inner verifier to build an outer Spartan constraint system
//! 2. Combining inner and outer oracle specs into a single BaseFold ZK compiler
//! 3. At verification time, running the inner verifier through a [`ZKWrappedVerifierChannel`] that
//!    records all values as outer public inputs, then finishing with outer Spartan verification
//!
//! [`ZKWrappedVerifierChannel`]: binius_spartan_verifier::wrapper::ZKWrappedVerifierChannel

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::BinaryField128bGhash as B128;
use binius_iop::{
	basefold_compiler::BaseFoldZKVerifierCompiler,
	channel::OracleSpec,
	fri::{self, MinProofSizeStrategy},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_spartan_frontend::{compiler::compile, constraint_system::BlindingInfo};
use binius_spartan_verifier::{
	IOPVerifier as IronSpartanIOPVerifier,
	constraint_system::ConstraintSystemPadded,
	wrapper::{IronSpartanBuilderChannel, ZKWrappedVerifierChannel},
};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, checked_arithmetics::log2_ceil_usize};
use digest::{Digest, Output, block_api::BlockSizeUser};

use crate::{
	config::LOG_WORDS_PER_ELEM,
	hash::PseudoCompressionFunction,
	verify::{IOPVerifier, SECURITY_BITS},
};

/// Zero-knowledge verifier for Binius64 constraint systems.
///
/// Wraps the Binius64 IOP verifier with a Spartan-based ZK wrapper. Call [`Self::setup`] with
/// a constraint system, then [`Self::verify`] with public inputs and a proof transcript.
#[derive(Debug, Clone)]
pub struct ZKVerifier<MerkleHash, MerkleCompress>
where
	MerkleHash: Digest + BlockSizeUser,
	MerkleCompress: PseudoCompressionFunction<Output<MerkleHash>, 2>,
{
	inner_iop_verifier: IOPVerifier,
	outer_iop_verifier: IronSpartanIOPVerifier<B128>,
	basefold_compiler:
		BaseFoldZKVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress>>,
}

impl<MerkleHash, MerkleCompress> ZKVerifier<MerkleHash, MerkleCompress>
where
	MerkleHash: Digest + BlockSizeUser,
	MerkleCompress: PseudoCompressionFunction<Output<MerkleHash>, 2>,
	Output<MerkleHash>: DeserializeBytes,
{
	/// Constructs a ZK verifier for a constraint system.
	pub fn setup(
		mut constraint_system: ConstraintSystem,
		log_inv_rate: usize,
		compression: MerkleCompress,
	) -> Result<Self, Error> {
		let _setup_guard = tracing::debug_span!("Setup ZK verifier").entered();

		constraint_system.validate_and_prepare()?;

		let n_public = constraint_system.value_vec_layout.offset_witness;
		let log_public_words = log2_ceil_usize(n_public);
		assert!(n_public.is_power_of_two());
		assert!(log_public_words >= LOG_WORDS_PER_ELEM);

		let inner_iop_verifier = IOPVerifier::new(constraint_system, log_public_words);

		// Symbolically execute the inner verifier to build the outer constraint system.
		let dummy_public_words =
			vec![Word::from_u64(0); 1 << inner_iop_verifier.log_public_words()];

		let outer_builder = {
			let _ = tracing::debug_span!("Build ZK wrapper circuit").entered();
			let mut builder_channel = IronSpartanBuilderChannel::new();
			inner_iop_verifier
				.verify(&dummy_public_words, &mut builder_channel)
				.expect("symbolic verify should not fail");
			builder_channel.finish()
		};
		let (outer_cs, _) = {
			let _ = tracing::debug_span!("Compile ZK wrapper circuit").entered();
			compile(outer_builder)
		};

		tracing::debug!(
			n_public = outer_cs.n_public(),
			n_precommit = outer_cs.n_precommit(),
			n_private = outer_cs.n_private(),
			n_mul_constraints = outer_cs.mul_constraints().len(),
			"ZK wrapper circuit stats"
		);

		// Pad the outer constraint system for zero-knowledge.
		let n_test_queries = fri::calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
		let blinding_info = BlindingInfo {
			n_dummy_wires: n_test_queries,
			n_dummy_constraints: 2,
		};
		let outer_cs = ConstraintSystemPadded::new(outer_cs, blinding_info);
		let outer_iop_verifier = IronSpartanIOPVerifier::new(outer_cs);

		// Transcript layout: outer precommit oracle first (committed at wrapper construction),
		// then all inner oracles, then the remaining outer oracles (private, mask).
		let outer_oracle_specs = outer_iop_verifier.oracle_specs();
		let oracle_specs: Vec<OracleSpec> = [
			vec![outer_oracle_specs[0]],
			inner_iop_verifier.oracle_specs(),
			outer_oracle_specs[1..].to_vec(),
		]
		.concat();

		let merkle_scheme = BinaryMerkleTreeScheme::new(compression);
		let basefold_compiler = BaseFoldZKVerifierCompiler::new(
			merkle_scheme,
			oracle_specs,
			log_inv_rate,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		Ok(Self {
			inner_iop_verifier,
			outer_iop_verifier,
			basefold_compiler,
		})
	}

	/// Returns a reference to the inner IOP verifier.
	pub fn inner_iop_verifier(&self) -> &IOPVerifier {
		&self.inner_iop_verifier
	}

	/// Returns a reference to the outer spartan IOP verifier.
	pub fn outer_iop_verifier(&self) -> &IronSpartanIOPVerifier<B128> {
		&self.outer_iop_verifier
	}

	/// Returns the BaseFold ZK verifier compiler.
	pub fn basefold_compiler(
		&self,
	) -> &BaseFoldZKVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress>>
	{
		&self.basefold_compiler
	}

	/// Returns the constraint system.
	pub fn constraint_system(&self) -> &ConstraintSystem {
		self.inner_iop_verifier.constraint_system()
	}

	/// Returns log2 of the number of public words.
	pub fn log_public_words(&self) -> usize {
		self.inner_iop_verifier.log_public_words()
	}

	/// Verifies a ZK proof against the constraint system.
	pub fn verify<Challenger_: Challenger>(
		&self,
		public: &[Word],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		// Create BaseFoldZK channel and wrap with outer verifier.
		let channel = self.basefold_compiler.create_channel(transcript);
		let mut wrapped_channel = ZKWrappedVerifierChannel::new(channel, &self.outer_iop_verifier)?;

		// Run the inner IOP verification through the wrapped channel.
		{
			let inner_cs = self.inner_iop_verifier.constraint_system();
			let _scope = tracing::debug_span!(
				"Binius64",
				n_witness_words = inner_cs.value_vec_layout.committed_total_len,
				n_bitand = inner_cs.and_constraints.len(),
				n_intmul = inner_cs.mul_constraints.len(),
			)
			.entered();

			self.inner_iop_verifier
				.verify(public, &mut wrapped_channel)?;
		};

		// Finish runs the outer spartan verification.
		{
			let outer_cs = self.outer_iop_verifier.constraint_system();
			let _scope = tracing::debug_span!(
				"ZK Wrapper",
				n_witness = outer_cs.n_private(),
				n_constraints = outer_cs.mul_constraints().len(),
			)
			.entered();

			wrapped_channel.finish()?;
		}

		Ok(())
	}
}

/// Error type for ZK verification.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("inner verification error: {0}")]
	InnerVerification(#[from] crate::error::Error),
	#[error("outer verification error: {0}")]
	OuterVerification(#[from] binius_spartan_verifier::Error),
	#[error("constraint system error: {0}")]
	ConstraintSystem(#[from] binius_core::ConstraintSystemError),
}
