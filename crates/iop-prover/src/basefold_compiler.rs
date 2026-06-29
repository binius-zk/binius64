// Copyright 2026 The Binius Developers

//! BaseFold compiler for IOP provers.
//!
//! This module provides [`BaseFoldProverCompiler`], which precomputes FRI parameters and can
//! create prover channel instances.

use std::marker::PhantomData;

use binius_field::{BinaryField, PackedField};
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler, channel::OracleSpec, fri::FRIParams,
	merkle_tree::MerkleTreeScheme,
};
use binius_math::ntt::AdditiveNTT;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{basefold_channel::BaseFoldProverChannel, merkle_tree::MerkleTreeProver};

/// A compiler that creates BaseFold ZK prover channels with precomputed parameters.
///
/// This compiler builds a single combined FRI over all oracles, with ZK oracles configured for
/// zero-knowledge mode.
#[derive(Debug)]
pub struct BaseFoldProverCompiler<P, NTT, MerkleProver_>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<Field = P::Scalar> + Sync,
	MerkleProver_: MerkleTreeProver<P::Scalar>,
{
	ntt: NTT,
	merkle_prover: MerkleProver_,
	oracle_specs: Vec<OracleSpec>,
	/// The combined FRI parameters over **all** oracles.
	fri_params: FRIParams<P::Scalar>,
	_p_marker: PhantomData<P>,
}

impl<F, P, NTT, MerkleScheme, MerkleProver_> BaseFoldProverCompiler<P, NTT, MerkleProver_>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
{
	/// Creates a new compiler with precomputed combined FRI parameters.
	///
	/// Each oracle's batch size is derived from its ZK flag: a ZK oracle fixes `log_batch_size = 1`
	/// (message ‖ equal-length mask), a non-ZK oracle takes a flexible batch size.
	pub fn new(
		ntt: NTT,
		merkle_prover: MerkleProver_,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Self {
		assert!(
			!oracle_specs.is_empty(),
			"BaseFoldProverCompiler requires at least one oracle spec"
		);

		// The single combined FRI parameters over all oracles. `optimal_for_batch` derives each
		// oracle's batch size from its ZK flag: ZK oracles fix `log_batch_size = 1` (message ‖
		// equal-length mask); non-ZK oracles take a flexible batch size.
		let (fri_params, _) = FRIParams::optimal_for_batch(
			ntt.domain_context(),
			merkle_prover.scheme(),
			&oracle_specs,
			log_inv_rate,
			n_test_queries,
		);

		Self {
			ntt,
			merkle_prover,
			oracle_specs,
			fri_params,
			_p_marker: PhantomData,
		}
	}

	/// Creates a prover compiler from a verifier compiler.
	///
	/// This reuses the precomputed FRI parameters and oracle specifications.
	pub fn from_verifier_compiler(
		verifier_compiler: &BaseFoldVerifierCompiler<F, MerkleScheme>,
		ntt: NTT,
		merkle_prover: MerkleProver_,
	) -> Self {
		Self {
			ntt,
			merkle_prover,
			oracle_specs: verifier_compiler.oracle_specs().to_vec(),
			fri_params: verifier_compiler.fri_params().clone(),
			_p_marker: PhantomData,
		}
	}

	/// Returns a reference to the NTT.
	pub fn ntt(&self) -> &NTT {
		&self.ntt
	}

	/// Returns a reference to the Merkle prover.
	pub fn merkle_prover(&self) -> &MerkleProver_ {
		&self.merkle_prover
	}

	/// Returns a reference to the oracle specifications.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Returns a reference to the precomputed combined FRI parameters.
	pub fn fri_params(&self) -> &FRIParams<F> {
		&self.fri_params
	}

	/// Creates a ZK prover channel from this compiler, a transcript, and an RNG.
	///
	/// The `rng` is used to seed an internal `StdRng` for mask generation.
	pub fn create_channel<'a, Challenger_: Challenger>(
		&'a self,
		transcript: &'a mut ProverTranscript<Challenger_>,
		rng: impl Rng,
	) -> BaseFoldProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_> {
		BaseFoldProverChannel::from_compiler(self, transcript, rng)
	}

	/// Creates a prover channel for a compiler whose oracles are all non-ZK.
	///
	/// A mask is drawn from the channel's RNG only when committing a ZK oracle.
	/// With no ZK oracle the RNG is never read, so its seed cannot affect the proof.
	/// The seed is therefore fixed, and no randomness needs to be supplied by the caller.
	///
	/// # Panics
	///
	/// Panics if any configured oracle is ZK.
	/// A ZK oracle would draw its mask from the fixed seed, which destroys the hiding property.
	/// So this constructor refuses to build a channel that could mask deterministically.
	pub fn create_channel_without_zk<'a, Challenger_: Challenger>(
		&'a self,
		transcript: &'a mut ProverTranscript<Challenger_>,
	) -> BaseFoldProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_> {
		// A ZK oracle masks with the RNG, so a fixed seed here would silently break hiding.
		assert!(
			self.oracle_specs().iter().all(|spec| !spec.is_zk),
			"create_channel_without_zk requires every oracle to be non-ZK"
		);

		// No mask is ever drawn, so the seed is arbitrary; reuse the seeded-RNG constructor.
		self.create_channel(transcript, StdRng::seed_from_u64(0))
	}
}
