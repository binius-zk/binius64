// Copyright 2026 The Binius Developers

//! BaseFold compiler for IOP provers.
//!
//! This module provides [`BaseFoldProverCompiler`], which precomputes FRI parameters and can
//! create prover channel instances.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_field::{BinaryField, PackedField};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler, channel::OracleSpec, fri::FRIParams,
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_math::ntt::AdditiveNTT;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use digest::Output;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
	basefold_channel::BaseFoldProverChannel,
	merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
};

/// A compiler that creates BaseFold ZK prover channels with precomputed parameters.
///
/// This compiler builds a single combined FRI over all oracles, with ZK oracles configured for
/// zero-knowledge mode.
#[derive(Debug)]
pub struct BaseFoldProverCompiler<P, NTT>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<Field = P::Scalar> + Sync,
{
	ntt: NTT,
	oracle_specs: Vec<OracleSpec>,
	/// The combined FRI parameters over **all** oracles.
	fri_params: FRIParams<P::Scalar>,
	_marker: PhantomData<P>,
}

impl<F, P, NTT> BaseFoldProverCompiler<P, NTT>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	/// Creates a new compiler with precomputed combined FRI parameters.
	///
	/// The `merkle_scheme` is consulted only for proof-size estimation while choosing the FRI
	/// parameters; it is not stored. Each oracle's batch size is derived from its ZK flag: a ZK
	/// oracle fixes `log_batch_size = 1` (message ‖ equal-length mask), a non-ZK oracle takes a
	/// flexible batch size.
	pub fn new<H>(
		ntt: NTT,
		merkle_scheme: BinaryMerkleTreeScheme<F, H>,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Self
	where
		H: HashSuite,
	{
		assert!(
			!oracle_specs.is_empty(),
			"BaseFoldProverCompiler requires at least one oracle spec"
		);

		// The single combined FRI parameters over all oracles. `optimal_for_batch` derives each
		// oracle's batch size from its ZK flag: ZK oracles fix `log_batch_size = 1` (message ‖
		// equal-length mask); non-ZK oracles take a flexible batch size.
		let (fri_params, _) = FRIParams::optimal_for_batch(
			ntt.domain_context(),
			&merkle_scheme,
			&oracle_specs,
			log_inv_rate,
			n_test_queries,
		);

		Self {
			ntt,
			oracle_specs,
			fri_params,
			_marker: PhantomData,
		}
	}

	/// Creates a prover compiler from a verifier compiler.
	///
	/// This reuses the precomputed FRI parameters and oracle specifications.
	pub fn from_verifier_compiler(
		verifier_compiler: &BaseFoldVerifierCompiler<F>,
		ntt: NTT,
	) -> Self {
		Self {
			ntt,
			oracle_specs: verifier_compiler.oracle_specs().to_vec(),
			fri_params: verifier_compiler.fri_params().clone(),
			_marker: PhantomData,
		}
	}

	/// Returns a reference to the NTT.
	pub const fn ntt(&self) -> &NTT {
		&self.ntt
	}

	/// Returns a reference to the oracle specifications.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Returns a reference to the precomputed combined FRI parameters.
	pub const fn fri_params(&self) -> &FRIParams<F> {
		&self.fri_params
	}

	/// Creates a ZK prover channel over the given Merkle channel and an RNG.
	///
	/// The returned channel drives all prover interaction through `channel`, committing and opening
	/// oracles with this compiler's NTT, oracle specs, and combined FRI parameters. The caller
	/// constructs the Merkle channel, so it decides how commitments are produced. The `rng` is used
	/// to seed an internal `StdRng` for mask generation.
	pub fn create_channel<Channel>(
		&self,
		channel: Channel,
		rng: impl Rng,
	) -> BaseFoldProverChannel<'_, F, P, NTT, Channel>
	where
		Channel: MerkleIPProverChannel<F>,
	{
		BaseFoldProverChannel::new(
			channel,
			&self.ntt,
			self.oracle_specs.clone(),
			self.fri_params.clone(),
			rng,
		)
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
	pub fn create_channel_without_zk<Channel>(
		&self,
		channel: Channel,
	) -> BaseFoldProverChannel<'_, F, P, NTT, Channel>
	where
		Channel: MerkleIPProverChannel<F>,
	{
		// A ZK oracle masks with the RNG, so a fixed seed here would silently break hiding.
		assert!(
			self.oracle_specs().iter().all(|spec| !spec.is_zk),
			"create_channel_without_zk requires every oracle to be non-ZK"
		);

		// No mask is ever drawn, so the seed is arbitrary; reuse the seeded-RNG constructor.
		self.create_channel(channel, StdRng::seed_from_u64(0))
	}

	/// Creates a ZK prover channel over a transcript, for the common case.
	///
	/// The transcript (owned or mutably borrowed) is wrapped in a
	/// [`ProverMerkleTranscriptChannel`] with a non-hiding Merkle tree prover for the given hash
	/// suite, then passed to [`Self::create_channel`].
	pub fn create_channel_from_transcript<H, Challenger_, T>(
		&self,
		transcript: T,
		rng: impl Rng,
	) -> BaseFoldProverChannel<'_, F, P, NTT, ProverMerkleTranscriptChannel<T, Challenger_, F, H>>
	where
		H: HashSuite,
		Challenger_: Challenger,
		T: BorrowMut<ProverTranscript<Challenger_>>,
		Output<H::LeafHash>: SerializeBytes,
	{
		self.create_channel(ProverMerkleTranscriptChannel::new(transcript), rng)
	}

	/// Creates a non-ZK prover channel over a transcript, for the common case.
	///
	/// The transcript handling matches [`Self::create_channel_from_transcript`]; the channel is
	/// built with [`Self::create_channel_without_zk`] and panics under the same conditions.
	pub fn create_channel_without_zk_from_transcript<H, Challenger_, T>(
		&self,
		transcript: T,
	) -> BaseFoldProverChannel<'_, F, P, NTT, ProverMerkleTranscriptChannel<T, Challenger_, F, H>>
	where
		H: HashSuite,
		Challenger_: Challenger,
		T: BorrowMut<ProverTranscript<Challenger_>>,
		Output<H::LeafHash>: SerializeBytes,
	{
		self.create_channel_without_zk(ProverMerkleTranscriptChannel::new(transcript))
	}
}
