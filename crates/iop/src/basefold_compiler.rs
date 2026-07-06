// Copyright 2026 The Binius Developers

//! BaseFold compiler for IOP verifiers.
//!
//! This module provides [`BaseFoldVerifierCompiler`], which precomputes FRI parameters and can
//! create verifier channel instances.

use std::borrow::BorrowMut;

use binius_field::BinaryField;
use binius_hash::binary_merkle_tree::HashSuite;
use binius_math::{BinarySubspace, ntt::domain_context::GenericOnTheFly};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes};
use digest::Output;

use crate::{
	basefold_channel::BaseFoldVerifierChannel,
	channel::OracleSpec,
	fri::{AritySelectionStrategy, FRIParams},
	merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
	merkle_tree::BinaryMerkleTreeScheme,
};

/// A compiler that creates BaseFold ZK verifier channels with precomputed parameters.
///
/// This compiler builds a single combined FRI over all oracles. ZK oracles configure FRI
/// parameters for zero-knowledge mode (`log_msg_len + 1` as the message length and
/// `log_batch_size = 1`); non-ZK oracles take a flexible batch size with no mask.
#[derive(Clone)]
pub struct BaseFoldVerifierCompiler<F>
where
	F: BinaryField,
{
	oracle_specs: Vec<OracleSpec>,
	fri_params: FRIParams<F>,
}

impl<F> BaseFoldVerifierCompiler<F>
where
	F: BinaryField,
{
	/// Creates a new compiler with precomputed combined FRI parameters.
	///
	/// The `merkle_scheme` is consulted only for proof-size estimation while choosing the FRI
	/// parameters; it is not stored. Each oracle's batch size is derived from its ZK flag: a ZK
	/// oracle fixes `log_batch_size = 1` (message ‖ equal-length mask), a non-ZK oracle takes a
	/// flexible batch size. Requires at least one oracle spec.
	pub fn new<H, Strategy>(
		merkle_scheme: BinaryMerkleTreeScheme<F, H>,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		n_test_queries: usize,
		_arity_strategy: &Strategy,
	) -> Self
	where
		H: HashSuite,
		Strategy: AritySelectionStrategy,
	{
		assert!(
			!oracle_specs.is_empty(),
			"BaseFoldVerifierCompiler requires at least one oracle spec"
		);

		// ZK oracles add 1 to the message length for the interleaved mask; non-ZK oracles do not.
		// Compute max code length across all oracles.
		let max_log_code_len = oracle_specs
			.iter()
			.map(|spec| spec.log_msg_len + usize::from(spec.is_zk))
			.max()
			.expect("oracle_specs is non-empty")
			+ log_inv_rate;
		let subspace = BinarySubspace::with_dim(max_log_code_len);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);

		// The single combined FRI parameters over all oracles. `optimal_for_batch` chooses the fold
		// arities to minimize proof size, so `_arity_strategy` is not consulted here. It derives
		// each oracle's batch size from its ZK flag: ZK oracles fix `log_batch_size = 1` (message
		// ‖ mask), non-ZK oracles take a flexible batch size.
		let (fri_params, _) = FRIParams::optimal_for_batch(
			&domain_context,
			&merkle_scheme,
			&oracle_specs,
			log_inv_rate,
			n_test_queries,
		);

		Self {
			oracle_specs,
			fri_params,
		}
	}

	/// Returns a reference to the oracle specifications.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Returns a reference to the precomputed combined FRI parameters.
	pub const fn fri_params(&self) -> &FRIParams<F> {
		&self.fri_params
	}

	/// Returns the Reed-Solomon code subspace of the combined FRI parameters (the largest needed).
	pub fn max_subspace(&self) -> &BinarySubspace<F> {
		self.fri_params.rs_code().subspace()
	}

	/// Creates a ZK verifier channel over the given Merkle channel.
	///
	/// The returned channel drives all prover interaction through `channel`, opening oracles with
	/// this compiler's oracle specs and combined FRI parameters. The caller constructs the Merkle
	/// channel, so it decides how commitments are received and verified.
	pub fn create_channel<Channel>(
		&self,
		channel: Channel,
	) -> BaseFoldVerifierChannel<'_, F, Channel>
	where
		Channel: MerkleIPVerifierChannel<F, Elem = F>,
	{
		BaseFoldVerifierChannel::new(channel, &self.oracle_specs, &self.fri_params)
	}

	/// Creates a ZK verifier channel over a transcript, for the common case.
	///
	/// The transcript (owned or mutably borrowed) is wrapped in a
	/// [`VerifierMerkleTranscriptChannel`] with a non-hiding [`BinaryMerkleTreeScheme`] for the
	/// given hash suite, then passed to [`Self::create_channel`].
	pub fn create_channel_from_transcript<H, Challenger_, T>(
		&self,
		transcript: T,
	) -> BaseFoldVerifierChannel<'_, F, VerifierMerkleTranscriptChannel<T, Challenger_, F, H>>
	where
		F: FixedSizeSerializeBytes,
		H: HashSuite,
		Challenger_: Challenger,
		T: BorrowMut<VerifierTranscript<Challenger_>>,
		Output<H::LeafHash>: DeserializeBytes,
	{
		self.create_channel(VerifierMerkleTranscriptChannel::new(transcript))
	}
}
