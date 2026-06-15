// Copyright 2026 The Binius Developers

//! BaseFold compilers for IOP verifiers.
//!
//! This module provides [`BaseFoldVerifierCompiler`] and [`BaseFoldZKVerifierCompiler`], which
//! precompute FRI parameters and can create verifier channel instances.

use binius_field::BinaryField;
use binius_math::{BinarySubspace, ntt::domain_context::GenericOnTheFly};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::DeserializeBytes;

use crate::{
	basefold_channel::BaseFoldVerifierChannel,
	basefold_zk_channel::BaseFoldZKVerifierChannel,
	channel::OracleSpec,
	fri::{AritySelectionStrategy, FRIParams},
	merkle_tree::MerkleTreeScheme,
	size_tracking_channel::SizeTrackingChannel,
};

/// A compiler that creates BaseFold verifier channels with precomputed parameters.
///
/// The compiler holds the Merkle scheme, oracle specifications, and precomputed FRI
/// parameters. It can create multiple channels for different verification sessions.
///
/// # Type Parameters
///
/// - `F`: The binary field type
/// - `MerkleScheme_`: The Merkle tree scheme for commitments
#[derive(Debug, Clone)]
pub struct BaseFoldVerifierCompiler<F, MerkleScheme_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F>,
{
	merkle_scheme: MerkleScheme_,
	oracle_specs: Vec<OracleSpec>,
	fri_params: Vec<FRIParams<F>>,
}

impl<F, MerkleScheme_> BaseFoldVerifierCompiler<F, MerkleScheme_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F>,
{
	/// Creates a new compiler with precomputed FRI parameters.
	///
	/// # Arguments
	///
	/// * `merkle_scheme` - The Merkle tree scheme (owned)
	/// * `oracle_specs` - Specifications for each oracle to be committed
	/// * `log_inv_rate` - Log2 of the inverse Reed-Solomon code rate
	/// * `n_test_queries` - Number of FRI test queries for soundness
	/// * `arity_strategy` - Strategy for selecting FRI fold arities
	pub fn new<Strategy>(
		merkle_scheme: MerkleScheme_,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		n_test_queries: usize,
		arity_strategy: &Strategy,
	) -> Self
	where
		Strategy: AritySelectionStrategy,
	{
		let max_log_code_len = oracle_specs
			.iter()
			.map(|spec| spec.log_msg_len)
			.max()
			.unwrap_or(0)
			+ log_inv_rate;
		let subspace = BinarySubspace::with_dim(max_log_code_len);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);

		let fri_params = oracle_specs
			.iter()
			.map(|spec| {
				FRIParams::with_strategy(
					&domain_context,
					&merkle_scheme,
					spec.log_msg_len,
					None,
					log_inv_rate,
					n_test_queries,
					arity_strategy,
				)
			})
			.collect();

		Self {
			merkle_scheme,
			oracle_specs,
			fri_params,
		}
	}

	/// Returns a reference to the oracle specifications.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Returns a reference to the precomputed FRI parameters.
	pub fn fri_params(&self) -> &[FRIParams<F>] {
		&self.fri_params
	}

	/// Returns a reference to the Merkle scheme.
	pub fn merkle_scheme(&self) -> &MerkleScheme_ {
		&self.merkle_scheme
	}

	/// Returns the Reed-Solomon code subspace with the largest dimension.
	///
	/// This is useful for creating an NTT that can handle all oracles.
	pub fn max_subspace(&self) -> &BinarySubspace<F> {
		self.fri_params
			.iter()
			.max_by_key(|p| p.rs_code().log_len())
			.map(|p| p.rs_code().subspace())
			.expect("fri_params is non-empty")
	}

	/// Creates a [`SizeTrackingChannel`] from this compiler's oracle specs.
	///
	/// This is useful for estimating proof sizes without running the full protocol.
	/// After verification, call [`SizeTrackingChannel::proof_size()`] to read the
	/// accumulated byte count.
	pub fn create_size_tracking_channel(&self) -> SizeTrackingChannel<'_, F, MerkleScheme_> {
		SizeTrackingChannel::new(self.oracle_specs.clone(), &self.fri_params, &self.merkle_scheme)
	}

	/// Creates a verifier channel from this compiler and a transcript.
	///
	/// The channel borrows the transcript and compiler's precomputed parameters,
	/// avoiding redundant computation and cloning.
	pub fn create_channel<'a, Challenger_>(
		&'a self,
		transcript: &'a mut VerifierTranscript<Challenger_>,
	) -> BaseFoldVerifierChannel<'a, F, MerkleScheme_, Challenger_>
	where
		MerkleScheme_::Digest: DeserializeBytes,
		Challenger_: Challenger,
	{
		BaseFoldVerifierChannel::from_precomputed(
			transcript,
			&self.merkle_scheme,
			&self.oracle_specs,
			&self.fri_params,
		)
	}
}

/// A compiler that creates BaseFold ZK verifier channels with precomputed parameters.
///
/// Unlike [`BaseFoldVerifierCompiler`], this compiler always configures FRI parameters
/// for zero-knowledge mode: `log_msg_len + 1` as the message length and
/// `log_batch_size = 1`.
#[derive(Debug, Clone)]
pub struct BaseFoldZKVerifierCompiler<F, MerkleScheme_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F>,
{
	merkle_scheme: MerkleScheme_,
	oracle_specs: Vec<OracleSpec>,
	fri_params: FRIParams<F>,
}

impl<F, MerkleScheme_> BaseFoldZKVerifierCompiler<F, MerkleScheme_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F>,
{
	/// Creates a new ZK compiler with precomputed FRI parameters.
	///
	/// All oracle specs are treated as ZK: the combined FRI parameters use `log_msg_len + 1` and
	/// `log_batch_size = 1` per oracle. Requires at least one oracle spec.
	pub fn new<Strategy>(
		merkle_scheme: MerkleScheme_,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		n_test_queries: usize,
		_arity_strategy: &Strategy,
	) -> Self
	where
		Strategy: AritySelectionStrategy,
	{
		assert!(
			!oracle_specs.is_empty(),
			"BaseFoldZKVerifierCompiler requires at least one oracle spec"
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
			merkle_scheme,
			oracle_specs,
			fri_params,
		}
	}

	/// Returns a reference to the oracle specifications.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Returns a reference to the precomputed combined FRI parameters.
	pub fn fri_params(&self) -> &FRIParams<F> {
		&self.fri_params
	}

	/// Returns a reference to the Merkle scheme.
	pub fn merkle_scheme(&self) -> &MerkleScheme_ {
		&self.merkle_scheme
	}

	/// Returns the Reed-Solomon code subspace of the combined FRI parameters (the largest needed).
	pub fn max_subspace(&self) -> &BinarySubspace<F> {
		self.fri_params.rs_code().subspace()
	}

	/// Creates a [`SizeTrackingChannel`] from this compiler's oracle specs.
	pub fn create_size_tracking_channel(&self) -> SizeTrackingChannel<'_, F, MerkleScheme_> {
		SizeTrackingChannel::new(
			self.oracle_specs.clone(),
			std::slice::from_ref(&self.fri_params),
			&self.merkle_scheme,
		)
	}

	/// Creates a ZK verifier channel from this compiler and a transcript.
	pub fn create_channel<'a, Challenger_>(
		&'a self,
		transcript: &'a mut VerifierTranscript<Challenger_>,
	) -> BaseFoldZKVerifierChannel<'a, F, MerkleScheme_, Challenger_>
	where
		MerkleScheme_::Digest: DeserializeBytes,
		Challenger_: Challenger,
	{
		BaseFoldZKVerifierChannel::from_precomputed(
			transcript,
			&self.merkle_scheme,
			&self.oracle_specs,
			&self.fri_params,
		)
	}
}
