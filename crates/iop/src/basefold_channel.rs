// Copyright 2026 The Binius Developers

//! BaseFold-based implementation of the IOP verifier channel.
//!
//! This module provides [`BaseFoldVerifierChannel`], which implements [`IOPVerifierChannel`] using
//! FRI commitment and BaseFold opening protocols.

use binius_field::BinaryField;
use binius_ip::{MultilinearEvalClaim, channel::IPVerifierChannel};
use binius_math::ntt::AdditiveNTT;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::DeserializeBytes;

use crate::{
	basefold,
	channel::{Error, IOPVerifierChannel, OracleSpec},
	fri::{AritySelectionStrategy, FRIParams},
	merkle_tree::MerkleTreeScheme,
};

/// Oracle handle returned by [`BaseFoldVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
	index: usize,
}

/// A verifier channel that uses BaseFold for oracle commitment and opening.
///
/// This channel wraps a [`VerifierTranscript`] and provides oracle operations using
/// FRI commitment (Reed-Solomon encoding + Merkle tree) and BaseFold opening protocols.
///
/// # Type Parameters
///
/// - `F`: The binary field type
/// - `NTT`: The additive NTT for Reed-Solomon encoding
/// - `MerkleScheme_`: The Merkle tree scheme for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldVerifierChannel<F, NTT, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F>,
	MerkleScheme_: MerkleTreeScheme<F>,
	Challenger_: Challenger,
{
	/// Verifier transcript for Fiat-Shamir.
	transcript: VerifierTranscript<Challenger_>,
	/// NTT (owned). Currently only used during construction for FRIParams computation.
	#[allow(dead_code)]
	ntt: NTT,
	/// Merkle tree scheme (owned).
	merkle_scheme: MerkleScheme_,
	/// Oracle specifications.
	oracle_specs: Vec<OracleSpec>,
	/// Precomputed FRI params per oracle.
	fri_params: Vec<FRIParams<F>>,
	/// Received oracle commitments.
	oracle_commitments: Vec<MerkleScheme_::Digest>,
	/// Next oracle index.
	next_oracle_index: usize,
}

impl<F, NTT, MerkleScheme_, Challenger_> BaseFoldVerifierChannel<F, NTT, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F>,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	/// Creates a new BaseFold verifier channel.
	///
	/// # Arguments
	///
	/// * `transcript` - The verifier transcript for Fiat-Shamir
	/// * `ntt` - The additive NTT for Reed-Solomon encoding (owned)
	/// * `merkle_scheme` - The Merkle tree scheme (owned)
	/// * `oracle_specs` - Specifications for each oracle to be committed
	/// * `log_inv_rate` - Log2 of the inverse Reed-Solomon code rate
	/// * `n_test_queries` - Number of FRI test queries for soundness
	/// * `arity_strategy` - Strategy for selecting FRI fold arities
	pub fn new(
		transcript: VerifierTranscript<Challenger_>,
		ntt: NTT,
		merkle_scheme: MerkleScheme_,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		n_test_queries: usize,
		arity_strategy: &impl AritySelectionStrategy,
	) -> Self {
		let fri_params = oracle_specs
			.iter()
			.map(|spec| {
				let log_msg_len = if spec.is_zk {
					spec.log_msg_len + 1
				} else {
					spec.log_msg_len
				};
				let log_batch_size = if spec.is_zk { Some(1) } else { None };
				FRIParams::with_strategy(
					&ntt,
					&merkle_scheme,
					log_msg_len,
					log_batch_size,
					log_inv_rate,
					n_test_queries,
					arity_strategy,
				)
				.expect("FRI params should be valid for given oracle spec")
			})
			.collect();

		Self {
			transcript,
			ntt,
			merkle_scheme,
			oracle_specs,
			fri_params,
			oracle_commitments: Vec::new(),
			next_oracle_index: 0,
		}
	}

	/// Returns a reference to the underlying transcript.
	pub fn transcript(&self) -> &VerifierTranscript<Challenger_> {
		&self.transcript
	}

	/// Consumes the channel and returns the underlying transcript.
	pub fn into_transcript(self) -> VerifierTranscript<Challenger_> {
		self.transcript
	}
}

impl<F, NTT, MerkleScheme_, Challenger_> IPVerifierChannel<F>
	for BaseFoldVerifierChannel<F, NTT, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F>,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar_slice(n)
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		self.transcript
			.message()
			.read()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn sample(&mut self) -> F {
		CanSample::sample(&mut self.transcript)
	}
}

impl<F, NTT, MerkleScheme_, Challenger_> IOPVerifierChannel<F>
	for BaseFoldVerifierChannel<F, NTT, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F>,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Oracle = BaseFoldOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining oracle specs"
		);

		let index = self.next_oracle_index;

		// Read commitment from transcript
		let commitment = self
			.transcript
			.message()
			.read::<MerkleScheme_::Digest>()
			.map_err(|_| Error::ProofEmpty)?;

		self.oracle_commitments.push(commitment);
		self.next_oracle_index += 1;

		Ok(BaseFoldOracle { index })
	}

	fn finish(
		mut self,
		oracle_relations: &[(Self::Oracle, F)],
	) -> Result<Vec<MultilinearEvalClaim<F>>, Error> {
		assert!(
			self.remaining_oracle_specs().is_empty(),
			"finish called but {} oracle specs remaining",
			self.remaining_oracle_specs().len()
		);

		let mut claims = Vec::with_capacity(oracle_relations.len());

		// Process each oracle relation with its own BaseFold verification
		for (oracle, eval_claim) in oracle_relations {
			let index = oracle.index;
			assert!(
				index < self.oracle_commitments.len(),
				"oracle index {index} out of bounds, expected < {}",
				self.oracle_commitments.len()
			);

			let spec = &self.oracle_specs[index];
			let fri_params = &self.fri_params[index];
			let commitment = self.oracle_commitments[index].clone();

			// Run BaseFold verification
			let reduced_output = if spec.is_zk {
				basefold::verify_zk(
					fri_params,
					&self.merkle_scheme,
					commitment,
					*eval_claim,
					&mut self.transcript,
				)?
			} else {
				basefold::verify(
					fri_params,
					&self.merkle_scheme,
					commitment,
					*eval_claim,
					&mut self.transcript,
				)?
			};

			// The transparent polynomial evaluation point is derived from the sumcheck challenges
			let mut eval_point = reduced_output.challenges;
			eval_point.reverse();

			// Create the multilinear evaluation claim for the transparent polynomial
			// The caller is responsible for verifying FRI-sumcheck consistency using
			// sumcheck_fri_consistency()
			claims.push(MultilinearEvalClaim {
				eval: reduced_output.final_fri_value,
				point: eval_point,
			});
		}

		Ok(claims)
	}
}
