// Copyright 2026 The Binius Developers

//! BaseFold compiler for IOP provers.
//!
//! This module provides [`BaseFoldProverCompiler`], which precomputes FRI parameters
//! and can create [`BaseFoldProverChannel`] instances for proving.

use std::marker::PhantomData;

use binius_field::{BinaryField, PackedField};
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler, channel::OracleSpec, fri::FRIParams,
	merkle_tree::MerkleTreeScheme,
};
use binius_math::ntt::AdditiveNTT;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;

use crate::{basefold_channel::BaseFoldProverChannel, merkle_tree::MerkleTreeProver};

/// A compiler that creates BaseFold prover channels with precomputed parameters.
///
/// The compiler holds the NTT, Merkle prover, oracle specifications, and precomputed FRI
/// parameters. It can create multiple channels for different proving sessions.
///
/// # Type Parameters
///
/// - `F`: The binary field type
/// - `P`: The packed field type with `Scalar = F`
/// - `NTT`: The additive NTT for Reed-Solomon encoding
/// - `MerkleProver_`: The Merkle tree prover for commitments
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
	fri_params: Vec<FRIParams<P::Scalar>>,
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
	/// Creates a new compiler with precomputed FRI parameters.
	///
	/// # Arguments
	///
	/// * `ntt` - The additive NTT for Reed-Solomon encoding (owned)
	/// * `merkle_prover` - The Merkle tree prover (owned)
	/// * `oracle_specs` - Specifications for each oracle to be committed
	/// * `log_inv_rate` - Log2 of the inverse Reed-Solomon code rate
	/// * `n_test_queries` - Number of FRI test queries for soundness
	pub fn new(
		ntt: NTT,
		merkle_prover: MerkleProver_,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Self {
		use binius_iop::fri::MinProofSizeStrategy;

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
					merkle_prover.scheme(),
					log_msg_len,
					log_batch_size,
					log_inv_rate,
					n_test_queries,
					&MinProofSizeStrategy,
				)
				.expect("FRI params should be valid for given oracle spec")
			})
			.collect();

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
	/// This reuses the precomputed FRI parameters and oracle specifications from
	/// the verifier compiler, avoiding redundant computation.
	///
	/// # Arguments
	///
	/// * `verifier_compiler` - The verifier compiler to copy parameters from
	/// * `ntt` - The additive NTT for Reed-Solomon encoding (owned)
	/// * `merkle_prover` - The Merkle tree prover (owned)
	pub fn from_verifier_compiler(
		verifier_compiler: &BaseFoldVerifierCompiler<F, MerkleScheme>,
		ntt: NTT,
		merkle_prover: MerkleProver_,
	) -> Self {
		Self {
			ntt,
			merkle_prover,
			oracle_specs: verifier_compiler.oracle_specs().to_vec(),
			fri_params: verifier_compiler.fri_params().to_vec(),
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

	/// Returns a reference to the precomputed FRI parameters.
	pub fn fri_params(&self) -> &[FRIParams<F>] {
		&self.fri_params
	}

	/// Creates a prover channel from this compiler and a transcript.
	///
	/// The channel borrows the NTT, Merkle prover, and transcript from the compiler.
	/// Uses precomputed FRI parameters, avoiding redundant computation.
	pub fn create_channel<'a, Challenger_: Challenger>(
		&'a self,
		transcript: &'a mut ProverTranscript<Challenger_>,
	) -> BaseFoldProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_> {
		BaseFoldProverChannel::from_compiler(self, transcript)
	}
}
