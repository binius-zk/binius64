// Copyright 2025 Irreducible Inc.

//! Polynomial commitment scheme prover for binary field multilinears using BaseFold.

use std::ops::Deref;

use binius_field::{BinaryField, PackedExtension, PackedField};
use binius_math::{FieldBuffer, multilinear::eq::eq_ind_partial_eval, ntt::AdditiveNTT};
use binius_prover::{
	fri::{self, CommitOutput, FRIFoldProver},
	merkle_tree::MerkleTreeProver,
	protocols::basefold::BaseFoldProver,
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use binius_verifier::{fri::FRIParams, merkle_tree::MerkleTreeScheme};

use crate::Error;

/// Prover for the BaseFold polynomial commitment scheme for binary field multilinears.
///
/// This prover commits multilinear polynomials over a binary field using FRI and proves evaluations
/// at given points using the BaseFold protocol.
pub struct PCSProver<'a, F, NTT, MTProver>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F> + Sync,
	MTProver: MerkleTreeProver<F>,
{
	ntt: &'a NTT,
	merkle_prover: &'a MTProver,
	fri_params: &'a FRIParams<F>,
}

impl<'a, F, NTT, MerkleScheme, MerkleProver> PCSProver<'a, F, NTT, MerkleProver>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver: MerkleTreeProver<F, Scheme = MerkleScheme>,
{
	/// Creates a new PCS prover.
	///
	/// ## Arguments
	///
	/// * `ntt` - the NTT for the FRI parameters
	/// * `merkle_prover` - the merkle tree prover
	/// * `fri_params` - the FRI parameters
	pub fn new(
		ntt: &'a NTT,
		merkle_prover: &'a MerkleProver,
		fri_params: &'a FRIParams<F>,
	) -> Self {
		let rs_code = fri_params.rs_code();
		assert_eq!(&ntt.subspace(rs_code.log_len()), rs_code.subspace());
		Self {
			ntt,
			merkle_prover,
			fri_params,
		}
	}

	/// Commit to a multilinear polynomial using FRI.
	///
	/// ## Arguments
	///
	/// * `multilinear` - a packed field buffer representing a binary field multilinear polynomial
	pub fn commit<P, Data>(
		&self,
		multilinear: FieldBuffer<P, Data>,
	) -> Result<CommitOutput<P, MerkleScheme::Digest, MerkleProver::Committed>, Error>
	where
		P: PackedField<Scalar = F> + PackedExtension<F>,
		Data: Deref<Target = [P]>,
	{
		fri::commit_interleaved(self.fri_params, self.ntt, self.merkle_prover, multilinear.to_ref())
			.map_err(Into::into)
	}

	/// Prove the committed polynomial's evaluation at a given point.
	///
	/// ## Arguments
	///
	/// * `committed_codeword` - the committed codeword from FRI
	/// * `committed` - the committed merkle tree
	/// * `multilinear` - the multilinear polynomial (must match what was committed)
	/// * `evaluation_point` - the evaluation point in F^n_vars
	/// * `evaluation_claim` - the claimed evaluation of the multilinear at evaluation_point
	/// * `transcript` - the prover's transcript
	pub fn prove<P, Challenger_>(
		&self,
		committed_codeword: &'a [P],
		committed: &'a MerkleProver::Committed,
		multilinear: FieldBuffer<P>,
		evaluation_point: &[F],
		evaluation_claim: F,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = F> + PackedExtension<F>,
		Challenger_: Challenger,
	{
		assert_eq!(
			multilinear.log_len(),
			evaluation_point.len(),
			"multilinear has {} variables but evaluation point has {} coordinates",
			multilinear.log_len(),
			evaluation_point.len()
		);

		// Compute eq_ind for the evaluation point
		let eq_ind = eq_ind_partial_eval::<P>(evaluation_point);

		// Create FRI fold prover
		let fri_folder = FRIFoldProver::new(
			self.fri_params,
			self.ntt,
			self.merkle_prover,
			committed_codeword,
			committed,
		)?;

		// Create and run BaseFold prover
		let basefold_prover =
			BaseFoldProver::new(multilinear, eq_ind, evaluation_claim, fri_folder);
		basefold_prover.prove(transcript)?;

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, PackedExtension, PackedField, Random, arch::OptimalPackedB128};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
	};
	use binius_prover::{
		hash::parallel_compression::ParallelCompressionAdaptor,
		merkle_tree::prover::BinaryMerkleTreeProver,
	};
	use binius_spartan_verifier::{
		config::{B128, StdChallenger},
		pcs,
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{
		fri::FRIParams,
		hash::{StdCompression, StdDigest},
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 32;

	fn run_pcs_prove_and_verify<P>(
		multilinear: FieldBuffer<P>,
		evaluation_point: Vec<B128>,
		evaluation_claim: B128,
	) -> Result<(), Box<dyn std::error::Error>>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B128>,
	{
		let merkle_prover = BinaryMerkleTreeProver::<B128, StdDigest, _>::new(
			ParallelCompressionAdaptor::new(StdCompression::default()),
		);

		let subspace = BinarySubspace::with_dim(multilinear.log_len() + LOG_INV_RATE)?;
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		let fri_params = FRIParams::choose_with_constant_fold_arity(
			&ntt,
			multilinear.log_len(),
			SECURITY_BITS,
			LOG_INV_RATE,
			2,
		)?;

		let pcs_prover = PCSProver::new(&ntt, &merkle_prover, &fri_params);

		// Commit
		let CommitOutput {
			commitment: codeword_commitment,
			committed: codeword_committed,
			codeword,
		} = pcs_prover.commit(multilinear.to_ref())?;

		// Prove
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover_transcript.message().write(&codeword_commitment);
		pcs_prover.prove(
			codeword.as_ref(),
			&codeword_committed,
			multilinear,
			&evaluation_point,
			evaluation_claim,
			&mut prover_transcript,
		)?;

		// Verify
		let mut verifier_transcript = prover_transcript.into_verifier();
		let retrieved_codeword_commitment = verifier_transcript.message().read()?;

		pcs::verify(
			&mut verifier_transcript,
			evaluation_claim,
			&evaluation_point,
			retrieved_codeword_commitment,
			&fri_params,
			merkle_prover.scheme(),
		)?;

		Ok(())
	}

	fn test_setup<P>(n_vars: usize) -> (FieldBuffer<P>, Vec<B128>, B128)
	where
		P: PackedField<Scalar = B128>,
	{
		let mut rng = StdRng::from_seed([0; 32]);

		// Create random multilinear
		let len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
		let data: Vec<P> = (0..len).map(|_| P::random(&mut rng)).collect();
		let multilinear = FieldBuffer::new(n_vars, data.into_boxed_slice())
			.expect("Failed to create FieldBuffer");

		// Create random evaluation point
		let evaluation_point: Vec<B128> = (0..n_vars)
			.map(|_| <B128 as Random>::random(&mut rng))
			.collect();

		let eval_point_eq = eq_ind_partial_eval(&evaluation_point);
		let evaluation_claim = inner_product_buffers(&multilinear, &eval_point_eq);

		(multilinear, evaluation_point, evaluation_claim)
	}

	fn dubiously_modify_claim<P>(claim: &mut B128)
	where
		P: PackedField<Scalar = B128>,
	{
		*claim += P::Scalar::ONE;
	}

	#[test]
	fn test_pcs_valid_proof() {
		type P = OptimalPackedB128;

		let n_vars = 8;
		let (multilinear, evaluation_point, evaluation_claim) = test_setup::<P>(n_vars);

		run_pcs_prove_and_verify::<P>(multilinear, evaluation_point, evaluation_claim).unwrap();
	}

	#[test]
	fn test_pcs_invalid_proof() {
		type P = OptimalPackedB128;

		let n_vars = 8;
		let (multilinear, evaluation_point, mut evaluation_claim) = test_setup::<P>(n_vars);

		dubiously_modify_claim::<P>(&mut evaluation_claim);
		let result = run_pcs_prove_and_verify::<P>(multilinear, evaluation_point, evaluation_claim);
		assert!(result.is_err(), "expected invalid proof to fail verification");
	}

	#[test]
	fn test_pcs_small_polynomial() {
		type P = OptimalPackedB128;

		let n_vars = 4;
		let (multilinear, evaluation_point, evaluation_claim) = test_setup::<P>(n_vars);

		run_pcs_prove_and_verify::<P>(multilinear, evaluation_point, evaluation_claim).unwrap();
	}
}
