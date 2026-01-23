// Copyright 2025 Irreducible Inc.

pub mod config;
pub mod pcs;
pub mod wiring;

use binius_field::{BinaryField, Field};
use binius_math::{
	BinarySubspace, FieldSlice,
	multilinear::evaluate::evaluate,
	ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
};
use binius_spartan_frontend::constraint_system::{
	BlindingInfo, ConstraintSystem, ConstraintSystemPadded,
};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::{DeserializeBytes, checked_arithmetics::checked_log_2};
use binius_verifier::{
	fri::{self, FRIParams, MinProofSizeStrategy, calculate_n_test_queries},
	hash::PseudoCompressionFunction,
	merkle_tree::BinaryMerkleTreeScheme,
	protocols::{basefold, mlecheck, mlecheck::mask_buffer_dimensions, sumcheck},
};
use digest::{Digest, Output, core_api::BlockSizeUser};

pub const SECURITY_BITS: usize = 96;

/// Output of the multiplication constraint check verification.
#[derive(Debug, Clone)]
pub struct MulcheckOutput<F: Field> {
	/// Evaluation of operand A at the challenge point.
	pub a_eval: F,
	/// Evaluation of operand B at the challenge point.
	pub b_eval: F,
	/// Evaluation of operand C at the challenge point.
	pub c_eval: F,
	/// Evaluation of the mask polynomial at the challenge point.
	pub mask_eval: F,
	/// The challenge point from the sumcheck reduction.
	pub r_x: Vec<F>,
}

/// Struct for verifying instances of a particular constraint system.
///
/// The [`Self::setup`] constructor determines public parameters for proving instances of the given
/// constraint system. Then [`Self::verify`] is called one or more times with individual instances.
#[derive(Debug, Clone)]
pub struct Verifier<F: Field, MerkleHash, MerkleCompress> {
	constraint_system: ConstraintSystemPadded,
	fri_params: FRIParams<F>,
	mulcheck_mask_fri_params: FRIParams<F>,
	/// Mask buffer dimensions (m_n, m_d) for the ZK mulcheck mask polynomial.
	mask_dims: (usize, usize),
	merkle_scheme: BinaryMerkleTreeScheme<F, MerkleHash, MerkleCompress>,
}

impl<F, MerkleHash, MerkleCompress> Verifier<F, MerkleHash, MerkleCompress>
where
	F: BinaryField,
	MerkleHash: Digest + BlockSizeUser,
	MerkleCompress: PseudoCompressionFunction<Output<MerkleHash>, 2>,
	Output<MerkleHash>: DeserializeBytes,
{
	/// Constructs a verifier for a constraint system.
	///
	/// See [`Verifier`] struct documentation for details.
	pub fn setup(
		constraint_system: ConstraintSystem,
		log_inv_rate: usize,
		compression: MerkleCompress,
	) -> Result<Self, Error> {
		// Modify the constraint system for zero-knowledge.
		let n_fri_queries = fri::calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
		let blinding_info = BlindingInfo {
			n_dummy_wires: n_fri_queries,
			// TODO: Document why these are necessary
			n_dummy_constraints: 2,
		};
		let constraint_system = ConstraintSystemPadded::new(constraint_system, blinding_info);

		// The message contains the witness and a random mask of equal size to the witness.
		// For ZK mode, the batch size is 1 (witness and mask are the two interleaved elements).
		let log_witness_size = constraint_system.log_size() as usize;
		let log_batch_size = 1;
		let log_dim = log_witness_size; // RS code dimension equals witness size
		let log_code_len = log_dim + log_inv_rate;
		let merkle_scheme = BinaryMerkleTreeScheme::new(compression);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);

		// Calculate mask buffer dimensions using the shared function.
		let log_mul_constraints = checked_log_2(constraint_system.mul_constraints().len());
		let mask_degree = 2; // quadratic composition
		let mask_dims = mask_buffer_dimensions(log_mul_constraints, mask_degree, n_test_queries);
		let (m_n, m_d) = mask_dims;
		// log_batch_size accounts for the masks_mask (BaseFold mask for the mask commitment)
		let log_mask_dim = m_n + m_d;
		let log_mask_code_len = log_mask_dim + log_batch_size + log_inv_rate;

		// Create a single NTT with the max domain size for both witness and mask.
		let max_log_code_len = log_code_len.max(log_mask_code_len);
		let subspace = BinarySubspace::with_dim(max_log_code_len);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		let fri_params = FRIParams::with_strategy(
			&ntt,
			&merkle_scheme,
			log_dim + log_batch_size,
			Some(log_batch_size),
			log_inv_rate,
			n_test_queries,
			&MinProofSizeStrategy,
		)?;

		let mulcheck_mask_fri_params = FRIParams::with_strategy(
			&ntt,
			&merkle_scheme,
			log_mask_dim + log_batch_size,
			Some(log_batch_size),
			log_inv_rate,
			n_test_queries,
			&MinProofSizeStrategy,
		)?;

		Ok(Self {
			constraint_system,
			fri_params,
			mulcheck_mask_fri_params,
			mask_dims,
			merkle_scheme,
		})
	}

	pub fn constraint_system(&self) -> &ConstraintSystemPadded {
		&self.constraint_system
	}

	pub fn fri_params(&self) -> &FRIParams<F> {
		&self.fri_params
	}

	pub fn mulcheck_mask_fri_params(&self) -> &FRIParams<F> {
		&self.mulcheck_mask_fri_params
	}

	/// Returns the mask buffer dimensions (m_n, m_d) for the ZK mulcheck mask polynomial.
	pub fn mask_dims(&self) -> (usize, usize) {
		self.mask_dims
	}

	pub fn verify<Challenger_: Challenger>(
		&self,
		public: &[F],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		let _verify_guard =
			tracing::info_span!("Verify", operation = "verify", perfetto_category = "operation")
				.entered();

		let cs = self.constraint_system();

		// Check that the public input length is correct
		if public.len() != 1 << cs.log_public() {
			return Err(Error::IncorrectPublicInputLength {
				expected: 1 << self.constraint_system.log_public(),
				actual: public.len(),
			});
		}

		// Verifier observes the public input (includes it in Fiat-Shamir).
		transcript.observe().write_slice(public);

		// Receive the trace commitment.
		let trace_commitment = transcript.message().read::<Output<MerkleHash>>()?;

		// Receive the mask commitment.
		let mask_commitment = transcript.message().read::<Output<MerkleHash>>()?;

		// Verify the multiplication constraints.
		let MulcheckOutput {
			a_eval,
			b_eval,
			c_eval,
			mask_eval,
			r_x,
		} = self.verify_mulcheck(transcript)?;

		// Sample the public input check challenge and evaluate the public input at the challenge
		// point.
		let r_public = transcript.sample_vec(cs.log_public() as usize);

		let public = FieldSlice::from_slice(cs.log_public() as usize, public);
		let public_eval = evaluate(&public, &r_public);

		// Verify the wiring check, public input check, and witness commitment opening with a
		// combined BaseFold reduction.
		let wiring_output = wiring::verify(
			&self.fri_params,
			&self.merkle_scheme,
			trace_commitment,
			&[a_eval, b_eval, c_eval],
			public_eval,
			transcript,
		)?;
		wiring::check_eval(&self.constraint_system, &r_public, &r_x, &wiring_output)?;

		// Verify the mask opening: mask_eval = <masks_buffer, libra_eval_r>
		self.verify_mask_opening(mask_commitment, mask_eval, &r_x, transcript)?;

		Ok(())
	}

	fn verify_mulcheck<Challenger_: Challenger>(
		&self,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<MulcheckOutput<F>, Error> {
		let log_mul_constraints = checked_log_2(self.constraint_system.mul_constraints().len());

		// Sample random evaluation point
		let r_mulcheck = transcript.sample_vec(log_mul_constraints);

		// Verify the zerocheck for the multiplication constraints.
		let mlecheck::VerifyZKOutput {
			eval,
			mask_eval,
			challenges: mut r_x,
		} = mlecheck::verify_zk(&r_mulcheck, 2, F::ZERO, transcript)?;

		// Reverse because sumcheck binds high-to-low variable indices.
		r_x.reverse();

		// Read the claimed evaluations
		let [a_eval, b_eval, c_eval] = transcript.message().read()?;

		if a_eval * b_eval - c_eval != eval {
			return Err(Error::IncorrectMulCheckEvaluation);
		}

		Ok(MulcheckOutput {
			a_eval,
			b_eval,
			c_eval,
			mask_eval,
			r_x,
		})
	}

	/// Verifies the mask opening proof: mask_eval = <masks_buffer, libra_eval_r>.
	///
	/// Uses ZK BaseFold to verify that the claimed mask evaluation is consistent with
	/// the committed mask polynomial.
	fn verify_mask_opening<Challenger_: Challenger>(
		&self,
		mask_commitment: Output<MerkleHash>,
		mask_eval: F,
		r_x: &[F],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		let (_m_n, m_d) = self.mask_dims;
		let n_vars = r_x.len();
		let mask_degree = 2; // quadratic composition

		// Verify the BaseFold proof
		let basefold::ReducedOutput {
			final_fri_value,
			final_sumcheck_value,
			challenges,
		} = basefold::verify_zk(
			&self.mulcheck_mask_fri_params,
			&self.merkle_scheme,
			mask_commitment,
			mask_eval,
			transcript,
		)?;

		// Skip batch challenge (first element) to get the query point
		// The remaining challenges form the query point (high-to-low order from sumcheck)
		let query_point: Vec<F> = challenges[1..].iter().rev().copied().collect();

		// Split into query_k (low-order bits) and query_j (high-order bits)
		let (query_k, query_j) = query_point.split_at(m_d);

		// Compute the expected libra_eval at the query point
		let expected_libra_eval =
			mlecheck::libra_eval::<F, F>(r_x, query_j, query_k, n_vars, mask_degree);

		// Verify consistency: final_fri_value * expected_libra_eval == final_sumcheck_value
		if final_fri_value * expected_libra_eval != final_sumcheck_value {
			return Err(Error::IncorrectMaskOpening);
		}

		Ok(())
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("FRI error: {0}")]
	FRI(#[from] fri::Error),
	#[error("PCS error: {0}")]
	PCS(#[from] pcs::Error),
	#[error("Sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("BaseFold error: {0}")]
	BaseFold(#[from] basefold::Error),
	#[error("wiring error: {0}")]
	Wiring(#[from] wiring::Error),
	#[error("Transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("incorrect public inputs length: expected {expected}, got {actual}")]
	IncorrectPublicInputLength { expected: usize, actual: usize },
	#[error("incorrect reduction output of the multiplication check")]
	IncorrectMulCheckEvaluation,
	#[error("mask opening verification failed")]
	IncorrectMaskOpening,
}
