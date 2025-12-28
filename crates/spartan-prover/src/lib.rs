// Copyright 2025 Irreducible Inc.

mod error;
pub mod pcs;
mod wiring;

use std::marker::PhantomData;

use binius_field::{BinaryField, Field, PackedExtension, PackedField};
use binius_math::{
	FieldBuffer, FieldSlice,
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
};
use binius_prover::{
	fri::{self, CommitOutput, FRIFoldProver},
	hash::{ParallelDigest, parallel_compression::ParallelPseudoCompression},
	merkle_tree::prover::BinaryMerkleTreeProver,
	protocols::sumcheck::{prove_single_mlecheck, quadratic_mle::QuadraticMleCheckProver},
};
use binius_spartan_frontend::constraint_system::{MulConstraint, WitnessIndex};
use binius_spartan_verifier::Verifier;
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::{SerializeBytes, checked_arithmetics::checked_log_2, rayon::prelude::*};
use digest::{Digest, FixedOutputReset, Output, core_api::BlockSizeUser};
pub use error::*;

use crate::wiring::WiringTranspose;

/// Struct for proving instances of a particular constraint system.
///
/// The [`Self::setup`] constructor pre-processes reusable structures for proving instances of the
/// given constraint system. Then [`Self::prove`] is called one or more times with individual
/// instances.
#[derive(Debug)]
pub struct Prover<P, ParallelMerkleCompress, ParallelMerkleHasher: ParallelDigest>
where
	P: PackedField,
	ParallelMerkleCompress: ParallelPseudoCompression<Output<ParallelMerkleHasher::Digest>, 2>,
{
	verifier:
		Verifier<P::Scalar, ParallelMerkleHasher::Digest, ParallelMerkleCompress::Compression>,
	ntt: NeighborsLastMultiThread<GenericPreExpanded<P::Scalar>>,
	merkle_prover: BinaryMerkleTreeProver<P::Scalar, ParallelMerkleHasher, ParallelMerkleCompress>,
	wiring_transpose: WiringTranspose,
	_p_marker: PhantomData<P>,
}

impl<F, P, MerkleHash, ParallelMerkleCompress, ParallelMerkleHasher>
	Prover<P, ParallelMerkleCompress, ParallelMerkleHasher>
where
	F: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<F>,
	MerkleHash: Digest + BlockSizeUser + FixedOutputReset,
	ParallelMerkleHasher: ParallelDigest<Digest = MerkleHash>,
	ParallelMerkleCompress: ParallelPseudoCompression<Output<MerkleHash>, 2>,
	Output<MerkleHash>: SerializeBytes,
{
	/// Constructs a prover corresponding to a constraint system verifier.
	///
	/// See [`Prover`] struct documentation for details.
	pub fn setup(
		verifier: Verifier<F, MerkleHash, ParallelMerkleCompress::Compression>,
		compression: ParallelMerkleCompress,
	) -> Result<Self, Error> {
		let subspace = verifier.fri_params().rs_code().subspace();
		let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		let merkle_prover = BinaryMerkleTreeProver::<_, ParallelMerkleHasher, _>::new(compression);

		// Compute wiring transpose from constraint system
		let cs = verifier.constraint_system();
		let wiring_transpose = WiringTranspose::transpose(cs.size(), cs.mul_constraints());

		Ok(Prover {
			verifier,
			ntt,
			merkle_prover,
			wiring_transpose,
			_p_marker: PhantomData,
		})
	}

	pub fn prove<Challenger_: Challenger>(
		&self,
		witness: &[F],
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		let _prove_guard =
			tracing::info_span!("Prove", operation = "prove", perfetto_category = "operation")
				.entered();

		let cs = self.verifier.constraint_system();

		// Check that the witness length matches the constraint system
		let expected_size = cs.size();
		if witness.len() != expected_size {
			return Err(Error::ArgumentError {
				arg: "witness".to_string(),
				msg: format!("witness has {} elements, expected {}", witness.len(), expected_size),
			});
		}

		let log_mul_constraints = checked_log_2(cs.mul_constraints().len());

		// Prover observes the public input (includes it in Fiat-Shamir).
		let public = &witness[..1 << cs.log_public()];
		transcript.observe().write_slice(public);

		// Pack witness into field elements
		// TODO: Populate witness directly into a FieldBuffer
		let witness_packed = pack_witness::<_, P>(cs.log_size() as usize, witness);

		// Commit the witness
		let CommitOutput {
			commitment: trace_commitment,
			committed: codeword_committed,
			codeword,
		} = fri::commit_interleaved(
			self.verifier.fri_params(),
			&self.ntt,
			&self.merkle_prover,
			witness_packed.to_ref(),
		)?;
		transcript.message().write(&trace_commitment);

		// Prove the multiplication constraints
		let (mulcheck_evals, r_x) = self.prove_mulcheck(
			cs.mul_constraints(),
			witness_packed.to_ref(),
			log_mul_constraints,
			transcript,
		)?;

		// Run wiring check protocol
		let r_public = transcript.sample_vec(cs.log_public() as usize);

		let fri_prover = FRIFoldProver::new(
			self.verifier.fri_params(),
			&self.ntt,
			&self.merkle_prover,
			codeword.as_ref(),
			&codeword_committed,
		)?;
		wiring::prove(
			&self.wiring_transpose,
			fri_prover,
			&r_public,
			&r_x,
			witness_packed.clone(),
			&mulcheck_evals,
			transcript,
		)?;

		Ok(())
	}

	fn prove_mulcheck<Challenger_: Challenger>(
		&self,
		mul_constraints: &[MulConstraint<WitnessIndex>],
		witness: FieldSlice<P>,
		log_mul_constraints: usize,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<([F; 3], Vec<F>), Error> {
		let mulcheck_witness = wiring::build_mulcheck_witness(mul_constraints, witness);

		// Sample random evaluation point for mulcheck
		let r_mulcheck = transcript.sample_vec(log_mul_constraints);

		// Create the QuadraticMleCheckProver for the mul gate: a * b - c
		let mlecheck_prover = QuadraticMleCheckProver::new(
			[mulcheck_witness.a, mulcheck_witness.b, mulcheck_witness.c],
			|[a, b, c]| a * b - c, // composition
			|[a, b, _c]| a * b,    // infinity_composition (quadratic term only)
			r_mulcheck,
			F::ZERO, // eval_claim: zerocheck
		)?;

		// Run the MLE-check protocol
		let mlecheck_output = prove_single_mlecheck(mlecheck_prover, transcript)?;

		// Extract the reduced evaluation point and multilinear evaluations
		let mut r_x = mlecheck_output.challenges;
		r_x.reverse(); // Match verifier's order

		let [a_eval, b_eval, c_eval]: [F; 3] = mlecheck_output
			.multilinear_evals
			.try_into()
			.expect("mlecheck returns 3 evaluations");

		// Write the multilinear evaluations to transcript
		transcript.message().write(&[a_eval, b_eval, c_eval]);

		let mulcheck_evals = [a_eval, b_eval, c_eval];

		Ok((mulcheck_evals, r_x))
	}
}

fn pack_witness<F: Field, P: PackedField<Scalar = F>>(
	log_witness_elems: usize,
	witness: &[F],
) -> FieldBuffer<P> {
	// Precondition: witness length must match expected size
	let expected_size = 1 << log_witness_elems;
	assert_eq!(
		witness.len(),
		expected_size,
		"witness length {} does not match expected size {}",
		witness.len(),
		expected_size
	);

	let len = 1 << log_witness_elems.saturating_sub(P::LOG_WIDTH);
	let mut packed_witness = Vec::<P>::with_capacity(len);

	packed_witness
		.spare_capacity_mut()
		.into_par_iter()
		.enumerate()
		.for_each(|(i, dst)| {
			let offset = i << P::LOG_WIDTH;
			let value = P::from_fn(|j| witness[offset + j]);

			dst.write(value);
		});

	// SAFETY: We just initialized all elements
	unsafe {
		packed_witness.set_len(len);
	};

	FieldBuffer::new(log_witness_elems, packed_witness.into_boxed_slice())
		.expect("FieldBuffer::new should succeed with correct log_witness_elems")
}
