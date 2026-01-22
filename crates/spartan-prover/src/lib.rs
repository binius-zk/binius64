// Copyright 2025 Irreducible Inc.

mod error;
pub mod pcs;
mod wiring;

use std::{
	iter::{repeat_n, repeat_with},
	marker::PhantomData,
};

use binius_field::{BinaryField, Field, PackedExtension, PackedField};
use binius_math::{
	FieldBuffer, FieldSlice,
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
};
use binius_prover::{
	fri::{self, CommitOutput, FRIFoldProver},
	hash::{ParallelDigest, parallel_compression::ParallelPseudoCompression},
	merkle_tree::prover::BinaryMerkleTreeProver,
	protocols::sumcheck::{quadratic_mle::QuadraticMleCheckProver, zk_mlecheck},
};
use binius_spartan_frontend::constraint_system::{BlindingInfo, MulConstraint, WitnessIndex};
use binius_spartan_verifier::Verifier;
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::{
	SerializeBytes,
	checked_arithmetics::checked_log_2,
	rand::par_rand,
	rayon::{self, prelude::*},
};
use digest::{Digest, FixedOutputReset, Output, core_api::BlockSizeUser};
pub use error::*;
use itertools::chain;
use rand::{CryptoRng, rngs::StdRng};

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
		let cs = verifier.constraint_system();
		let subspace = verifier.fri_params().rs_code().subspace();
		let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		let merkle_prover = BinaryMerkleTreeProver::<_, ParallelMerkleHasher, _>::new(compression);

		// Compute wiring transpose from constraint system
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
		mut rng: impl CryptoRng,
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

		// Create mask polynomial for ZK mulcheck
		let mulcheck_mask = zk_mlecheck::Mask::<P>::random(log_mul_constraints, 2, &mut rng);

		// Pack witness into field elements and add blinding
		let blinding_info = cs.blinding_info();
		let witness_packed = pack_and_blind_witness::<_, P>(
			cs.log_size() as usize,
			witness,
			blinding_info,
			cs.n_public() as usize,
			cs.n_private() as usize,
			&mut rng,
		);

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
			&mulcheck_mask,
			transcript,
		)?;

		// Run wiring check protocol
		let r_public = transcript.sample_vec(cs.log_public() as usize);

		let fri_prover = FRIFoldProver::new(
			self.verifier.fri_params(),
			&self.ntt,
			&self.merkle_prover,
			codeword,
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
		mask: &zk_mlecheck::Mask<P>,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<([F; 3], Vec<F>), Error> {
		let mulcheck_witness = wiring::build_mulcheck_witness(mul_constraints, witness);

		// Sample random evaluation point for mulcheck
		let r_mulcheck = transcript.sample_vec(mask.n_vars());

		// Create the QuadraticMleCheckProver for the mul gate: a * b - c
		let mlecheck_prover = QuadraticMleCheckProver::new(
			[mulcheck_witness.a, mulcheck_witness.b, mulcheck_witness.c],
			|[a, b, c]| a * b - c, // composition
			|[a, b, _c]| a * b,    // infinity_composition (quadratic term only)
			r_mulcheck,
			F::ZERO, // eval_claim: zerocheck
		)?;

		// Run the ZK MLE-check protocol
		let mlecheck_output = zk_mlecheck::prove(mlecheck_prover, mask, transcript)?;

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

fn pack_and_blind_witness<F: Field, P: PackedField<Scalar = F>>(
	log_witness_elems: usize,
	witness: &[F],
	blinding_info: &BlindingInfo,
	n_public: usize,
	n_private: usize,
	mut rng: impl CryptoRng,
) -> FieldBuffer<P> {
	let packed_witness = if log_witness_elems < P::LOG_WIDTH {
		let elems_iter = witness.iter().copied();
		let zeros_iter = repeat_n(F::ZERO, (1 << log_witness_elems) - witness.len());
		let mask_iter = repeat_with(|| F::random(&mut rng)).take(1 << log_witness_elems);

		let elems = P::from_scalars(chain!(elems_iter, zeros_iter, mask_iter));
		vec![elems]
	} else {
		let packed_len = 1 << (log_witness_elems - P::LOG_WIDTH);

		let elems_iter = witness
			.par_chunks(P::WIDTH)
			.map(|chunk| P::from_scalars(chunk.iter().copied()));
		let zeros_iter = rayon::iter::repeat_n(P::zero(), packed_len - elems_iter.len());

		// Append a random mask to the end of the witness buffer, of equal length to the witness.
		let mask_iter = par_rand::<StdRng, _, _>(packed_len, &mut rng, P::random);

		elems_iter
			.chain(zeros_iter)
			.chain(mask_iter)
			.collect::<Vec<_>>()
	};

	let mut witness_packed =
		FieldBuffer::new(log_witness_elems + 1, packed_witness.into_boxed_slice());

	// Add blinding values
	let base = n_public + n_private;

	// Set random values for non-constraint dummy wires
	for i in 0..blinding_info.n_dummy_wires {
		witness_packed.set(base + i, F::random(&mut rng));
	}

	// Set random values for dummy constraint wires (A * B = C)
	let constraint_wire_base = base + blinding_info.n_dummy_wires;
	for i in 0..blinding_info.n_dummy_constraints {
		let a = F::random(&mut rng);
		let b = F::random(&mut rng);
		let c = a * b;

		witness_packed.set(constraint_wire_base + 3 * i, a);
		witness_packed.set(constraint_wire_base + 3 * i + 1, b);
		witness_packed.set(constraint_wire_base + 3 * i + 2, c);
	}

	witness_packed
}
