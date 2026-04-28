// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Spartan-based proof generation for Binius64 constraint systems.
//!
//! This crate provides the [`Prover`] struct for generating zero-knowledge proofs
//! using the Spartan protocol adapted for Binius64's constraint system. It is the
//! prover-side counterpart to `binius_spartan_verifier`.
//!
//! # When to use this crate
//!
//! Use this crate when you have a constraint system built with `binius_spartan_frontend`
//! and need to generate a Spartan-based proof. This is an alternative to the main
//! `binius_prover` crate.
//!
//! # Key types
//!
//! - [`Prover`] - Main proving interface; call [`Prover::setup`] with a verifier, then
//!   [`Prover::prove`] with witness data
//! - [`IOPProver`] - Core IOP proving logic, independent of the compilation strategy
//!
//! # Related crates
//!
//! - `binius_spartan_verifier` - Verification counterpart
//! - `binius_spartan_frontend` - Constraint system builder for Spartan
//! - `binius_prover` - Alternative proving backend

#![warn(rustdoc::missing_crate_level_docs)]

mod error;
mod wiring;
pub mod wrapper;

use std::{
	iter::{repeat_n, repeat_with},
	ops::Deref,
};

use binius_field::{BinaryField, Field, PackedExtension, PackedField};
use binius_hash::{ParallelDigest, parallel_compression::ParallelPseudoCompression};
use binius_iop_prover::{
	basefold_compiler::BaseFoldZKProverCompiler, channel::IOPProverChannel,
	merkle_tree::prover::BinaryMerkleTreeProver,
};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{quadratic_mle::QuadraticMleCheckProver, zk_mlecheck},
};
use binius_math::{
	FieldBuffer, FieldSlice,
	multilinear::eq::eq_ind_partial_eval,
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
	univariate::evaluate_univariate,
};
use binius_spartan_frontend::constraint_system::{
	MulConstraint, Witness, WitnessIndex, WitnessSegment,
};
use binius_spartan_verifier::{
	Verifier,
	constraint_system::{BlindingInfo, ConstraintSystemPadded},
	wiring::evaluate_wiring_mle_public,
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{
	SerializeBytes,
	checked_arithmetics::checked_log_2,
	rayon::{self, prelude::*},
};
use digest::{Digest, FixedOutputReset, Output, block_api::BlockSizeUser};
pub use error::*;
use itertools::chain;
use rand::CryptoRng;

use crate::wiring::{WiringTranspose, fold_constraints};

type ProverNTT<F> = NeighborsLastMultiThread<GenericPreExpanded<F>>;
type ProverMerkleProver<F, ParallelMerkleHasher, ParallelMerkleCompress> =
	BinaryMerkleTreeProver<F, ParallelMerkleHasher, ParallelMerkleCompress>;

/// IOP prover for a particular constraint system.
///
/// This struct encapsulates the constraint system and pre-computed wiring transpose,
/// providing the core proving logic independent of the specific IOP compilation strategy.
/// Most users should use [`Prover`] instead, which wraps this with a BaseFold compiler.
#[derive(Debug)]
pub struct IOPProver<F: Field> {
	constraint_system: ConstraintSystemPadded<F>,
	precommit_wiring_transpose: WiringTranspose,
	private_wiring_transpose: WiringTranspose,
}

/// Struct for proving instances of a particular constraint system.
///
/// The [`Self::setup`] constructor pre-processes reusable structures for proving instances of the
/// given constraint system. Then [`Self::prove`] is called one or more times with individual
/// instances.
pub struct Prover<P, ParallelMerkleCompress, ParallelMerkleHasher>
where
	P: PackedField<Scalar: BinaryField>,
	ParallelMerkleHasher: ParallelDigest<Digest: Digest + BlockSizeUser + FixedOutputReset>,
	ParallelMerkleCompress: ParallelPseudoCompression<Output<ParallelMerkleHasher::Digest>, 2>,
{
	iop_prover: IOPProver<P::Scalar>,
	#[allow(clippy::type_complexity)]
	basefold_compiler: BaseFoldZKProverCompiler<
		P,
		ProverNTT<P::Scalar>,
		ProverMerkleProver<P::Scalar, ParallelMerkleHasher, ParallelMerkleCompress>,
	>,
}

impl<F: Field> IOPProver<F> {
	/// Constructs an IOP prover for a constraint system.
	pub fn new(constraint_system: ConstraintSystemPadded<F>) -> Self {
		let precommit_wiring_transpose = WiringTranspose::transpose(
			WitnessSegment::Precommit,
			constraint_system.precommit_size(),
			constraint_system.mul_constraints(),
		);
		let private_wiring_transpose = WiringTranspose::transpose(
			WitnessSegment::Private,
			constraint_system.private_size(),
			constraint_system.mul_constraints(),
		);
		Self {
			constraint_system,
			precommit_wiring_transpose,
			private_wiring_transpose,
		}
	}

	pub fn constraint_system(&self) -> &ConstraintSystemPadded<F> {
		&self.constraint_system
	}

	/// Packs and commits the precommit segment of a witness on the channel.
	///
	/// This must be called before [`Self::prove`], and the returned oracle handle and packed
	/// buffer must be passed into `prove`. Callers that wrap the IOP (e.g. the ZK wrapper) can
	/// invoke this separately so the precommit oracle handle is available before the rest of
	/// the protocol runs.
	pub fn commit_precommit<P, Channel>(
		&self,
		witness: &Witness<F>,
		rng: &mut impl CryptoRng,
		channel: &mut Channel,
	) -> (Channel::Oracle, FieldBuffer<P>)
	where
		F: BinaryField,
		P: PackedField<Scalar = F> + PackedExtension<F>,
		Channel: IOPProverChannel<P>,
	{
		let cs = &self.constraint_system;
		// Precommit segment has no dummy mul-constraint blinding (see ConstraintSystemPadded).
		let precommit_blinding = BlindingInfo {
			n_dummy_wires: cs.blinding_info().n_dummy_wires,
			n_dummy_constraints: 0,
		};
		let precommit_packed = pack_and_blind_witness::<_, P>(
			cs.log_precommit() as usize,
			witness.precommit(),
			cs.n_precommit() as usize,
			&precommit_blinding,
			rng,
		);
		let precommit_oracle = channel.send_oracle(precommit_packed.to_ref());
		(precommit_oracle, precommit_packed)
	}

	/// Proves using an IOP channel interface.
	///
	/// This is the core proving logic, independent of the specific IOP compilation strategy.
	/// For most users, [`Prover::prove`] is the simpler interface.
	///
	/// # Arguments
	///
	/// * `witness` - The witness values for the constraint system
	/// * `precommit_oracle` - Oracle handle obtained from [`Self::commit_precommit`]
	/// * `precommit_packed` - Packed precommit buffer obtained from [`Self::commit_precommit`]
	/// * `rng` - Random number generator for blinding
	/// * `channel` - The IOP prover channel (public input must be observed on transcript before
	///   creating the channel; the precommit oracle must already have been committed on it via
	///   [`Self::commit_precommit`])
	pub fn prove<P, Channel>(
		&self,
		witness: Witness<F>,
		precommit_oracle: Channel::Oracle,
		precommit_packed: FieldBuffer<P>,
		mut rng: impl CryptoRng,
		mut channel: Channel,
	) -> Result<(), Error>
	where
		F: BinaryField,
		P: PackedField<Scalar = F> + PackedExtension<F>,
		Channel: IOPProverChannel<P>,
	{
		let _prove_guard =
			tracing::info_span!("Prove", operation = "prove", perfetto_category = "operation")
				.entered();

		let cs = &self.constraint_system;

		// Check that the witness segments have the expected sizes
		let expected_public_size = 1 << cs.log_public() as usize;
		let expected_precommit_size = cs.precommit_size();
		let expected_private_size = cs.private_size();
		if witness.public().len() != expected_public_size {
			return Err(Error::ArgumentError {
				arg: "witness".to_string(),
				msg: format!(
					"public segment has {} elements, expected {}",
					witness.public().len(),
					expected_public_size
				),
			});
		}
		if witness.precommit().len() != expected_precommit_size {
			return Err(Error::ArgumentError {
				arg: "witness".to_string(),
				msg: format!(
					"precommit segment has {} elements, expected {}",
					witness.precommit().len(),
					expected_precommit_size
				),
			});
		}
		if witness.private().len() != expected_private_size {
			return Err(Error::ArgumentError {
				arg: "witness".to_string(),
				msg: format!(
					"private segment has {} elements, expected {}",
					witness.private().len(),
					expected_private_size
				),
			});
		}

		let log_mul_constraints = checked_log_2(cs.mul_constraints().len());

		// Create mask buffer for the ZK mulcheck mask polynomial.
		let (m_n, m_d) = cs.mask_dims();
		let mask_degree = 2; // quadratic composition
		let log_masks_buffer_size = m_n + m_d;

		let masks_buffer = FieldBuffer::<P>::new(
			log_masks_buffer_size,
			repeat_with(|| P::random(&mut rng))
				.take(1 << log_masks_buffer_size.saturating_sub(P::LOG_WIDTH))
				.collect(),
		);

		let mulcheck_mask =
			zk_mlecheck::Mask::new(log_mul_constraints, mask_degree, masks_buffer.to_ref());

		// Pack private witness into field elements and add blinding
		let blinding_info = cs.blinding_info();
		let private_packed = pack_and_blind_witness::<_, P>(
			cs.log_private() as usize,
			witness.private(),
			cs.n_private() as usize,
			blinding_info,
			&mut rng,
		);

		// Send the private and mask oracles to the channel. The precommit oracle was committed
		// by the caller via `commit_precommit` and passed in as `precommit_oracle`.
		let private_oracle = channel.send_oracle(private_packed.to_ref());
		let mask_oracle = channel.send_oracle(masks_buffer.to_ref());

		// Prove the multiplication constraints
		let (mulcheck_evals, mask_eval, r_x) = prove_mulcheck::<F, P, _>(
			cs.mul_constraints(),
			witness.public(),
			precommit_packed.to_ref(),
			private_packed.to_ref(),
			mulcheck_mask,
			&mut channel,
		)?;

		// λ is the batching challenge for the constraint operands
		let lambda = channel.sample();

		// Batch together the constraint operand evaluation claims.
		let batched_sum = evaluate_univariate(&mulcheck_evals, lambda);

		// Compute eq indicator tensor for r_x (shared across all segment evaluations)
		let r_x_tensor = eq_ind_partial_eval::<F>(&r_x);

		// Compute rₓ^⊤ (M_A + λ M_B + λ² M_C) x
		let public_eval = evaluate_wiring_mle_public(
			cs.mul_constraints(),
			witness.public(),
			lambda,
			r_x_tensor.as_ref(),
		);

		// Compute the precommit segment's contribution to the wiring check.
		// The prover sends this as a scalar; the oracle relation then verifies it.
		let precommit_wiring_poly =
			fold_constraints(&self.precommit_wiring_transpose, lambda, r_x_tensor.as_ref());
		let precommit_claim = binius_math::inner_product::inner_product_buffers(
			&precommit_packed.to_ref(),
			&precommit_wiring_poly,
		);
		channel.send_one(precommit_claim);

		let private_claim = batched_sum - public_eval - precommit_claim;

		// Fold private wiring constraints
		let private_wiring_poly =
			fold_constraints(&self.private_wiring_transpose, lambda, r_x_tensor.as_ref());

		// Compute the mask folding polynomial (libra_eval tensor)
		let n_vars = r_x.len();
		let libra_eval_tensor =
			zk_mlecheck::expand_libra_eval::<P>(&r_x, n_vars, mask_degree, m_n, m_d);

		// Prove all oracle relations
		channel.prove_oracle_relations([
			(precommit_oracle, precommit_packed, precommit_wiring_poly, precommit_claim),
			(private_oracle, private_packed, private_wiring_poly, private_claim),
			(mask_oracle, masks_buffer, libra_eval_tensor, mask_eval),
		]);

		Ok(())
	}
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
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;

		// Get the largest subspace from the verifier compiler for NTT creation
		let subspace = verifier.iop_compiler().max_subspace();
		let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		let merkle_prover = BinaryMerkleTreeProver::<_, ParallelMerkleHasher, _>::new(compression);

		// Create the BaseFold ZK compiler from verifier compiler (reuses oracle_specs and
		// fri_params)
		let basefold_compiler = BaseFoldZKProverCompiler::from_verifier_compiler(
			verifier.iop_compiler(),
			ntt,
			merkle_prover,
		);

		let iop_prover = IOPProver::new(verifier.constraint_system().clone());

		Ok(Prover {
			iop_prover,
			basefold_compiler,
		})
	}

	/// Returns a reference to the IOP prover.
	pub fn iop_prover(&self) -> &IOPProver<P::Scalar> {
		&self.iop_prover
	}

	/// Returns a reference to the BaseFold ZK prover compiler.
	pub fn iop_compiler(
		&self,
	) -> &BaseFoldZKProverCompiler<
		P,
		ProverNTT<F>,
		ProverMerkleProver<F, ParallelMerkleHasher, ParallelMerkleCompress>,
	> {
		&self.basefold_compiler
	}

	/// Generates a proof for a witness against the constraint system.
	///
	/// # Arguments
	///
	/// * `witness` - The witness values for the constraint system
	/// * `rng` - Random number generator for blinding
	/// * `transcript` - The prover transcript for Fiat-Shamir
	///
	/// # Preconditions
	///
	/// * The witness length must match the constraint system size
	pub fn prove<Challenger_: Challenger>(
		&self,
		witness: Witness<F>,
		mut rng: impl CryptoRng,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		// Prover observes the public input (includes it in Fiat-Shamir).
		let public = witness.public();
		transcript.observe().write_slice(public);

		// Create ZK channel (owns the RNG for mask generation), commit the precommit oracle,
		// and delegate to the IOP prover.
		let mut channel = self.basefold_compiler.create_channel(transcript, &mut rng);
		let (precommit_oracle, precommit_packed) =
			self.iop_prover
				.commit_precommit::<P, _>(&witness, &mut rng, &mut channel);
		self.iop_prover
			.prove::<P, _>(witness, precommit_oracle, precommit_packed, rng, channel)
	}
}

fn prove_mulcheck<F, P, Channel>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	public: &[F],
	precommit_packed: FieldSlice<P>,
	private_packed: FieldSlice<P>,
	mask: zk_mlecheck::Mask<P, impl Deref<Target = [P]>>,
	channel: &mut Channel,
) -> Result<([F; 3], F, Vec<F>), Error>
where
	F: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<F>,
	Channel: IPProverChannel<F>,
{
	let mulcheck_witness =
		wiring::build_mulcheck_witness(mul_constraints, public, precommit_packed, private_packed);

	// Sample random evaluation point for mulcheck
	let r_mulcheck = channel.sample_many(mask.n_vars());

	// Create the QuadraticMleCheckProver for the mul gate: a * b - c
	let mlecheck_prover = QuadraticMleCheckProver::new(
		[mulcheck_witness.a, mulcheck_witness.b, mulcheck_witness.c],
		|[a, b, c]| a * b - c, // composition
		|[a, b, _c]| a * b,    // infinity_composition (quadratic term only)
		r_mulcheck,
		F::ZERO, // eval_claim: zerocheck
	)?;

	// Run the ZK MLE-check protocol
	let mlecheck_output = zk_mlecheck::prove(mlecheck_prover, mask, channel)?;

	// Extract the reduced evaluation point and multilinear evaluations
	let mut r_x = mlecheck_output.challenges;
	r_x.reverse(); // Match verifier's order

	let [a_eval, b_eval, c_eval]: [F; 3] = mlecheck_output
		.multilinear_evals
		.try_into()
		.expect("mlecheck returns 3 evaluations");

	// Write the multilinear evaluations to channel
	channel.send_many(&[a_eval, b_eval, c_eval]);

	let mulcheck_evals = [a_eval, b_eval, c_eval];
	let mask_eval = mlecheck_output.mask_eval;

	Ok((mulcheck_evals, mask_eval, r_x))
}

/// Packs witness values into a [`FieldBuffer`] and adds blinding values for dummy wires.
fn pack_and_blind_witness<F: Field, P: PackedField<Scalar = F>>(
	log_private: usize,
	private: &[F],
	n_private: usize,
	blinding_info: &BlindingInfo,
	mut rng: impl CryptoRng,
) -> FieldBuffer<P> {
	let packed = if log_private < P::LOG_WIDTH {
		let elems_iter = private.iter().copied();
		let zeros_iter = repeat_n(F::ZERO, (1 << log_private) - private.len());

		let elems = P::from_scalars(chain!(elems_iter, zeros_iter));
		vec![elems]
	} else {
		let packed_len = 1 << (log_private - P::LOG_WIDTH);

		let elems_iter = private
			.par_chunks(P::WIDTH)
			.map(|chunk| P::from_scalars(chunk.iter().copied()));
		let zeros_iter = rayon::iter::repeat_n(P::zero(), packed_len - elems_iter.len());

		elems_iter.chain(zeros_iter).collect::<Vec<_>>()
	};

	let mut buffer = FieldBuffer::new(log_private, packed.into_boxed_slice());

	// Add blinding values after the actual private wires
	// Set random values for non-constraint dummy wires
	for i in 0..blinding_info.n_dummy_wires {
		buffer.set(n_private + i, F::random(&mut rng));
	}

	// Set random values for dummy constraint wires (A * B = C)
	let constraint_wire_base = n_private + blinding_info.n_dummy_wires;
	for i in 0..blinding_info.n_dummy_constraints {
		let a = F::random(&mut rng);
		let b = F::random(&mut rng);
		let c = a * b;

		buffer.set(constraint_wire_base + 3 * i, a);
		buffer.set(constraint_wire_base + 3 * i + 1, b);
		buffer.set(constraint_wire_base + 3 * i + 2, c);
	}

	buffer
}
