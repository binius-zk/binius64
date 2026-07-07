// Copyright 2026 The Binius Developers

//! Zero-knowledge proving configuration for Binius64 constraint systems.
//!
//! This module provides [`ZKProver`], which wraps the Binius64 IOP prover with a
//! Spartan-based zero-knowledge wrapper. The prover counterpart to
//! [`binius_verifier::zk_config::ZKVerifier`].

use std::{marker::PhantomData, sync::Arc};

use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_field::{BinaryField128bGhash as B128, PackedField};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop_prover::basefold_compiler::BaseFoldProverCompiler;
use binius_math::ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded};
use binius_spartan_frontend::constraint_system::WitnessLayout;
use binius_spartan_prover::wrapper::{ReplayChannel, ZKWrappedProverChannel};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, SerializeBytes, serialization::SerializationError};
use binius_verifier::{IOPVerifier, zk_config::ZKVerifier};
use bytes::{Buf, BufMut};
use digest::Output;
use rand::CryptoRng;

use crate::{
	IOPProver,
	protocols::shift::{KeyCollection, build_key_collection},
};

type ProverNTT<F> = NeighborsLastMultiThread<GenericPreExpanded<F>>;

/// Zero-knowledge prover for Binius64 constraint systems.
///
/// Wraps the Binius64 IOP prover with a Spartan-based ZK wrapper. Call [`Self::setup`] with
/// a [`ZKVerifier`], then [`Self::prove`] with witness data and a proof transcript.
pub struct ZKProver<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
{
	inner_iop_prover: IOPProver,
	inner_iop_verifier: IOPVerifier,
	outer_iop_prover: binius_spartan_prover::IOPProver<B128>,
	/// Shared with the verifier via an `Arc`, not owned outright.
	/// Setup skips deep-cloning the layout, and each `prove` shares it with a reference-count
	/// bump.
	outer_layout: Arc<WitnessLayout<B128>>,
	basefold_compiler: BaseFoldProverCompiler<P, ProverNTT<B128>>,
	/// The prover creates its Merkle transcript channels with the hash suite `H`.
	_hash_marker: PhantomData<H>,
}

impl<P, H> ZKProver<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes,
{
	/// Constructs a ZK prover from a [`ZKVerifier`].
	pub fn setup(zk_verifier: ZKVerifier<H>) -> Result<Self, Error> {
		let key_collection = {
			let _guard = tracing::debug_span!("Build key collection").entered();
			build_key_collection(zk_verifier.inner_iop_verifier().constraint_system())
		};
		Self::setup_with_key_collection(zk_verifier, key_collection)
	}

	/// Constructs a ZK prover from a verifier and a prebuilt [`KeyCollection`], skipping the
	/// dominant key-collection build. Private: external callers use [`Self::setup`], or
	/// [`DeserializeBytes::deserialize`] to reuse a serialized prover.
	fn setup_with_key_collection(
		zk_verifier: ZKVerifier<H>,
		key_collection: KeyCollection,
	) -> Result<Self, Error> {
		// Build the inner IOPProver.
		let inner_iop_verifier = zk_verifier.inner_iop_verifier().clone();
		let inner_iop_prover = IOPProver::new(inner_iop_verifier.clone(), key_collection);

		// Reuse the padded outer constraint system and layout the verifier already built.
		//
		// Invariant: the verifier builds, pads, and lays out the wrapper circuit once.
		// - It keeps that result for its own verification.
		// - The prover's outer circuit is identical to it.
		// - Reusing the result skips a redundant rebuild.
		//
		// The layout is shared via `Arc`, so the prover and verifier hold one allocation, not two.
		let outer_cs = zk_verifier.outer_iop_verifier().constraint_system().clone();
		let outer_layout = zk_verifier.outer_layout_arc();
		let outer_iop_prover = binius_spartan_prover::IOPProver::new(outer_cs);

		// Build the BaseFold prover compiler from the verifier compiler.
		let subspace = zk_verifier.basefold_compiler().max_subspace();
		let domain_context = {
			let _guard = tracing::debug_span!("Precompute NTT domain").entered();
			GenericPreExpanded::generate_from_subspace(subspace)
		};
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);
		let basefold_compiler =
			BaseFoldProverCompiler::from_verifier_compiler(zk_verifier.basefold_compiler(), ntt);

		Ok(Self {
			inner_iop_prover,
			inner_iop_verifier,
			outer_iop_prover,
			outer_layout,
			basefold_compiler,
			_hash_marker: PhantomData,
		})
	}

	/// Returns a reference to the inner IOP prover.
	pub const fn inner_iop_prover(&self) -> &IOPProver {
		&self.inner_iop_prover
	}

	/// Returns a reference to the KeyCollection.
	pub const fn key_collection(&self) -> &crate::protocols::shift::KeyCollection {
		self.inner_iop_prover.key_collection()
	}

	/// Generates a ZK proof for a witness.
	pub fn prove<Challenger_: Challenger>(
		&self,
		witness: ValueVec,
		mut rng: impl CryptoRng,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		// Clone public words before moving witness into prove().
		let public_words = witness.public().to_vec();

		// Create BaseFold prover channel and wrap with outer prover.
		let basefold_channel = self
			.basefold_compiler
			.create_channel_from_transcript::<H, Challenger_, _>(transcript, &mut rng);
		let mut wrapped_channel = ZKWrappedProverChannel::new(
			basefold_channel,
			&self.outer_iop_prover,
			Arc::clone(&self.outer_layout),
			&mut rng,
			{
				let inner_iop_verifier = &self.inner_iop_verifier;
				move |replay_channel: &mut ReplayChannel<B128>| {
					inner_iop_verifier
						.verify(&public_words, replay_channel)
						.expect("replay verification should not fail");
				}
			},
		);

		// Run the inner IOP proof through the wrapped channel.
		{
			let inner_cs = self.inner_iop_prover.constraint_system();
			let _scope = tracing::debug_span!(
				"Binius64",
				n_hidden_words = inner_cs.value_vec_layout.n_hidden_words,
				n_bitand = inner_cs.and_constraints.len(),
				n_intmul = inner_cs.mul_constraints.len(),
			)
			.entered();

			self.inner_iop_prover
				.prove::<P, _>(witness, &mut wrapped_channel)?;
		}

		// Finish runs the outer spartan proof.
		{
			let outer_cs = self.outer_iop_prover.constraint_system();
			let _scope = tracing::debug_span!(
				"ZK Wrapper",
				n_witness = outer_cs.n_private(),
				n_constraints = outer_cs.mul_constraints().len(),
			)
			.entered();

			wrapped_channel.finish(rng)?;
		}

		Ok(())
	}

	/// Generates a ZK signature of knowledge over `message`.
	///
	/// Binds `message` into the transcript before any other data, then runs the ordinary
	/// [`Self::prove`] logic. See [`binius_verifier::signature`] for details.
	pub fn prove_sig<Challenger_: Challenger>(
		&self,
		witness: ValueVec,
		message: &[u8],
		rng: impl CryptoRng,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		binius_verifier::signature::observe_message::<H, _>(&mut transcript.observe(), message);
		self.prove(witness, rng, transcript)
	}
}

/// Serializes the seed a [`ZKProver`] is built from — constraint system, `log_inv_rate`, and the
/// prebuilt [`KeyCollection`]. On [`DeserializeBytes::deserialize`] the key collection (the
/// dominant setup cost) is reused as-is while the cheaper derived state is recomputed.
impl<P, H> SerializeBytes for ZKProver<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes + DeserializeBytes,
{
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		const VERSION: u32 = 1;
		VERSION.serialize(&mut write_buf)?;
		self.inner_iop_verifier
			.constraint_system()
			.serialize(&mut write_buf)?;
		self.basefold_compiler
			.fri_params()
			.rs_code()
			.log_inv_rate()
			.serialize(&mut write_buf)?;
		self.inner_iop_prover.key_collection().serialize(write_buf)
	}
}

impl<P, H> DeserializeBytes for ZKProver<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes + DeserializeBytes,
{
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		const VERSION: u32 = 1;
		let version = u32::deserialize(&mut read_buf)?;
		if version != VERSION {
			return Err(SerializationError::InvalidConstruction {
				name: "ZKProver::version",
			});
		}
		let constraint_system = ConstraintSystem::deserialize(&mut read_buf)?;
		let log_inv_rate = usize::deserialize(&mut read_buf)?;
		let key_collection = KeyCollection::deserialize(&mut read_buf)?;
		let zk_verifier = ZKVerifier::setup(constraint_system, log_inv_rate)
			.map_err(|_| SerializationError::InvalidConstruction { name: "ZKProver" })?;
		Self::setup_with_key_collection(zk_verifier, key_collection)
			.map_err(|_| SerializationError::InvalidConstruction { name: "ZKProver" })
	}
}

/// Error type for ZK proving.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("inner proving error: {0}")]
	InnerProving(#[from] crate::error::Error),
	#[error("outer proving error: {0}")]
	OuterProving(#[from] binius_spartan_prover::Error),
}
