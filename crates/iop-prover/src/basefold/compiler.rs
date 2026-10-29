// Copyright 2026 The Binius Developers

//! BaseFold compiler for IOP provers.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_compute::Allocator;
use binius_field::{BinaryField, PackedField};
use binius_hash_prover::ParallelHashSuite;
use binius_iop::{
	basefold::{compiler::BaseFoldVerifierCompiler, layout::RoundLayout},
	channel::OracleSchedule,
	fri::FRIParams,
};
use binius_math::ntt::AdditiveNTT;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use digest::Output;
use rand::{CryptoRng, SeedableRng, rngs::StdRng};

use crate::{
	basefold::channel::BaseFoldProverChannel,
	merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
	merkle_tree::prover::BinaryMerkleTreeProver,
};

/// The channel the `*_from_transcript` constructors return.
///
/// A BaseFold channel over a transcript-backed Merkle channel, where `A` backs both the BaseFold
/// working buffers and the nodes of every Merkle tree committed through it.
pub type TranscriptBaseFoldProverChannel<'a, F, P, NTT, T, Challenger_, H, A> =
	BaseFoldProverChannel<'a, F, P, NTT, ProverMerkleTranscriptChannel<T, Challenger_, F, H, A>, A>;

/// A compiler that creates BaseFold ZK prover channels with precomputed parameters.
///
/// This compiler commits each round of the schedule as one oracle and builds a single combined
/// FRI over the round oracles, with ZK ones configured for zero-knowledge mode. It is built from a
/// [`BaseFoldVerifierCompiler`], so both sides share one layout.
#[derive(Debug)]
pub struct BaseFoldProverCompiler<P, NTT>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<Field = P::Scalar> + Sync,
{
	ntt: NTT,
	schedule: OracleSchedule,
	layouts: Vec<RoundLayout>,
	/// The combined FRI parameters over **all** round oracles.
	fri_params: FRIParams<P::Scalar>,
	_marker: PhantomData<P>,
}

impl<F, P, NTT> BaseFoldProverCompiler<P, NTT>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	/// Creates a prover compiler from a verifier compiler.
	///
	/// This reuses the precomputed FRI parameters, oracle schedule and round layouts.
	pub fn from_verifier_compiler(
		verifier_compiler: &BaseFoldVerifierCompiler<F>,
		ntt: NTT,
	) -> Self {
		Self {
			ntt,
			schedule: verifier_compiler.oracle_schedule().clone(),
			layouts: verifier_compiler.layouts().to_vec(),
			fri_params: verifier_compiler.fri_params().clone(),
			_marker: PhantomData,
		}
	}

	/// Returns a reference to the NTT.
	pub const fn ntt(&self) -> &NTT {
		&self.ntt
	}

	/// Returns a reference to the precomputed combined FRI parameters.
	pub const fn fri_params(&self) -> &FRIParams<F> {
		&self.fri_params
	}

	/// Creates a ZK prover channel over the given Merkle channel and an RNG.
	///
	/// The returned channel drives all prover interaction through `channel`, committing and opening
	/// oracles with this compiler's NTT, schedule, round layouts, and combined FRI parameters. The
	/// caller constructs the Merkle channel, so it decides how commitments are produced.
	///
	/// The RNG seeds the channel's own generator, whose only output is the ZK masks.
	/// A mask is what hides a committed witness at the positions the verifier opens.
	/// Hiding is therefore only as strong as this RNG, so it must be a cryptographic one.
	pub fn create_channel<Channel, A>(
		&self,
		channel: Channel,
		rng: impl CryptoRng,
		alloc: A,
	) -> BaseFoldProverChannel<'_, F, P, NTT, Channel, A>
	where
		Channel: MerkleIPProverChannel<F>,
		A: Allocator,
	{
		BaseFoldProverChannel::new(
			channel,
			&self.ntt,
			&self.schedule,
			&self.layouts,
			self.fri_params.clone(),
			rng,
			alloc,
		)
	}

	/// Creates a prover channel for a compiler whose oracles are all non-ZK.
	///
	/// A mask is drawn from the channel's RNG only when committing a ZK oracle.
	/// With no ZK oracle the RNG is never read, so its seed cannot affect the proof.
	/// The seed is therefore fixed, and no randomness needs to be supplied by the caller.
	///
	/// # Panics
	///
	/// Panics if any configured oracle is ZK.
	/// A ZK oracle would draw its mask from the fixed seed, which destroys the hiding property.
	/// So this constructor refuses to build a channel that could mask deterministically.
	pub fn create_channel_without_zk<Channel, A>(
		&self,
		channel: Channel,
		alloc: A,
	) -> BaseFoldProverChannel<'_, F, P, NTT, Channel, A>
	where
		Channel: MerkleIPProverChannel<F>,
		A: Allocator,
	{
		// A ZK oracle masks with the RNG, so a fixed seed here would silently break hiding.
		assert!(
			self.layouts.iter().all(|layout| !layout.is_zk),
			"create_channel_without_zk requires every oracle to be non-ZK"
		);

		// No mask is ever drawn, so the seed is arbitrary; reuse the seeded-RNG constructor.
		self.create_channel(channel, StdRng::seed_from_u64(0), alloc)
	}

	/// Creates a ZK prover channel over a transcript, for the common case.
	///
	/// The transcript may be owned or mutably borrowed.
	/// It is wrapped in a [`ProverMerkleTranscriptChannel`] for the given hash suite.
	/// That channel is then passed to [`Self::create_channel`].
	/// `alloc` backs both the channel's working buffers and the nodes of every Merkle tree it
	/// commits, so one pool serves the whole opening.
	pub fn create_channel_from_transcript<H, Challenger_, T, A>(
		&self,
		transcript: T,
		rng: impl CryptoRng,
		alloc: A,
	) -> TranscriptBaseFoldProverChannel<'_, F, P, NTT, T, Challenger_, H, A>
	where
		H: ParallelHashSuite,
		Challenger_: Challenger,
		T: BorrowMut<ProverTranscript<Challenger_>>,
		Output<H::LeafHash>: SerializeBytes,
		A: Allocator,
	{
		self.create_channel(
			ProverMerkleTranscriptChannel::with_merkle_prover(
				transcript,
				BinaryMerkleTreeProver::with_allocator(alloc),
			),
			rng,
			alloc,
		)
	}

	/// Creates a non-ZK prover channel over a transcript, for the common case.
	///
	/// The transcript handling matches [`Self::create_channel_from_transcript`]; the channel is
	/// built with [`Self::create_channel_without_zk`] and panics under the same conditions.
	pub fn create_channel_without_zk_from_transcript<H, Challenger_, T, A>(
		&self,
		transcript: T,
		alloc: A,
	) -> TranscriptBaseFoldProverChannel<'_, F, P, NTT, T, Challenger_, H, A>
	where
		H: ParallelHashSuite,
		Challenger_: Challenger,
		T: BorrowMut<ProverTranscript<Challenger_>>,
		Output<H::LeafHash>: SerializeBytes,
		A: Allocator,
	{
		self.create_channel_without_zk(
			ProverMerkleTranscriptChannel::with_merkle_prover(
				transcript,
				BinaryMerkleTreeProver::with_allocator(alloc),
			),
			alloc,
		)
	}
}
