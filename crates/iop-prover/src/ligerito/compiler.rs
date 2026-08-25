// Copyright 2026 The Binius Developers

//! Ligerito compiler for IOP provers.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::{
	channel::OracleSpec,
	ligerito::{LigeritoParams, compiler::LigeritoVerifierCompiler},
	merkle_tree::MerkleTreeScheme,
	soundness::{Grinding, SoundnessRegime},
};
use binius_math::ntt::AdditiveNTT;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use digest::Output;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
	ligerito::channel::LigeritoProverChannel,
	merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
	merkle_tree::prover::BinaryMerkleTreeProver,
};

/// The channel the transcript constructor returns.
///
/// A Ligerito channel over a transcript-backed Merkle channel.
/// Its allocator backs the opening's working buffers and every Merkle tree node it commits.
pub type TranscriptLigeritoProverChannel<'a, F, P, NTT, T, Challenger_, H, A> =
	LigeritoProverChannel<'a, F, P, NTT, ProverMerkleTranscriptChannel<T, Challenger_, F, H, A>, A>;

/// A compiler that creates Ligerito prover channels from a precomputed ladder.
///
/// The mirror of the verifier's compiler, holding the additive transform in addition to the ladder.
#[derive(Debug)]
pub struct LigeritoProverCompiler<P, NTT>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<Field = P::Scalar> + Sync,
{
	/// The transform every level encodes over, sized for the ladder's longest codeword.
	ntt: NTT,
	/// The oracles every channel this compiler makes will commit, in the order they are sent.
	oracle_specs: Vec<OracleSpec>,
	/// The ladder each of those openings runs down.
	params: LigeritoParams,
	/// Ties the packed field to the compiler without storing a value of it.
	_marker: PhantomData<P>,
}

impl<F, P, NTT> LigeritoProverCompiler<P, NTT>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	/// Creates a compiler from an explicit ladder.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` is non-empty.
	/// * The longest message is the ladder's message length.
	/// * `ntt`'s domain covers every level's codeword domain.
	pub fn new(ntt: NTT, oracle_specs: Vec<OracleSpec>, params: LigeritoParams) -> Self {
		// The two sides must agree on the ladder, so the checks are the verifier's, run here too.
		let verifier = LigeritoVerifierCompiler::<F>::new(oracle_specs, params);
		Self::from_verifier_compiler(&verifier, ntt)
	}

	/// Creates a compiler whose ladder is the proof-size-minimizing one for the longest oracle.
	///
	/// `None` means no ladder over that message reaches the security target.
	///
	/// `grinding` is what the ladder will pay per level; pass [`Grinding::NONE`] for none.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` is non-empty.
	/// * `l0_log_inv_rate` is a usable inverse rate and `security_bits` is positive.
	/// * `ntt`'s domain covers every level's codeword domain.
	pub fn optimal<MerkleScheme>(
		ntt: NTT,
		merkle_scheme: &MerkleScheme,
		oracle_specs: Vec<OracleSpec>,
		l0_log_inv_rate: usize,
		regime: SoundnessRegime,
		security_bits: usize,
		grinding: Grinding,
	) -> Option<Self>
	where
		MerkleScheme: MerkleTreeScheme<F>,
	{
		let verifier = LigeritoVerifierCompiler::<F>::optimal(
			merkle_scheme,
			oracle_specs,
			l0_log_inv_rate,
			regime,
			security_bits,
			grinding,
		)?;
		Some(Self::from_verifier_compiler(&verifier, ntt))
	}

	/// Creates a prover compiler from a verifier compiler, reusing its ladder and oracle specs.
	pub fn from_verifier_compiler(
		verifier_compiler: &LigeritoVerifierCompiler<F>,
		ntt: NTT,
	) -> Self {
		Self {
			ntt,
			oracle_specs: verifier_compiler.oracle_specs().to_vec(),
			params: verifier_compiler.params().clone(),
			_marker: PhantomData,
		}
	}

	/// Returns a reference to the additive transform.
	pub const fn ntt(&self) -> &NTT {
		&self.ntt
	}

	/// Returns a reference to the oracle specifications.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Returns a reference to the precomputed ladder.
	pub const fn params(&self) -> &LigeritoParams {
		&self.params
	}

	/// Creates a prover channel over the given Merkle channel and a generator for its masks.
	///
	/// The returned channel drives all prover interaction through the one it is given.
	/// So the caller decides how commitments are produced.
	/// `rng` seeds the generator every zero-knowledge oracle draws its mask from.
	pub fn create_channel<Channel, A>(
		&self,
		channel: Channel,
		rng: impl Rng,
		alloc: A,
	) -> LigeritoProverChannel<'_, F, P, NTT, Channel, A>
	where
		Channel: MerkleIPProverChannel<F, Word = Word>,
		A: Allocator,
	{
		LigeritoProverChannel::new(
			channel,
			&self.ntt,
			self.oracle_specs.clone(),
			&self.params,
			rng,
			alloc,
		)
	}

	/// Creates a prover channel for a compiler whose oracles are all committed in the clear.
	///
	/// A mask is drawn only when committing a zero-knowledge oracle.
	/// With none of those the generator is never read, so its seed cannot reach the proof.
	/// The seed is therefore fixed, and the caller supplies no randomness at all.
	///
	/// # Panics
	///
	/// Panics if any configured oracle asks for zero knowledge.
	/// Such an oracle would mask from a fixed seed, which is no masking at all.
	pub fn create_channel_without_zk<Channel, A>(
		&self,
		channel: Channel,
		alloc: A,
	) -> LigeritoProverChannel<'_, F, P, NTT, Channel, A>
	where
		Channel: MerkleIPProverChannel<F, Word = Word>,
		A: Allocator,
	{
		assert!(
			self.oracle_specs.iter().all(|spec| !spec.is_zk),
			"precondition: a channel built without randomness opens no zero-knowledge oracle"
		);

		// No mask is ever drawn, so the seed is arbitrary.
		self.create_channel(channel, StdRng::seed_from_u64(0), alloc)
	}

	/// Creates a prover channel over a transcript, for the common case.
	///
	/// The transcript may be owned or mutably borrowed.
	/// It is wrapped in a Merkle transcript channel for the given hash suite.
	/// That channel is then handed to the general constructor.
	/// The allocator backs the opening's working buffers and every Merkle tree node it commits.
	/// So one pool serves the whole opening.
	pub fn create_channel_from_transcript<H, Challenger_, T, A>(
		&self,
		transcript: T,
		rng: impl Rng,
		alloc: A,
	) -> TranscriptLigeritoProverChannel<'_, F, P, NTT, T, Challenger_, H, A>
	where
		H: HashSuite,
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

	/// Creates a channel over a transcript for a compiler whose oracles are all in the clear.
	///
	/// The transcript handling matches the constructor that takes a generator.
	/// The channel is built without one, and refuses a zero-knowledge oracle for that reason.
	pub fn create_channel_without_zk_from_transcript<H, Challenger_, T, A>(
		&self,
		transcript: T,
		alloc: A,
	) -> TranscriptLigeritoProverChannel<'_, F, P, NTT, T, Challenger_, H, A>
	where
		H: HashSuite,
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
