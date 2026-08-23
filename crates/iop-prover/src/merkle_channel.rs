// Copyright 2026 The Binius Developers

//! Channel abstraction for provers of protocols using Merkle commitments.
//!
//! This module provides the [`MerkleIPProverChannel`] trait, the prover-side counterpart of
//! `binius_iop::merkle_channel::MerkleIPVerifierChannel`. It extends [`IPProverChannel`] with the
//! ability to send Merkle commitments and openings of the committed leaves.
//!
//! The [`ProverMerkleTranscriptChannel`] implementation wraps a [`ProverTranscript`] and commits
//! with a [`BinaryMerkleTreeProver`]: commitment roots are written as observed messages, while
//! opening proofs are written as unobserved decommitment advice bound to the already-observed
//! roots.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_compute::{Allocator, GlobalAllocator};
use binius_core::word::Word;
use binius_field::{Field, PackedField};
use binius_hash::binary_merkle_tree::{BinaryMerkleTree, HashSuite};
use binius_iop::merkle_tree::MerkleTreeScheme;
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::FieldSlice;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{SerializeBytes, checked_arithmetics::checked_log_2};
use digest::Output;

use crate::merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver};

/// An extension of [`WordIPProverChannel`] that can send and open Merkle commitments.
///
/// Query indices are [`Self::Word`](WordIPProverChannel::Word)s, matching the verifier's
/// `MerkleIPVerifierChannel`.
pub trait MerkleIPProverChannel<F: Field>: WordIPProverChannel<F> {
	/// A Merkle commitment, carrying the data required to open it later.
	type Commitment;

	/// Commits `data` as a Merkle tree with leaves of exactly `leaf_size` scalars each and sends
	/// the commitment.
	///
	/// The tree depth is `log2(data.len() / leaf_size)`.
	///
	/// ## Preconditions
	///
	/// * `data.len()` must be a multiple of `leaf_size`, and the resulting leaf count must be a
	///   power of two.
	fn send_merkle_commitment<P: PackedField<Scalar = F>>(
		&mut self,
		data: FieldSlice<P>,
		leaf_size: usize,
	) -> Self::Commitment;

	/// Sends a multi-opening of committed leaves, bound by a Merkle commitment.
	///
	/// All indices must be less than `2^depth` for the commitment's tree depth. The verifier
	/// receives `indices.len() * leaf_size` field elements via its matching `recv_openings` call.
	///
	/// ## Preconditions
	///
	/// * `data` must be the buffer passed to [`Self::send_merkle_commitment`] for this commitment.
	fn send_openings<P: PackedField<Scalar = F>>(
		&mut self,
		commitment: &Self::Commitment,
		data: FieldSlice<P>,
		indices: &[Self::Word],
	);

	/// Sends the full committed vector, bound by a Merkle commitment.
	///
	/// ## Preconditions
	///
	/// * `data` must be the buffer passed to [`Self::send_merkle_commitment`] for this commitment.
	fn send_committed_vector<P: PackedField<Scalar = F>>(
		&mut self,
		commitment: &Self::Commitment,
		data: FieldSlice<P>,
	);
}

/// A [`MerkleIPProverChannel`] over a [`ProverTranscript`], committing with a
/// [`BinaryMerkleTreeProver`].
///
/// The transcript is held through a [`BorrowMut`] bound, so the channel can own the transcript or
/// mutably borrow one.
pub struct ProverMerkleTranscriptChannel<
	T,
	Challenger_,
	F,
	H: HashSuite,
	A: Allocator = GlobalAllocator,
> {
	transcript: T,
	merkle_prover: BinaryMerkleTreeProver<F, H, A>,
	_challenger_marker: PhantomData<Challenger_>,
}

impl<T, Challenger_, F, H: HashSuite> ProverMerkleTranscriptChannel<T, Challenger_, F, H> {
	/// Constructs a channel over the transcript with a default Merkle tree prover.
	pub fn new(transcript: T) -> Self {
		Self::with_merkle_prover(transcript, BinaryMerkleTreeProver::new())
	}
}

impl<T, Challenger_, F, H: HashSuite, A: Allocator>
	ProverMerkleTranscriptChannel<T, Challenger_, F, H, A>
{
	/// Constructs a channel over the transcript with the given Merkle tree prover.
	///
	/// The prover carries the allocator every committed tree draws its nodes from.
	pub const fn with_merkle_prover(
		transcript: T,
		merkle_prover: BinaryMerkleTreeProver<F, H, A>,
	) -> Self {
		Self {
			transcript,
			merkle_prover,
			_challenger_marker: PhantomData,
		}
	}

	/// Returns the wrapped transcript.
	pub fn into_transcript(self) -> T {
		self.transcript
	}
}

/// A Merkle commitment produced by [`ProverMerkleTranscriptChannel`], carrying the committed tree
/// required to open it.
pub struct ProverMerkleCommitment<Committed> {
	committed: Committed,
	depth: usize,
	log_leaf_size: usize,
}

impl<F, T, Challenger_, H, A> IPProverChannel<F>
	for ProverMerkleTranscriptChannel<T, Challenger_, F, H, A>
where
	F: Field,
	T: BorrowMut<ProverTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
	A: Allocator,
{
	fn send_one(&mut self, elem: F) {
		self.transcript.borrow_mut().send_one(elem)
	}

	fn send_many(&mut self, elems: &[F]) {
		self.transcript.borrow_mut().send_many(elems)
	}

	fn observe_one(&mut self, val: F) {
		self.transcript.borrow_mut().observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) {
		self.transcript.borrow_mut().observe_many(vals)
	}

	fn sample(&mut self) -> F {
		IPProverChannel::sample(self.transcript.borrow_mut())
	}
}

impl<F, T, Challenger_, H, A> WordIPProverChannel<F>
	for ProverMerkleTranscriptChannel<T, Challenger_, F, H, A>
where
	F: Field,
	T: BorrowMut<ProverTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
	A: Allocator,
{
	type Word = Word;

	fn observe_words(&mut self, words: &[Word]) {
		WordIPProverChannel::<F>::observe_words(self.transcript.borrow_mut(), words);
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		WordIPProverChannel::<F>::sample_bits(self.transcript.borrow_mut(), bits)
	}
}

impl<F, T, Challenger_, H, A> MerkleIPProverChannel<F>
	for ProverMerkleTranscriptChannel<T, Challenger_, F, H, A>
where
	F: Field,
	T: BorrowMut<ProverTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
	A: Allocator,
	Output<H::LeafHash>: SerializeBytes,
{
	type Commitment = ProverMerkleCommitment<BinaryMerkleTree<Output<H::LeafHash>, A>>;

	fn send_merkle_commitment<P: PackedField<Scalar = F>>(
		&mut self,
		data: FieldSlice<P>,
		leaf_size: usize,
	) -> Self::Commitment {
		assert!(leaf_size.is_power_of_two(), "precondition: leaf_size must be a power of two");
		let log_leaf_size = checked_log_2(leaf_size);
		let (commitment, committed) = self.merkle_prover.commit_field_buffer(data, log_leaf_size);
		self.transcript
			.borrow_mut()
			.message()
			.write(&commitment.root);
		ProverMerkleCommitment {
			committed,
			depth: commitment.depth,
			log_leaf_size,
		}
	}

	fn send_openings<P: PackedField<Scalar = F>>(
		&mut self,
		commitment: &Self::Commitment,
		data: FieldSlice<P>,
		indices: &[Word],
	) {
		let tree_depth = commitment.depth;
		debug_assert_eq!(tree_depth, data.log_len() - commitment.log_leaf_size);
		let indices = indices
			.iter()
			.map(|index| index.as_u64() as usize)
			.collect::<Vec<_>>();
		assert!(indices.iter().all(|&index| index < 1 << tree_depth)); // precondition

		// Write the optimal internal layer once, then the leaf values and opening proof for each
		// queried index, mirroring the verifier's `recv_openings`.
		let scheme = self.merkle_prover.scheme();
		let layer_depth = scheme.optimal_verify_layer(indices.len(), tree_depth);
		let layer = self.merkle_prover.layer(&commitment.committed, layer_depth);
		let mut advice = self.transcript.borrow_mut().decommitment();
		advice.write_slice(layer);
		for &index in &indices {
			let leaf = data.chunk(commitment.log_leaf_size, index);
			advice.write_scalar_iter(leaf.iter_scalars());
			self.merkle_prover.prove_opening(
				&commitment.committed,
				layer_depth,
				index,
				&mut advice,
			);
		}
	}

	fn send_committed_vector<P: PackedField<Scalar = F>>(
		&mut self,
		commitment: &Self::Commitment,
		data: FieldSlice<P>,
	) {
		debug_assert_eq!(commitment.depth, data.log_len() - commitment.log_leaf_size);

		// The data itself is the whole opening.
		// The verifier recomputes the root from it, so no further advice follows.
		let mut advice = self.transcript.borrow_mut().decommitment();
		advice.write_scalar_iter(data.iter_scalars());
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use binius_field::{Ghash128b as B128, PackedBinaryGhash2x128b};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel};
	use binius_ip::channel::WordIPVerifierChannel;
	use binius_math::{FieldBuffer, test_utils::random_scalars};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::{MerkleIPProverChannel, ProverMerkleTranscriptChannel};

	type StdChallenger = HasherChallenger<StdDigest>;
	type P = PackedBinaryGhash2x128b;
	type VerifierChannel<T> = VerifierMerkleTranscriptChannel<T, StdChallenger, B128, StdHashSuite>;
	type ProverChannel<T> = ProverMerkleTranscriptChannel<T, StdChallenger, B128, StdHashSuite>;

	const LOG_LEN: usize = 8;
	const LOG_LEAF_SIZE: usize = 2;
	const LEAF_SIZE: usize = 1 << LOG_LEAF_SIZE;
	const DEPTH: usize = LOG_LEN - LOG_LEAF_SIZE;
	const N_QUERIES: usize = 5;

	fn sample_indices<Channel: MerkleIPProverChannel<B128>>(
		channel: &mut Channel,
	) -> Vec<Channel::Word> {
		(0..N_QUERIES).map(|_| channel.sample_bits(DEPTH)).collect()
	}

	#[test]
	fn test_merkle_channel_roundtrip() {
		let mut rng = StdRng::seed_from_u64(0);

		let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
		let data = FieldBuffer::<P, _>::from_values(&scalars);

		// Prover side: commit, sample query indices, open them, then send the vector in full.
		let mut prover_channel =
			ProverChannel::new(ProverTranscript::new(StdChallenger::default()));
		let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
		let indices = sample_indices(&mut prover_channel);
		prover_channel.send_openings(&commitment, data.to_ref(), &indices);
		prover_channel.send_committed_vector(&commitment, data.to_ref());

		// Verifier side: mirror the interaction and check the opened values against the data.
		let transcript = prover_channel.into_transcript().into_verifier();
		let mut verifier_channel = VerifierChannel::new(transcript);
		let commitment = verifier_channel
			.recv_merkle_commitment(LEAF_SIZE, DEPTH)
			.unwrap();
		let verifier_indices = (0..N_QUERIES)
			.map(|_| verifier_channel.sample_bits(DEPTH))
			.collect::<Vec<_>>();
		assert_eq!(verifier_indices, indices);

		let values = verifier_channel
			.recv_openings(&commitment, &indices)
			.unwrap();
		assert_eq!(values.len(), N_QUERIES * LEAF_SIZE);
		for (chunk, index) in values.chunks(LEAF_SIZE).zip(&indices) {
			let index = index.as_u64() as usize;
			assert_eq!(chunk, &scalars[index * LEAF_SIZE..(index + 1) * LEAF_SIZE]);
		}

		let vector = verifier_channel.recv_committed_vector(&commitment).unwrap();
		assert_eq!(vector, scalars);

		verifier_channel.into_transcript().finalize().unwrap();
	}

	#[test]
	fn test_merkle_channel_borrowed_transcript() {
		let mut rng = StdRng::seed_from_u64(0);

		let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
		let data = FieldBuffer::<P, _>::from_values(&scalars);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		{
			let mut prover_channel = ProverChannel::new(&mut prover_transcript);
			let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
			let indices = sample_indices(&mut prover_channel);
			prover_channel.send_openings(&commitment, data.to_ref(), &indices);
		}

		let mut verifier_transcript = prover_transcript.into_verifier();
		{
			let mut verifier_channel = VerifierChannel::new(&mut verifier_transcript);
			let commitment = verifier_channel
				.recv_merkle_commitment(LEAF_SIZE, DEPTH)
				.unwrap();
			let indices = (0..N_QUERIES)
				.map(|_| verifier_channel.sample_bits(DEPTH))
				.collect::<Vec<_>>();
			let values = verifier_channel
				.recv_openings(&commitment, &indices)
				.unwrap();
			for (chunk, index) in values.chunks(LEAF_SIZE).zip(&indices) {
				let index = index.as_u64() as usize;
				assert_eq!(chunk, &scalars[index * LEAF_SIZE..(index + 1) * LEAF_SIZE]);
			}
		}
		verifier_transcript.finalize().unwrap();
	}

	#[test]
	fn test_merkle_channel_rejects_openings_at_wrong_index() {
		let mut rng = StdRng::seed_from_u64(0);

		let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
		let data = FieldBuffer::<P, _>::from_values(&scalars);

		let mut prover_channel =
			ProverChannel::new(ProverTranscript::new(StdChallenger::default()));
		let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
		let indices = sample_indices(&mut prover_channel);
		prover_channel.send_openings(&commitment, data.to_ref(), &indices);

		let transcript = prover_channel.into_transcript().into_verifier();
		let mut verifier_channel = VerifierChannel::new(transcript);
		let commitment = verifier_channel
			.recv_merkle_commitment(LEAF_SIZE, DEPTH)
			.unwrap();
		let _ = (0..N_QUERIES)
			.map(|_| verifier_channel.sample_bits(DEPTH))
			.collect::<Vec<_>>();

		// Requesting openings at indices other than the ones the prover opened must fail.
		let wrong_indices = indices
			.iter()
			.map(|&index| index ^ Word::ONE)
			.collect::<Vec<_>>();
		assert!(
			verifier_channel
				.recv_openings(&commitment, &wrong_indices)
				.is_err()
		);

		// Drop the transcript without finalizing; the tampered read left it misaligned.
		let _ = verifier_channel.into_transcript();
	}

	#[test]
	fn test_merkle_channel_rejects_wrong_root() {
		let mut rng = StdRng::seed_from_u64(0);

		let scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
		let data = FieldBuffer::<P, _>::from_values(&scalars);
		let other_scalars = random_scalars::<B128>(&mut rng, 1 << LOG_LEN);
		let other_data = FieldBuffer::<P, _>::from_values(&other_scalars);

		// Commit one buffer but open the other, so the openings do not match the commitment.
		let mut prover_channel =
			ProverChannel::new(ProverTranscript::new(StdChallenger::default()));
		let commitment = prover_channel.send_merkle_commitment(data.to_ref(), LEAF_SIZE);
		let other_commitment =
			prover_channel.send_merkle_commitment(other_data.to_ref(), LEAF_SIZE);
		let indices = sample_indices(&mut prover_channel);
		prover_channel.send_openings(&other_commitment, other_data.to_ref(), &indices);
		let _ = commitment;

		let transcript = prover_channel.into_transcript().into_verifier();
		let mut verifier_channel = VerifierChannel::new(transcript);
		let commitment = verifier_channel
			.recv_merkle_commitment(LEAF_SIZE, DEPTH)
			.unwrap();
		let _other_commitment = verifier_channel
			.recv_merkle_commitment(LEAF_SIZE, DEPTH)
			.unwrap();
		let indices = (0..N_QUERIES)
			.map(|_| verifier_channel.sample_bits(DEPTH))
			.collect::<Vec<_>>();

		// The openings on the tape are bound to `other_commitment`, so verifying them against
		// `commitment` must fail.
		assert!(
			verifier_channel
				.recv_openings(&commitment, &indices)
				.is_err()
		);

		let _ = verifier_channel.into_transcript();
	}
}
