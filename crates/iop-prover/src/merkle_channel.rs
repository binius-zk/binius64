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

use binius_field::{Field, PackedField};
use binius_hash::binary_merkle_tree::{BinaryMerkleTree, HashSuite};
use binius_iop::merkle_tree::MerkleTreeScheme;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::FieldSlice;
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSampleBits, Challenger},
};
use binius_utils::{SerializeBytes, checked_arithmetics::log2_strict_usize};
use digest::Output;
use rand::CryptoRng;

use crate::merkle_tree::{MerkleTreeProver, commit_field_buffer, prover::BinaryMerkleTreeProver};

/// An extension of [`IPProverChannel`] that can send and open Merkle commitments.
pub trait MerkleIPProverChannel<F: Field>: IPProverChannel<F> {
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
		indices: &[usize],
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

	/// Samples a uniform integer with the given number of bits.
	///
	/// Protocols use this to sample query indices for [`Self::send_openings`], matching the
	/// verifier's samples.
	fn sample_bits(&mut self, bits: usize) -> usize;
}

/// A [`MerkleIPProverChannel`] over a [`ProverTranscript`], committing with a
/// [`BinaryMerkleTreeProver`].
///
/// The transcript is held through a [`BorrowMut`] bound, so the channel can own the transcript or
/// mutably borrow one.
pub struct ProverMerkleTranscriptChannel<T, Challenger_, F, H: HashSuite> {
	transcript: T,
	merkle_prover: BinaryMerkleTreeProver<F, H>,
	_challenger_marker: PhantomData<Challenger_>,
}

impl<T, Challenger_, F, H: HashSuite> ProverMerkleTranscriptChannel<T, Challenger_, F, H> {
	/// Constructs a channel over the transcript with a non-hiding Merkle tree prover.
	pub fn new(transcript: T) -> Self {
		Self::with_merkle_prover(transcript, BinaryMerkleTreeProver::new())
	}

	/// Constructs a channel over the transcript with a hiding Merkle tree prover, salting each
	/// leaf with `salt_len` random field elements drawn from `rng`.
	pub fn hiding(transcript: T, rng: impl CryptoRng, salt_len: usize) -> Self {
		Self::with_merkle_prover(transcript, BinaryMerkleTreeProver::hiding(rng, salt_len))
	}

	/// Constructs a channel over the transcript with the given Merkle tree prover.
	pub const fn with_merkle_prover(
		transcript: T,
		merkle_prover: BinaryMerkleTreeProver<F, H>,
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

impl<F, T, Challenger_, H> IPProverChannel<F>
	for ProverMerkleTranscriptChannel<T, Challenger_, F, H>
where
	F: Field,
	T: BorrowMut<ProverTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
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

impl<F, T, Challenger_, H> MerkleIPProverChannel<F>
	for ProverMerkleTranscriptChannel<T, Challenger_, F, H>
where
	F: Field,
	T: BorrowMut<ProverTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes,
{
	type Commitment = ProverMerkleCommitment<BinaryMerkleTree<Output<H::LeafHash>, F>>;

	fn send_merkle_commitment<P: PackedField<Scalar = F>>(
		&mut self,
		data: FieldSlice<P>,
		leaf_size: usize,
	) -> Self::Commitment {
		assert!(leaf_size.is_power_of_two(), "precondition: leaf_size must be a power of two");
		let log_leaf_size = log2_strict_usize(leaf_size);
		let (commitment, committed) = commit_field_buffer(&self.merkle_prover, data, log_leaf_size);
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
		indices: &[usize],
	) {
		let tree_depth = commitment.depth;
		debug_assert_eq!(tree_depth, data.log_len() - commitment.log_leaf_size);
		assert!(indices.iter().all(|&index| index < 1 << tree_depth)); // precondition

		// Write the optimal internal layer once, then the leaf values and opening proof for each
		// queried index, mirroring the verifier's `recv_openings`.
		let scheme = self.merkle_prover.scheme();
		let layer_depth = scheme.optimal_verify_layer(indices.len(), tree_depth);
		let layer = self.merkle_prover.layer(&commitment.committed, layer_depth);
		let mut advice = self.transcript.borrow_mut().decommitment();
		advice.write_slice(layer);
		for &index in indices {
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

		// Write the data in full, then whatever binding data the verifier's `verify_vector` reads
		// while recomputing the root (the per-leaf salts, empty for non-hiding trees).
		let mut advice = self.transcript.borrow_mut().decommitment();
		advice.write_scalar_iter(data.iter_scalars());
		self.merkle_prover
			.prove_vector(&commitment.committed, &mut advice);
	}

	fn sample_bits(&mut self, bits: usize) -> usize {
		CanSampleBits::sample_bits(self.transcript.borrow_mut(), bits) as usize
	}
}

#[cfg(test)]
mod tests;
