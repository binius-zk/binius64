// Copyright 2026 The Binius Developers

//! Channel abstraction for verifiers of protocols using Merkle commitments.
//!
//! This module provides the [`MerkleIPVerifierChannel`] trait, which extends
//! [`IPVerifierChannel`] with the ability to receive Merkle commitments and openings of the
//! committed leaves. Protocols like FRI and BaseFold interact with the prover through these
//! operations instead of reading commitments and opening proofs from a transcript directly.
//!
//! The [`VerifierMerkleTranscriptChannel`] implementation wraps a [`VerifierTranscript`] and
//! verifies openings with a [`BinaryMerkleTreeScheme`]: commitment roots are read as observed
//! messages, while opening proofs are read as unobserved decommitment advice bound to the
//! already-observed roots.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_field::{Field, util::FieldFn};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_ip::channel::IPVerifierChannel;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSampleBits, Challenger},
};
use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes};
use digest::Output;

use crate::merkle_tree::{BinaryMerkleTreeScheme, Commitment, MerkleTreeScheme};

/// An extension of [`IPVerifierChannel`] that can receive and open Merkle commitments.
pub trait MerkleIPVerifierChannel<F: Field>: IPVerifierChannel<F> {
	/// A Merkle commitment.
	type Commitment: Clone;

	/// Receives a Merkle commitment for a tree with the given depth and leaf size.
	///
	/// The leaves of the Merkle tree each contain exactly `leaf_size` `F` elements.
	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Self::Commitment, Error>;

	/// Receives a multi-opening of leaves, bound by a Merkle commitment.
	///
	/// Each commitment is associated with the `leaf_size` and `depth` requested when received with
	/// [`Self::recv_merkle_commitment`]. All indices must be less than `2^depth`.
	///
	/// Returns `indices.len() * leaf_size` field elements, where each chunk of `leaf_size`
	/// contiguous elements corresponds to one provided index.
	fn recv_openings(
		&mut self,
		commitment: &Self::Commitment,
		indices: &[usize],
	) -> Result<Vec<F>, Error>;

	/// Receives the full committed vector, bound by a Merkle commitment.
	///
	/// Returns `leaf_size << depth` field elements, in leaf order.
	fn recv_committed_vector(&mut self, commitment: &Self::Commitment) -> Result<Vec<F>, Error>;

	/// Samples a uniform integer with the given number of bits.
	///
	/// Protocols use this to sample query indices for [`Self::recv_openings`].
	fn sample_bits(&mut self, bits: usize) -> usize;
}

/// A [`MerkleIPVerifierChannel`] over a [`VerifierTranscript`], verifying openings with a
/// [`BinaryMerkleTreeScheme`].
///
/// The transcript is held through a [`BorrowMut`] bound, so the channel can own the transcript or
/// mutably borrow one.
pub struct VerifierMerkleTranscriptChannel<T, Challenger_, F, H: HashSuite> {
	transcript: T,
	scheme: BinaryMerkleTreeScheme<F, H>,
	_challenger_marker: PhantomData<Challenger_>,
}

impl<T, Challenger_, F, H: HashSuite> VerifierMerkleTranscriptChannel<T, Challenger_, F, H> {
	/// Constructs a channel over the transcript with a non-hiding Merkle tree scheme.
	pub fn new(transcript: T) -> Self {
		Self::with_scheme(transcript, BinaryMerkleTreeScheme::new())
	}

	/// Constructs a channel over the transcript with the given Merkle tree scheme.
	pub const fn with_scheme(transcript: T, scheme: BinaryMerkleTreeScheme<F, H>) -> Self {
		Self {
			transcript,
			scheme,
			_challenger_marker: PhantomData,
		}
	}

	/// Returns the wrapped transcript.
	pub fn into_transcript(self) -> T {
		self.transcript
	}
}

/// A Merkle commitment received over a transcript channel.
#[derive(Debug, Clone)]
pub struct TranscriptMerkleCommitment<Digest> {
	/// The commitment root and tree depth.
	pub commitment: Commitment<Digest>,
	/// The number of `F` elements in each leaf.
	pub leaf_size: usize,
}

impl<F, T, Challenger_, H> IPVerifierChannel<F>
	for VerifierMerkleTranscriptChannel<T, Challenger_, F, H>
where
	F: Field,
	T: BorrowMut<VerifierTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
{
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		self.transcript.borrow_mut().recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		self.transcript.borrow_mut().recv_many(n)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		self.transcript.borrow_mut().recv_array()
	}

	fn sample(&mut self) -> F {
		IPVerifierChannel::sample(self.transcript.borrow_mut())
	}

	fn observe_one(&mut self, val: F) -> F {
		self.transcript.borrow_mut().observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		self.transcript.borrow_mut().observe_many(vals)
	}

	fn assert_zero(&mut self, val: F) -> Result<(), binius_ip::channel::Error> {
		self.transcript.borrow_mut().assert_zero(val)
	}

	fn compute_public_value(&mut self, inputs: &[F], f: impl FieldFn<F>) -> F {
		self.transcript.borrow_mut().compute_public_value(inputs, f)
	}
}

impl<F, T, Challenger_, H> MerkleIPVerifierChannel<F>
	for VerifierMerkleTranscriptChannel<T, Challenger_, F, H>
where
	F: Field + FixedSizeSerializeBytes,
	T: BorrowMut<VerifierTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
	Output<H::LeafHash>: DeserializeBytes,
{
	type Commitment = TranscriptMerkleCommitment<Output<H::LeafHash>>;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Self::Commitment, Error> {
		let root = self.transcript.borrow_mut().message().read()?;
		Ok(TranscriptMerkleCommitment {
			commitment: Commitment { root, depth },
			leaf_size,
		})
	}

	fn recv_openings(
		&mut self,
		commitment: &Self::Commitment,
		indices: &[usize],
	) -> Result<Vec<F>, Error> {
		let tree_depth = commitment.commitment.depth;
		assert!(indices.iter().all(|&index| index < 1 << tree_depth)); // precondition

		// Read and verify the optimal internal layer once, then verify every opening against it.
		let layer_depth = self.scheme.optimal_verify_layer(indices.len(), tree_depth);
		let mut advice = self.transcript.borrow_mut().decommitment();
		let layer_digests = advice.read_vec(1 << layer_depth)?;
		self.scheme
			.verify_layer(&commitment.commitment.root, layer_depth, &layer_digests)?;

		let mut values = Vec::with_capacity(indices.len() * commitment.leaf_size);
		for &index in indices {
			let leaf = advice.read_scalar_slice::<F>(commitment.leaf_size)?;
			self.scheme.verify_opening(
				index,
				&leaf,
				layer_depth,
				tree_depth,
				&layer_digests,
				&mut advice,
			)?;
			values.extend_from_slice(&leaf);
		}
		Ok(values)
	}

	fn recv_committed_vector(&mut self, commitment: &Self::Commitment) -> Result<Vec<F>, Error> {
		let len = commitment.leaf_size << commitment.commitment.depth;
		let mut advice = self.transcript.borrow_mut().decommitment();
		let data = advice.read_scalar_slice::<F>(len)?;
		self.scheme.verify_vector(
			&commitment.commitment.root,
			&data,
			commitment.leaf_size,
			&mut advice,
		)?;
		Ok(data)
	}

	fn sample_bits(&mut self, bits: usize) -> usize {
		CanSampleBits::sample_bits(self.transcript.borrow_mut(), bits) as usize
	}
}

/// Error type for Merkle channel operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("IP channel error: {0}")]
	IPChannel(#[from] binius_ip::channel::Error),
	#[error("Merkle tree error: {0}")]
	MerkleTree(#[from] crate::merkle_tree::Error),
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
}
