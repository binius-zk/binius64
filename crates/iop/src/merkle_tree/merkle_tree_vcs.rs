// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_transcript::{Buf, TranscriptReader};
use binius_utils::FixedSizeSerializeBytes;

use super::error::Error;

/// A Merkle tree commitment.
///
/// This struct includes the depth of the tree to guard against attacks that exploit the
/// indistinguishability of leaf digests from inner node digests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<Digest> {
	/// The root digest of the Merkle tree.
	pub root: Digest,
	/// The depth of the Merkle tree.
	pub depth: usize,
}

/// A Merkle tree scheme.
pub trait MerkleTreeScheme<T: FixedSizeSerializeBytes> {
	type Digest: Clone + PartialEq + Eq;

	/// Returns the optimal layer that the verifier should verify only once.
	fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize;

	/// Returns the total byte-size of a proof for multiple opening queries.
	///
	/// ## Arguments
	///
	/// * `len` - the length of the committed vector
	/// * `n_queries` - the number of opening queries
	/// * `layer_depth` - the depth of the internal layer the verifier decommits once and verifies
	///   all openings against (see [`Self::optimal_verify_layer`])
	///
	/// ## Preconditions
	///
	/// * `len` must be a power of two.
	/// * `layer_depth` must be at most `log2(len)`.
	fn proof_size(&self, len: usize, n_queries: usize, layer_depth: usize) -> usize;

	/// Verify the opening of the full vector.
	///
	/// ## Preconditions
	///
	/// * `data.len()` must be a multiple of `batch_size`.
	fn verify_vector<B: Buf>(
		&self,
		root: &Self::Digest,
		data: &[T],
		batch_size: usize,
		proof: &mut TranscriptReader<B>,
	) -> Result<(), Error>;

	/// Verify a given layer of the Merkle tree.
	///
	/// When a protocol requires verification of many openings at independent and randomly sampled
	/// indices, it is more efficient for the verifier to verifier an internal layer once, then
	/// verify all openings with respect to that layer.
	///
	/// ## Preconditions
	///
	/// * `layer_digests.len()` must equal `2^layer_depth`.
	fn verify_layer(
		&self,
		root: &Self::Digest,
		layer_depth: usize,
		layer_digests: &[Self::Digest],
	) -> Result<(), Error>;

	/// Verify an opening proof for an entry in a committed vector at the given index.
	///
	/// ## Preconditions
	///
	/// * `layer_digests.len()` must equal `2^layer_depth`.
	/// * `index` must be less than `2^tree_depth`.
	fn verify_opening<B: Buf>(
		&self,
		index: usize,
		values: &[T],
		layer_depth: usize,
		tree_depth: usize,
		layer_digests: &[Self::Digest],
		proof: &mut TranscriptReader<B>,
	) -> Result<(), Error>;
}
