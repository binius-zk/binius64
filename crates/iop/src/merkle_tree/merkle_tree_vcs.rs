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
///
/// The committed values have a fixed serialized width, so a leaf's byte encoding is unambiguous
/// and two distinct leaves can never present the same bytes to the hash.
pub trait MerkleTreeScheme<T: FixedSizeSerializeBytes> {
	/// The digest of a leaf or an inner node.
	type Digest: Clone + PartialEq + Eq;

	/// Returns the optimal layer that the verifier should verify only once.
	///
	/// Decommitting a layer at depth `d` costs `2^d` digests but shortens every one of the
	/// `n_queries` branches by `d`, so the proof holds
	///
	/// ```text
	/// (tree_depth - d) * n_queries + 2^d
	/// ```
	///
	/// digests. That is minimized at `d = ceil(log2(n_queries))`, clamped to the tree depth.
	fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize;

	/// Returns the total byte-size of a proof for multiple opening queries.
	///
	/// This counts the decommitment advice only: the layer digests, the per-query branches, and
	/// any per-query salt. The opened leaf values are sent by the caller, so they are not counted.
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

	/// Returns the total byte-size of a proof for opening the full vector.
	///
	/// This counts the decommitment advice that [`Self::verify_vector`] reads, which is any
	/// per-leaf salt. The committed values themselves are sent by the caller, so they are not
	/// counted.
	///
	/// ## Arguments
	///
	/// * `len` - the length of the committed vector
	fn vector_proof_size(&self, len: usize) -> usize;

	/// Verify the opening of the full vector.
	///
	/// ## Preconditions
	///
	/// * `batch_size` must be non-zero.
	/// * `data.len()` must be a multiple of `batch_size`.
	/// * `data.len() / batch_size` must be a non-zero power of two.
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
	/// * `layer_depth` must be at most `tree_depth`.
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
