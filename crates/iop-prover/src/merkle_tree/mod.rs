// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::PackedField;
use binius_iop::merkle_tree::{Commitment, MerkleTreeScheme};
use binius_math::FieldSlice;
use binius_transcript::{BufMut, TranscriptWriter};
use binius_utils::{FixedSizeSerializeBytes, rayon::prelude::*};

pub mod prover;
#[cfg(test)]
mod tests;

/// The digest type produced by a Merkle tree prover's scheme.
pub type ProverDigest<T, M> = <<M as MerkleTreeProver<T>>::Scheme as MerkleTreeScheme<T>>::Digest;

/// A Merkle tree prover for a particular scheme.
///
/// This is separate from [`MerkleTreeScheme`] so that it may be implemented using a
/// hardware-accelerated backend.
pub trait MerkleTreeProver<T: FixedSizeSerializeBytes> {
	type Scheme: MerkleTreeScheme<T>;
	/// Data generated during commitment required to generate opening proofs.
	type Committed;

	/// Returns the Merkle tree scheme used by the prover.
	fn scheme(&self) -> &Self::Scheme;

	/// Commit a vector of values.
	///
	/// ## Preconditions
	///
	/// * `data.len()` must be a multiple of `batch_size`, and the resulting leaf count (`data.len()
	///   / batch_size`) must be a power of two.
	fn commit(
		&self,
		data: &[T],
		batch_size: usize,
	) -> (Commitment<ProverDigest<T, Self>>, Self::Committed)
	where
		T: Clone + Sync,
	{
		self.commit_iterated(
			data.par_chunks_exact(batch_size)
				.map(|chunk| chunk.iter().cloned()),
			batch_size,
		)
	}

	/// Commits a field buffer, packing `2^log_leaf_len` scalars into each leaf.
	///
	/// Scalars fill leaves in order:
	///
	/// ```text
	/// leaf i  <-  buffer[i * 2^log_leaf_len .. (i+1) * 2^log_leaf_len]
	/// ```
	///
	/// The leaf count is `2^(log_len - log_leaf_len)`, hence a power of two by construction.
	/// That is what [`commit_iterated`](Self::commit_iterated) requires of it.
	///
	/// ## Preconditions
	///
	/// * `log_leaf_len` must be at most the buffer's log length.
	fn commit_field_buffer<P>(
		&self,
		buffer: FieldSlice<P>,
		log_leaf_len: usize,
	) -> (Commitment<ProverDigest<T, Self>>, Self::Committed)
	where
		P: PackedField<Scalar = T>,
	{
		// Invariant: leaves are counted in scalar space, never in backing words.
		// A buffer narrower than one packed word therefore commits no dead lanes.
		self.commit_iterated(buffer.par_chunk_scalars(log_leaf_len), 1 << log_leaf_len)
	}

	/// Commit interleaved elements from iterator by val
	///
	/// Each leaf is built from exactly `n_items_per_input` elements, which lets the leaf hasher
	/// specialize for short, constant-length leaves.
	///
	/// ## Preconditions
	///
	/// * The number of leaves must be a power of two.
	/// * Each iterator in `leaves` yields exactly `n_items_per_input` elements.
	fn commit_iterated<ParIter>(
		&self,
		leaves: ParIter,
		n_items_per_input: usize,
	) -> (Commitment<ProverDigest<T, Self>>, Self::Committed)
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = T, IntoIter: Send>>;

	/// Returns the internal digest layer at the given depth.
	///
	/// ## Preconditions
	///
	/// * `layer_depth` must be at most the committed tree's depth.
	fn layer<'a>(
		&self,
		committed: &'a Self::Committed,
		layer_depth: usize,
	) -> &'a [<Self::Scheme as MerkleTreeScheme<T>>::Digest];

	/// Generate an opening proof for an entry in a committed vector at the given index.
	///
	/// ## Arguments
	///
	/// * `committed` - helper data generated during commitment
	/// * `layer_depth` - depth of the layer to prove inclusion in
	/// * `index` - the entry index
	///
	/// ## Preconditions
	///
	/// * `index` must be within the committed tree and `layer_depth` at most its depth.
	fn prove_opening<B: BufMut>(
		&self,
		committed: &Self::Committed,
		layer_depth: usize,
		index: usize,
		proof: &mut TranscriptWriter<B>,
	);

	/// Generate an opening proof for the full committed vector.
	///
	/// This writes the binding data that [`MerkleTreeScheme::verify_vector`] reads alongside the
	/// data itself while recomputing the root — the per-leaf salts, empty for non-hiding trees.
	fn prove_vector<B: BufMut>(&self, committed: &Self::Committed, proof: &mut TranscriptWriter<B>);
}
