// Copyright 2025 Irreducible Inc.

use binius_iop::merkle_tree::{Commitment, MerkleTreeScheme};
use binius_transcript::{BufMut, TranscriptWriter};
use binius_utils::rayon::prelude::*;

pub mod prover;
#[cfg(test)]
mod tests;

/// A Merkle tree prover for a particular scheme.
///
/// This is separate from [`MerkleTreeScheme`] so that it may be implemented using a
/// hardware-accelerated backend.
pub trait MerkleTreeProver<T> {
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
	#[allow(clippy::type_complexity)]
	fn commit(
		&self,
		data: &[T],
		batch_size: usize,
	) -> (Commitment<<Self::Scheme as MerkleTreeScheme<T>>::Digest>, Self::Committed)
	where
		T: Clone + Sync,
	{
		self.commit_iterated(
			data.par_chunks_exact(batch_size)
				.map(|chunk| chunk.iter().cloned()),
		)
	}

	/// Commit interleaved elements from iterator by val
	///
	/// ## Preconditions
	///
	/// * The number of leaves must be a power of two.
	#[allow(clippy::type_complexity)]
	fn commit_iterated<ParIter>(
		&self,
		leaves: ParIter,
	) -> (Commitment<<Self::Scheme as MerkleTreeScheme<T>>::Digest>, Self::Committed)
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
}
