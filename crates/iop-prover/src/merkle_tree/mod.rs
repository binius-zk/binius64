// Copyright 2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_iop::merkle_tree::{Commitment, MerkleTreeScheme};
use binius_math::FieldSlice;
use binius_transcript::{BufMut, TranscriptWriter};
use binius_utils::{FixedSizeSerializeBytes, rayon::prelude::*};

pub mod prover;
#[cfg(test)]
mod tests;

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
			batch_size,
		)
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
	#[allow(clippy::type_complexity)]
	fn commit_iterated<ParIter>(
		&self,
		leaves: ParIter,
		n_items_per_input: usize,
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

	/// Generate an opening proof for the full committed vector.
	///
	/// This writes the binding data that [`MerkleTreeScheme::verify_vector`] reads alongside the
	/// data itself while recomputing the root — the per-leaf salts, empty for non-hiding trees.
	fn prove_vector<B: BufMut>(&self, committed: &Self::Committed, proof: &mut TranscriptWriter<B>);
}

/// Commits a field buffer to a Merkle tree, packing `1 << log_batch_size` scalars into each leaf.
///
/// The buffer's scalars are grouped into leaves of `1 << log_batch_size` elements each, in order.
/// This dispatches on whether a leaf spans at least one full packed element (`log_batch_size >=
/// P::LOG_WIDTH`) or fits within a single packed element, choosing the parallel scalar iterator
/// that avoids splitting work below the packing granularity.
///
/// ## Preconditions
///
/// * The resulting leaf count (`buffer.len() / (1 << log_batch_size)`) must be a power of two.
#[allow(clippy::type_complexity)]
pub fn commit_field_buffer<F, P, MerkleProver>(
	merkle_prover: &MerkleProver,
	buffer: FieldSlice<P>,
	log_batch_size: usize,
) -> (
	Commitment<<MerkleProver::Scheme as MerkleTreeScheme<F>>::Digest>,
	MerkleProver::Committed,
)
where
	F: Field,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F>,
{
	if log_batch_size >= P::LOG_WIDTH {
		let iterated_big_chunks = to_par_scalar_big_chunks(buffer.as_ref(), 1 << log_batch_size);
		merkle_prover.commit_iterated(iterated_big_chunks, 1 << log_batch_size)
	} else {
		let iterated_small_chunks =
			to_par_scalar_small_chunks(buffer.as_ref(), 1 << log_batch_size);
		merkle_prover.commit_iterated(iterated_small_chunks, 1 << log_batch_size)
	}
}

/// Creates a parallel iterator over scalars of subfield elements. Assumes chunk_size to be a power
/// of two.
fn to_par_scalar_big_chunks<P>(
	packed_slice: &[P],
	chunk_size: usize,
) -> impl IndexedParallelIterator<Item: Iterator<Item = P::Scalar> + Send + '_>
where
	P: PackedField,
{
	packed_slice
		.par_chunks(chunk_size / P::WIDTH)
		.map(|chunk| PackedField::iter_slice(chunk))
}

fn to_par_scalar_small_chunks<P>(
	packed_slice: &[P],
	chunk_size: usize,
) -> impl IndexedParallelIterator<Item: Iterator<Item = P::Scalar> + Send + '_>
where
	P: PackedField,
{
	(0..packed_slice.len() * P::WIDTH)
		.into_par_iter()
		.step_by(chunk_size)
		.map(move |start_index| {
			let packed_item = &packed_slice[start_index / P::WIDTH];
			packed_item
				.iter()
				.skip(start_index % P::WIDTH)
				.take(chunk_size)
		})
}
