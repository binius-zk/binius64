// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, fmt::Debug, marker::PhantomData};

use binius_hash::{CompressionFunction, binary_merkle_tree::HashSuite, hash_serialize};
use binius_transcript::{Buf, TranscriptReader};
use binius_utils::{
	FixedSizeSerializeBytes,
	checked_arithmetics::{log2_ceil_usize, log2_strict_usize},
};
use digest::{Digest, Output};
use getset::CopyGetters;

use super::{
	error::{Error, VerificationError},
	merkle_tree_vcs::MerkleTreeScheme,
};

#[derive(Clone, CopyGetters)]
pub struct BinaryMerkleTreeScheme<T, H: HashSuite> {
	#[getset(get = "pub")]
	compression: H::Compression,
	#[getset(get_copy = "pub")]
	salt_len: usize,
	// This makes it so that `BinaryMerkleTreeScheme` remains Send + Sync regardless of `T`.
	// See https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
	_phantom: PhantomData<fn() -> T>,
}

impl<T, H: HashSuite> Default for BinaryMerkleTreeScheme<T, H> {
	fn default() -> Self {
		Self::new()
	}
}

impl<T, H: HashSuite> BinaryMerkleTreeScheme<T, H> {
	pub fn new() -> Self {
		Self::hiding(0)
	}

	pub fn hiding(salt_len: usize) -> Self {
		Self {
			compression: H::Compression::default(),
			salt_len,
			_phantom: PhantomData,
		}
	}
}

impl<T, H> BinaryMerkleTreeScheme<T, H>
where
	T: FixedSizeSerializeBytes,
	H: HashSuite,
{
	fn compute_leaf_digest<B: Buf>(
		&self,
		values: &[T],
		proof: &mut TranscriptReader<B>,
	) -> Result<Output<H::LeafHash>, Error> {
		let salt = proof.read_vec::<T>(self.salt_len)?;
		hash_serialize::<T, H::LeafHash>(values.iter().chain(&salt)).map_err(Error::Serialization)
	}
}

impl<T, H> MerkleTreeScheme<T> for BinaryMerkleTreeScheme<T, H>
where
	T: FixedSizeSerializeBytes,
	H: HashSuite,
{
	type Digest = Output<H::LeafHash>;

	/// This layer allows minimizing the proof size.
	fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize {
		log2_ceil_usize(n_queries).min(tree_depth)
	}

	/// ## Preconditions
	/// * `len` must be a power of two.
	/// * `layer_depth` must be at most `log2(len)`.
	fn proof_size(&self, len: usize, n_queries: usize, layer_depth: usize) -> usize {
		assert!(len.is_power_of_two(), "precondition: len must be a power of two");

		let log_len = log2_strict_usize(len);

		assert!(layer_depth <= log_len, "precondition: layer_depth must be at most log2(len)");

		((log_len - layer_depth) * n_queries + (1 << layer_depth))
			* <H::LeafHash as Digest>::output_size()
	}

	fn verify_vector<B: Buf>(
		&self,
		root: &Self::Digest,
		data: &[T],
		batch_size: usize,
		proof: &mut TranscriptReader<B>,
	) -> Result<(), Error> {
		assert!(
			data.len().is_multiple_of(batch_size),
			"precondition: data length must be a multiple of batch_size"
		);

		let digests = data
			.chunks(batch_size)
			.map(|chunk| self.compute_leaf_digest(chunk, proof))
			.collect::<Result<Vec<_>, _>>()?;

		if fold_digests_vector_inplace(&self.compression, digests) != *root {
			return Err(VerificationError::InvalidProof.into());
		}
		Ok(())
	}

	fn verify_layer(
		&self,
		root: &Self::Digest,
		layer_depth: usize,
		layer_digests: &[Self::Digest],
	) -> Result<(), Error> {
		assert_eq!(
			layer_digests.len(),
			1 << layer_depth,
			"precondition: layer_digests must have 2^layer_depth entries"
		);

		let computed_root = fold_digests_vector_inplace(&self.compression, layer_digests.to_vec());
		if computed_root != *root {
			return Err(VerificationError::InvalidProof.into());
		}
		Ok(())
	}

	fn verify_opening<B: Buf>(
		&self,
		mut index: usize,
		values: &[T],
		layer_depth: usize,
		tree_depth: usize,
		layer_digests: &[Self::Digest],
		proof: &mut TranscriptReader<B>,
	) -> Result<(), Error> {
		assert_eq!(
			layer_digests.len(),
			1 << layer_depth,
			"precondition: layer_digests must have 2^layer_depth entries"
		);
		assert!(index < (1 << tree_depth), "precondition: index must be less than 2^tree_depth");

		let mut leaf_digest = self.compute_leaf_digest(values, proof)?;
		for branch_node in proof.read_vec(tree_depth - layer_depth)? {
			leaf_digest = self.compression.compress(if index & 1 == 0 {
				[leaf_digest, branch_node]
			} else {
				[branch_node, leaf_digest]
			});
			index >>= 1;
		}

		(leaf_digest == layer_digests[index])
			.then_some(())
			.ok_or_else(|| VerificationError::InvalidProof.into())
	}
}

/// Compute the Merkle root over a vector of leaf digests.
///
/// Consumes digests because it modifies the vector in place.
///
/// # Preconditions
/// - `digests.len()` is a power of two
fn fold_digests_vector_inplace<C, D>(compression: &C, mut digests: Vec<D>) -> D
where
	C: CompressionFunction<D, 2>,
	D: Clone + Default + Send + Sync + Debug,
{
	let log_len = log2_strict_usize(digests.len()); // pre-condition
	for layer in (0..log_len).rev() {
		for i in 0..1 << layer {
			digests[i] = compression.compress(array::from_fn(|j| digests[2 * i + j].clone()));
		}
	}
	digests[0].clone()
}
