// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{fmt::Debug, mem::MaybeUninit};

use binius_field::Field;
use binius_utils::{
	checked_arithmetics::checked_log_2,
	rayon::{prelude::*, slice::ParallelSlice},
};
use digest::{Digest, FixedOutputReset, Output, block_api::BlockSizeUser};

use super::{
	compress::CompressionFunction, parallel_compression::ParallelPseudoCompression,
	parallel_digest::ParallelDigest,
};

/// A bundle of hash and compression types used to build and verify a binary Merkle tree.
///
/// Most callers want to vary the underlying hash family (SHA-256, etc.) as a single unit
/// rather than independently picking a leaf hash, a compression function, and their parallel
/// counterparts. `HashSuite` bundles the four related types so that user-facing prover and
/// verifier APIs can take a single `H: HashSuite` parameter instead of two or three loose hash
/// trait parameters.
pub trait HashSuite {
	/// Sequential hash used to compute leaf digests during verification.
	type LeafHash: Digest + BlockSizeUser + FixedOutputReset + Send;
	/// Sequential 2-to-1 compression used to fold inner Merkle nodes during verification.
	type Compression: CompressionFunction<Output<Self::LeafHash>, 2> + Default;
	/// Parallel counterpart of [`Self::LeafHash`] used during proving.
	type ParLeafHash: ParallelDigest<Digest = Self::LeafHash> + Default;
	/// Parallel counterpart of [`Self::Compression`] used during proving.
	type ParCompression: ParallelPseudoCompression<Output<Self::LeafHash>, 2, Compression = Self::Compression>
		+ Default;
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Index exceeds Merkle tree base size: {max}")]
	IndexOutOfRange { max: usize },
	#[error("values length must be a multiple of the batch size")]
	IncorrectBatchSize,
	#[error("The argument length must be a power of two.")]
	PowerOfTwoLengthRequired,
	#[error("The layer does not exist in the Merkle tree")]
	IncorrectLayerDepth,
}

/// A binary Merkle tree that commits batches of vectors.
///
/// # Overview
///
/// The entries sharing an index across a batch are hashed together into one leaf digest.
/// A binary tree is then folded over those leaf digests.
///
/// All committed vectors must have the same length, and that length must be a power of two.
#[derive(Debug, Clone)]
pub struct BinaryMerkleTree<D> {
	/// Base-2 logarithm of the number of leaves.
	pub log_len: usize,
	/// The inner nodes, arranged as a flattened array of layers with the root at the end.
	pub inner_nodes: Vec<D>,
}

/// Commits a slice of values, cutting it into consecutive leaves.
///
/// # Arguments
///
/// * `elements` - the values to commit, in leaf order.
/// * `batch_size` - how many consecutive values are hashed together into one leaf.
///
/// # Returns
///
/// The committed tree, or a failure describing which shape requirement the input broke.
///
/// # Errors
///
/// * The value count is not a multiple of the batch size.
/// * The resulting leaf count is not a power of two.
pub fn build<F, H>(
	elements: &[F],
	batch_size: usize,
) -> Result<BinaryMerkleTree<Output<H::LeafHash>>, Error>
where
	F: Field,
	H: HashSuite,
{
	// Every leaf holds the same number of values, so the split has to come out even.
	if !elements.len().is_multiple_of(batch_size) {
		return Err(Error::IncorrectBatchSize);
	}

	// A binary tree only spans a power-of-two number of leaves.
	let len = elements.len() / batch_size;
	if !len.is_power_of_two() {
		return Err(Error::PowerOfTwoLengthRequired);
	}

	// Hand the leaves over one contiguous chunk at a time.
	build_from_iterator::<_, H, _>(
		elements
			.par_chunks(batch_size)
			.map(|chunk| chunk.iter().copied()),
		batch_size,
	)
}

/// Commits leaves drawn from a parallel iterator, one iterator item per leaf.
///
/// # Overview
///
/// The tree is laid out as one flat buffer of layers, widest first:
///
/// ```text
///     [ leaf digests | layer 1 | ... | root ]
///        2^log_len      2^(log_len-1)    1
/// ```
///
/// Each layer is written into the buffer's spare capacity.
/// It is then read back as the input to the layer above it.
///
/// # Arguments
///
/// * `iterated_chunks` - one iterator per leaf, each yielding that leaf's values.
/// * `n_items_per_input` - how many values every leaf iterator yields.
///
/// # Panics
///
/// Panics unless the number of leaves is a power of two.
pub fn build_from_iterator<F, H, ParIter>(
	iterated_chunks: ParIter,
	n_items_per_input: usize,
) -> Result<BinaryMerkleTree<Output<H::LeafHash>>, Error>
where
	F: Field,
	H: HashSuite,
	ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
{
	let log_len = checked_log_2(iterated_chunks.len()); // precondition

	// A binary tree over 2^log_len leaves has 2^(log_len+1) - 1 nodes in total.
	let total_length = (1 << (log_len + 1)) - 1;
	let mut inner_nodes = Vec::with_capacity(total_length);

	// Fill the widest layer first, straight into uninitialized capacity.
	hash_leaves::<F, H, _>(
		iterated_chunks,
		n_items_per_input,
		&mut inner_nodes.spare_capacity_mut()[..(1 << log_len)],
	);

	let (prev_layer, mut remaining) = inner_nodes.spare_capacity_mut().split_at_mut(1 << log_len);

	let mut prev_layer = unsafe {
		// SAFETY: prev-layer was initialized by hash_leaves
		prev_layer.assume_init_mut()
	};
	// Fold one layer per round, each half the width of the one below it.
	let parallel_compression = H::ParCompression::default();
	for i in 1..(log_len + 1) {
		let (next_layer, next_remaining) = remaining.split_at_mut(1 << (log_len - i));
		remaining = next_remaining;

		parallel_compression.parallel_compress(prev_layer, next_layer);

		prev_layer = unsafe {
			// SAFETY: next_layer was just initialized by compress_layer
			next_layer.assume_init_mut()
		};
	}

	unsafe {
		// SAFETY: inner_nodes should be entirely initialized by now
		// Note that we don't incrementally update inner_nodes.len() since
		// that doesn't play well with using split_at_mut on spare capacity.
		inner_nodes.set_len(total_length);
	}
	Ok(BinaryMerkleTree {
		log_len,
		inner_nodes,
	})
}

impl<D: Clone> BinaryMerkleTree<D> {
	pub fn root(&self) -> D {
		self.inner_nodes
			.last()
			.expect("MerkleTree inner nodes can't be empty")
			.clone()
	}

	pub fn layer(&self, layer_depth: usize) -> Result<&[D], Error> {
		if layer_depth > self.log_len {
			return Err(Error::IncorrectLayerDepth);
		}
		let range_start = self.inner_nodes.len() + 1 - (1 << (layer_depth + 1));

		Ok(&self.inner_nodes[range_start..range_start + (1 << layer_depth)])
	}

	/// Get a Merkle branch for the given index
	///
	/// Throws if the index is out of range
	pub fn branch(&self, index: usize, layer_depth: usize) -> Result<Vec<D>, Error> {
		if index >= 1 << self.log_len || layer_depth > self.log_len {
			return Err(Error::IndexOutOfRange {
				max: (1 << self.log_len) - 1,
			});
		}

		let branch = (0..self.log_len - layer_depth)
			.map(|j| {
				let node_index = (((1 << j) - 1) << (self.log_len + 1 - j)) | (index >> j) ^ 1;
				self.inner_nodes[node_index].clone()
			})
			.collect();

		Ok(branch)
	}
}

/// Hashes the elements in chunks of a vector into digests.
///
/// Given a vector of elements and an output buffer of N hash digests, this splits the elements
/// into N equal-sized chunks and hashes each chunks into the corresponding output digest.
///
/// Every leaf holds exactly `n_items_per_input` values, so its byte length is a constant.
/// Passing that length to the hasher lets it specialize for short leaves.
///
/// # Preconditions
/// - Each iterator in `iterated_chunks` yields exactly `n_items_per_input` elements.
#[tracing::instrument("hash_leaves", skip_all, level = "debug")]
fn hash_leaves<F, H, ParIter>(
	iterated_chunks: ParIter,
	n_items_per_input: usize,
	digests: &mut [MaybeUninit<Output<H::LeafHash>>],
) where
	F: Field,
	H: HashSuite,
	ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
{
	// The constant leaf length lets the hasher skip per-leaf length bookkeeping and padding.
	let hasher = H::ParLeafHash::default();
	hasher.digest_with_const_len(n_items_per_input, iterated_chunks, digests);
}
