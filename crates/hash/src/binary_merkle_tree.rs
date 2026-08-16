// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{fmt, mem::MaybeUninit, slice};

use binius_compute::{Allocator, GlobalAllocator, VecLike};
use binius_field::Field;
use binius_utils::{
	checked_arithmetics::checked_log_2,
	rayon::{self, prelude::*, slice::ParallelSlice},
};
use digest::{
	Digest, FixedOutputReset, Output, OutputSizeUser,
	array::{Array, ArraySize},
	block_api::BlockSizeUser,
};

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
	///
	/// `Sync` because one instance is shared across threads while folding the tree.
	type ParCompression: ParallelPseudoCompression<Output<Self::LeafHash>, 2, Compression = Self::Compression>
		+ Default
		+ Sync;
}

/// Reason a Merkle tree operation rejects its arguments.
///
/// Every variant carries the numbers that failed the check alongside the bound they broke.
/// A log line therefore says which call was malformed without the reader rerunning it.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// A committed batch of values does not split into whole leaves.
	#[error("{n_values} values do not split into leaves of {batch_size}")]
	IncorrectBatchSize {
		/// The number of values handed over for commitment.
		n_values: usize,
		/// The number of values every leaf must hold.
		batch_size: usize,
	},
	/// The leaves a commitment splits into do not span a binary tree.
	#[error("the leaf count {n_leaves} is not a power of two")]
	PowerOfTwoLengthRequired {
		/// The number of leaves the values split into.
		n_leaves: usize,
	},
	/// A layer is asked for below the leaves of the tree.
	#[error("layer depth {layer_depth} exceeds the tree depth {log_len}")]
	IncorrectLayerDepth {
		/// The depth asked for, counted from the root.
		layer_depth: usize,
		/// The depth of the tree, which is its deepest valid layer.
		log_len: usize,
	},
	/// A branch is asked for at a leaf the tree does not have.
	#[error("leaf index {index} is outside the 2^{log_len} leaves")]
	IndexOutOfRange {
		/// The leaf position asked for.
		index: usize,
		/// Base-2 logarithm of the number of leaves the tree has.
		log_len: usize,
	},
}

/// A binary Merkle tree that commits batches of vectors.
///
/// # Overview
///
/// The entries sharing an index across a batch are hashed together into one leaf digest.
/// A binary tree is then folded over those leaf digests.
///
/// All committed vectors must have the same length, and that length must be a power of two.
///
/// The nodes are drawn from an [`Allocator`], so a prover can back the whole tree with pooled
/// memory instead of the global heap. The default is [`GlobalAllocator`], which is a plain [`Vec`].
pub struct BinaryMerkleTree<D: Send, A: Allocator = GlobalAllocator> {
	/// Base-2 logarithm of the number of leaves.
	pub log_len: usize,
	/// The inner nodes, arranged as a flattened array of layers with the root at the end.
	pub inner_nodes: A::Vec<D>,
}

/// Written through the node slice rather than the buffer, so no allocator has to be [`Debug`]
/// itself for a tree over it to be.
impl<D: fmt::Debug + Send, A: Allocator> fmt::Debug for BinaryMerkleTree<D, A> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("BinaryMerkleTree")
			.field("log_len", &self.log_len)
			.field("inner_nodes", &&*self.inner_nodes)
			.finish()
	}
}

impl<N: ArraySize, A: Allocator> BinaryMerkleTree<Array<u8, N>, A> {
	/// Commits a slice of values, cutting it into consecutive leaves.
	///
	/// # Arguments
	///
	/// * `elements` - the values to commit, in leaf order.
	/// * `batch_size` - how many consecutive values are hashed together into one leaf.
	/// * `alloc` - the allocator the tree's nodes are drawn from.
	///
	/// # Errors
	///
	/// * The value count is not a multiple of the batch size.
	/// * The resulting leaf count is not a power of two.
	pub fn new<F, H>(elements: &[F], batch_size: usize, alloc: &A) -> Result<Self, Error>
	where
		F: Field,
		H: HashSuite<LeafHash: OutputSizeUser<OutputSize = N>>,
	{
		// Every leaf holds the same number of values, so the split has to come out even.
		// An empty leaf divides no value count, so it is rejected here rather than dividing by it.
		if batch_size == 0 || !elements.len().is_multiple_of(batch_size) {
			return Err(Error::IncorrectBatchSize {
				n_values: elements.len(),
				batch_size,
			});
		}

		// A binary tree only spans a power-of-two number of leaves.
		let len = elements.len() / batch_size;
		if !len.is_power_of_two() {
			return Err(Error::PowerOfTwoLengthRequired { n_leaves: len });
		}

		// Hand the leaves over one contiguous chunk at a time.
		Ok(Self::from_leaves::<F, H, _>(
			elements
				.par_chunks(batch_size)
				.map(|chunk| chunk.iter().copied()),
			batch_size,
			alloc,
		))
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
	/// * `leaves` - one iterator per leaf, each yielding that leaf's values.
	/// * `n_items_per_input` - how many values every leaf iterator yields.
	/// * `alloc` - the allocator the tree's nodes are drawn from.
	///
	/// # Panics
	///
	/// Panics unless the number of leaves is a power of two.
	pub fn from_leaves<F, H, ParIter>(leaves: ParIter, n_items_per_input: usize, alloc: &A) -> Self
	where
		F: Field,
		H: HashSuite<LeafHash: OutputSizeUser<OutputSize = N>>,
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
	{
		// Panics unless the leaf count is a power of two, which the binary layout requires.
		let log_len = checked_log_2(leaves.len());

		// A binary tree over 2^log_len leaves has 2^(log_len+1) - 1 nodes in total.
		// The whole tree is one allocation, which the layer loop then fills front to back.
		let total_length = (1 << (log_len + 1)) - 1;
		let mut inner_nodes = alloc.alloc::<Array<u8, N>>(total_length);

		// Fill the widest layer first, straight into uninitialized capacity.
		{
			let _span = tracing::debug_span!("hash_leaves").entered();

			// Every leaf holds exactly the same number of values, so its byte length is a constant.
			// Handing that length to the hasher lets it specialize for short leaves.
			H::ParLeafHash::default().digest_with_const_len(
				n_items_per_input,
				leaves,
				&mut inner_nodes.spare_capacity_mut()[..(1 << log_len)],
			);
		}

		let (leaf_layer, remaining) = inner_nodes.spare_capacity_mut().split_at_mut(1 << log_len);

		let leaves: &[Array<u8, N>] = unsafe {
			// SAFETY: the leaf digests were just written over the whole widest layer
			leaf_layer.assume_init_mut()
		};

		// One synchronization point: the leaves are already written, so build every level
		// above them as a single recursive tree of rayon tasks.
		// A node is written the instant both its children are ready, with no other barrier.
		if log_len > 0 {
			let parallel_compression = H::ParCompression::default();
			let ctx = BuildContext {
				leaves,
				log_len,
				nodes: SendPtr(remaining.as_mut_ptr()),
				compression: &parallel_compression,
			};
			ctx.build_node(0, log_len);
		}

		unsafe {
			// SAFETY: inner_nodes should be entirely initialized by now
			// Note that we don't incrementally update inner_nodes.len() since
			// that doesn't play well with using split_at_mut on spare capacity.
			inner_nodes.set_len(total_length);
		}
		Self {
			log_len,
			inner_nodes,
		}
	}
}

/// Leaf-span depth below which a subtree folds without spawning further rayon tasks.
///
/// Below this depth, a subtree's whole span already fills one SIMD-batched compression
/// call per level, so splitting it further would only add task overhead.
///
/// 6 was picked from per-layer costs measured on a 2^17-leaf SHA-256 tree.
const SUBTREE_LOG_DEPTH: usize = 6;

/// A raw pointer that can cross into another rayon task.
///
/// Every write through it targets a disjoint element, checked by hand at each write site.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Sync> Sync for SendPtr<T> {}

/// State shared by every recursive call building the tree above the leaves.
struct BuildContext<'a, D, C> {
	/// The leaf digests, already written.
	leaves: &'a [D],
	/// Base-2 logarithm of the leaf count.
	log_len: usize,
	/// The first node above the leaf layer, addressed by [`node_offset`].
	nodes: SendPtr<MaybeUninit<D>>,
	/// Two-to-one compression, used both one pair at a time and in SIMD-batched groups.
	compression: &'a C,
}

/// Flat-buffer offset of the node spanning `1 << log_width` leaves starting at `leaves_start`.
///
/// The offset is relative to the first node above the leaf layer, matching `nodes` in
/// [`BuildContext`].
const fn node_offset(log_len: usize, leaves_start: usize, log_width: usize) -> usize {
	// Depth counted from the root, matching `BinaryMerkleTree::layer`.
	let layer_depth = log_len - log_width;
	let layer_start = (1 << log_len) - (1 << (layer_depth + 1));
	layer_start + (leaves_start >> log_width)
}

impl<D: Clone, C: ParallelPseudoCompression<D, 2>> BuildContext<'_, D, C> {
	/// Folds `1 << log_width` leaves down to their single root without spawning rayon tasks.
	///
	/// Every level still runs through the hash suite's SIMD-batched compression, since a small
	/// subtree's whole level is one batch, not one task.
	fn fold_small_subtree(&self, leaves_start: usize, log_width: usize) -> D {
		let mut cur = &self.leaves[leaves_start..leaves_start + (1 << log_width)];
		for depth in 1..=log_width {
			let width = 1 << (log_width - depth);
			let offset = node_offset(self.log_len, leaves_start, depth);
			let out = unsafe {
				// SAFETY: this subtree owns a leaf range disjoint from every other subtree's, so
				// its node offset at every depth is disjoint from every other subtree's.
				slice::from_raw_parts_mut(self.nodes.0.add(offset), width)
			};
			self.compression.parallel_compress(cur, out);
			cur = unsafe {
				// SAFETY: the compression above wrote every slot of this layer
				out.assume_init_mut()
			};
		}
		cur[0].clone()
	}
}

impl<D: Clone + Send + Sync, C: ParallelPseudoCompression<D, 2> + Sync> BuildContext<'_, D, C> {
	/// Builds every node from `1 << log_width` leaves starting at `leaves_start` up to their root.
	///
	/// Levels past [`SUBTREE_LOG_DEPTH`] recurse as two independent halves, run in parallel by
	/// rayon, with no synchronization beyond a node waiting on its own two children.
	fn build_node(&self, leaves_start: usize, log_width: usize) -> D {
		if log_width <= SUBTREE_LOG_DEPTH {
			return self.fold_small_subtree(leaves_start, log_width);
		}

		let half = log_width - 1;
		let mid = leaves_start + (1 << half);
		let (left, right) =
			rayon::join(|| self.build_node(leaves_start, half), || self.build_node(mid, half));

		let node = self.compression.compression().compress([left, right]);

		let offset = node_offset(self.log_len, leaves_start, log_width);
		unsafe {
			// SAFETY: this subtree owns a leaf range disjoint from every other subtree's, so its
			// node offset is disjoint from every other subtree's.
			(*self.nodes.0.add(offset)).write(node.clone());
		}
		node
	}
}

impl<D: Clone + Send, A: Allocator> BinaryMerkleTree<D, A> {
	/// Clones the root digest, which sits last in the flattened layers.
	pub fn root(&self) -> D {
		self.inner_nodes
			.last()
			.expect("MerkleTree inner nodes can't be empty")
			.clone()
	}

	/// Borrows one whole layer of the tree, counting depth from the root.
	///
	/// # Errors
	///
	/// * The depth lies below the leaves.
	pub fn layer(&self, layer_depth: usize) -> Result<&[D], Error> {
		if layer_depth > self.log_len {
			return Err(Error::IncorrectLayerDepth {
				layer_depth,
				log_len: self.log_len,
			});
		}
		let range_start = self.inner_nodes.len() + 1 - (1 << (layer_depth + 1));

		Ok(&self.inner_nodes[range_start..range_start + (1 << layer_depth)])
	}

	/// Collects the sibling digests opening a leaf up to the layer at `layer_depth`.
	///
	/// # Errors
	///
	/// * The leaf index lies outside the tree.
	/// * The depth lies below the leaves.
	pub fn branch(&self, index: usize, layer_depth: usize) -> Result<Vec<D>, Error> {
		if index >= 1 << self.log_len {
			return Err(Error::IndexOutOfRange {
				index,
				log_len: self.log_len,
			});
		}
		if layer_depth > self.log_len {
			return Err(Error::IncorrectLayerDepth {
				layer_depth,
				log_len: self.log_len,
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

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash as B128;

	use super::*;
	use crate::sha256::{Sha256Compression, Sha256HashSuite};

	/// Commits `n_values` distinct field elements in leaves of `batch_size` values each.
	fn commit(
		n_values: usize,
		batch_size: usize,
	) -> Result<BinaryMerkleTree<Output<sha2::Sha256>>, Error> {
		let elements = (0..n_values)
			.map(|i| B128::new(i as u128))
			.collect::<Vec<_>>();
		BinaryMerkleTree::new::<B128, Sha256HashSuite>(&elements, batch_size, &GlobalAllocator)
	}

	#[test]
	fn test_new_rejects_a_ragged_final_leaf() {
		// 7 values leave a partial leaf behind once 3 whole leaves of 2 are cut.
		let err = commit(7, 2).unwrap_err();

		let Error::IncorrectBatchSize {
			n_values,
			batch_size,
		} = err
		else {
			panic!("expected IncorrectBatchSize, got {err}");
		};
		assert_eq!(n_values, 7);
		assert_eq!(batch_size, 2);
	}

	#[test]
	fn test_new_rejects_an_empty_leaf() {
		// A leaf holding no value divides no value count, so the check runs before the division.
		let err = commit(0, 0).unwrap_err();

		let Error::IncorrectBatchSize {
			n_values,
			batch_size,
		} = err
		else {
			panic!("expected IncorrectBatchSize, got {err}");
		};
		assert_eq!(n_values, 0);
		assert_eq!(batch_size, 0);
	}

	#[test]
	fn test_new_rejects_a_leaf_count_off_the_power_of_two_grid() {
		// 6 values in leaves of 2 split evenly, but 3 leaves do not span a binary tree.
		let err = commit(6, 2).unwrap_err();

		let Error::PowerOfTwoLengthRequired { n_leaves } = err else {
			panic!("expected PowerOfTwoLengthRequired, got {err}");
		};
		assert_eq!(n_leaves, 3);
	}

	#[test]
	fn test_layer_spans_the_root_down_to_the_leaves() {
		// 8 values in leaves of 2 give 4 leaves, so the tree is 2 layers deep.
		let tree = commit(8, 2).unwrap();
		assert_eq!(tree.log_len, 2);

		// Layer d holds 2^d digests, from the single root down to the 4 leaves.
		assert_eq!(tree.layer(0).unwrap(), &[tree.root()]);
		assert_eq!(tree.layer(1).unwrap().len(), 2);
		assert_eq!(tree.layer(2).unwrap().len(), 4);

		// One layer past the leaves is the first depth the tree does not hold.
		let err = tree.layer(3).unwrap_err();

		let Error::IncorrectLayerDepth {
			layer_depth,
			log_len,
		} = err
		else {
			panic!("expected IncorrectLayerDepth, got {err}");
		};
		assert_eq!(layer_depth, 3);
		assert_eq!(log_len, 2);
	}

	#[test]
	fn test_branch_opens_every_leaf_up_to_the_named_layer() {
		let tree = commit(8, 2).unwrap();

		// Opening a leaf to the root names one sibling per layer crossed.
		for index in 0..4 {
			assert_eq!(tree.branch(index, 0).unwrap().len(), 2);
			assert_eq!(tree.branch(index, 1).unwrap().len(), 1);
			assert_eq!(tree.branch(index, 2).unwrap().len(), 0);
		}

		// Sibling leaves cross the same inner node, so their top sibling agrees.
		assert_eq!(tree.branch(0, 0).unwrap()[1], tree.branch(1, 0).unwrap()[1]);
	}

	#[test]
	fn test_branch_rejects_a_leaf_past_the_last_one() {
		// 4 leaves are indexed 0..=3, so 4 is the first index outside the tree.
		let tree = commit(8, 2).unwrap();
		let err = tree.branch(4, 0).unwrap_err();

		let Error::IndexOutOfRange { index, log_len } = err else {
			panic!("expected IndexOutOfRange, got {err}");
		};
		assert_eq!(index, 4);
		assert_eq!(log_len, 2);
	}

	#[test]
	fn test_branch_blames_the_depth_when_the_leaf_is_valid() {
		// The index is in range, so the depth is the argument the diagnostic must name.
		let tree = commit(8, 2).unwrap();
		let err = tree.branch(0, 3).unwrap_err();

		let Error::IncorrectLayerDepth {
			layer_depth,
			log_len,
		} = err
		else {
			panic!("expected IncorrectLayerDepth, got {err}");
		};
		assert_eq!(layer_depth, 3);
		assert_eq!(log_len, 2);
	}

	#[test]
	fn test_new_matches_a_naive_layer_by_layer_fold_across_the_recursion_cutoff() {
		// Sizes chosen to land on both sides of the small-subtree cutoff:
		// a single leaf, two leaves, exactly at the cutoff, and twice past it.
		for log_len in [0, 1, SUBTREE_LOG_DEPTH, 2 * SUBTREE_LOG_DEPTH] {
			let tree = commit(1 << log_len, 1).unwrap();

			// Reference: fold the same leaves one pair at a time, one full layer at a time.
			let compression = Sha256Compression::default();
			let mut layer = tree.layer(log_len).unwrap().to_vec();
			let mut expected_layers = vec![layer.clone()];
			while layer.len() > 1 {
				layer = layer
					.chunks_exact(2)
					.map(|pair| compression.compress([pair[0], pair[1]]))
					.collect();
				expected_layers.push(layer.clone());
			}

			for depth in 0..=log_len {
				assert_eq!(
					tree.layer(depth).unwrap(),
					expected_layers[log_len - depth].as_slice(),
					"log_len {log_len}, depth {depth}"
				);
			}
		}
	}
}
