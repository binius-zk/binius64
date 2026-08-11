// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::{self, Debug, Formatter},
	marker::PhantomData,
};

use binius_hash::{CompressionFunction, binary_merkle_tree::HashSuite, hash_serialize};
use binius_transcript::{Buf, TranscriptReader};
use binius_utils::{
	FixedSizeSerializeBytes,
	checked_arithmetics::{checked_log_2, log2_ceil_usize},
};
use digest::{Digest, Output};

use super::{
	error::{Error, VerificationError},
	merkle_tree_vcs::MerkleTreeScheme,
};

/// A binary Merkle tree vector commitment, as seen by the verifier.
///
/// A committed vector is cut into equal-size batches of values.
/// Each batch is hashed into one leaf digest.
/// Pairs of digests are then folded upward until a single root digest remains.
pub struct BinaryMerkleTreeScheme<T, H: HashSuite> {
	/// Two-to-one function folding a pair of child digests into their parent digest.
	compression: H::Compression,
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
		Self {
			// The compression function is stateless, so one default instance serves every call.
			compression: H::Compression::default(),
			_phantom: PhantomData,
		}
	}

	/// Folds a layer of digests down to the single root above it.
	///
	/// Each round pairs neighbours and replaces them with their parent, halving the layer:
	///
	/// ```text
	///     [d_0, d_1, ..., d_{n-1}]  ->  [C(d_0, d_1), ..., C(d_{n-2}, d_{n-1})]
	/// ```
	///
	/// After `log_2(n)` rounds exactly one digest is left, and that digest is the root.
	///
	/// # Panics
	///
	/// Panics unless the number of digests is a non-zero power of two.
	///
	/// # Performance
	///
	/// One allocation of `n / 2` digests in total, reused by every round after the first.
	fn fold_to_root(&self, digests: &[Output<H::LeafHash>]) -> Output<H::LeafHash> {
		// A layer that is not a power of two cannot be paired off cleanly.
		// An empty layer spans no subtree at all.
		assert!(
			digests.len().is_power_of_two(),
			"precondition: the number of digests must be a non-zero power of two"
		);

		// A lone digest already is the root of its subtree; folding it would invent a level.
		if let [root] = digests {
			return root.clone();
		}

		// The first round reads the caller's slice and writes into fresh space.
		// That caps the scratch buffer at half the input length.
		let mut layer = digests
			.chunks_exact(2)
			.map(|pair| {
				self.compression
					.compress([pair[0].clone(), pair[1].clone()])
			})
			.collect::<Vec<_>>();

		// Later rounds halve the buffer in place.
		// A parent lands strictly below both children it replaces, so nothing is overwritten early.
		while layer.len() > 1 {
			let half = layer.len() / 2;
			for i in 0..half {
				layer[i] = self
					.compression
					.compress([layer[2 * i].clone(), layer[2 * i + 1].clone()]);
			}
			// Drop the tail the round just consumed, keeping the allocation.
			layer.truncate(half);
		}

		layer
			.pop()
			.expect("a non-empty layer folds down to exactly one digest")
	}
}

impl<T, H: HashSuite> Clone for BinaryMerkleTreeScheme<T, H> {
	fn clone(&self) -> Self {
		// Written out rather than derived: a derived copy would demand a cloneable value type.
		// No value of that type is ever held.
		// The compression function is always cloneable through its own trait bound.
		Self {
			compression: self.compression.clone(),
			_phantom: PhantomData,
		}
	}
}

impl<T, H: HashSuite> Debug for BinaryMerkleTreeScheme<T, H> {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		// Written out for the same reason as the copy above.
		// The compression function carries no formatting bound, and the scheme holds no state
		// beyond it.
		f.debug_struct("BinaryMerkleTreeScheme").finish()
	}
}

impl<T, H> BinaryMerkleTreeScheme<T, H>
where
	T: FixedSizeSerializeBytes,
	H: HashSuite,
{
	/// Hashes one leaf from the values it holds.
	fn compute_leaf_digest(&self, values: &[T]) -> Result<Output<H::LeafHash>, Error> {
		Ok(hash_serialize::<T, H::LeafHash>(values)?)
	}
}

impl<T, H> MerkleTreeScheme<T> for BinaryMerkleTreeScheme<T, H>
where
	T: FixedSizeSerializeBytes,
	H: HashSuite,
{
	type Digest = Output<H::LeafHash>;

	fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize {
		// Raising the layer by one level doubles its width but shortens every branch by one.
		// The two effects balance where the layer width first reaches the query count.
		//
		// A layer can never sit below the leaves, hence the clamp.
		log2_ceil_usize(n_queries).min(tree_depth)
	}

	fn proof_size(&self, len: usize, n_queries: usize, layer_depth: usize) -> usize {
		assert!(len.is_power_of_two(), "precondition: len must be a power of two");

		// Depth of the tree spanning the committed vector.
		let log_len = checked_log_2(len);

		assert!(layer_depth <= log_len, "precondition: layer_depth must be at most log2(len)");

		// Each query walks from its leaf up to the decommitted layer, one sibling per level.
		// The layer itself is sent once, for all queries together.
		//
		//     branches: (log_len - layer_depth) * n_queries
		//     layer   : 2^layer_depth
		((log_len - layer_depth) * n_queries + (1 << layer_depth))
			* <H::LeafHash as Digest>::output_size()
	}

	fn verify_vector(
		&self,
		root: &Self::Digest,
		data: &[T],
		batch_size: usize,
	) -> Result<(), Error> {
		// A zero-size batch would slice the data into unboundedly many empty leaves.
		assert_ne!(batch_size, 0, "precondition: batch_size must be non-zero");
		// Every leaf holds the same number of values, so the split has to come out even.
		assert!(
			data.len().is_multiple_of(batch_size),
			"precondition: data length must be a multiple of batch_size"
		);
		// A binary tree only spans a power-of-two number of leaves.
		assert!(
			(data.len() / batch_size).is_power_of_two(),
			"precondition: data.len() / batch_size must be a non-zero power of two"
		);

		// Rebuild every leaf digest from the revealed values.
		let digests = data
			.chunks(batch_size)
			.map(|chunk| self.compute_leaf_digest(chunk))
			.collect::<Result<Vec<_>, _>>()?;

		// Rebuilding the whole tree and landing on the committed root is what binds the data.
		if self.fold_to_root(&digests) != *root {
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
		// A layer that many levels below the root holds exactly that many digests.
		assert_eq!(
			layer_digests.len(),
			1 << layer_depth,
			"precondition: layer_digests must have 2^layer_depth entries"
		);

		// Folding the claimed layer must reproduce the committed root.
		// The fold takes one round per level, so a layer only passes at the depth it claims.
		if self.fold_to_root(layer_digests) != *root {
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
		// A layer that many levels below the root holds exactly that many digests.
		assert_eq!(
			layer_digests.len(),
			1 << layer_depth,
			"precondition: layer_digests must have 2^layer_depth entries"
		);
		// The climb runs from the leaves up to the layer, so the layer cannot sit below them.
		assert!(layer_depth <= tree_depth, "precondition: layer_depth must be at most tree_depth");
		// A tree of that depth has exactly that many leaves to address.
		assert!(index < (1 << tree_depth), "precondition: index must be less than 2^tree_depth");

		// Bottom of the authentication path: the leaf the opening claims.
		let mut digest = self.compute_leaf_digest(values)?;

		// Climb one level per round, folding in the sibling the advice supplies.
		//
		//     level k:  running digest + sibling_k  ->  running digest at level k+1
		//
		// The low bit of the running index says which side the running digest sits on.
		for _ in layer_depth..tree_depth {
			let sibling = proof.read::<Self::Digest>()?;
			// An even index means the running digest is the left child of its parent.
			digest = self.compression.compress(if index & 1 == 0 {
				[digest, sibling]
			} else {
				[sibling, digest]
			});
			// Discard the bit just consumed, exposing the next level's side bit.
			index >>= 1;
		}

		// The climb dropped one bit per level, so what is left addresses the decommitted layer.
		// Matching the entry there binds the leaf to the already-verified layer, hence the root.
		if digest != layer_digests[index] {
			return Err(VerificationError::InvalidProof.into());
		}
		Ok(())
	}
}
