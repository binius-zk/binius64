// Copyright 2026 The Binius Developers

//! The batched counterpart of the verifier's hash pair.

use binius_hash::HashSuite;
use digest::Output;

use crate::{parallel_compression::ParallelPseudoCompression, parallel_digest::ParallelDigest};

/// A hash suite with the batched hashes a prover needs on top of the sequential ones.
///
/// Verification walks one Merkle path, so the sequential pair is enough for it.
/// Proving folds whole layers at once, where batching independent messages per call is what
/// keeps the hash off the critical path.
///
/// Each batched hash is required to agree with its sequential counterpart byte for byte, so
/// the two sides of a proof commit to the same tree.
pub trait ParallelHashSuite: HashSuite {
	/// Batched counterpart of the sequential leaf hash.
	type ParLeafHash: ParallelDigest<Digest = Self::LeafHash> + Default;
	/// Batched counterpart of the sequential inner-node compression.
	///
	/// `Sync` because one instance is shared across threads while folding the tree.
	type ParCompression: ParallelPseudoCompression<Output<Self::LeafHash>, 2, Compression = Self::Compression>
		+ Default
		+ Sync;
}
