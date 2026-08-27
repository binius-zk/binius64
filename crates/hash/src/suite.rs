// Copyright 2026 The Binius Developers

//! The hash pair a Merkle commitment is built from.

use digest::{Digest, FixedOutputReset, Output, block_api::BlockSizeUser};

use crate::compress::CompressionFunction;

/// The two hashes a Merkle commitment needs: one for leaves, one for inner nodes.
///
/// Verification walks a single path, so both are sequential.
/// Proving folds whole layers at once and needs a batched counterpart for each, which the
/// prover-side crate adds through its own extension of this trait.
pub trait HashSuite {
	/// Sequential hash used to compute leaf digests.
	type LeafHash: Digest + BlockSizeUser + FixedOutputReset + Send;
	/// Sequential 2-to-1 compression used to fold inner Merkle nodes.
	type Compression: CompressionFunction<Output<Self::LeafHash>, 2> + Default;
}
