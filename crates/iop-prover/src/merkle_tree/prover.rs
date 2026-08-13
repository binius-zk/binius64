// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::{Allocator, GlobalAllocator};
use binius_field::Field;
use binius_hash::binary_merkle_tree::{BinaryMerkleTree, HashSuite};
use binius_iop::merkle_tree::{BinaryMerkleTreeScheme, Commitment};
use binius_transcript::{BufMut, TranscriptWriter};
use binius_utils::rayon::iter::IndexedParallelIterator;
use digest::Output;
use getset::Getters;

use super::{MerkleTreeProver, ProverDigest};

/// Builds Merkle trees over an allocator, which every tree it commits draws its nodes from.
///
/// The allocator is state rather than a per-call argument because [`MerkleTreeProver::Committed`]
/// names the tree, and an associated type cannot depend on a method's generic parameter.
#[derive(Getters)]
pub struct BinaryMerkleTreeProver<T, H: HashSuite, A: Allocator = GlobalAllocator> {
	#[getset(get = "pub")]
	scheme: BinaryMerkleTreeScheme<T, H>,
	alloc: A,
}

impl<T, H: HashSuite> BinaryMerkleTreeProver<T, H, GlobalAllocator> {
	/// Commits trees on the global heap.
	pub fn new() -> Self {
		Self::with_allocator(GlobalAllocator)
	}
}

impl<T, H: HashSuite, A: Allocator> BinaryMerkleTreeProver<T, H, A> {
	/// Commits trees whose nodes are drawn from `alloc`.
	///
	/// Pass `&BufferPool` to recycle node buffers across the proofs one prover runs.
	pub fn with_allocator(alloc: A) -> Self {
		Self {
			scheme: BinaryMerkleTreeScheme::new(),
			alloc,
		}
	}
}

impl<T, H: HashSuite> Default for BinaryMerkleTreeProver<T, H, GlobalAllocator> {
	fn default() -> Self {
		Self::new()
	}
}

impl<F, H, A> MerkleTreeProver<F> for BinaryMerkleTreeProver<F, H, A>
where
	F: Field,
	H: HashSuite,
	A: Allocator,
{
	type Scheme = BinaryMerkleTreeScheme<F, H>;
	type Committed = BinaryMerkleTree<Output<H::LeafHash>, A>;

	fn scheme(&self) -> &Self::Scheme {
		&self.scheme
	}

	fn layer<'a>(&self, committed: &'a Self::Committed, depth: usize) -> &'a [Output<H::LeafHash>] {
		committed
			.layer(depth)
			.expect("precondition: layer_depth must be at most the committed tree's depth")
	}

	fn prove_opening<B: BufMut>(
		&self,
		committed: &Self::Committed,
		layer_depth: usize,
		index: usize,
		proof: &mut TranscriptWriter<B>,
	) {
		let branch = committed
			.branch(index, layer_depth)
			.expect("precondition: index and layer_depth must be within the committed tree");
		proof.write_slice(&branch);
	}

	fn commit_iterated<ParIter>(
		&self,
		leaves: ParIter,
		n_items_per_input: usize,
	) -> (Commitment<ProverDigest<F, Self>>, Self::Committed)
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
	{
		let tree = BinaryMerkleTree::from_leaves::<F, H, _>(leaves, n_items_per_input, &self.alloc);

		let commitment = Commitment {
			root: tree.root(),
			depth: tree.log_len,
		};

		(commitment, tree)
	}
}
