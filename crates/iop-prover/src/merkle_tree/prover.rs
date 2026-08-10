// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::Field;
use binius_hash::binary_merkle_tree::{self, BinaryMerkleTree, HashSuite};
use binius_iop::merkle_tree::{BinaryMerkleTreeScheme, Commitment, MerkleTreeScheme};
use binius_transcript::{BufMut, TranscriptWriter};
use binius_utils::rayon::iter::IndexedParallelIterator;
use digest::Output;
use getset::Getters;

use super::MerkleTreeProver;

#[derive(Getters)]
pub struct BinaryMerkleTreeProver<T, H: HashSuite> {
	#[getset(get = "pub")]
	scheme: BinaryMerkleTreeScheme<T, H>,
}

impl<T, H: HashSuite> BinaryMerkleTreeProver<T, H> {
	pub fn new() -> Self {
		Self {
			scheme: BinaryMerkleTreeScheme::new(),
		}
	}
}

impl<T, H: HashSuite> Default for BinaryMerkleTreeProver<T, H> {
	fn default() -> Self {
		Self::new()
	}
}

impl<F, H> MerkleTreeProver<F> for BinaryMerkleTreeProver<F, H>
where
	F: Field,
	H: HashSuite,
{
	type Scheme = BinaryMerkleTreeScheme<F, H>;
	type Committed = BinaryMerkleTree<Output<H::LeafHash>>;

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

	#[allow(clippy::type_complexity)]
	fn commit_iterated<ParIter>(
		&self,
		leaves: ParIter,
		n_items_per_input: usize,
	) -> (Commitment<<Self::Scheme as MerkleTreeScheme<F>>::Digest>, Self::Committed)
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
	{
		let tree = binary_merkle_tree::build_from_iterator::<F, H, _>(leaves, n_items_per_input)
			.expect("precondition: the number of leaves must be a power of two");

		let commitment = Commitment {
			root: tree.root(),
			depth: tree.log_len,
		};

		(commitment, tree)
	}
}
