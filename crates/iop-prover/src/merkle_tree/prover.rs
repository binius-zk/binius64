// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::sync::Mutex;

use binius_field::Field;
use binius_hash::binary_merkle_tree::{self, BinaryMerkleTree, HashSuite};
use binius_iop::merkle_tree::{BinaryMerkleTreeScheme, Commitment, Error, MerkleTreeScheme};
use binius_transcript::{BufMut, TranscriptWriter};
use binius_utils::rayon::iter::IndexedParallelIterator;
use digest::Output;
use getset::Getters;
use rand::{CryptoRng, SeedableRng, rngs::StdRng};

use super::MerkleTreeProver;

#[derive(Getters)]
pub struct BinaryMerkleTreeProver<T, H: HashSuite> {
	#[getset(get = "pub")]
	scheme: BinaryMerkleTreeScheme<T, H>,
	salt_rng: Mutex<StdRng>,
}

impl<T, H: HashSuite> BinaryMerkleTreeProver<T, H> {
	pub fn new() -> Self {
		Self {
			scheme: BinaryMerkleTreeScheme::new(),
			// We can construct a dummy Rng with a deterministic seed because it will be unused.
			salt_rng: Mutex::new(StdRng::seed_from_u64(0)),
		}
	}

	pub fn hiding(mut rng: impl CryptoRng, salt_len: usize) -> Self {
		Self {
			scheme: BinaryMerkleTreeScheme::hiding(salt_len),
			salt_rng: Mutex::new(StdRng::from_rng(&mut rng)),
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
	type Committed = BinaryMerkleTree<Output<H::LeafHash>, F>;

	fn scheme(&self) -> &Self::Scheme {
		&self.scheme
	}

	fn layer<'a>(
		&self,
		committed: &'a Self::Committed,
		depth: usize,
	) -> Result<&'a [Output<H::LeafHash>], Error> {
		Ok(committed.layer(depth)?)
	}

	fn prove_opening<B: BufMut>(
		&self,
		committed: &Self::Committed,
		layer_depth: usize,
		index: usize,
		proof: &mut TranscriptWriter<B>,
	) -> Result<(), Error> {
		let salt = committed.get_salt(index >> layer_depth);
		proof.write_slice(salt);

		let branch = committed.branch(index, layer_depth)?;
		proof.write_slice(&branch);
		Ok(())
	}

	#[allow(clippy::type_complexity)]
	fn commit_iterated<ParIter>(
		&self,
		leaves: ParIter,
	) -> Result<(Commitment<<Self::Scheme as MerkleTreeScheme<F>>::Digest>, Self::Committed), Error>
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
	{
		let salt_rng = {
			// If mutex is poisoned, panic.
			let mut root_rng = self.salt_rng.lock().unwrap();
			StdRng::from_rng(&mut *root_rng)
		};
		let tree = binary_merkle_tree::build_from_iterator::<F, H, _, _>(
			leaves,
			self.scheme.salt_len(),
			salt_rng,
		)?;

		let commitment = Commitment {
			root: tree.root(),
			depth: tree.log_len,
		};

		Ok((commitment, tree))
	}
}
