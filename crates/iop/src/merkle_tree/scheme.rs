// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Verifier side of the binary Merkle tree vector commitment.

use std::{
	fmt::{self, Debug, Formatter},
	marker::PhantomData,
	num::NonZeroUsize,
};

use binius_hash::{CompressionFunction, HashBuffer, binary_merkle_tree::HashSuite};
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
/// # Overview
///
/// A committed vector is cut into equal-size batches of values.
/// Each batch is hashed into one leaf digest.
/// Pairs of digests are then folded upward until a single root digest remains.
///
/// ```text
///                     root
///                   /      \
///            C(d_0,d_1)   C(d_2,d_3)
///             /     \       /     \
///           d_0    d_1    d_2    d_3     <- leaf digests
///            |      |      |      |
///         batch_0  ...          batch_3  <- committed values, plus salt when hiding
/// ```
///
/// # Why the values need a fixed serialized width
///
/// Every committed value serializes to the same number of bytes.
/// A leaf's byte string is therefore an injective encoding of the values and salt it holds.
/// Two different leaves can never present identical bytes to the hash.
pub struct BinaryMerkleTreeScheme<T, H: HashSuite> {
	/// Two-to-one function folding a pair of child digests into their parent digest.
	compression: H::Compression,
	/// Number of uniform values appended to each leaf before hashing.
	///
	/// Zero when the commitment is not hiding.
	salt_len: usize,
	/// Records the committed value type without ever holding one.
	///
	/// The function-pointer form keeps the scheme thread-safe whatever that type is.
	/// See <https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns>.
	_phantom: PhantomData<fn() -> T>,
}

impl<T, H: HashSuite> BinaryMerkleTreeScheme<T, H> {
	/// Builds a non-hiding scheme, hashing each leaf with no added randomness.
	pub fn new() -> Self {
		// A salt length of zero is exactly the non-hiding case.
		Self::with_salt_len(0)
	}

	/// Builds a hiding scheme, appending uniform random values to every leaf before hashing.
	///
	/// # Overview
	///
	/// The salt is what hides a leaf.
	/// Guessing a leaf's values is not enough to confirm the guess without the salt too.
	///
	/// # Arguments
	///
	/// * `salt_len` - how many uniform values to append to each leaf.
	///
	/// # Why the length is caller-chosen
	///
	/// The salt must carry at least as many bits of entropy as the target security level.
	/// One value contributes its full width in bits, so the field size fixes the count.
	pub fn hiding(salt_len: NonZeroUsize) -> Self {
		Self::with_salt_len(salt_len.get())
	}

	/// Number of uniform values appended to each leaf before hashing.
	///
	/// # Returns
	///
	/// Zero when the scheme is not hiding, otherwise the configured salt length.
	pub const fn salt_len(&self) -> usize {
		self.salt_len
	}

	/// Shared constructor covering both the hiding and the non-hiding case.
	fn with_salt_len(salt_len: usize) -> Self {
		Self {
			// The compression function is stateless, so one default instance serves every call.
			compression: H::Compression::default(),
			salt_len,
			_phantom: PhantomData,
		}
	}

	/// Folds a layer of digests down to the single root above it.
	///
	/// # Overview
	///
	/// Each round pairs neighbours and replaces them with their parent, halving the layer:
	///
	/// ```text
	///     [d_0, d_1, ..., d_{n-1}]  ->  [C(d_0, d_1), ..., C(d_{n-2}, d_{n-1})]
	/// ```
	///
	/// After `log_2(n)` rounds exactly one digest is left, and that digest is the root.
	///
	/// # Returns
	///
	/// The root digest of the subtree spanned by the given layer.
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

impl<T, H: HashSuite> Default for BinaryMerkleTreeScheme<T, H> {
	fn default() -> Self {
		// The non-hiding scheme is the one with no parameter left to choose.
		Self::new()
	}
}

impl<T, H: HashSuite> Clone for BinaryMerkleTreeScheme<T, H> {
	fn clone(&self) -> Self {
		// Written out rather than derived: a derived copy would demand a cloneable value type.
		// No value of that type is ever held.
		// The compression function is always cloneable through its own trait bound.
		Self {
			compression: self.compression.clone(),
			salt_len: self.salt_len,
			_phantom: PhantomData,
		}
	}
}

impl<T, H: HashSuite> Debug for BinaryMerkleTreeScheme<T, H> {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		// Written out for the same reason as the copy above.
		// The compression function carries no formatting bound.
		// That leaves the salt length as the only state worth printing.
		f.debug_struct("BinaryMerkleTreeScheme")
			.field("salt_len", &self.salt_len)
			.finish_non_exhaustive()
	}
}

impl<T, H> BinaryMerkleTreeScheme<T, H>
where
	T: FixedSizeSerializeBytes,
	H: HashSuite,
{
	/// Hashes one leaf from the values it holds and the salt the proof supplies.
	///
	/// # Arguments
	///
	/// * `values` - the committed values of this leaf, in commitment order.
	/// * `proof` - decommitment advice, positioned at this leaf's salt.
	///
	/// # Returns
	///
	/// The leaf digest, or a failure if the salt could not be read.
	fn compute_leaf_digest<B: Buf>(
		&self,
		values: &[T],
		proof: &mut TranscriptReader<B>,
	) -> Result<Output<H::LeafHash>, Error> {
		let mut hasher = H::LeafHash::new();
		{
			// The buffer groups writes into hash blocks and flushes on scope exit.
			// The salt therefore streams from the proof into the hash with no staging buffer.
			let mut buffer = HashBuffer::new(&mut hasher);

			// Values first, salt second: the same order the prover used when committing.
			for value in values {
				value.serialize(&mut buffer)?;
			}
			for _ in 0..self.salt_len {
				proof.read::<T>()?.serialize(&mut buffer)?;
			}
		}
		Ok(hasher.finalize())
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
		let n_digests = (log_len - layer_depth) * n_queries + (1 << layer_depth);

		// A hiding scheme additionally reveals the salt of every opened leaf.
		let salt_bytes = n_queries * self.salt_len * T::BYTE_SIZE;

		n_digests * <H::LeafHash as Digest>::output_size() + salt_bytes
	}

	fn vector_proof_size(&self, len: usize) -> usize {
		// Every leaf is revealed, so no branch is needed.
		// The only advice left is one salt per leaf.
		len * self.salt_len * T::BYTE_SIZE
	}

	fn verify_vector<B: Buf>(
		&self,
		root: &Self::Digest,
		data: &[T],
		batch_size: usize,
		proof: &mut TranscriptReader<B>,
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
		// Each leaf's salt is read from the advice as the rebuild reaches it.
		let digests = data
			.chunks(batch_size)
			.map(|chunk| self.compute_leaf_digest(chunk, proof))
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
		let mut digest = self.compute_leaf_digest(values, proof)?;

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

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash as B128, Field};
	use binius_hash::{StdDigest, StdHashSuite, hash_serialize};
	use binius_math::test_utils::random_scalars;
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;

	type Suite = StdHashSuite;
	type LeafHash = <Suite as HashSuite>::LeafHash;
	type Node = Output<LeafHash>;
	type Scheme = BinaryMerkleTreeScheme<B128, Suite>;
	type Challenger = HasherChallenger<StdDigest>;

	/// A Merkle tree built independently of the scheme under test.
	///
	/// # Why an independent build
	///
	/// Checking the verifier against trees it built itself would pass even with a wrong fold.
	/// This one goes straight from the hash and compression primitives to the layers.
	struct RefTree {
		/// Digest layers, leaves first and the single-element root layer last.
		layers: Vec<Vec<Node>>,
		/// Committed values of each leaf, in leaf order.
		leaves: Vec<Vec<B128>>,
		/// Salt of each leaf, in leaf order, all empty when the tree is not salted.
		salts: Vec<Vec<B128>>,
	}

	impl RefTree {
		/// Commits a full tree of uniform leaves.
		///
		/// # Arguments
		///
		/// * `log_len` - base-2 logarithm of the number of leaves.
		/// * `batch_size` - number of committed values per leaf.
		/// * `salt_len` - number of salt values per leaf, zero for an unsalted tree.
		fn random(rng: &mut impl Rng, log_len: usize, batch_size: usize, salt_len: usize) -> Self {
			let compression = <Suite as HashSuite>::Compression::default();

			// Draw the committed payload and the salt separately, one entry per leaf.
			let leaves = (0..1 << log_len)
				.map(|_| random_scalars::<B128>(&mut *rng, batch_size))
				.collect::<Vec<_>>();
			let salts = (0..1 << log_len)
				.map(|_| random_scalars::<B128>(&mut *rng, salt_len))
				.collect::<Vec<_>>();

			// A leaf hashes its values first and its salt second, matching the verifier's order.
			let leaf_digests = leaves
				.iter()
				.zip(&salts)
				.map(|(values, salt)| {
					hash_serialize::<B128, LeafHash>(values.iter().chain(salt))
						.expect("field elements serialize into a growable buffer")
				})
				.collect::<Vec<_>>();

			// Fold neighbouring pairs upward until the topmost layer holds one digest.
			// Every intermediate layer is kept, since openings are checked against them.
			let mut layers = vec![leaf_digests];
			while layers.last().expect("layers is never empty").len() > 1 {
				let next = layers
					.last()
					.expect("layers is never empty")
					.chunks_exact(2)
					.map(|pair| compression.compress([pair[0], pair[1]]))
					.collect();
				layers.push(next);
			}

			Self {
				layers,
				leaves,
				salts,
			}
		}

		/// Number of levels between the leaves and the root.
		fn depth(&self) -> usize {
			// One layer is stored per level, plus the leaf layer itself.
			self.layers.len() - 1
		}

		/// The root digest of the tree.
		fn root(&self) -> Node {
			// The topmost layer holds exactly one digest.
			self.layers[self.depth()][0]
		}

		/// The digests sitting the given number of levels below the root.
		fn layer(&self, layer_depth: usize) -> Vec<Node> {
			// Layers are stored leaves-first, so counting down from the root inverts the index.
			self.layers[self.depth() - layer_depth].clone()
		}

		/// The sibling digests on the path from one leaf up to the given layer.
		fn branch(&self, index: usize, layer_depth: usize) -> Vec<Node> {
			// At level `j` the path sits at `index >> j`.
			// Its sibling is that same position with the low bit flipped.
			(0..self.depth() - layer_depth)
				.map(|j| self.layers[j][(index >> j) ^ 1])
				.collect()
		}

		/// Every committed value of the tree, concatenated in leaf order.
		fn flat_data(&self) -> Vec<B128> {
			self.leaves.concat()
		}
	}

	/// Builds a scheme for a salt length, hiding exactly when that length is non-zero.
	fn scheme(salt_len: usize) -> Scheme {
		match NonZeroUsize::new(salt_len) {
			Some(salt_len) => Scheme::hiding(salt_len),
			None => Scheme::new(),
		}
	}

	/// Writes the advice one single-leaf opening consumes: the leaf salt, then the branch.
	fn write_opening(
		transcript: &mut ProverTranscript<Challenger>,
		tree: &RefTree,
		index: usize,
		layer_depth: usize,
	) {
		// The verifier reads the salt while hashing the leaf, then the siblings while climbing.
		let mut writer = transcript.decommitment();
		writer.write_slice(&tree.salts[index]);
		writer.write_slice(&tree.branch(index, layer_depth));
	}

	/// Writes the advice a full-vector opening consumes: every leaf's salt, in leaf order.
	fn write_vector(transcript: &mut ProverTranscript<Challenger>, tree: &RefTree) {
		// No branch is sent: the verifier rebuilds every leaf and folds the whole tree itself.
		let mut writer = transcript.decommitment();
		for salt in &tree.salts {
			writer.write_slice(salt);
		}
	}

	/// A transcript the tests write proof bytes into.
	fn prover_transcript() -> ProverTranscript<Challenger> {
		ProverTranscript::new(Challenger::default())
	}

	/// Asserts a verification failed for the one reason a tampered proof should fail.
	fn expect_invalid_proof(result: Result<(), Error>) {
		// Every field of the variant is pinned, so a differently-shaped failure still fails.
		match result {
			Err(Error::Verification(VerificationError::InvalidProof)) => {}
			other => panic!("expected an invalid-proof rejection, got {other:?}"),
		}
	}

	#[test]
	fn verify_opening_accepts_every_leaf_at_every_layer_depth() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: an honest branch verifies from any leaf against any layer of its tree.
		//
		// Fixture state: 4 trees of 8 leaves, covering batched leaves and salted leaves.
		for (batch_size, salt_len) in [(1, 0), (1, 2), (4, 0), (3, 1)] {
			let tree = RefTree::random(&mut rng, 3, batch_size, salt_len);
			let scheme = scheme(salt_len);

			// Depth 0 is the root and depth 3 is the leaf layer.
			// The sweep covers the longest branch, the empty branch, and every depth between.
			for layer_depth in 0..=tree.depth() {
				let layer = tree.layer(layer_depth);
				for index in 0..1 << tree.depth() {
					let mut transcript = prover_transcript();
					write_opening(&mut transcript, &tree, index, layer_depth);

					let mut verifier = transcript.into_verifier();
					scheme
						.verify_opening(
							index,
							&tree.leaves[index],
							layer_depth,
							tree.depth(),
							&layer,
							&mut verifier.decommitment(),
						)
						.unwrap();
					// Nothing may be left over: the verifier must read exactly what was written.
					verifier.finalize().unwrap();
				}
			}
		}
	}

	#[test]
	fn verify_opening_accepts_a_single_leaf_tree() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a one-leaf tree has no levels, so its salted leaf digest is the root.
		//
		// Fixture state: 1 leaf of 2 values, salted with 1 more, depth 0.
		//
		//     advice = [salt]     (no siblings to send)
		//     layer  = [root]     (the layer at depth 0 is the root itself)
		let tree = RefTree::random(&mut rng, 0, 2, 1);
		let scheme = scheme(1);

		let mut transcript = prover_transcript();
		write_opening(&mut transcript, &tree, 0, 0);

		let mut verifier = transcript.into_verifier();
		scheme
			.verify_opening(0, &tree.leaves[0], 0, 0, &[tree.root()], &mut verifier.decommitment())
			.unwrap();
		verifier.finalize().unwrap();
	}

	#[test]
	fn verify_opening_rejects_a_corrupted_branch() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 3, 2, 0);
		let scheme = scheme(0);

		// Invariant: every level of the climb feeds the next, so no sibling can be swapped.
		//
		// Fixture state: 8 leaves, unsalted, opening leaf 5 against the root.
		//
		// Mutation: flip the lowest bit of one sibling, one level at a time.
		//
		//     level 0:  [BAD, ok , ok ]
		//     level 1:  [ok , BAD, ok ]
		//     level 2:  [ok , ok , BAD]
		//     -> each must climb to a root differing from the committed one
		for level in 0..tree.depth() {
			let mut branch = tree.branch(5, 0);
			branch[level][0] ^= 1;

			let mut transcript = prover_transcript();
			transcript.decommitment().write_slice(&branch);

			let mut verifier = transcript.into_verifier();
			expect_invalid_proof(scheme.verify_opening(
				5,
				&tree.leaves[5],
				0,
				tree.depth(),
				&[tree.root()],
				&mut verifier.decommitment(),
			));
			verifier.finalize().unwrap();
		}
	}

	#[test]
	fn verify_opening_rejects_corrupted_leaf_values() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 3, 2, 0);
		let scheme = scheme(0);

		// Invariant: the branch binds the leaf's contents, not just its position.
		//
		// Fixture state: 8 leaves of 2 values each, unsalted, opening leaf 5.
		let mut transcript = prover_transcript();
		write_opening(&mut transcript, &tree, 5, 0);

		// Mutation: keep the honest branch, but claim a different first value for the leaf.
		let mut values = tree.leaves[5].clone();
		values[0] += B128::ONE;

		let mut verifier = transcript.into_verifier();
		expect_invalid_proof(scheme.verify_opening(
			5,
			&values,
			0,
			tree.depth(),
			&[tree.root()],
			&mut verifier.decommitment(),
		));
		verifier.finalize().unwrap();
	}

	#[test]
	fn verify_opening_rejects_a_corrupted_salt() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 3, 1, 2);
		let scheme = scheme(2);

		// Invariant: the salt is hashed into the leaf, so it is bound just like the values are.
		//
		// Fixture state: 8 leaves of 1 value each, salted with 2, opening leaf 5.
		//
		// Mutation: send leaf 6's salt alongside leaf 5's honest branch.
		//
		//     advice = [salt_6, branch_5 ...]
		//     -> the leaf digest differs, so the climb lands off the root
		let mut transcript = prover_transcript();
		{
			let mut writer = transcript.decommitment();
			writer.write_slice(&tree.salts[6]);
			writer.write_slice(&tree.branch(5, 0));
		}

		let mut verifier = transcript.into_verifier();
		expect_invalid_proof(scheme.verify_opening(
			5,
			&tree.leaves[5],
			0,
			tree.depth(),
			&[tree.root()],
			&mut verifier.decommitment(),
		));
		verifier.finalize().unwrap();
	}

	#[test]
	fn verify_opening_rejects_an_opening_against_a_corrupted_layer() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 3, 1, 0);
		let scheme = scheme(0);

		// Invariant: the climb ends by matching the layer entry the leaf index selects.
		//
		// Fixture state: 8 leaves, unsalted, opening leaf 5 against the layer at depth 2.
		//
		// Mutation: flip a bit in the one layer entry the climb lands on.
		//
		//     leaf 5 climbs 1 level -> layer index 5 >> 1 = 2
		let mut layer = tree.layer(2);
		layer[5 >> 1][0] ^= 1;

		let mut transcript = prover_transcript();
		write_opening(&mut transcript, &tree, 5, 2);

		let mut verifier = transcript.into_verifier();
		expect_invalid_proof(scheme.verify_opening(
			5,
			&tree.leaves[5],
			2,
			tree.depth(),
			&layer,
			&mut verifier.decommitment(),
		));
		verifier.finalize().unwrap();
	}

	#[test]
	fn verify_layer_accepts_the_committed_layer() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 4, 1, 0);
		let scheme = scheme(0);

		// Invariant: folding any honest layer reproduces the committed root.
		//
		// Fixture state: 16 leaves, unsalted, depth 4.
		//
		// Depth 0 is the degenerate case where the layer is the root and nothing folds.
		for layer_depth in 0..=tree.depth() {
			scheme
				.verify_layer(&tree.root(), layer_depth, &tree.layer(layer_depth))
				.unwrap();
		}
	}

	#[test]
	fn verify_layer_rejects_a_corrupted_layer() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 4, 1, 0);
		let scheme = scheme(0);

		// Invariant: a single wrong digest anywhere in the layer changes the folded root.
		//
		// Fixture state: 16 leaves, unsalted, depth 4.
		//
		// Mutation: flip the lowest bit of the layer's first digest, at every depth in turn.
		for layer_depth in 0..=tree.depth() {
			let mut layer = tree.layer(layer_depth);
			layer[0][0] ^= 1;
			expect_invalid_proof(scheme.verify_layer(&tree.root(), layer_depth, &layer));
		}
	}

	#[test]
	fn verify_vector_accepts_the_committed_vector() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: rebuilding every leaf from the revealed values reproduces the root.
		//
		// Fixture state: 4 trees sweeping leaf count, batch size, and salt length.
		//
		//     (log_len, batch_size, salt_len)
		//     the two log_len = 0 cases are single-leaf trees, where nothing folds at all
		for (log_len, batch_size, salt_len) in [(0, 1, 0), (0, 3, 2), (3, 1, 0), (3, 4, 2)] {
			let tree = RefTree::random(&mut rng, log_len, batch_size, salt_len);
			let scheme = scheme(salt_len);

			let mut transcript = prover_transcript();
			write_vector(&mut transcript, &tree);

			let mut verifier = transcript.into_verifier();
			scheme
				.verify_vector(
					&tree.root(),
					&tree.flat_data(),
					batch_size,
					&mut verifier.decommitment(),
				)
				.unwrap();
			verifier.finalize().unwrap();
		}
	}

	#[test]
	fn verify_vector_rejects_a_corrupted_value() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 3, 2, 1);
		let scheme = scheme(1);

		// Invariant: the root binds every value in the vector, not only the ones later queried.
		//
		// Fixture state: 8 leaves of 2 values each, salted with 1, so 16 values in total.
		let mut transcript = prover_transcript();
		write_vector(&mut transcript, &tree);

		// Mutation: alter one value in the middle of the vector.
		//
		//     value 7 lives in leaf 3 (7 / 2 = 3) -> that leaf digest changes -> the root changes
		let mut data = tree.flat_data();
		data[7] += B128::ONE;

		let mut verifier = transcript.into_verifier();
		expect_invalid_proof(scheme.verify_vector(
			&tree.root(),
			&data,
			2,
			&mut verifier.decommitment(),
		));
		verifier.finalize().unwrap();
	}

	#[test]
	fn verify_vector_rejects_a_corrupted_salt() {
		let mut rng = StdRng::seed_from_u64(0);
		let tree = RefTree::random(&mut rng, 3, 1, 2);
		let scheme = scheme(2);

		// Invariant: a full-vector opening binds each leaf's salt as tightly as its values.
		//
		// Fixture state: 8 leaves of 1 value each, salted with 2.
		//
		// Mutation: replace the last leaf's salt with the first leaf's.
		//
		//     advice = [salt_0, salt_1, ..., salt_6, salt_0]
		//                                            ^^^^^^ should be salt_7
		let last = tree.salts.len() - 1;
		let mut transcript = prover_transcript();
		{
			let mut writer = transcript.decommitment();
			for (i, salt) in tree.salts.iter().enumerate() {
				writer.write_slice(if i == last { &tree.salts[0] } else { salt });
			}
		}

		let mut verifier = transcript.into_verifier();
		expect_invalid_proof(scheme.verify_vector(
			&tree.root(),
			&tree.flat_data(),
			1,
			&mut verifier.decommitment(),
		));
		verifier.finalize().unwrap();
	}

	#[test]
	fn proof_size_matches_the_advice_a_multi_opening_consumes() {
		let mut rng = StdRng::seed_from_u64(0);
		let indices = [1, 4, 7, 7];

		// Invariant: the predicted byte count equals what a multi-opening actually writes.
		//
		// Fixture state: 8 leaves of 2 values each, 4 queries, one unsalted and one salted run.
		//
		// The salted run is the one that silently under-counts if the salt term is forgotten.
		for salt_len in [0, 3] {
			let tree = RefTree::random(&mut rng, 3, 2, salt_len);
			let scheme = scheme(salt_len);
			let layer_depth = scheme.optimal_verify_layer(indices.len(), tree.depth());

			// Advice layout, exactly as the verifier consumes it:
			//
			//     [layer digests] [salt_1 branch_1] ... [salt_7 branch_7]
			let mut transcript = prover_transcript();
			transcript
				.decommitment()
				.write_slice(&tree.layer(layer_depth));
			for &index in &indices {
				write_opening(&mut transcript, &tree, index, layer_depth);
			}

			let advice = transcript.finalize();
			assert_eq!(
				advice.len(),
				scheme.proof_size(1 << tree.depth(), indices.len(), layer_depth)
			);

			// A byte count only means something if those exact bytes verify, so replay them.
			let mut verifier = VerifierTranscript::new(Challenger::default(), advice);
			let layer_digests = verifier
				.decommitment()
				.read_vec::<Node>(1 << layer_depth)
				.unwrap();
			scheme
				.verify_layer(&tree.root(), layer_depth, &layer_digests)
				.unwrap();
			for &index in &indices {
				scheme
					.verify_opening(
						index,
						&tree.leaves[index],
						layer_depth,
						tree.depth(),
						&layer_digests,
						&mut verifier.decommitment(),
					)
					.unwrap();
			}
			verifier.finalize().unwrap();
		}
	}

	#[test]
	fn vector_proof_size_matches_the_advice_a_full_opening_consumes() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the predicted byte count equals what a full-vector opening actually writes.
		//
		// Fixture state: 8 leaves of 2 values each, one unsalted and one salted run.
		//
		//     unsalted -> no advice at all
		//     salted   -> 8 leaves * 3 salt values each
		for salt_len in [0, 3] {
			let tree = RefTree::random(&mut rng, 3, 2, salt_len);
			let scheme = scheme(salt_len);

			let mut transcript = prover_transcript();
			write_vector(&mut transcript, &tree);

			let advice = transcript.finalize();
			assert_eq!(advice.len(), scheme.vector_proof_size(1 << tree.depth()));

			// Replay the same bytes to confirm the count covers a proof that really verifies.
			let mut verifier = VerifierTranscript::new(Challenger::default(), advice);
			scheme
				.verify_vector(&tree.root(), &tree.flat_data(), 2, &mut verifier.decommitment())
				.unwrap();
			verifier.finalize().unwrap();
		}
	}

	#[test]
	fn salt_len_reports_the_configured_length() {
		// Invariant: the non-hiding constructions report no salt.
		// The hiding one reports exactly the length it was given.
		assert_eq!(Scheme::new().salt_len(), 0);
		assert_eq!(Scheme::default().salt_len(), 0);
		assert_eq!(Scheme::hiding(NonZeroUsize::new(4).unwrap()).salt_len(), 4);
	}

	proptest! {
		#[test]
		fn optimal_verify_layer_minimizes_proof_size(
			n_queries in 1usize..64,
			tree_depth in 0usize..10,
			salt_len in 0usize..3,
		) {
			// Invariant: the chosen layer depth is one that really minimizes the proof.
			let scheme = scheme(salt_len);
			let len = 1 << tree_depth;

			let chosen = scheme.optimal_verify_layer(n_queries, tree_depth);
			// A layer never sits below the leaves.
			prop_assert!(chosen <= tree_depth);

			// Search every admissible depth for the smallest proof.
			let best = (0..=tree_depth)
				.map(|layer_depth| scheme.proof_size(len, n_queries, layer_depth))
				.min()
				.expect("the range always includes layer_depth 0");

			// Two depths can tie, so compare the sizes rather than the depths themselves.
			prop_assert_eq!(scheme.proof_size(len, n_queries, chosen), best);
		}
	}
}
