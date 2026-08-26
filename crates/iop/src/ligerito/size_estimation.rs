// Copyright 2026 The Binius Developers

use binius_field::BinaryField;

use super::{
	common::{LigeritoLevel, LigeritoParams},
	opening,
};
use crate::{merkle_tree::MerkleTreeScheme, soundness::Grinding};

/// Serialized byte sizes of the two atoms a proof is built from.
pub(super) struct ByteSizes {
	/// Byte-size of one Merkle digest.
	pub digest: usize,
	/// Byte-size of one serialized field element.
	pub value: usize,
}

impl ByteSizes {
	/// Reads both sizes off the field and the Merkle scheme a proof is written with.
	pub fn new<F, VCS>() -> Self
	where
		F: BinaryField,
		VCS: MerkleTreeScheme<F>,
	{
		let mut buf = Vec::new();
		F::default()
			.serialize(&mut buf)
			.expect("default element can be serialized to a resizable buffer");
		Self {
			digest: std::mem::size_of::<VCS::Digest>(),
			value: buf.len(),
		}
	}
}

/// Stands for any level below the root, where the ladder search tabulates its subproblems.
///
/// Only level 0 versus not-level-0 changes a price, so every deeper level shares one index.
pub(super) const DEEPER_LEVEL: usize = 1;

/// Field elements one fold round's message costs at `level_index`.
///
/// A degree-`d` round message is `d` elements, whichever protocol sends it.
/// Sumcheck truncates the high coefficient and recovers it from the claimed sum.
/// An MLE-check truncates the low one and recovers it from the claimed evaluation.
///
/// Level 0 folds under the equality indicator `eq(z)` alone, which is what makes it an MLE-check.
/// Gluing a query-induced basis into the weight destroys that structure at every deeper level.
/// Both degrees are read from the protocol itself, so the price cannot drift from what is sent.
const fn round_degree(level_index: usize) -> usize {
	match level_index {
		0 => opening::MLECHECK_DEGREE,
		_ => opening::PRODUCT_DEGREE,
	}
}

/// The proof bytes one committed level contributes, as the `level_index`-th of its ladder.
///
/// Per formula (19) of [NA25], a level sends:
///
/// ```text
///     root        one digest
///     rows        n_queries * 2^log_lanes field elements
///     branches    one Merkle multi-proof over 2^(log_msg_cols + log_inv_rate) leaves
///     sumcheck    round_degree(level_index) field elements per fold round, log_lanes of them
///     nonces      one u64 per proof of work, which `LigeritoLevel::n_grind_nonces` counts
/// ```
///
/// The index is carried only for that last line; see [`round_degree`] for why it matters.
///
/// The Merkle tree is indexed by codeword position, one leaf per position across all lanes.
/// So it is `log_lanes` levels shorter than the level's element count.
/// Sizing it by the element count would charge `log_lanes` extra hashes per branch.
/// The search would then minimize a proof size no prover produces.
///
/// [NA25]: <https://eprint.iacr.org/2025/1187>
pub(super) fn level_size<F, VCS>(
	level: &LigeritoLevel,
	level_index: usize,
	vcs: &VCS,
	sizes: &ByteSizes,
	grinding: Grinding,
) -> usize
where
	F: BinaryField,
	VCS: MerkleTreeScheme<F>,
{
	let log_n_leaves = level.log_codeword_len();
	let layer_depth = vcs.optimal_verify_layer(level.n_queries, log_n_leaves);
	let merkle_size = vcs.proof_size(1 << log_n_leaves, level.n_queries, layer_depth);

	let rows_size = level.n_queries * (1 << level.log_lanes) * sizes.value;
	let sumcheck_size = round_degree(level_index) * level.log_lanes * sizes.value;
	let grinding_size = level.n_grind_nonces(grinding) * size_of::<u64>();

	sizes.digest + rows_size + merkle_size + sumcheck_size + grinding_size
}

/// Computes the exact byte-size of a Ligerito proof without running the prover.
///
/// This accounts for:
///
/// - **Message channel**: one Merkle root per committed level (digests observed by Fiat-Shamir).
/// - **Decommitment channel**: per level, the opened rows and one Merkle multi-proof.
/// - **Sumcheck transcript**: [`round_degree`] elements per fold round, which level 0 discounts.
/// - **Residual**: its commitment, then `2^log_residual_dim` elements in the clear.
/// - **Proof of work**: one `u64` nonce per grind, at the depths the parameters fix.
///
/// Exact is meant literally.
/// `the_estimate_equals_the_proof_the_prover_writes` proves real openings and compares this.
pub(super) fn proof_size<F, VCS>(params: &LigeritoParams, vcs: &VCS) -> usize
where
	F: BinaryField,
	VCS: MerkleTreeScheme<F>,
{
	let sizes = ByteSizes::new::<F, VCS>();

	let levels_size = params
		.levels()
		.iter()
		.enumerate()
		.map(|(level_index, level)| level_size(level, level_index, vcs, &sizes, params.grinding()))
		.sum::<usize>();
	levels_size + residual_size(params.log_residual_dim(), &sizes)
}

/// Bytes the terminating residual costs.
///
/// It is committed before any query position is sampled.
/// So it costs a digest on top of its elements.
/// The reported size and the search's objective both price it here, so they cannot disagree.
pub(super) const fn residual_size(log_residual_dim: usize, sizes: &ByteSizes) -> usize {
	sizes.digest + (1 << log_residual_dim) * sizes.value
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;
	use proptest::prelude::*;

	use super::*;
	use crate::{
		ligerito::LigeritoLevel, merkle_tree::BinaryMerkleTreeScheme, soundness::SoundnessRegime,
	};

	type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

	const UDR: SoundnessRegime = SoundnessRegime::UniqueDecoding;

	/// The target `binius-verifier` ships, which the shapes below are priced at.
	const SECURITY_BITS: usize = 96;

	fn test_merkle_scheme() -> TestMerkleScheme {
		BinaryMerkleTreeScheme::new()
	}

	#[test]
	fn single_level_ladder_prices_the_residual_in_the_clear() {
		let merkle_scheme = test_merkle_scheme();
		// One level, 2^9 columns by 2^3 lanes, rate 1/2, residual 2^9 elements in the clear.
		let level = LigeritoLevel::new(9, 3, 1, UDR, SECURITY_BITS, Grinding::NONE);
		let params = LigeritoParams::new(vec![level], UDR, SECURITY_BITS);
		assert_eq!(params.log_residual_dim(), 9);

		let value_size = size_of::<B128>();
		let digest_size = size_of::<<TestMerkleScheme as MerkleTreeScheme<B128>>::Digest>();
		let queries = UDR.n_queries(SECURITY_BITS, 1);
		let layer_depth = merkle_scheme.optimal_verify_layer(queries, 10);
		// The codeword root, the opened rows, their multi-proof, one round polynomial per folded
		// lane, then the residual: its own commitment, and its elements in the clear.
		let expected = digest_size
			+ queries * 8 * value_size
			+ merkle_scheme.proof_size(1 << 10, queries, layer_depth)
			+ round_degree(0) * 3 * value_size
			+ digest_size
			+ (1 << 9) * value_size;
		assert_eq!(params.proof_size(&merkle_scheme), expected);
	}

	// Level 0 folds under an equality indicator and the levels below it do not, so the two are
	// charged different round messages. Two ladders identical but for where a level sits pin the
	// gap, since nothing else in `level_size` reads the index.
	#[test]
	fn a_deeper_level_is_charged_a_wider_round_message() {
		let merkle_scheme = test_merkle_scheme();
		let value_size = size_of::<B128>();

		// The same shape, once as the only level and once as the second of two.
		let deep = LigeritoLevel::new(10, 3, 2, UDR, SECURITY_BITS, Grinding::NONE);
		let alone = LigeritoParams::new(vec![deep], UDR, SECURITY_BITS);
		let below = LigeritoParams::new(
			vec![
				LigeritoLevel::new(13, 3, 1, UDR, SECURITY_BITS, Grinding::NONE),
				deep,
			],
			UDR,
			SECURITY_BITS,
		);

		// Level 0 of the two-level ladder, priced on its own, plus the extra element per fold
		// round the deeper copy pays.
		let level_zero = LigeritoParams::new(
			vec![LigeritoLevel::new(
				13,
				3,
				1,
				UDR,
				SECURITY_BITS,
				Grinding::NONE,
			)],
			UDR,
			SECURITY_BITS,
		);
		let level_zero_size = level_zero.proof_size(&merkle_scheme)
			- size_of::<<TestMerkleScheme as MerkleTreeScheme<B128>>::Digest>()
			- (1 << 13) * value_size;
		let widening = (round_degree(DEEPER_LEVEL) - round_degree(0)) * deep.log_lanes * value_size;

		assert_eq!(
			below.proof_size(&merkle_scheme),
			level_zero_size + alone.proof_size(&merkle_scheme) + widening
		);
	}

	proptest! {
		#[test]
		fn proof_size_is_positive_and_grows_with_queries(
			log_msg_cols in 8usize..=20,
			log_lanes in 1usize..=7,
			log_inv_rate in 1usize..=4,
			extra_queries in 1usize..=64,
		) {
			let merkle_scheme = test_merkle_scheme();
			let base = LigeritoLevel {
				log_msg_cols,
				log_lanes,
				log_inv_rate,
				n_queries: 1,
			};
			let mut more = base;
			more.n_queries = 1 + extra_queries;
			// Both levels must fit their codeword, which 2^(8 + 1) positions comfortably do.
			prop_assert!(more.is_feasible());

			let small = LigeritoParams::new(
				vec![base],
				SoundnessRegime::UniqueDecoding,
				SECURITY_BITS,
			);
			let large = LigeritoParams::new(
				vec![more],
				SoundnessRegime::UniqueDecoding,
				SECURITY_BITS,
			);
			let small_size = small.proof_size(&merkle_scheme);
			prop_assert!(small_size > 0);
			prop_assert!(large.proof_size(&merkle_scheme) > small_size);
		}
	}
}
