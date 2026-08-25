// Copyright 2026 The Binius Developers

//! What checking a Ligerito opening costs the verifier, level by level.
//!
//! Proof size prices what crosses the wire.
//! This prices what the verifier does with it.
//! The units are the ones a recursion circuit pays in:
//!
//! - calls to the hash function;
//! - bit-decomposition sums over fixed constants.
//!
//! Both are counted from the ladder alone, without running a prover.
//!
//! # What the verifier hashes
//!
//! A level's query round reads one internal Merkle layer.
//! It folds that layer to the root, then climbs from each opened leaf up to it.
//! At tree depth `d`, layer depth `l` and `t` opened rows that is
//!
//! ```text
//!     leaf hashes  = t
//!     compressions = (2^l - 1)  +  t * (d - l)
//! ```
//!
//! where the first term folds the layer and the second walks the `t` branches.
//!
//! The residual is different in kind.
//! It arrives in the clear, so its whole tree is rebuilt rather than one branch of it.
//! At residual dimension `r` that is `2^r` leaf hashes and `2^r - 1` compressions.
//! Every other row grows with the logarithm of its level's length, and this one with `2^r`.
//! So past a modest `r` the cleartext residual is most of the verifier's hashing.
//!
//! # What the verifier decomposes
//!
//! A query index arrives as an opaque word.
//! The induced basis needs one subspace-polynomial evaluation per column variable.
//! Each is a subset sum over precomputed constants, so a level costs `t * cols` of them.
//!
//! That count is the whole reason the closed-form basis is the verifier's only route.
//! Expanding the basis densely would cost `2^cols` field elements per level.
//! That is not a circuit anyone can build.

use binius_field::BinaryField;

use super::common::LigeritoParams;
use crate::merkle_tree::MerkleTreeScheme;

/// The verifier work one row of the cost table accounts for.
///
/// A row is either one committed level's query round, or the cleartext residual's check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerifierCost {
	/// Leaf digests computed, one per Merkle leaf the verifier rebuilds from revealed values.
	pub leaf_hashes: usize,
	/// Two-to-one compressions computed while folding digests toward a committed root.
	pub compressions: usize,
	/// Subset sums over fixed constants, driven by the bits of a query index.
	pub subset_sums: usize,
}

impl VerifierCost {
	/// Calls to the hash function, of either kind.
	///
	/// The two kinds cost differently in a circuit.
	/// Their sum is the figure to compare across ladder shapes.
	pub const fn hash_calls(&self) -> usize {
		self.leaf_hashes + self.compressions
	}

	/// The row-by-row sum of a cost table.
	pub fn total(rows: &[Self]) -> Self {
		rows.iter().fold(Self::default(), |acc, row| Self {
			leaf_hashes: acc.leaf_hashes + row.leaf_hashes,
			compressions: acc.compressions + row.compressions,
			subset_sums: acc.subset_sums + row.subset_sums,
		})
	}
}

/// The per-level cost table, with the residual as its last row.
///
/// See the module documentation for the formula behind each entry.
pub(super) fn verifier_cost<F, VCS>(params: &LigeritoParams, vcs: &VCS) -> Vec<VerifierCost>
where
	F: BinaryField,
	VCS: MerkleTreeScheme<F>,
{
	let mut rows = params
		.levels()
		.iter()
		.map(|level| {
			let tree_depth = level.log_codeword_len();
			// The verifier reads one internal layer and climbs to it, rather than to the root.
			let layer_depth = vcs.optimal_verify_layer(level.n_queries, tree_depth);
			VerifierCost {
				// One leaf per opened row, whatever the leaf holds.
				leaf_hashes: level.n_queries,
				// Fold the layer to the root once, then walk one branch per opened row.
				compressions: ((1 << layer_depth) - 1)
					+ level.n_queries * (tree_depth - layer_depth),
				// One subspace-polynomial evaluation per column variable, per opened row.
				subset_sums: level.n_queries * level.log_msg_cols,
			}
		})
		.collect::<Vec<_>>();

	// The residual is checked against its own commitment by rebuilding the entire tree, because
	// every one of its leaves is revealed rather than one branch of them.
	let residual_leaves = 1 << params.log_residual_dim();
	rows.push(VerifierCost {
		leaf_hashes: residual_leaves,
		compressions: residual_leaves - 1,
		subset_sums: 0,
	});
	rows
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;

	use super::*;
	use crate::{
		ligerito::LigeritoLevel, merkle_tree::BinaryMerkleTreeScheme, soundness::SoundnessRegime,
	};

	fn scheme() -> BinaryMerkleTreeScheme<B128, StdHashSuite> {
		BinaryMerkleTreeScheme::new()
	}

	/// A ladder whose level `i` commits at inverse rate `2^(i + 1)` and opens `n_queries` rows.
	///
	/// `lanes[i]` is level `i`'s fold amount, and `log_msg_cols` is level 0's column count.
	fn ladder(log_msg_cols: usize, lanes: &[usize], n_queries: usize) -> LigeritoParams {
		let mut log_msg_cols = log_msg_cols;
		let levels = lanes
			.iter()
			.enumerate()
			.map(|(i, &log_lanes)| {
				// Level 0 keeps the column count it was given; every deeper one loses its lanes.
				if i > 0 {
					log_msg_cols -= log_lanes;
				}
				LigeritoLevel {
					log_msg_cols,
					log_lanes,
					log_inv_rate: i + 1,
					n_queries,
				}
			})
			.collect();
		LigeritoParams::new(levels, SoundnessRegime::UniqueDecoding, 32)
	}

	#[test]
	fn a_single_query_walks_one_full_branch() {
		// Fixture state: one level, 2^6 columns at rate 1/2, so a tree of depth 7, opened once.
		//
		// A single query puts the decommitted layer at the root, so there is no layer to fold and
		// the branch is the whole climb.
		//
		//     layer depth 0 -> 2^0 - 1 = 0 compressions to fold the layer
		//     1 query       -> 7 - 0   = 7 compressions to climb
		let params = ladder(6, &[1], 1);
		let rows = params.verifier_cost(&scheme());

		assert_eq!(rows.len(), 2, "one row per level, plus the residual");
		assert_eq!(
			rows[0],
			VerifierCost {
				leaf_hashes: 1,
				compressions: 7,
				subset_sums: 6,
			}
		);
		// The residual is 2^6 leaves rebuilt in full, and a full binary tree over `n` leaves has
		// `n - 1` internal nodes.
		assert_eq!(
			rows[1],
			VerifierCost {
				leaf_hashes: 64,
				compressions: 63,
				subset_sums: 0,
			}
		);
	}

	#[test]
	fn the_decommitted_layer_trades_branch_length_for_layer_width() {
		// Invariant: raising the decommitted layer by one doubles the layer fold but shortens
		// every branch by one compression, and the scheme puts it where the two balance.
		//
		// Fixture state: one level of tree depth 7, opened 8 times.
		//
		//     layer depth 3 -> 2^3 - 1     = 7 compressions to fold the layer
		//     8 queries     -> 8 * (7 - 3) = 32 compressions to climb
		let params = ladder(6, &[1], 8);
		let rows = params.verifier_cost(&scheme());

		assert_eq!(rows[0].compressions, 7 + 32);
		// One subset sum per column variable, per opened row.
		assert_eq!(rows[0].subset_sums, 8 * 6);
	}

	#[test]
	fn a_longer_level_zero_costs_a_branch_hash_rather_than_a_dense_expansion() {
		// Invariant: the verifier's per-level work is logarithmic in the level's length. Doubling
		// a level's columns adds one compression per query and one subset sum per query, where
		// materializing the induced basis would double the level's cost outright.
		//
		// Fixture state: one level opened 12 times, at 2^6 and then 2^7 columns.
		let small = ladder(6, &[1], 12).verifier_cost(&scheme());
		let large = ladder(7, &[1], 12).verifier_cost(&scheme());

		// The layer depth is the same on both, so the whole difference is branch length.
		assert_eq!(large[0].compressions - small[0].compressions, 12);
		assert_eq!(large[0].subset_sums - small[0].subset_sums, 12);
		// The dense route's own vector, meanwhile, went from 64 entries to 128.
		assert!(large[0].subset_sums < 1 << 7);
	}

	#[test]
	fn the_cleartext_residual_overtakes_every_committed_level() {
		// Invariant: every committed level hashes proportionally to the logarithm of its length,
		// while the residual hashes proportionally to its length. So the residual dimension, not
		// the message length, is what a recursive circuit's hash budget turns on.
		//
		// Fixture state: a four-level ladder over 2^18 message elements, residual 2^10.
		let params = ladder(16, &[2, 2, 2, 2], 30);
		let rows = params.verifier_cost(&scheme());

		let residual = rows
			.last()
			.expect("the table always ends with the residual");
		let levels = &rows[..rows.len() - 1];
		for (i, level) in levels.iter().enumerate() {
			assert!(
				residual.hash_calls() > level.hash_calls(),
				"level {i} hashes {} against the residual's {}",
				level.hash_calls(),
				residual.hash_calls()
			);
		}

		// And it is most of the total, not merely the largest single row.
		let total = VerifierCost::total(&rows);
		assert!(2 * residual.hash_calls() > total.hash_calls());
	}

	#[test]
	fn a_shallower_residual_moves_the_verifier_off_the_dense_term() {
		// Invariant: the residual row is the only one that grows with a power of two, so shrinking
		// it is the one lever that changes the shape of the verifier's cost.
		//
		// Fixture state: the same 2^18 message, folded to 2^10 and then to 2^6.
		let deep = VerifierCost::total(&ladder(16, &[2, 2, 2, 2], 30).verifier_cost(&scheme()));
		let shallow =
			VerifierCost::total(&ladder(16, &[2, 2, 2, 2, 2, 2], 30).verifier_cost(&scheme()));

		// Two extra levels of queries cost less than the 2^10 residual they removed.
		assert!(shallow.hash_calls() < deep.hash_calls());
		// The subset-sum count moves the other way: extra levels are extra query rounds.
		assert!(shallow.subset_sums > deep.subset_sums);
	}
}
