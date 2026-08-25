// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use binius_field::BinaryField;

use super::{
	common::{LigeritoLevel, LigeritoParams},
	opening,
};
use crate::{merkle_tree::MerkleTreeScheme, soundness::SoundnessRegime};

/// The largest `log_lanes` the ladder search considers.
///
/// An opened row costs `2^log_lanes` field elements, so its cost doubles with every extra lane.
/// The Merkle branch that lane saves shortens by only one hash.
/// Past 7 the rows dominate at every size and rate this repo commits at.
/// So the cap sits far outside the optimum rather than binding on it.
pub const MAX_LOG_LANES: usize = 7;

/// The largest `log_inv_rate` the ladder search considers.
///
/// Query counts flatten out as the rate falls.
/// At 100-bit security the unique-decoding count moves from 105 to 101 between 1/32 and 1/256.
/// Each extra `log_inv_rate`, meanwhile, doubles the level's encoding work and Merkle tree.
/// Nothing is gained past 8.
pub const MAX_LOG_INV_RATE: usize = 8;

/// Serialized byte sizes of the two atoms a proof is built from.
struct ByteSizes {
	/// Byte-size of one Merkle digest.
	digest: usize,
	/// Byte-size of one serialized field element.
	value: usize,
}

impl ByteSizes {
	fn new<F, VCS>() -> Self
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
const DEEPER_LEVEL: usize = 1;

/// Round-polynomial degree of a fold round past level 0.
///
/// The weight there is `eq_remaining + alpha * w`, where `w` is a query-induced basis.
/// A product of two general multilinears is degree 2 in the bound variable.
const DEEP_ROUND_DEGREE: usize = 2;

/// Field elements one fold round's message costs at `level_index`.
///
/// A degree-`d` round message is `d` elements, whichever protocol sends it.
/// Sumcheck truncates the high coefficient and recovers it from the claimed sum.
/// An MLE-check truncates the low one and recovers it from the claimed evaluation.
///
/// Level 0 folds under the equality indicator `eq(z)` alone, which is what makes it an MLE-check.
/// Its composite is then the multilinear itself, of degree [`opening::DEGREE`].
/// Gluing a query-induced basis into the weight destroys that structure at every deeper level.
/// So those pay [`DEEP_ROUND_DEGREE`] instead, and the difference is one element per fold round.
const fn round_degree(level_index: usize) -> usize {
	match level_index {
		0 => opening::DEGREE,
		_ => DEEP_ROUND_DEGREE,
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
fn level_size<F, VCS>(
	level: &LigeritoLevel,
	level_index: usize,
	vcs: &VCS,
	sizes: &ByteSizes,
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

	sizes.digest + rows_size + merkle_size + sumcheck_size
}

/// Computes the exact byte-size of a Ligerito proof without running the prover.
///
/// This accounts for:
///
/// - **Message channel**: one Merkle root per committed level (digests observed by Fiat-Shamir).
/// - **Decommitment channel**: per level, the opened rows and one Merkle multi-proof.
/// - **Sumcheck transcript**: [`round_degree`] elements per fold round, which level 0 discounts.
/// - **Residual**: its commitment, then `2^log_residual_dim` elements in the clear.
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
		.map(|(level_index, level)| level_size(level, level_index, vcs, &sizes))
		.sum::<usize>();
	levels_size + residual_size(params.log_residual_dim(), &sizes)
}

/// One entry of the ladder search: the best continuation from a subproblem, and its cost.
#[derive(Debug, Clone, Copy)]
struct Decision {
	/// Total proof bytes of this level and everything after it, residual included.
	cost: usize,
	/// The lane count chosen for this level.
	log_lanes: usize,
	/// The inverse rate chosen for this level.
	log_inv_rate: usize,
	/// Whether a further committed level follows, rather than the residual terminating here.
	recurse: bool,
}

/// Bytes the terminating residual costs.
///
/// It is committed before any query position is sampled.
/// So it costs a digest on top of its elements.
/// The reported size and the search's objective both price it here, so they cannot disagree.
const fn residual_size(log_residual_dim: usize, sizes: &ByteSizes) -> usize {
	sizes.digest + (1 << log_residual_dim) * sizes.value
}

/// Everything the ladder search needs that does not vary between subproblems.
struct Search<'a, F, MerkleScheme> {
	/// The Merkle scheme whose branch sizes price every level.
	merkle_scheme: &'a MerkleScheme,
	/// Serialized sizes of a digest and a field element.
	sizes: ByteSizes,
	/// Query count per `log_inv_rate`. Index 0 is unused: rate 1 is not a proximity test.
	n_queries: Vec<usize>,
	/// The regime the query counts and the soundness ceiling are derived in.
	regime: SoundnessRegime,
	/// The target every level must reach, in bits.
	security_bits: usize,
	/// Ties `F` to the Merkle scheme without storing a value of it.
	_field: PhantomData<F>,
}

impl<F, MerkleScheme> Search<'_, F, MerkleScheme>
where
	F: BinaryField,
	MerkleScheme: MerkleTreeScheme<F>,
{
	/// Whether a level can reach [`Self::security_bits`] at all.
	///
	/// Two independent ways to fail, and neither is fixable by opening more rows.
	/// A level cannot sample more distinct positions than its codeword has.
	/// And its correlated-agreement term is a ceiling set by the codeword length and the field.
	fn reaches_target(&self, level: &LigeritoLevel) -> bool {
		let base = self.regime.correlated_agreement_bits(
			level.log_msg_len(),
			level.log_inv_rate,
			F::N_BITS,
		);
		// Same row union `LigeritoParams::correlated_agreement_bits` charges, so the search and
		// the reported ceiling cannot disagree about which levels clear the target.
		let algebra = base - (level.log_lanes.saturating_sub(1)) as f64;
		level.is_feasible() && algebra >= self.security_bits as f64
	}

	/// The best level to commit next, given `log_total` message elements left and a rate floor.
	///
	/// `min_log_inv_rate ..= max_log_inv_rate` is the range of rates this level may commit at.
	/// Level 0 gets a pinned singleton, and deeper levels get `previous + 1 ..= MAX_LOG_INV_RATE`.
	///
	/// `level_index` prices this level's fold rounds and nothing else.
	/// A subproblem is always solved as a deeper level, since only the root call is level 0.
	///
	/// `best` holds the already-solved subproblems, indexed by `[log_total][rate_floor]`.
	/// Every subproblem read here has a smaller `log_total`, since a level folds at least one axis.
	///
	/// Returns `None` when no level reaches the target for this subproblem.
	fn choose_level(
		&self,
		best: &[Vec<Option<Decision>>],
		log_total: usize,
		min_log_inv_rate: usize,
		max_log_inv_rate: usize,
		level_index: usize,
	) -> Option<Decision> {
		let mut chosen: Option<Decision> = None;
		let mut consider = |candidate: Decision| {
			if chosen.is_none_or(|current: Decision| candidate.cost < current.cost) {
				chosen = Some(candidate);
			}
		};

		for log_lanes in 1..=MAX_LOG_LANES.min(log_total) {
			let log_msg_cols = log_total - log_lanes;
			for log_inv_rate in min_log_inv_rate..=max_log_inv_rate {
				let level = LigeritoLevel {
					log_msg_cols,
					log_lanes,
					log_inv_rate,
					n_queries: self.n_queries[log_inv_rate],
				};
				if !self.reaches_target(&level) {
					continue;
				}
				let here = level_size(&level, level_index, self.merkle_scheme, &self.sizes);

				// Terminate: fold this level and send the remaining columns in the clear.
				consider(Decision {
					cost: here + residual_size(log_msg_cols, &self.sizes),
					log_lanes,
					log_inv_rate,
					recurse: false,
				});

				// Or recurse: the remaining columns become the next level's message, committed at
				// a strictly lower rate.
				if let Some(next) = best[log_msg_cols][log_inv_rate + 1] {
					consider(Decision {
						cost: here + next.cost,
						log_lanes,
						log_inv_rate,
						recurse: true,
					});
				}
			}
		}

		chosen
	}
}

/// Ladder minimizing the estimated proof size, with level 0's rate pinned by the caller.
///
/// Level 0's rate is pinned because level 0's encoding dominates prover time.
/// A ladder only compares to today's FRI at the same L0 rate, where it does the same L0 work.
/// The deep levels are small, so they are free to drop the rate as far as the search likes.
///
/// Returns the parameters together with the estimated proof size in bytes.
/// That size is exactly [`LigeritoParams::proof_size`] of the returned parameters.
///
/// ## The search
///
/// This is a memoized dynamic program over `(log_total, rate_floor)`.
/// The state is how many message elements are left, and the lowest rate the ladder still
/// allows. It mirrors what `fri`'s `ReductionOptimizer` does for fold arities, with two
/// differences.
///
/// The first difference is that the state carries a rate floor at all.
/// A Ligerito level chooses its own rate, where an FRI round cannot.
/// So an FRI round minimizes over arity alone.
/// Each state here minimizes over a two-dimensional grid of `(log_lanes, log_inv_rate)`.
///
/// The second is that the minimization is exhaustive, not the early-exit `min_concave`
/// performs. The `log_lanes` axis was measured monotone, so that early exit would in fact be
/// sound on it. The measurement swept every state up to `log_n = 32`, both regimes, at 100 and
/// 128 bits. It never found the objective turning back down after rising.
/// The scan stays exhaustive anyway, for two reasons.
/// The minimization is genuinely two-dimensional, where `min_concave` handles one axis.
/// And a state's low rates are infeasible while its high ones are not, leaving a hole.
/// An early-exit scan cannot walk over that hole unless it is handed a filtered range first.
/// Meanwhile a state costs at most `MAX_LOG_LANES * MAX_LOG_INV_RATE` evaluations either way.
/// A shape assumption buys nothing here, so this code does not make one.
///
/// The search caps `log_lanes` at [`MAX_LOG_LANES`] and `log_inv_rate` at [`MAX_LOG_INV_RATE`].
/// See those constants for why neither cap binds.
///
/// Returns `None` when no ladder reaches `security_bits`, for either of two reasons.
/// A level must have at least as many codeword positions as it opens queries, which rules out
/// small `log_n`.
/// And a level's correlated-agreement term must itself clear the target, which rules out large
/// `log_n` over a field this size.
/// [`crate::soundness`] explains why the second one cannot be bought back with more queries.
///
/// ## Panics
///
/// Panics if the preconditions below are violated.
///
/// ## Preconditions
///
/// * `l0_log_inv_rate` is in `1..=MAX_LOG_INV_RATE`.
/// * `security_bits` is positive.
pub(super) fn optimal_ladder<F, MerkleScheme>(
	merkle_scheme: &MerkleScheme,
	log_n: usize,
	l0_log_inv_rate: usize,
	regime: SoundnessRegime,
	security_bits: usize,
) -> Option<(LigeritoParams, usize)>
where
	F: BinaryField,
	MerkleScheme: MerkleTreeScheme<F>,
{
	assert!(
		(1..=MAX_LOG_INV_RATE).contains(&l0_log_inv_rate),
		"precondition: l0_log_inv_rate must be in 1..={MAX_LOG_INV_RATE}, got {l0_log_inv_rate}"
	);
	assert!(security_bits > 0, "precondition: security_bits must be positive");

	// Query counts depend only on the rate, so derive them once. Index 0 is unused: rate 1 is
	// not a proximity test at all.
	let search = Search::<F, MerkleScheme> {
		merkle_scheme,
		sizes: ByteSizes::new::<F, MerkleScheme>(),
		n_queries: (0..=MAX_LOG_INV_RATE)
			.map(|log_inv_rate| match log_inv_rate {
				0 => 0,
				_ => regime.n_queries(security_bits, log_inv_rate),
			})
			.collect(),
		regime,
		security_bits,
		_field: PhantomData,
	};

	// `best[log_total][rate_floor]`. The rate floor runs to `MAX_LOG_INV_RATE + 1`, one past
	// the last usable rate, so that a level committed at the last rate can look up "no level
	// may follow" without a bounds check.
	let mut best = vec![vec![None::<Decision>; MAX_LOG_INV_RATE + 2]; log_n + 1];
	for log_total in 0..=log_n {
		for rate_floor in 1..=MAX_LOG_INV_RATE {
			// Every tabulated subproblem is reached by recursing, so it is never level 0.
			best[log_total][rate_floor] =
				search.choose_level(&best, log_total, rate_floor, MAX_LOG_INV_RATE, DEEPER_LEVEL);
		}
	}

	// Level 0 is the same subproblem with its rate pinned to a single value.
	let root = search.choose_level(&best, log_n, l0_log_inv_rate, l0_log_inv_rate, 0)?;

	let estimated_size = root.cost;

	let mut levels = Vec::new();
	let mut decision = root;
	let mut log_total = log_n;
	loop {
		let log_msg_cols = log_total - decision.log_lanes;
		levels.push(LigeritoLevel {
			log_msg_cols,
			log_lanes: decision.log_lanes,
			log_inv_rate: decision.log_inv_rate,
			n_queries: search.n_queries[decision.log_inv_rate],
		});
		if !decision.recurse {
			break;
		}
		log_total = log_msg_cols;
		decision = best[log_msg_cols][decision.log_inv_rate + 1]
			.expect("a recursing decision was scored against a solved subproblem");
	}

	Some((LigeritoParams::new(levels, regime, security_bits), estimated_size))
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;
	use proptest::prelude::*;

	use super::*;
	use crate::{channel::OracleSpec, fri::FRIParams, merkle_tree::BinaryMerkleTreeScheme};

	type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

	fn test_merkle_scheme() -> TestMerkleScheme {
		BinaryMerkleTreeScheme::new()
	}

	const UDR: SoundnessRegime = SoundnessRegime::UniqueDecoding;
	const JOHNSON: SoundnessRegime = SoundnessRegime::Johnson { eta: 0.02 };

	/// The target `binius-verifier` ships, which the pinned tables below are sized to.
	///
	/// The plan document tabulates 100 bits instead. Over `B128` that target is out of reach for
	/// the larger shapes, which the infeasibility test below pins.
	const SECURITY_BITS: usize = 96;

	// Re-checks the constructor's invariants from the outside, so a ladder the search produced is
	// held to the same standard as one a caller wrote by hand.
	fn assert_invariants(params: &LigeritoParams) {
		let levels = params.levels();
		assert!(!levels.is_empty());
		for pair in levels.windows(2) {
			let (prev, next) = (&pair[0], &pair[1]);
			assert_eq!(next.log_msg_cols + next.log_lanes, prev.log_msg_cols);
			assert!(next.log_inv_rate > prev.log_inv_rate);
		}
		for level in levels {
			assert!(level.is_feasible());
			assert!(level.log_lanes >= 1);
			assert!(level.log_lanes <= MAX_LOG_LANES);
			assert!(level.log_inv_rate <= MAX_LOG_INV_RATE);
			assert_eq!(
				level.n_queries,
				params
					.regime()
					.n_queries(params.security_bits(), level.log_inv_rate)
			);
		}
		assert_eq!(params.log_residual_dim(), levels.last().expect("non-empty").log_msg_cols);
	}

	#[test]
	fn ladder_is_valid_and_its_estimate_is_exact() {
		let merkle_scheme = test_merkle_scheme();
		for regime in [SoundnessRegime::UniqueDecoding, JOHNSON] {
			for log_n in [12, 17, 20, 24, 28, 30] {
				for l0_log_inv_rate in [1, 2, 4] {
					let Some((params, estimate)) = LigeritoParams::optimal_ladder::<B128, _>(
						&merkle_scheme,
						log_n,
						l0_log_inv_rate,
						regime,
						SECURITY_BITS,
					) else {
						continue;
					};
					assert_invariants(&params);
					// Level 0's rate is the caller's to pin, and the message is fully covered.
					assert_eq!(params.levels()[0].log_inv_rate, l0_log_inv_rate);
					assert_eq!(params.log_msg_len(), log_n);
					// The search's own objective is the byte count, not an approximation of it.
					assert_eq!(estimate, params.proof_size(&merkle_scheme));
				}
			}
		}
	}

	// The plan document's table, priced by this repo's own Merkle scheme rather than by the
	// document's simplified model. L0's rate is pinned to 1/2, so L0 encoding costs exactly what
	// today's FRI costs and only the small deep levels drop the rate.
	#[test]
	fn pinned_ladder_sizes_at_l0_rate_one_half() {
		let merkle_scheme = test_merkle_scheme();

		// Level 0's lane count comes out at 3 or 4, matching the arity `fri`'s own optimizer picks.
		let expected = [
			(17, 161_488),
			(20, 236_944),
			(24, 348_272),
			(28, 472_928),
			(30, 542_800),
		];
		for (log_n, bytes) in expected {
			let (_, size) = LigeritoParams::optimal_ladder::<B128, _>(
				&merkle_scheme,
				log_n,
				1,
				UDR,
				SECURITY_BITS,
			)
			.expect("pinned shapes are feasible");
			assert_eq!(size, bytes, "log_n={log_n}");
		}
	}

	#[test]
	fn the_johnson_regime_has_no_ladder_over_this_field() {
		let merkle_scheme = test_merkle_scheme();

		// The reference implementation's `eta = 0.02` puts `m = ceil(sqrt(rho)/eta)` at 36 at rate
		// 1/2, and `m^5 * n / |F|` is then nowhere near the target. So there is no ladder at all.
		for log_n in [17, 20, 24, 28, 30] {
			let ladder = LigeritoParams::optimal_ladder::<B128, _>(
				&merkle_scheme,
				log_n,
				1,
				JOHNSON,
				SECURITY_BITS,
			);
			assert!(ladder.is_none(), "log_n={log_n}");
		}

		// Sweeping `eta` barely rescues it. Past `log_n = 20` no slack clears the target at all,
		// and at 20 the slack that does clear costs several times what unique decoding costs.
		for log_n in [24, 28, 30] {
			assert_eq!(SoundnessRegime::optimal_johnson(SECURITY_BITS, log_n, 1, 128), None);
		}
		let (_, swept) =
			SoundnessRegime::optimal_johnson(SECURITY_BITS, 20, 1, 128).expect("20 clears, barely");
		assert!(swept > 4 * UDR.n_queries(SECURITY_BITS, 1), "swept={swept}");
	}

	#[test]
	fn lower_l0_rate_gives_a_smaller_proof() {
		let merkle_scheme = test_merkle_scheme();
		// Unique decoding only: the Johnson regime has no feasible ladder over this field, which
		// the test above pins.
		for regime in [UDR] {
			for log_n in [17, 20, 24, 28] {
				// The trend holds while the ladder still has rate room below L0 to recurse into.
				let sizes = (1..=4)
					.map(|l0_log_inv_rate| {
						LigeritoParams::optimal_ladder::<B128, _>(
							&merkle_scheme,
							log_n,
							l0_log_inv_rate,
							regime,
							SECURITY_BITS,
						)
						.expect("feasible")
						.1
					})
					.collect::<Vec<_>>();
				// Lowering L0's rate pays down to 1/8 at every size measured.
				for pair in sizes[..3].windows(2) {
					assert!(pair[1] <= pair[0], "regime={regime:?} log_n={log_n} sizes={sizes:?}");
				}
				assert!(sizes[2] < sizes[0], "regime={regime:?} log_n={log_n} sizes={sizes:?}");
				// It stops paying at 1/16 for the largest shapes: the fold row union eats the
				// headroom the extra rate step would have bought.
				if log_n <= 24 {
					assert!(sizes[3] <= sizes[2], "regime={regime:?} log_n={log_n} {sizes:?}");
				}

				// It reverses at the far end, and the reason is the rate cap rather than the rate.
				// Pinning L0 at MAX_LOG_INV_RATE leaves no strictly lower rate for a second level,
				// so the ladder is forced to one level and a huge cleartext residual.
				let Some((capped, capped_size)) = LigeritoParams::optimal_ladder::<B128, _>(
					&merkle_scheme,
					log_n,
					MAX_LOG_INV_RATE,
					regime,
					SECURITY_BITS,
				) else {
					continue;
				};
				assert_eq!(capped.n_levels(), 1);
				assert!(capped_size > sizes[3], "regime={regime:?} log_n={log_n}");
			}
		}
	}

	#[test]
	fn single_level_ladder_prices_the_residual_in_the_clear() {
		let merkle_scheme = test_merkle_scheme();
		// One level, 2^9 columns by 2^3 lanes, rate 1/2, residual 2^9 elements in the clear.
		let level = LigeritoLevel::new(9, 3, 1, UDR, SECURITY_BITS);
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
		let deep = LigeritoLevel::new(10, 3, 2, UDR, SECURITY_BITS);
		let alone = LigeritoParams::new(vec![deep], UDR, SECURITY_BITS);
		let below = LigeritoParams::new(
			vec![LigeritoLevel::new(13, 3, 1, UDR, SECURITY_BITS), deep],
			UDR,
			SECURITY_BITS,
		);

		// Level 0 of the two-level ladder, priced on its own, plus the extra element per fold
		// round the deeper copy pays.
		let level_zero = LigeritoParams::new(
			vec![LigeritoLevel::new(13, 3, 1, UDR, SECURITY_BITS)],
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

	#[test]
	fn smallest_feasible_log_n() {
		let merkle_scheme = test_merkle_scheme();
		// At 96 bits and rate 1/2 the unique-decoding regime opens 232 queries, so a level needs
		// 2^8 codeword positions. With one lane folded away that puts the floor at log_n = 8.
		let (params, size) =
			LigeritoParams::optimal_ladder::<B128, _>(&merkle_scheme, 8, 1, UDR, SECURITY_BITS)
				.expect("log_n = 8 is the smallest feasible shape");
		assert_invariants(&params);
		// The search stops at one level, not because a second is infeasible but because it costs
		// more: the 2^7-element residual is cheaper than another root, rows, and multi-proof.
		assert_eq!(params.n_levels(), 1);
		assert_eq!(params.levels()[0].log_lanes, 1);
		assert_eq!(params.log_residual_dim(), 7);
		assert_eq!(size, params.proof_size(&merkle_scheme));
	}

	#[test]
	fn log_n_below_the_feasibility_floor_has_no_ladder() {
		// Every candidate level would open more rows than its codeword has positions.
		let ladder = LigeritoParams::optimal_ladder::<B128, _>(
			&test_merkle_scheme(),
			7,
			1,
			UDR,
			SECURITY_BITS,
		);
		assert!(ladder.is_none());
	}

	#[test]
	fn the_correlated_agreement_ceiling_bounds_log_n_from_above() {
		let merkle_scheme = test_merkle_scheme();

		// Over B128 the ceiling falls one bit per doubling, so a target picks out a largest shape.
		// At 96 bits and L0 rate 1/2 nothing past log_n = 32 has a ladder at all.
		let ladder = |log_n, target| {
			LigeritoParams::optimal_ladder::<B128, _>(&merkle_scheme, log_n, 1, UDR, target)
				.is_some()
		};
		assert!(ladder(32, 96));
		assert!(!ladder(33, 96));

		// And the cutoff is not a cliff: the shape degenerates well before it. At log_n = 32 the
		// only levels that still clear the target fold one lane at a time, so the search is forced
		// into a huge cleartext residual rather than a ladder. Pinning that keeps a caller from
		// reading the returned size as a usable configuration.
		let (_, degenerate) =
			LigeritoParams::optimal_ladder::<B128, _>(&merkle_scheme, 32, 1, UDR, SECURITY_BITS)
				.expect("log_n = 32 still has a ladder, of a sort");
		let (_, sane) =
			LigeritoParams::optimal_ladder::<B128, _>(&merkle_scheme, 30, 1, UDR, SECURITY_BITS)
				.expect("log_n = 30 has a real ladder");
		assert!(degenerate > 100 * sane, "degenerate={degenerate} sane={sane}");

		// Raising the target lowers that boundary, which is the whole point of tracking the term.
		assert!(ladder(28, 100));
		assert!(!ladder(29, 100));

		// And 128 bits is out of reach at every size, as `crate::soundness` documents.
		for log_n in [12, 20, 28] {
			assert!(!ladder(log_n, 128), "log_n={log_n}");
		}
	}

	// The full 2x2 of {FRI, Ligerito} x {unique decoding, Johnson}, all priced by this repo's own
	// estimators. Run with `--nocapture` to read the table.
	//
	// The Johnson rows price the *query phase only*, which is what the plan document tabulates.
	// Over this field that regime has no correlated-agreement headroom at all, so its Ligerito
	// cell reads infeasible and its FRI cells are a lower bound on bytes rather than a shippable
	// configuration. `crate::soundness` and `PROXIMITY_GAPS.md` carry the reason.
	//
	// The regime enters the FRI side only through the query count, so feeding it a Johnson count
	// is the entire change needed to price FRI in that regime.
	//
	// Two things to read carefully in the output. Rows (2) and (4) lower the rate at L0 too, so
	// they are not comparable on prover time to rows (1), (3), (5), or (6): L0 encoding scales
	// with the codeword, and rate 1/8 measures ~3.75x rate 1/2. And row (4)'s winner sits at the
	// top of the swept range, so that row is bounded by the sweep, not by an interior optimum.
	//
	// Byte counts are deliberately not pinned: six configurations by five sizes is a lot of
	// literals to maintain, and each one would break on any retuning of either search. Only the
	// orderings are asserted, and only the ones that hold at every size measured.
	#[test]
	fn fri_versus_ligerito_table() {
		let merkle_scheme = test_merkle_scheme();

		// FRI at one fixed rate, a single non-ZK oracle, its own proof-size-minimizing arities.
		let fri_size = |log_n: usize, log_inv_rate: usize, regime: SoundnessRegime| {
			let (params, _) = FRIParams::<B128>::optimal_for_batch(
				&merkle_scheme,
				&[OracleSpec::new(log_n)],
				log_inv_rate,
				regime.n_queries(100, log_inv_rate),
			);
			params.proof_size(&merkle_scheme)
		};
		// The best constant rate FRI can pick, paired with the rate that won it.
		let fri_best = |log_n: usize, regime: SoundnessRegime| {
			(1..=MAX_LOG_INV_RATE)
				.map(|log_inv_rate| (fri_size(log_n, log_inv_rate, regime), log_inv_rate))
				.min()
				.expect("the rate range is non-empty")
		};
		// A ladder exists exactly when some level 0 is feasible, which is cheap to check up front.
		// Checking it here keeps an infeasible cell a printed row rather than a panic.
		let ladder_feasible = |log_n: usize, l0: usize, regime: SoundnessRegime| {
			(1..=MAX_LOG_LANES.min(log_n)).any(|lanes| {
				LigeritoLevel::new(log_n - lanes, lanes, l0, regime, 100).is_feasible()
			})
		};

		println!();
		println!(
			"{:>5}  {:<31}  {:>9}  {:>7}  {:>7}",
			"log_n", "configuration", "bytes", "KiB", "vs (1)"
		);
		for log_n in [17usize, 20, 24, 28, 30] {
			let baseline = fri_size(log_n, 1, UDR);
			let (best_udr, best_udr_rate) = fri_best(log_n, UDR);
			let (best_john, best_john_rate) = fri_best(log_n, JOHNSON);
			let ladder = |regime| {
				ladder_feasible(log_n, 1, regime)
					.then(|| {
						LigeritoParams::optimal_ladder::<B128, _>(
							&merkle_scheme,
							log_n,
							1,
							regime,
							SECURITY_BITS,
						)
					})
					.flatten()
					.map(|(_, size)| size)
			};

			let rows = [
				("(1) FRI       UDR      rate 1/2".to_owned(), Some(baseline)),
				(format!("(2) FRI       UDR      rate 1/{}", 1 << best_udr_rate), Some(best_udr)),
				("(3) FRI       Johnson  rate 1/2".to_owned(), Some(fri_size(log_n, 1, JOHNSON))),
				(format!("(4) FRI       Johnson  rate 1/{}", 1 << best_john_rate), Some(best_john)),
				("(5) Ligerito  UDR      L0 1/2".to_owned(), ladder(UDR)),
				("(6) Ligerito  Johnson  L0 1/2".to_owned(), ladder(JOHNSON)),
			];
			for (label, size) in &rows {
				match size {
					Some(bytes) => println!(
						"{log_n:>5}  {label:<31}  {bytes:>9}  {:>7.1}  {:>6.2}x",
						*bytes as f64 / 1024.0,
						baseline as f64 / *bytes as f64,
					),
					None => println!("{log_n:>5}  {label:<31}  {:>9}", "infeasible"),
				}
			}

			let lig_udr = ladder(UDR).expect("a unique-decoding ladder is feasible at these sizes");

			// A ladder beats FRI at the same L0 rate, in the same regime. This is the honest
			// comparison, since equal L0 rate means equal L0 encoding cost.
			assert!(lig_udr < baseline, "log_n={log_n} lig={lig_udr} fri={baseline}");
			// Rows (3), (4) and (6) price a regime this field cannot support. They are printed
			// because the plan document tabulates them, and asserted only on their query counts.
			assert!(fri_size(log_n, 1, JOHNSON) < baseline, "log_n={log_n}");
			assert_eq!(ladder(JOHNSON), None, "log_n={log_n}");
			// Rows (2) and (4) are deliberately NOT asserted against the ladder. They can and do
			// beat it, by lowering the rate at L0 too, which is exactly where the encoding cost
			// lives. Only their weak ordering against their own rate-1/2 row is a real invariant.
			assert!(best_udr <= baseline, "log_n={log_n}");
			assert!(best_john <= fri_size(log_n, 1, JOHNSON), "log_n={log_n}");
		}
	}

	proptest! {
		#[test]
		fn ladder_always_satisfies_the_invariants(
			log_n in 12usize..=32,
			l0_log_inv_rate in 1usize..=4,
			eta in 0.005f64..0.1,
			use_johnson in any::<bool>(),
			security_bits in 80usize..=128,
		) {
			let regime = if use_johnson {
				SoundnessRegime::Johnson { eta }
			} else {
				SoundnessRegime::UniqueDecoding
			};
			let merkle_scheme = test_merkle_scheme();
			let Some((params, estimate)) = LigeritoParams::optimal_ladder::<B128, _>(
				&merkle_scheme,
				log_n,
				l0_log_inv_rate,
				regime,
				security_bits,
			) else {
				// Infeasible shapes are a documented outcome, not a failure to search.
				return Ok(());
			};
			assert_invariants(&params);
			prop_assert_eq!(params.log_msg_len(), log_n);
			prop_assert_eq!(estimate, params.proof_size(&merkle_scheme));
		}

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
