// Copyright 2026 The Binius Developers

//! The search that picks a ladder: what the caller asks for, and the program that answers.

use std::marker::PhantomData;

use binius_field::BinaryField;
use getset::CopyGetters;

use super::{
	common::{LigeritoLevel, LigeritoParams},
	ladder_cost::LadderCost,
	ladder_objective::LadderObjective,
	size_estimation::{ByteSizes, DEEPER_LEVEL, level_size, residual_size},
};
use crate::{
	merkle_tree::MerkleTreeScheme,
	soundness::{Grinding, SoundnessRegime},
};

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

/// A request for the best ladder over a message, and the price list it is judged by.
///
/// Four of these fields describe the protocol the ladder has to be sound for.
/// The fifth, the objective, describes what "best" means, and it is the caller's to choose.
/// A search that says nothing about it minimizes proof bytes, and nothing else.
///
/// Level 0's rate is not searched over: the caller pins it.
/// Level 0 encodes the ladder's longest codeword, so its rate sets the prover's dominant cost.
/// A ladder is only comparable to a fold at the rate where the two do the same level-0 work.
/// Everything below level 0 is the search's to choose.
#[derive(Debug, Clone, Copy, PartialEq, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct LadderSearch {
	/// log2 the inverse rate level 0 is pinned to.
	l0_log_inv_rate: usize,
	/// The proximity-testing regime every query count and every ceiling is derived in.
	regime: SoundnessRegime,
	/// The target every level must reach, in bits.
	security_bits: usize,
	/// The proof of work every level pays, which both halves of the target are credited with.
	grinding: Grinding,
	/// What the search minimizes over the shapes that reach the target.
	objective: LadderObjective,
}

impl LadderSearch {
	/// A search with no proof of work, minimizing proof bytes alone.
	///
	/// ## Preconditions
	///
	/// * `l0_log_inv_rate` is in `1..=MAX_LOG_INV_RATE`.
	/// * `security_bits` is positive.
	pub fn new(l0_log_inv_rate: usize, regime: SoundnessRegime, security_bits: usize) -> Self {
		assert!(
			(1..=MAX_LOG_INV_RATE).contains(&l0_log_inv_rate),
			"precondition: l0_log_inv_rate must be in 1..={MAX_LOG_INV_RATE}, got \
			 {l0_log_inv_rate}"
		);
		assert!(security_bits > 0, "precondition: security_bits must be positive");
		Self {
			l0_log_inv_rate,
			regime,
			security_bits,
			grinding: Grinding::NONE,
			objective: LadderObjective::BYTES_ONLY,
		}
	}

	/// The same search, with a proof of work paid at every level.
	///
	/// The grind is priced rather than assumed away.
	/// Its challenge half raises the ceiling a level has to clear to be priced at all.
	/// Its query half shrinks the row count every level opens.
	/// Both halves cost nonce bytes, which the returned cost counts.
	pub const fn with_grinding(mut self, grinding: Grinding) -> Self {
		self.grinding = grinding;
		self
	}

	/// The same search, ranking shapes by something other than their bytes.
	pub const fn with_objective(mut self, objective: LadderObjective) -> Self {
		self.objective = objective;
		self
	}

	/// The best ladder over `2^log_n` message elements, and what it costs on both axes.
	///
	/// `None` means no ladder reaches the target, for either of two reasons.
	/// A level cannot open more queries than its codeword has positions, ruling out small messages.
	/// And a level's correlated-agreement term must clear the target on its own.
	/// Over a field this size that rules out the largest messages.
	/// [`crate::soundness`] explains why the second one cannot be bought back with more queries.
	///
	/// ## The program
	///
	/// This is a memoized dynamic program over `(log_total, rate_floor)`.
	/// A state is the message elements left, paired with the lowest rate the ladder still allows.
	/// It mirrors the search that picks FRI's fold arities, with two differences.
	///
	/// The first is that the state carries a rate floor at all.
	/// A Ligerito level chooses its own rate, where an FRI round cannot.
	/// So an FRI round minimizes over arity alone.
	/// Each state here minimizes over a two-dimensional grid of `(log_lanes, log_inv_rate)`.
	///
	/// The second is that the minimization is exhaustive rather than an early-exit scan.
	/// The lane axis was measured monotone, so an early exit would in fact be sound along it.
	/// That sweep covered every state up to `log_n = 32`, both regimes, at 100 and 128 bits.
	/// The scan stays exhaustive anyway, for two reasons.
	/// The minimization is genuinely two-dimensional, where an early exit walks one axis.
	/// And a state's low rates can be infeasible while its high ones are not, leaving a hole.
	/// An early-exit scan cannot cross that hole unless it is handed a filtered range first.
	/// Meanwhile a state costs at most [`MAX_LOG_LANES`] times [`MAX_LOG_INV_RATE`] evaluations.
	/// A shape assumption buys nothing at that price, so this code makes none.
	///
	/// ## What the program needs of the objective
	///
	/// Tabulating subproblems is exact only when a best ladder is built from best sub-ladders.
	/// Both axes of the cost are sums over levels, and the objective is linear in both.
	/// So a ladder's score is its first level's score plus the score of everything under it.
	/// A shape stored for a subproblem then stays optimal whatever is later stacked on top of it.
	///
	/// An objective that is not a weighted sum need not have that property.
	/// A cap on encoding work is the natural example, and a cap is not additive.
	/// A search under one would carry the budget left in its state rather than in its score.
	pub fn solve<F, MerkleScheme>(
		&self,
		merkle_scheme: &MerkleScheme,
		log_n: usize,
	) -> Option<(LigeritoParams, LadderCost)>
	where
		F: BinaryField,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		// Query counts depend only on the rate, so derive them once. Index 0 is unused: rate 1
		// is not a proximity test at all.
		//
		// The query grind closes part of the target before a row is opened, so the rows cover
		// only what is left of it.
		let from_queries = self
			.security_bits
			.saturating_sub(self.grinding.query_bits());
		let solver = Solver::<F, MerkleScheme> {
			search: self,
			merkle_scheme,
			sizes: ByteSizes::new::<F, MerkleScheme>(),
			n_queries: (0..=MAX_LOG_INV_RATE)
				.map(|log_inv_rate| match log_inv_rate {
					0 => 0,
					_ => self.regime.n_queries(from_queries, log_inv_rate),
				})
				.collect(),
			_field: PhantomData,
		};

		// `best[log_total][rate_floor]`. The rate floor runs to `MAX_LOG_INV_RATE + 1`, one past
		// the last usable rate, so that a level committed at the last rate can look up "no level
		// may follow" without a bounds check.
		let mut best = vec![vec![None::<Decision>; MAX_LOG_INV_RATE + 2]; log_n + 1];
		for log_total in 0..=log_n {
			for rate_floor in 1..=MAX_LOG_INV_RATE {
				// Every tabulated subproblem is reached by recursing, so it is never level 0.
				best[log_total][rate_floor] = solver.choose_level(
					&best,
					log_total,
					rate_floor,
					MAX_LOG_INV_RATE,
					DEEPER_LEVEL,
				);
			}
		}

		// Level 0 is the same subproblem with its rate pinned to a single value.
		let root =
			solver.choose_level(&best, log_n, self.l0_log_inv_rate, self.l0_log_inv_rate, 0)?;
		let cost = root.cost;

		let mut levels = Vec::new();
		let mut decision = root;
		let mut log_total = log_n;
		loop {
			let log_msg_cols = log_total - decision.log_lanes;
			levels.push(LigeritoLevel {
				log_msg_cols,
				log_lanes: decision.log_lanes,
				log_inv_rate: decision.log_inv_rate,
				n_queries: solver.n_queries[decision.log_inv_rate],
			});
			if !decision.recurse {
				break;
			}
			log_total = log_msg_cols;
			decision = best[log_msg_cols][decision.log_inv_rate + 1]
				.expect("a recursing decision was scored against a solved subproblem");
		}

		let params = LigeritoParams::new(levels, self.regime, self.security_bits)
			.with_grinding(self.grinding);
		Some((params, cost))
	}
}

/// One entry of the tabulation: the best continuation from a subproblem, and what it costs.
#[derive(Debug, Clone, Copy)]
struct Decision {
	/// Both prices of this level and everything after it, residual included.
	cost: LadderCost,
	/// That cost under the objective, which is the single number the program minimizes.
	score: f64,
	/// The lane count chosen for this level.
	log_lanes: usize,
	/// The inverse rate chosen for this level.
	log_inv_rate: usize,
	/// Whether a further committed level follows, rather than the residual terminating here.
	recurse: bool,
}

/// Everything the program needs that does not vary between subproblems.
struct Solver<'a, F, MerkleScheme> {
	/// The request whose target, regime, grinding and objective price every level.
	search: &'a LadderSearch,
	/// The Merkle scheme whose branch sizes price every level.
	merkle_scheme: &'a MerkleScheme,
	/// Serialized sizes of a digest and a field element.
	sizes: ByteSizes,
	/// Query count per `log_inv_rate`. Index 0 is unused: rate 1 is not a proximity test.
	n_queries: Vec<usize>,
	/// Ties `F` to the Merkle scheme without storing a value of it.
	_field: PhantomData<F>,
}

impl<F, MerkleScheme> Solver<'_, F, MerkleScheme>
where
	F: BinaryField,
	MerkleScheme: MerkleTreeScheme<F>,
{
	/// Whether a level can reach the requested security at all.
	///
	/// Two independent ways to fail, and neither is fixable by opening more rows.
	/// A level cannot sample more distinct positions than its codeword has.
	/// And its correlated-agreement term is a ceiling set by the codeword length and the field.
	/// Proof of work raises that ceiling, which is why the challenge grind is credited here.
	fn reaches_target(&self, level: &LigeritoLevel) -> bool {
		let base = self.search.regime.correlated_agreement_bits(
			level.log_msg_len(),
			level.log_inv_rate,
			F::N_BITS,
		);
		// Same row union `LigeritoParams::correlated_agreement_bits` charges, so the search and
		// the reported ceiling cannot disagree about which levels clear the target.
		//
		// The challenge grind is added on the same side `LigeritoParams::achieved_security_bits`
		// adds it, so a level priced here is a level that really reaches the target.
		let algebra = base - (level.log_lanes.saturating_sub(1)) as f64
			+ self.search.grinding.challenge_bits() as f64;
		level.is_feasible() && algebra >= self.search.security_bits as f64
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
			if chosen.is_none_or(|current: Decision| candidate.score < current.score) {
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
				// What this level costs on its own: the bytes it writes, and the encoding the
				// prover runs to write them.
				let here = LadderCost {
					bytes: level_size(
						&level,
						level_index,
						self.merkle_scheme,
						&self.sizes,
						self.search.grinding,
					),
					encode_butterflies: level.encode_butterflies(),
				};

				// Terminate: fold this level and send the remaining columns in the clear.
				consider(self.decide(
					here + LadderCost::from_bytes(residual_size(log_msg_cols, &self.sizes)),
					log_lanes,
					log_inv_rate,
					false,
				));

				// Or recurse: the remaining columns become the next level's message, committed at
				// a strictly lower rate.
				if let Some(next) = best[log_msg_cols][log_inv_rate + 1] {
					consider(self.decide(here + next.cost, log_lanes, log_inv_rate, true));
				}
			}
		}

		chosen
	}

	/// Scores one candidate continuation, so that the comparison never re-derives the score.
	const fn decide(
		&self,
		cost: LadderCost,
		log_lanes: usize,
		log_inv_rate: usize,
		recurse: bool,
	) -> Decision {
		Decision {
			cost,
			score: self.search.objective.bytes_equivalent(cost),
			log_lanes,
			log_inv_rate,
			recurse,
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;
	use proptest::prelude::*;

	use super::*;
	use crate::{
		channel::OracleSpec, fri::FRIParams, ligerito::VerifierCost,
		merkle_tree::BinaryMerkleTreeScheme,
	};

	type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

	fn test_merkle_scheme() -> TestMerkleScheme {
		BinaryMerkleTreeScheme::new()
	}

	const UDR: SoundnessRegime = SoundnessRegime::UniqueDecoding;
	const JOHNSON: SoundnessRegime = SoundnessRegime::Johnson { eta: 0.02 };

	/// The target `binius-verifier` ships, which the pinned tables below are sized to.
	///
	/// The plan document tabulates 100 bits instead.
	/// Over `B128` that target is out of reach for the larger shapes, as the test below pins.
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
			// Query grinding closes part of the target before a row is opened, so the rows only
			// cover what is left of it.
			let from_queries = params
				.security_bits()
				.saturating_sub(params.grinding().query_bits());
			assert_eq!(
				level.n_queries,
				params.regime().n_queries(from_queries, level.log_inv_rate)
			);
		}
		assert_eq!(params.log_residual_dim(), levels.last().expect("non-empty").log_msg_cols);
	}

	/// The lossy regime both reference Ligerito implementations ship a conjecture-free profile in.
	/// Priced here over this repo's own field.
	fn lossy_regime(log_n: usize) -> SoundnessRegime {
		let (regime, _) = SoundnessRegime::optimal_unique_decoding(120, log_n, 1, 128)
			.expect("120 bits is reachable with a constant loss");
		regime
	}

	#[test]
	fn a_ladder_that_grinds_reaches_a_target_the_algebra_alone_cannot() {
		// Invariant: the correlated-agreement term is a ceiling no query count raises, and over
		// B128 it falls short of 128 bits. Proof of work is the only lever that moves it, so a
		// 128-bit ladder exists exactly when enough of it is paid.
		//
		// Fixture state: a 2^24 message at L0 rate 1/2, in the lossy unique-decoding regime whose
		// ceiling is a flat 124.5 bits at every size.
		let merkle_scheme = test_merkle_scheme();
		let regime = lossy_regime(24);
		let ladder = |grinding| {
			LadderSearch::new(1, regime, 128)
				.with_grinding(grinding)
				.solve::<B128, _>(&merkle_scheme, 24)
		};

		// A level also pays the fold row union, so the ceiling a real ladder has to clear sits
		// below the flat 124.5 and the first grind that buys anything is larger than the 3.5 bits
		// the bare ceiling is short by.
		for challenge_bits in 0..8 {
			assert!(ladder(Grinding::new(challenge_bits, 0)).is_none(), "{challenge_bits}");
		}

		let grinding = Grinding::new(11, 0);
		let (params, cost) = ladder(grinding).expect("eleven bits of work reach 128");
		assert_invariants(&params);
		assert_eq!(params.grinding(), grinding);
		// The bits the ladder reports are the bits its transcript pays for, and they clear the
		// target rather than merely approaching it.
		assert!(params.achieved_security_bits(128) >= 128.0);
		assert_eq!(cost.bytes, params.proof_size(&merkle_scheme));
	}

	#[test]
	fn the_two_halves_of_the_grind_buy_different_parts_of_the_proof() {
		// Invariant: the two grinds are not interchangeable, and the search shows why. Challenge
		// bits raise the ceiling, which lets a level fold more lanes before the row union eats its
		// headroom. Query bits close part of the target outright, which lets every level open
		// fewer rows. Neither does the other's work.
		//
		// Fixture state: the same 2^24 message at 128 bits in the lossy regime.
		let merkle_scheme = test_merkle_scheme();
		let regime = lossy_regime(24);
		let ladder = |grinding| {
			LadderSearch::new(1, regime, 128)
				.with_grinding(grinding)
				.solve::<B128, _>(&merkle_scheme, 24)
				.expect("the profiles below all reach 128 bits")
				.0
		};

		// More challenge bits, wider lanes, and a smaller proof at every step.
		let widening = (8..=11)
			.map(|challenge_bits| ladder(Grinding::new(challenge_bits, 0)))
			.collect::<Vec<_>>();
		for pair in widening.windows(2) {
			assert!(pair[1].levels()[0].log_lanes >= pair[0].levels()[0].log_lanes);
			assert!(pair[1].proof_size(&merkle_scheme) < pair[0].proof_size(&merkle_scheme));
		}
		// And the row count never moves, because the query phase was never the binding term.
		let rows = widening[0].levels()[0].n_queries;
		for params in &widening {
			assert_eq!(params.levels()[0].n_queries, rows);
		}

		// Query bits do the opposite: the same shape, fewer rows, a smaller proof.
		let bare = ladder(Grinding::new(11, 0));
		// Seventeen bits is what one reference implementation grinds before drawing positions.
		let ground = ladder(Grinding::new(11, 17));
		assert_eq!(ground.n_levels(), bare.n_levels());
		assert_eq!(ground.levels()[0].log_lanes, bare.levels()[0].log_lanes);
		assert_eq!(bare.levels()[0].n_queries, 312);
		assert_eq!(ground.levels()[0].n_queries, 270);
		assert!(ground.proof_size(&merkle_scheme) < bare.proof_size(&merkle_scheme));
		assert!(ground.achieved_security_bits(128) >= 128.0);
	}

	#[test]
	fn grinding_a_term_that_does_not_bind_costs_bytes_and_buys_nothing() {
		// Invariant: proof of work is not free, and it only pays on the term that binds. At the
		// shipped target over B128 the query phase is the binding one, so challenge grinding buys
		// no bits at all and still writes a nonce per fold round. That is why `Grinding::NONE` is
		// the default rather than a small positive number.
		//
		// Fixture state: the shipped 96-bit target, unique decoding, a 2^24 message at rate 1/2.
		let merkle_scheme = test_merkle_scheme();
		let ladder = |grinding| {
			LadderSearch::new(1, UDR, 96)
				.with_grinding(grinding)
				.solve::<B128, _>(&merkle_scheme, 24)
				.expect("96 bits is reachable over a 2^24 message")
				.0
		};

		let (bare, ground) = (ladder(Grinding::NONE), ladder(Grinding::new(4, 0)));
		assert_eq!(bare.levels(), ground.levels());
		// One nonce per fold round, and nothing else changed.
		let nonces = ground.n_fold_rounds() * size_of::<u64>();
		assert_eq!(ground.proof_size(&merkle_scheme), bare.proof_size(&merkle_scheme) + nonces);
		// And the reported security does not move, because the algebraic term was already the
		// slacker of the two.
		let (bare_bits, ground_bits) =
			(bare.achieved_security_bits(128), ground.achieved_security_bits(128));
		assert!((bare_bits - ground_bits).abs() < 1e-9, "{bare_bits} {ground_bits}");
		assert!(bare_bits >= 96.0);
	}

	#[test]
	fn ladder_is_valid_and_its_estimate_is_exact() {
		let merkle_scheme = test_merkle_scheme();
		// The grinding profiles are swept alongside the shapes, since a ladder that grinds writes
		// nonces the estimate has to count.
		for grinding in [Grinding::NONE, Grinding::new(0, 17), Grinding::new(4, 0)] {
			for regime in [SoundnessRegime::UniqueDecoding, JOHNSON] {
				for log_n in [12, 17, 20, 24, 28, 30] {
					for l0_log_inv_rate in [1, 2, 4] {
						let Some((params, cost)) =
							LadderSearch::new(l0_log_inv_rate, regime, SECURITY_BITS)
								.with_grinding(grinding)
								.solve::<B128, _>(&merkle_scheme, log_n)
						else {
							continue;
						};
						assert_invariants(&params);
						// Level 0's rate is the caller's, and the message is fully covered.
						assert_eq!(params.levels()[0].log_inv_rate, l0_log_inv_rate);
						assert_eq!(params.log_msg_len(), log_n);
						assert_eq!(params.grinding(), grinding);
						// The search's own objective is the byte count, not an approximation.
						assert_eq!(cost.bytes, params.proof_size(&merkle_scheme));
					}
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
			let (_, cost) = LadderSearch::new(1, UDR, SECURITY_BITS)
				.solve::<B128, _>(&merkle_scheme, log_n)
				.expect("pinned shapes are feasible");
			assert_eq!(cost.bytes, bytes, "log_n={log_n}");
		}
	}

	#[test]
	fn the_johnson_regime_has_no_ladder_over_this_field() {
		let merkle_scheme = test_merkle_scheme();

		// The reference implementation's `eta = 0.02` puts `m = ceil(sqrt(rho)/eta)` at 36 at rate
		// 1/2, and `m^5 * n / |F|` is then nowhere near the target. So there is no ladder at all.
		for log_n in [17, 20, 24, 28, 30] {
			let ladder = LadderSearch::new(1, JOHNSON, SECURITY_BITS)
				.solve::<B128, _>(&merkle_scheme, log_n);
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
						LadderSearch::new(l0_log_inv_rate, regime, SECURITY_BITS)
							.solve::<B128, _>(&merkle_scheme, log_n)
							.expect("feasible")
							.1
							.bytes
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
				let Some((capped, capped_cost)) =
					LadderSearch::new(MAX_LOG_INV_RATE, regime, SECURITY_BITS)
						.solve::<B128, _>(&merkle_scheme, log_n)
				else {
					continue;
				};
				assert_eq!(capped.n_levels(), 1);
				assert!(capped_cost.bytes > sizes[3], "regime={regime:?} log_n={log_n}");
			}
		}
	}
	#[test]
	fn smallest_feasible_log_n() {
		let merkle_scheme = test_merkle_scheme();
		// At 96 bits and rate 1/2 the unique-decoding regime opens 232 queries, so a level needs
		// 2^8 codeword positions. With one lane folded away that puts the floor at log_n = 8.
		let (params, cost) = LadderSearch::new(1, UDR, SECURITY_BITS)
			.solve::<B128, _>(&merkle_scheme, 8)
			.expect("log_n = 8 is the smallest feasible shape");
		assert_invariants(&params);
		// The search stops at one level, not because a second is infeasible but because it costs
		// more: the 2^7-element residual is cheaper than another root, rows, and multi-proof.
		assert_eq!(params.n_levels(), 1);
		assert_eq!(params.levels()[0].log_lanes, 1);
		assert_eq!(params.log_residual_dim(), 7);
		assert_eq!(cost.bytes, params.proof_size(&merkle_scheme));
	}

	#[test]
	fn log_n_below_the_feasibility_floor_has_no_ladder() {
		// Every candidate level would open more rows than its codeword has positions.
		let ladder =
			LadderSearch::new(1, UDR, SECURITY_BITS).solve::<B128, _>(&test_merkle_scheme(), 7);
		assert!(ladder.is_none());
	}

	#[test]
	fn the_correlated_agreement_ceiling_bounds_log_n_from_above() {
		let merkle_scheme = test_merkle_scheme();

		// Over B128 the ceiling falls one bit per doubling, so a target picks out a largest shape.
		// At 96 bits and L0 rate 1/2 nothing past log_n = 32 has a ladder at all.
		let ladder = |log_n, target| {
			LadderSearch::new(1, UDR, target)
				.solve::<B128, _>(&merkle_scheme, log_n)
				.is_some()
		};
		assert!(ladder(32, 96));
		assert!(!ladder(33, 96));

		// And the cutoff is not a cliff: the shape degenerates well before it. At log_n = 32 the
		// only levels that still clear the target fold one lane at a time, so the search is forced
		// into a huge cleartext residual rather than a ladder. Pinning that keeps a caller from
		// reading the returned size as a usable configuration.
		let degenerate = LadderSearch::new(1, UDR, SECURITY_BITS)
			.solve::<B128, _>(&merkle_scheme, 32)
			.expect("log_n = 32 still has a ladder, of a sort")
			.1
			.bytes;
		let sane = LadderSearch::new(1, UDR, SECURITY_BITS)
			.solve::<B128, _>(&merkle_scheme, 30)
			.expect("log_n = 30 has a real ladder")
			.1
			.bytes;
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
		//
		// Both sides are sized to the same target. A comparison whose two arms are sized to
		// different targets is not a comparison of the schemes, it is a comparison of the targets.
		let fri_size = |log_n: usize, log_inv_rate: usize, regime: SoundnessRegime| {
			let (params, _) = FRIParams::<B128>::optimal_for_batch(
				&merkle_scheme,
				&[OracleSpec::new(log_n)],
				log_inv_rate,
				regime.n_queries(SECURITY_BITS, log_inv_rate),
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
				LigeritoLevel::new(log_n - lanes, lanes, l0, regime, 100, Grinding::NONE)
					.is_feasible()
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
						LadderSearch::new(1, regime, SECURITY_BITS)
							.solve::<B128, _>(&merkle_scheme, log_n)
					})
					.flatten()
					.map(|(_, cost)| cost.bytes)
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
	/// The byte-optimal ladder over a message of `2^log_n` elements, at level-0 rate 1/2.
	///
	/// This is what the shipped search returns, and what every priced search is read against.
	fn byte_optimal(
		merkle_scheme: &TestMerkleScheme,
		log_n: usize,
	) -> (LigeritoParams, LadderCost) {
		LadderSearch::new(1, UDR, SECURITY_BITS)
			.solve::<B128, _>(merkle_scheme, log_n)
			.expect("the sizes swept here are all feasible at 96 bits")
	}

	/// The ladder in the notation the plan document writes it in, residual included.
	fn shape(params: &LigeritoParams) -> String {
		let levels = params
			.levels()
			.iter()
			.map(|level| format!("({}, 1/{})", level.log_lanes, 1 << level.log_inv_rate))
			.collect::<Vec<_>>()
			.join(" -> ");
		format!("{levels}  res 2^{}", params.log_residual_dim())
	}

	/// The sizes both the plan document and the tables above are written at.
	const SWEPT_SIZES: [usize; 6] = [17, 20, 22, 24, 28, 30];

	#[test]
	fn saying_nothing_about_the_objective_minimizes_bytes() {
		// Invariant: the default search is the byte-optimal one, so a caller who never mentions
		// encoding gets the ladder a proof-size estimate alone picks. The pinned byte counts
		// above are that ladder's, and they are what a priced search is measured against.
		let merkle_scheme = test_merkle_scheme();
		for log_n in SWEPT_SIZES {
			let (params, cost) = byte_optimal(&merkle_scheme, log_n);
			let explicit = LadderSearch::new(1, UDR, SECURITY_BITS)
				.with_objective(LadderObjective::BYTES_ONLY)
				.solve::<B128, _>(&merkle_scheme, log_n)
				.expect("feasible");

			assert_eq!(explicit.0.levels(), params.levels(), "log_n={log_n}");
			assert_eq!(explicit.1, cost, "log_n={log_n}");
			// Both axes of the returned cost are the ladder's own, not the search's bookkeeping.
			assert_eq!(cost, params.ladder_cost(&merkle_scheme), "log_n={log_n}");
		}
	}

	// The plan document's own example, and the whole point of the priced objective in one row.
	// Byte-optimal it drops four rate steps below level 0 and then five more; priced, it drops
	// one and one, sends 4.3% more bytes, and encodes 22% less.
	#[test]
	fn pinned_ladders_over_a_message_of_four_million_elements() {
		let merkle_scheme = test_merkle_scheme();
		let (bytes_params, bytes_cost) = byte_optimal(&merkle_scheme, 22);
		assert_eq!(shape(&bytes_params), "(4, 1/2) -> (4, 1/16) -> (4, 1/32)  res 2^10");
		assert_eq!(bytes_cost.bytes, 290_432);
		assert_eq!(bytes_cost.encode_butterflies, 107_479_040);

		let (priced_params, priced_cost) = LadderSearch::new(1, UDR, SECURITY_BITS)
			.with_objective(LadderObjective::proportional(bytes_cost, 1.0))
			.solve::<B128, _>(&merkle_scheme, 22)
			.expect("feasible");
		assert_eq!(shape(&priced_params), "(4, 1/2) -> (4, 1/4) -> (4, 1/8)  res 2^10");
		assert_eq!(priced_cost.bytes, 303_040);
		assert_eq!(priced_cost.encode_butterflies, 83_492_864);

		// Level 0 is untouched: the caller pinned its rate, and the priced search kept its lanes.
		assert_eq!(priced_params.levels()[0], bytes_params.levels()[0]);
	}

	#[test]
	fn pricing_encoding_shortens_the_rate_ladder() {
		// Invariant: the deep rates the byte objective picks are bought with encoding it never
		// looks at. Charge for that encoding and the ladder stops dropping the rate so far, at
		// every size, for a few percent of the proof.
		//
		// Fixture state: an even trade, where a percentage point of the byte-optimal proof is
		// worth a percentage point of its encoding.
		let merkle_scheme = test_merkle_scheme();
		for log_n in SWEPT_SIZES {
			let (bytes_params, bytes_cost) = byte_optimal(&merkle_scheme, log_n);
			let (priced_params, priced_cost) = LadderSearch::new(1, UDR, SECURITY_BITS)
				.with_objective(LadderObjective::proportional(bytes_cost, 1.0))
				.solve::<B128, _>(&merkle_scheme, log_n)
				.expect("feasible");
			assert_invariants(&priced_params);

			// The deepest level commits at a strictly higher rate than the byte optimum's does.
			let deepest = |params: &LigeritoParams| {
				params
					.levels()
					.last()
					.expect("a ladder has a level")
					.log_inv_rate
			};
			assert!(
				deepest(&priced_params) < deepest(&bytes_params),
				"log_n={log_n} priced={} bytes={}",
				shape(&priced_params),
				shape(&bytes_params)
			);

			// Bytes can only rise, since the ladder it is measured against minimizes them.
			// A tenth of the proof is the most that costs at any size swept.
			assert!(priced_cost.bytes >= bytes_cost.bytes, "log_n={log_n}");
			assert!(
				priced_cost.bytes as f64 <= 1.10 * bytes_cost.bytes as f64,
				"log_n={log_n} {} vs {}",
				priced_cost.bytes,
				bytes_cost.bytes
			);
			// And what it buys is a sixth of the encoding at worst.
			assert!(
				priced_cost.encode_butterflies as f64
					<= 0.85 * bytes_cost.encode_butterflies as f64,
				"log_n={log_n} {} vs {}",
				priced_cost.encode_butterflies,
				bytes_cost.encode_butterflies
			);
		}
	}

	#[test]
	fn the_shorter_rate_ladder_does_not_cost_the_verifier() {
		// Invariant: the residual is what the verifier's hashing turns on, and a priced ladder is
		// free to leave it wider. It does not: the extra levels the byte objective bought with
		// encoding were not buying the verifier anything either, so trading them away is at worst
		// neutral and usually a saving.
		let merkle_scheme = test_merkle_scheme();
		for log_n in SWEPT_SIZES {
			let (bytes_params, bytes_cost) = byte_optimal(&merkle_scheme, log_n);
			let (priced_params, _) = LadderSearch::new(1, UDR, SECURITY_BITS)
				.with_objective(LadderObjective::proportional(bytes_cost, 1.0))
				.solve::<B128, _>(&merkle_scheme, log_n)
				.expect("feasible");

			let hashes = |params: &LigeritoParams| {
				VerifierCost::total(&params.verifier_cost(&merkle_scheme)).hash_calls()
			};
			let (priced, byte_optimal) = (hashes(&priced_params), hashes(&bytes_params));
			assert!(
				priced as f64 <= 1.02 * byte_optimal as f64,
				"log_n={log_n} priced={priced} byte_optimal={byte_optimal}"
			);
		}
	}

	#[test]
	fn each_objective_wins_under_its_own_score() {
		// Invariant: the dynamic program returns the true optimum for the objective it was
		// handed, so neither ladder can beat the other on the other's score. This is the property
		// tabulating subproblems would silently break if the score were not additive over levels.
		let merkle_scheme = test_merkle_scheme();
		for log_n in SWEPT_SIZES {
			let (_, bytes_cost) = byte_optimal(&merkle_scheme, log_n);
			let objective = LadderObjective::proportional(bytes_cost, 1.0);
			let (_, priced_cost) = LadderSearch::new(1, UDR, SECURITY_BITS)
				.with_objective(objective)
				.solve::<B128, _>(&merkle_scheme, log_n)
				.expect("feasible");

			assert!(
				objective.bytes_equivalent(priced_cost) <= objective.bytes_equivalent(bytes_cost),
				"log_n={log_n}"
			);
			let bytes = LadderObjective::BYTES_ONLY;
			assert!(bytes.bytes_equivalent(bytes_cost) <= bytes.bytes_equivalent(priced_cost));
		}
	}

	#[test]
	fn a_cheaper_encode_price_never_buys_more_encoding() {
		// Invariant: the feasible shapes do not depend on the price, so raising the encoding a
		// byte is worth can only move the optimum along the frontier. Encoding rises weakly and
		// bytes fall weakly, which is what makes the knob predictable to turn.
		//
		// Fixture state: an even trade, then eight times more encoding for the same byte.
		let merkle_scheme = test_merkle_scheme();
		for log_n in SWEPT_SIZES {
			let (_, bytes_cost) = byte_optimal(&merkle_scheme, log_n);
			let costs = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0].map(|encode_per_bytes| {
				LadderSearch::new(1, UDR, SECURITY_BITS)
					.with_objective(LadderObjective::proportional(bytes_cost, encode_per_bytes))
					.solve::<B128, _>(&merkle_scheme, log_n)
					.expect("feasible")
					.1
			});

			for pair in costs.windows(2) {
				assert!(pair[1].encode_butterflies >= pair[0].encode_butterflies, "log_n={log_n}");
				assert!(pair[1].bytes <= pair[0].bytes, "log_n={log_n}");
			}
			// The far end of the sweep is the byte objective itself, which charges nothing.
			assert!(costs.last().expect("non-empty").bytes >= bytes_cost.bytes);
		}
	}

	// The measurement that set the rate and the even proportional trade are two different ways to
	// price encoding, and they land on the same ladder at the size the measurement was taken.
	//
	// They do not stay together: a proof grows with the logarithm of the message and its encoding
	// grows with the message, so an absolute rate charges relatively more for encoding at every
	// size up. At 2^28 the measured rate is an even trade of about one byte for eight parts of
	// encoding.
	#[test]
	fn the_measured_rate_is_an_even_trade_where_it_was_measured() {
		let merkle_scheme = test_merkle_scheme();
		let (_, bytes_cost) = byte_optimal(&merkle_scheme, 22);
		let even = LadderObjective::proportional(bytes_cost, 1.0);

		// The two prices agree to within a tenth, which is closer than the measurement they come
		// from can distinguish.
		let ratio = even.bytes_per_butterfly() / LadderObjective::MEASURED.bytes_per_butterfly();
		assert!((0.9..=1.1).contains(&ratio), "ratio={ratio}");

		let ladder = |objective| {
			LadderSearch::new(1, UDR, SECURITY_BITS)
				.with_objective(objective)
				.solve::<B128, _>(&merkle_scheme, 22)
				.expect("feasible")
		};
		assert_eq!(ladder(even).0.levels(), ladder(LadderObjective::MEASURED).0.levels());
	}

	// Both objectives on both axes, at the sizes the plan document tabulates. Run with
	// `--nocapture` to read the table.
	//
	// The rows priced against the byte-optimal ladder are the ones to read: they say the same
	// thing at every size, where the measured absolute rate does not.
	//
	// Two things to read carefully. The byte column of a priced row can only rise, since the row
	// it is divided by is the byte minimum. And the cheapest trades are the ones near the top of
	// the price range, where a percent of proof buys tens of percent of encoding: the byte
	// objective is nearly flat in the encode direction around its own optimum, so most of the
	// encoding it spends is buying it almost nothing.
	#[test]
	fn objective_comparison_table() {
		let merkle_scheme = test_merkle_scheme();

		println!();
		println!(
			"{:>5}  {:<10}  {:<48}  {:>9}  {:>7}  {:>13}  {:>7}",
			"log_n", "objective", "ladder", "bytes", "vs (b)", "butterflies", "vs (b)"
		);
		for log_n in SWEPT_SIZES {
			let (_, bytes_cost) = byte_optimal(&merkle_scheme, log_n);
			for (label, objective) in [
				("bytes".to_owned(), LadderObjective::BYTES_ONLY),
				("measured".to_owned(), LadderObjective::MEASURED),
				("even".to_owned(), LadderObjective::proportional(bytes_cost, 1.0)),
				("4x encode".to_owned(), LadderObjective::proportional(bytes_cost, 4.0)),
			] {
				let (params, cost) = LadderSearch::new(1, UDR, SECURITY_BITS)
					.with_objective(objective)
					.solve::<B128, _>(&merkle_scheme, log_n)
					.expect("feasible");
				println!(
					"{log_n:>5}  {label:<10}  {:<48}  {:>9}  {:>6.3}x  {:>13}  {:>6.3}x",
					shape(&params),
					cost.bytes,
					cost.bytes as f64 / bytes_cost.bytes as f64,
					cost.encode_butterflies,
					cost.encode_butterflies as f64 / bytes_cost.encode_butterflies as f64,
				);
			}
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
			let search = LadderSearch::new(l0_log_inv_rate, regime, security_bits);
			let Some((params, cost)) = search.solve::<B128, _>(&merkle_scheme, log_n) else {
				// Infeasible shapes are a documented outcome, not a failure to search.
				return Ok(());
			};
			assert_invariants(&params);
			prop_assert_eq!(params.log_msg_len(), log_n);
			prop_assert_eq!(cost.bytes, params.proof_size(&merkle_scheme));
		}

	}
}
