// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use getset::CopyGetters;

use crate::fri::calculate_n_test_queries;

/// Which proximity-testing regime the query counts are derived in.
///
/// A proximity test bounds the chance that a word `delta`-far from the code survives one query.
/// The two variants differ only in how far out `delta` is pushed:
///
/// - [`Self::UniqueDecoding`] stops at the unique-decoding radius, where the bound is a theorem.
/// - [`Self::Johnson`] goes out to the Johnson bound, where the bound is a conjecture.
///
/// The variant is part of [`LigeritoParams`], so no caller reaches the conjectured path unnamed.
/// And [`Self::UniqueDecoding`] is the [`Default`], so no conjecture is adopted by omission.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SoundnessRegime {
	/// Unique-decoding radius. No conjecture.
	///
	/// The proximity parameter is `delta = (1 - rho)/2`, half the code's relative distance.
	/// A word that far from the code has at most one codeword near it.
	/// One query then rules out `-log2(1 - delta)` bits of soundness error.
	/// The resulting count is exactly what FRI uses today; see [`calculate_n_test_queries`].
	/// [DP24] Section 5.2 carries the concrete analysis.
	///
	/// [DP24]: <https://eprint.iacr.org/2024/504>
	#[default]
	UniqueDecoding,
	/// Johnson list-decoding bound with out-of-domain binding. Conjecture-dependent.
	///
	/// The proximity parameter is pushed out to `delta = 1 - sqrt(rho) - eta`.
	/// One query then rules out `-log2(sqrt(rho) + eta)` bits of soundness error.
	/// Past the unique-decoding radius the list of nearby codewords is no longer a singleton.
	/// Soundness then needs *mutual correlated agreement*: WHIR's Conjecture 4.12 ([ACFY24]).
	/// [BCHKS25] Corollary 1.4 is the unique-decoding statement that conjecture leaves behind.
	/// [BCHKS25] Theorem 4.6 is the Johnson-regime statement it reaches for.
	///
	/// Selecting this variant adopts that conjecture.
	/// It also requires the protocol to bind the prover out of domain after every commitment.
	/// These parameters do not by themselves provide that binding.
	///
	/// [ACFY24]: <https://eprint.iacr.org/2024/1586>
	/// [BCHKS25]: <https://eprint.iacr.org/2025/2055>
	Johnson {
		/// The slack `eta > 0` held back from the Johnson bound, in units of relative distance.
		///
		/// Smaller values give fewer queries and lean on a looser conjecture.
		/// `0.02` reproduces the query counts of the reference implementation's configs.
		eta: f64,
	},
}

impl SoundnessRegime {
	/// Bits of soundness error one row query rules out at inverse rate `2^log_inv_rate`.
	///
	/// Write `rho = 2^-log_inv_rate` for the code's rate.
	/// [`Self::UniqueDecoding`] then gives `-log2(1 - (1 - rho)/2)`.
	/// [`Self::Johnson`] gives `-log2(sqrt(rho) + eta)`.
	///
	/// ## Preconditions
	///
	/// * `log_inv_rate >= 1`, since at rate 1 a query rules out nothing.
	/// * For [`Self::Johnson`], `eta > 0` and `sqrt(rho) + eta < 1`.
	pub fn bits_per_query(self, log_inv_rate: usize) -> f64 {
		assert!(log_inv_rate >= 1, "precondition: log_inv_rate must be at least 1");
		let rho = 2.0f64.powi(-(log_inv_rate as i32));
		let per_query_err = match self {
			// The same expression `calculate_n_test_queries` uses, so the two cannot drift.
			Self::UniqueDecoding => 0.5 * (1.0 + rho),
			Self::Johnson { eta } => {
				assert!(eta > 0.0, "precondition: eta must be positive");
				let err = rho.sqrt() + eta;
				assert!(err < 1.0, "precondition: sqrt(rho) + eta must be less than 1, got {err}");
				err
			}
		};
		-per_query_err.log2()
	}

	/// Number of distinct row queries needed for `security_bits` bits of query-phase soundness.
	///
	/// This is `ceil(security_bits / bits_per_query)`.
	/// It accounts for the query phase alone.
	/// The folding phase, the proximity-gap term, and the rest are budgeted elsewhere.
	///
	/// For [`Self::UniqueDecoding`] this delegates to [`calculate_n_test_queries`].
	/// So a Ligerito level in that regime costs exactly as many queries as an FRI round.
	///
	/// ## Preconditions
	///
	/// * Those of [`Self::bits_per_query`].
	pub fn n_queries(self, security_bits: usize, log_inv_rate: usize) -> usize {
		match self {
			Self::UniqueDecoding => {
				assert!(log_inv_rate >= 1, "precondition: log_inv_rate must be at least 1");
				calculate_n_test_queries(security_bits, log_inv_rate)
			}
			Self::Johnson { .. } => {
				(security_bits as f64 / self.bits_per_query(log_inv_rate)).ceil() as usize
			}
		}
	}
}

/// One committed level of the Ligerito recursion.
///
/// The level's message is a matrix of `2^log_lanes` interleaved lanes by `2^log_msg_cols` columns.
/// Every lane is Reed–Solomon encoded to `2^(log_msg_cols + log_inv_rate)` positions.
/// One Merkle leaf holds one codeword position across all lanes.
/// So the tree has `2^(log_msg_cols + log_inv_rate)` leaves of `2^log_lanes` elements each.
/// An opened row is therefore `2^log_lanes` field elements.
///
/// The `log_lanes` fold challenges of this level are also its sumcheck rounds.
/// That is why the lane count and the fold amount are the same number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LigeritoLevel {
	/// log2 the number of message columns this level's matrix has.
	pub log_msg_cols: usize,
	/// log2 the number of interleaved lanes, which is also this level's fold amount.
	pub log_lanes: usize,
	/// log2 the inverse Reed-Solomon rate this level is committed at.
	pub log_inv_rate: usize,
	/// Number of row queries opened against this level.
	pub n_queries: usize,
}

impl LigeritoLevel {
	/// A level whose query count is derived from `regime` at this level's own rate.
	///
	/// ## Preconditions
	///
	/// * Those of [`SoundnessRegime::n_queries`].
	pub fn new(
		log_msg_cols: usize,
		log_lanes: usize,
		log_inv_rate: usize,
		regime: SoundnessRegime,
		security_bits: usize,
	) -> Self {
		Self {
			log_msg_cols,
			log_lanes,
			log_inv_rate,
			n_queries: regime.n_queries(security_bits, log_inv_rate),
		}
	}

	/// log2 the number of codeword positions, which is also the number of Merkle leaves.
	pub const fn log_codeword_len(&self) -> usize {
		self.log_msg_cols + self.log_inv_rate
	}

	/// log2 the total number of message elements this level commits.
	pub const fn log_msg_len(&self) -> usize {
		self.log_msg_cols + self.log_lanes
	}

	/// Whether this level has at least as many codeword positions as queries to open.
	///
	/// Queries are sampled without replacement.
	/// A level with fewer positions than queries is not a protocol at all.
	pub const fn is_feasible(&self) -> bool {
		self.n_queries <= pow2_saturating(self.log_codeword_len())
	}
}

/// Parameters for the Ligerito recursive matrix-commitment protocol.
///
/// ## Invariants
///
/// Level `i`'s fields are abbreviated `cols_i`, `lanes_i`, `rate_i`, and `queries_i` below.
/// [`Self::new`] enforces:
///
/// - `levels` is non-empty.
/// - Column counts chain: `cols_{i+1} + lanes_{i+1} == cols_i`.
/// - The rate ladder strictly increases: `rate_{i+1} > rate_i`.
/// - Every level is feasible: `2^(cols_i + rate_i) >= queries_i`.
///
/// And derives, rather than taking on faith:
///
/// - `log_residual_dim == cols_last`.
///
/// That last relation is the whole recursion in one line.
/// A level holds `2^(cols + lanes)` elements and folds `lanes` of them away, leaving `2^cols`.
/// For an intermediate level that remainder is the next level's message, hence the chaining.
/// For the last level there is no next level, so it is the residual sent in the clear.
#[derive(Debug, Clone, CopyGetters)]
pub struct LigeritoParams<F> {
	/// The committed levels, outermost first. Guaranteed non-empty.
	levels: Vec<LigeritoLevel>,
	/// log2 the number of field elements in the residual matrix sent in the clear.
	#[getset(get_copy = "pub")]
	log_residual_dim: usize,
	/// The proximity-testing regime the per-level query counts were derived in.
	#[getset(get_copy = "pub")]
	regime: SoundnessRegime,
	/// The target query-phase soundness, in bits.
	#[getset(get_copy = "pub")]
	security_bits: usize,
	_marker: PhantomData<F>,
}

impl<F> LigeritoParams<F> {
	/// Assembles parameters from an explicit ladder, checking every invariant.
	///
	/// ## Preconditions
	///
	/// * All the invariants listed on [`LigeritoParams`].
	pub fn new(levels: Vec<LigeritoLevel>, regime: SoundnessRegime, security_bits: usize) -> Self {
		assert!(!levels.is_empty(), "precondition: levels must be non-empty");

		for (i, pair) in levels.windows(2).enumerate() {
			let (prev, next) = (&pair[0], &pair[1]);

			// Invariant: a level folds `log_lanes` of its dimensions away, and what remains is
			// exactly the next level's message.
			assert!(
				next.log_msg_cols + next.log_lanes == prev.log_msg_cols,
				"precondition: column counts must chain, but level {} has log_msg_cols {} + \
				 log_lanes {} != level {i}'s log_msg_cols {}",
				i + 1,
				next.log_msg_cols,
				next.log_lanes,
				prev.log_msg_cols,
			);

			// Invariant: the rate strictly decreases down the ladder. This is the whole point of
			// recommitting per level, so a flat or rising rate is a mis-specified ladder.
			assert!(
				next.log_inv_rate > prev.log_inv_rate,
				"precondition: the rate ladder must be strictly increasing in log_inv_rate, but \
				 level {} has {} and level {i} has {}",
				i + 1,
				next.log_inv_rate,
				prev.log_inv_rate,
			);
		}

		for (i, level) in levels.iter().enumerate() {
			// Invariant: queries are sampled without replacement, so there must be at least as
			// many codeword positions as queries.
			assert!(
				level.is_feasible(),
				"precondition: level {i} is infeasible, it opens {} queries against a codeword of \
				 2^{} positions",
				level.n_queries,
				level.log_codeword_len(),
			);
		}

		let log_residual_dim = levels.last().expect("levels is non-empty").log_msg_cols;

		Self {
			levels,
			log_residual_dim,
			regime,
			security_bits,
			_marker: PhantomData,
		}
	}

	/// The committed levels, outermost first. Non-empty.
	pub fn levels(&self) -> &[LigeritoLevel] {
		&self.levels
	}

	/// The number of committed levels, which is also the number of Merkle roots sent.
	pub const fn n_levels(&self) -> usize {
		self.levels.len()
	}

	/// log2 the number of field elements in the committed message.
	pub fn log_msg_len(&self) -> usize {
		self.levels[0].log_msg_len()
	}

	/// The total number of sumcheck fold rounds, `sum_i log_lanes_i`.
	pub fn n_fold_rounds(&self) -> usize {
		self.levels.iter().map(|level| level.log_lanes).sum()
	}
}

/// `2^log`, saturating at `usize::MAX` rather than wrapping on an out-of-range exponent.
const fn pow2_saturating(log: usize) -> usize {
	if log < usize::BITS as usize {
		1 << log
	} else {
		usize::MAX
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash as B128;

	use super::*;

	#[test]
	fn udr_query_counts_agree_with_fri() {
		for security_bits in [100, 128] {
			for log_inv_rate in 1..=8 {
				assert_eq!(
					SoundnessRegime::UniqueDecoding.n_queries(security_bits, log_inv_rate),
					calculate_n_test_queries(security_bits, log_inv_rate),
					"security_bits={security_bits} log_inv_rate={log_inv_rate}"
				);
			}
		}
	}

	#[test]
	fn udr_query_counts_at_100_bits() {
		// The table in the plan document: rates 1/2 through 1/32 at 100-bit security.
		let counts = (1..=5)
			.map(|log_inv_rate| SoundnessRegime::UniqueDecoding.n_queries(100, log_inv_rate))
			.collect::<Vec<_>>();
		assert_eq!(counts, vec![241, 148, 121, 110, 105]);
	}

	#[test]
	fn johnson_query_counts_at_100_bits() {
		let regime = SoundnessRegime::Johnson { eta: 0.02 };
		// Reproduces the shipped configurations of the reference implementation.
		assert_eq!(regime.n_queries(100, 1), 218);
		assert_eq!(regime.n_queries(100, 2), 106);
		assert_eq!(regime.n_queries(100, 3), 71);
	}

	#[test]
	fn johnson_needs_fewer_queries_than_unique_decoding() {
		let johnson = SoundnessRegime::Johnson { eta: 0.02 };
		for log_inv_rate in 1..=8 {
			let udr = SoundnessRegime::UniqueDecoding.n_queries(100, log_inv_rate);
			assert!(johnson.n_queries(100, log_inv_rate) < udr, "log_inv_rate={log_inv_rate}");
		}
	}

	#[test]
	fn default_regime_adopts_no_conjecture() {
		assert_eq!(SoundnessRegime::default(), SoundnessRegime::UniqueDecoding);
	}

	// A three-level ladder that satisfies every invariant, used as the base for the rejection
	// tests below. Message 2^12, lanes 3/2/1, rates 1/2 -> 1/4 -> 1/8, residual 2^6.
	fn valid_levels() -> Vec<LigeritoLevel> {
		vec![
			LigeritoLevel::new(9, 3, 1, SoundnessRegime::UniqueDecoding, 100),
			LigeritoLevel::new(7, 2, 2, SoundnessRegime::UniqueDecoding, 100),
			LigeritoLevel::new(6, 1, 3, SoundnessRegime::UniqueDecoding, 100),
		]
	}

	#[test]
	fn valid_ladder_is_accepted_and_residual_is_derived() {
		let params =
			LigeritoParams::<B128>::new(valid_levels(), SoundnessRegime::UniqueDecoding, 100);
		assert_eq!(params.n_levels(), 3);
		assert_eq!(params.log_msg_len(), 12);
		// The residual is the last level's remaining columns.
		assert_eq!(params.log_residual_dim(), 6);
		// Every fold challenge is a sumcheck round: 3 + 2 + 1.
		assert_eq!(params.n_fold_rounds(), 6);
		assert_eq!(params.regime(), SoundnessRegime::UniqueDecoding);
		assert_eq!(params.security_bits(), 100);
	}

	#[test]
	#[should_panic(expected = "precondition: levels must be non-empty")]
	fn empty_ladder_is_rejected() {
		LigeritoParams::<B128>::new(Vec::new(), SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	#[should_panic(expected = "precondition: column counts must chain")]
	fn broken_column_chain_is_rejected() {
		let mut levels = valid_levels();
		// Level 1 should have 7 columns to absorb level 0's 9 minus its own 2 lanes.
		levels[1].log_msg_cols = 8;
		LigeritoParams::<B128>::new(levels, SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	#[should_panic(expected = "precondition: the rate ladder must be strictly increasing")]
	fn flat_rate_ladder_is_rejected() {
		let mut levels = valid_levels();
		// Level 1 recommits at level 0's rate, so the recursion buys nothing.
		levels[1].log_inv_rate = 1;
		LigeritoParams::<B128>::new(levels, SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	#[should_panic(expected = "precondition: level 2 is infeasible")]
	fn infeasible_level_is_rejected() {
		let mut levels = valid_levels();
		// 2^(6 + 3) = 512 positions cannot serve 513 distinct queries.
		levels[2].n_queries = 513;
		LigeritoParams::<B128>::new(levels, SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	fn feasibility_is_exact_at_the_boundary() {
		let mut level = LigeritoLevel {
			log_msg_cols: 6,
			log_lanes: 1,
			log_inv_rate: 3,
			n_queries: 512,
		};
		assert!(level.is_feasible());
		level.n_queries = 513;
		assert!(!level.is_feasible());
	}
}
