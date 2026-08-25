// Copyright 2026 The Binius Developers

use binius_field::BinaryField;
use getset::CopyGetters;

use super::{size_estimation, verifier_cost, verifier_cost::VerifierCost};
use crate::{
	merkle_tree::MerkleTreeScheme,
	soundness::{Grinding, SoundnessRegime},
};

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
	/// Query grinding closes part of the target before a single row is opened.
	/// So the rows only have to cover what is left of `security_bits`.
	/// Challenge grinding buys nothing here, because it moves the term the query count cannot.
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
		grinding: Grinding,
	) -> Self {
		let from_queries = security_bits.saturating_sub(grinding.query_bits());
		Self {
			log_msg_cols,
			log_lanes,
			log_inv_rate,
			n_queries: regime.n_queries(from_queries, log_inv_rate),
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
	/// Positions are sampled independently, so the same one can come up twice.
	/// The query count is derived under exactly that model: `n_queries` solves `(1 - delta)^t`.
	/// Nothing in it assumes the draws are distinct.
	///
	/// So a level opening more rows than its codeword has is not unsound, merely incoherent.
	/// The ladder search declines to price such a shape.
	pub const fn is_feasible(&self) -> bool {
		self.n_queries <= pow2_saturating(self.log_codeword_len())
	}

	/// Proof-of-work nonces this level writes, at the given grinding depth.
	///
	/// One stands before each of the `log_lanes` fold challenges.
	/// One more stands after the level's commitment goes out and before its queries are drawn.
	/// A depth of zero is not a grind and writes nothing.
	/// So an ungrinding ladder writes no nonces at all.
	pub const fn n_grind_nonces(&self, grinding: Grinding) -> usize {
		let challenge = match grinding.challenge_bits() {
			0 => 0,
			_ => self.log_lanes,
		};
		let query = match grinding.query_bits() {
			0 => 0,
			_ => 1,
		};
		challenge + query
	}
}

/// Parameters for the Ligerito recursive matrix-commitment protocol.
///
/// ## Invariants
///
/// Level `i`'s fields are abbreviated `cols_i`, `lanes_i`, `inv_rate_i`, and `queries_i` below.
/// [`Self::new`] enforces:
///
/// - `levels` is non-empty.
/// - Column counts chain: `cols_{i+1} + lanes_{i+1} == cols_i`.
/// - The rate falls down the ladder: `inv_rate_{i+1} > inv_rate_i`.
/// - Every level is feasible: `2^(cols_i + inv_rate_i) >= queries_i`.
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
pub struct LigeritoParams {
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
	/// The proof of work each level pays, split by the term each half buys back.
	#[getset(get_copy = "pub")]
	grinding: Grinding,
}

impl LigeritoParams {
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
			// Invariant: a level never opens more rows than its codeword has positions.
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
			grinding: Grinding::NONE,
		}
	}

	/// The same ladder, with a proof of work paid at every level.
	///
	/// [`Self::new`] leaves this at [`Grinding::NONE`].
	/// So a caller that says nothing writes the transcript an ungrinding protocol writes.
	///
	/// The query counts are not re-derived here.
	/// Setting a query grind therefore leaves the rows opening more than the target needs.
	/// [`Self::optimal_ladder`] is the one that sizes them against the grind.
	pub const fn with_grinding(mut self, grinding: Grinding) -> Self {
		self.grinding = grinding;
		self
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

	/// log2 the length of the longest codeword any level of this ladder commits.
	///
	/// One additive transform serves the whole ladder, and this is the domain it must cover.
	/// The levels encode over nested Gao-Mateer domains, so the largest of them covers them all.
	/// That is normally level 0, whose codeword is the longest.
	/// A level that drops the rate without folding any lanes has a longer one still.
	pub fn max_log_codeword_len(&self) -> usize {
		self.levels
			.iter()
			.map(LigeritoLevel::log_codeword_len)
			.max()
			.expect("levels is non-empty")
	}

	/// The ceiling the correlated-agreement term puts on this ladder, over a field of that size.
	///
	/// Every level is an independent proximity test, so the ladder is only as sound as its worst
	/// one.
	/// That is normally level 0, whose codeword is the longest.
	///
	/// No number of queries raises this.
	/// See [`crate::soundness`] for why, and `PROXIMITY_GAPS.md` for where it falls in practice.
	///
	/// Proof of work is the one thing that does raise it, and it is not counted here.
	/// [`Self::achieved_security_bits`] adds it, on the side it belongs to.
	///
	/// A level folding `2^log_lanes` interleaved lanes applies the underlying bound once per
	/// lane-fold round, so it pays a row union that the single-step bound does not carry.
	/// The two reference implementations disagree on the size of that union, charging a factor
	/// `log_lanes` and `2^(log_lanes - 1)` respectively.
	/// This charges the second, which is the pessimistic one.
	pub fn correlated_agreement_bits(&self, log_field_size: usize) -> f64 {
		self.levels
			.iter()
			.map(|level| {
				let base = self.regime.correlated_agreement_bits(
					level.log_msg_len(),
					level.log_inv_rate,
					log_field_size,
				);
				// The worst of the `log_lanes` fold rounds pays `2^(log_lanes - 1)`.
				base - (level.log_lanes.saturating_sub(1)) as f64
			})
			.fold(f64::INFINITY, f64::min)
	}

	/// The security this ladder reaches over a field of that size, counting both terms.
	///
	/// This is the worse of [`Self::correlated_agreement_bits`] and the query-phase soundness.
	/// Each of the two is credited with the half of [`Self::grinding`] that buys it.
	/// It can fall below [`Self::security_bits`], which is the *target* the queries were sized to.
	///
	/// A level grinds before every fold challenge.
	/// So every level's algebraic term gains the same [`Grinding::challenge_bits`].
	/// A level grinds once more before its queries are drawn.
	/// That is where [`Grinding::query_bits`] lands.
	pub fn achieved_security_bits(&self, log_field_size: usize) -> f64 {
		let algebra =
			self.correlated_agreement_bits(log_field_size) + self.grinding.challenge_bits() as f64;
		// The same grind stands before every level's queries, so crediting it to the worst level
		// is crediting it to all of them.
		let queries = self
			.levels
			.iter()
			.map(|level| level.n_queries as f64 * self.regime.bits_per_query(level.log_inv_rate))
			.fold(f64::INFINITY, f64::min)
			+ self.grinding.query_bits() as f64;
		algebra.min(queries)
	}

	/// The exact byte-size of a Ligerito proof at these parameters, without running the prover.
	///
	/// Counted on the message channel:
	/// - one Merkle root per committed level.
	///
	/// Counted on the decommitment channel, per level:
	/// - the opened rows, `n_queries * 2^log_lanes` field elements;
	/// - one Merkle multi-proof over the level's codeword positions.
	///
	/// Plus the sumcheck transcript: one element per fold round of level 0, two per round below it.
	/// Plus the residual: its commitment, then its elements in the clear.
	pub fn proof_size<F, VCS>(&self, vcs: &VCS) -> usize
	where
		F: BinaryField,
		VCS: MerkleTreeScheme<F>,
	{
		size_estimation::proof_size(self, vcs)
	}

	/// What checking a proof at these parameters costs the verifier, one row per level.
	///
	/// The rows are the committed levels in ladder order.
	/// One final row follows them, for the cleartext residual.
	///
	/// Counted from the ladder alone, the way the byte-size estimate is.
	/// So a shape can be priced before anyone implements it.
	///
	/// The units are the ones a recursion circuit pays in.
	/// Hash calls, and the bit decompositions a query index drives.
	/// The residual's row is the only one that grows with a power of two.
	pub fn verifier_cost<F, VCS>(&self, vcs: &VCS) -> Vec<VerifierCost>
	where
		F: BinaryField,
		VCS: MerkleTreeScheme<F>,
	{
		verifier_cost::verifier_cost(self, vcs)
	}

	/// The ladder minimizing [`Self::proof_size`], with level 0's rate pinned by the caller.
	///
	/// Level 0's rate is pinned because level 0's encoding dominates prover time.
	/// A ladder only compares to today's FRI at the same L0 rate, where it does the same L0 work.
	/// The deep levels are small, so they are free to drop the rate as far as the search likes.
	///
	/// Returns the parameters together with their [`Self::proof_size`] in bytes.
	///
	/// `None` means no ladder reaches `security_bits`, for either of two reasons.
	/// A level must have at least as many codeword positions as it opens queries, which rules out
	/// small `log_n`.
	/// And a level's algebraic term must itself clear the target, `grinding` included.
	/// That rules out large `log_n` over a field this size.
	/// See [`crate::soundness`] for why the second one cannot be bought back with more queries.
	///
	/// `grinding` is paid by the search rather than assumed away.
	/// Its challenge half raises the ceiling a level has to clear to be priced at all.
	/// Its query half shrinks the row count every level opens.
	/// Both halves cost nonce bytes, which the returned size counts.
	///
	/// ## Panics
	///
	/// Panics if `l0_log_inv_rate` is outside `1..=MAX_LOG_INV_RATE`, or `security_bits` is zero.
	pub fn optimal_ladder<F, MerkleScheme>(
		merkle_scheme: &MerkleScheme,
		log_n: usize,
		l0_log_inv_rate: usize,
		regime: SoundnessRegime,
		security_bits: usize,
		grinding: Grinding,
	) -> Option<(Self, usize)>
	where
		F: BinaryField,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		size_estimation::optimal_ladder(
			merkle_scheme,
			log_n,
			l0_log_inv_rate,
			regime,
			security_bits,
			grinding,
		)
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
	use binius_transcript::MAX_GRINDING_BITS;

	use super::*;

	// A three-level ladder that satisfies every invariant, used as the base for the rejection
	// tests below. Message 2^12, lanes 3/2/1, rates 1/2 -> 1/4 -> 1/8, residual 2^6.
	fn valid_levels() -> Vec<LigeritoLevel> {
		vec![
			LigeritoLevel::new(9, 3, 1, SoundnessRegime::UniqueDecoding, 100, Grinding::NONE),
			LigeritoLevel::new(7, 2, 2, SoundnessRegime::UniqueDecoding, 100, Grinding::NONE),
			LigeritoLevel::new(6, 1, 3, SoundnessRegime::UniqueDecoding, 100, Grinding::NONE),
		]
	}

	#[test]
	fn valid_ladder_is_accepted_and_residual_is_derived() {
		let params = LigeritoParams::new(valid_levels(), SoundnessRegime::UniqueDecoding, 100);
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
		LigeritoParams::new(Vec::new(), SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	#[should_panic(expected = "precondition: column counts must chain")]
	fn broken_column_chain_is_rejected() {
		let mut levels = valid_levels();
		// Level 1 should have 7 columns to absorb level 0's 9 minus its own 2 lanes.
		levels[1].log_msg_cols = 8;
		LigeritoParams::new(levels, SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	#[should_panic(expected = "precondition: the rate ladder must be strictly increasing")]
	fn flat_rate_ladder_is_rejected() {
		let mut levels = valid_levels();
		// Level 1 recommits at level 0's rate, so the recursion buys nothing.
		levels[1].log_inv_rate = 1;
		LigeritoParams::new(levels, SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	#[should_panic(expected = "precondition: level 2 is infeasible")]
	fn infeasible_level_is_rejected() {
		let mut levels = valid_levels();
		// 2^(6 + 3) = 512 positions cannot serve 513 distinct queries.
		levels[2].n_queries = 513;
		LigeritoParams::new(levels, SoundnessRegime::UniqueDecoding, 100);
	}

	#[test]
	fn the_transform_domain_covers_every_level() {
		// Codeword lengths of the three levels: 9 + 1, 7 + 2, 6 + 3 = 10, 9, 9.
		// Level 0 is the longest, which is the usual case.
		let params = LigeritoParams::new(valid_levels(), SoundnessRegime::UniqueDecoding, 100);
		assert_eq!(params.max_log_codeword_len(), 10);

		// A level that folds no lanes keeps its predecessor's column count and only drops the rate.
		// Its codeword is then longer than level 0's, so the maximum is not always level 0's.
		//
		//     level 0: 2^4 columns at rate 1/2  -> 2^5 positions
		//     level 1: 2^4 columns at rate 1/4  -> 2^6 positions
		let flat = vec![
			LigeritoLevel::new(4, 0, 1, SoundnessRegime::UniqueDecoding, 8, Grinding::NONE),
			LigeritoLevel::new(4, 0, 2, SoundnessRegime::UniqueDecoding, 8, Grinding::NONE),
		];
		let params = LigeritoParams::new(flat, SoundnessRegime::UniqueDecoding, 8);
		assert_eq!(params.max_log_codeword_len(), 6);
	}

	#[test]
	fn a_ladder_grinds_nothing_unless_it_is_asked_to() {
		// Invariant: the default has to be no proof of work at all, since a grinding transcript
		// and an ungrinding one are different protocols and only the caller knows which it wants.
		//
		// Fixture state: the three-level ladder above, built the ordinary way.
		let params = LigeritoParams::new(valid_levels(), SoundnessRegime::UniqueDecoding, 100);
		assert_eq!(params.grinding(), Grinding::NONE);
		for level in params.levels() {
			assert_eq!(level.n_grind_nonces(Grinding::NONE), 0);
		}
	}

	#[test]
	fn the_nonce_count_follows_the_two_call_sites() {
		// Invariant: a level grinds once before each of its fold challenges and once more before
		// its queries are drawn. So the two halves of a grind are told apart by count as well as
		// by position, everywhere but a level folding a single lane.
		//
		// Fixture state: level 0 of the ladder above, which folds three lanes.
		let level = valid_levels()[0];
		assert_eq!(level.log_lanes, 3);
		assert_eq!(level.n_grind_nonces(Grinding::new(4, 0)), 3);
		assert_eq!(level.n_grind_nonces(Grinding::new(0, 4)), 1);
		assert_eq!(level.n_grind_nonces(Grinding::new(4, 4)), 4);

		// A level that folds nothing has no fold challenge to stand before, so a challenge grind
		// writes nothing there while a query grind still writes its one nonce.
		let flat = LigeritoLevel {
			log_lanes: 0,
			..level
		};
		assert_eq!(flat.n_grind_nonces(Grinding::new(4, 0)), 0);
		assert_eq!(flat.n_grind_nonces(Grinding::new(0, 4)), 1);
	}

	#[test]
	fn the_two_grinds_are_credited_to_the_terms_they_buy() {
		// Invariant: charging one grind where the other belongs would report security nothing
		// produces. So the ladder credits the challenge half to the algebraic term and the query
		// half to the row term, and never the other way round.
		//
		// Fixture state: the ladder above at 100 bits over B128. Its message is small, so the
		// ceiling sits well above the target and the row term is the binding one. A query grind
		// therefore shows in the total and a challenge grind does not.
		let levels = valid_levels();
		let bare = LigeritoParams::new(levels, SoundnessRegime::UniqueDecoding, 100);
		let ceiling = bare.correlated_agreement_bits(128);
		let base = bare.achieved_security_bits(128);
		assert!(base < ceiling, "base={base} ceiling={ceiling}");

		// The ceiling itself never moves: it describes the code and the field, not the transcript.
		let query_ground = bare.clone().with_grinding(Grinding::new(0, 7));
		assert!((query_ground.correlated_agreement_bits(128) - ceiling).abs() < 1e-9);

		// What moves is the total, by exactly the bits ground, since the row term binds.
		let ground_bits = query_ground.achieved_security_bits(128);
		assert!((ground_bits - base - 7.0).abs() < 1e-9, "{ground_bits} {base}");
		assert!(ground_bits < ceiling, "the row term still binds after the grind");

		// The same depth on the other side buys nothing here, because it lands on the term that
		// was already the slacker of the two.
		let challenge_ground = bare.with_grinding(Grinding::new(7, 0));
		assert!((challenge_ground.achieved_security_bits(128) - base).abs() < 1e-9);
	}

	#[test]
	fn the_largest_grind_the_transcript_can_express_is_a_usable_ladder() {
		// Invariant: `Grinding::new` refuses a difficulty past what the transcript can sample, so
		// nothing downstream has to re-check the bound. What is left to pin is that the largest
		// one it does accept still describes a coherent ladder.
		//
		// Fixture state: the ladder above, ground as hard as the transcript allows.
		let grinding = Grinding::new(MAX_GRINDING_BITS, MAX_GRINDING_BITS);
		let params = LigeritoParams::new(valid_levels(), SoundnessRegime::UniqueDecoding, 100)
			.with_grinding(grinding);
		assert_eq!(params.grinding().challenge_bits(), MAX_GRINDING_BITS);
		assert_eq!(params.grinding().query_bits(), MAX_GRINDING_BITS);
		for level in params.levels() {
			assert_eq!(level.n_grind_nonces(grinding), level.log_lanes + 1);
		}

		// A query grind deeper than the target leaves the rows nothing to cover. That is a
		// misconfigured budget rather than a protocol, and it saturates rather than wrapping.
		let level = LigeritoLevel::new(
			9,
			3,
			1,
			SoundnessRegime::UniqueDecoding,
			MAX_GRINDING_BITS - 1,
			grinding,
		);
		assert_eq!(level.n_queries, 0);
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
