// Copyright 2026 The Binius Developers

//! The Ligerito opening protocol on the verifier side.
//!
//! The ladder proves an evaluation claim `<X_0, eq(z)> = y` about the multilinear level 0
//! committed. `X_0` is held as `2^log_lanes` interleaved lanes of `2^log_msg_cols` columns, and
//! each lane is a Reed-Solomon codeword of the same code.
//! Level `i` folds its lanes away and the remainder becomes level `i + 1`'s message, committed at a
//! strictly lower rate, until the last level's remainder is small enough to send in the clear.
//!
//! # Why the lane fold and the sumcheck are the same challenges
//!
//! A sumcheck binds one variable per round, highest first.
//! Running it for `log_lanes` rounds binds exactly the variables the lane index occupies, so the
//! claim it leaves behind is about the lane-folded matrix
//!
//! ```text
//!     X_1[j] = sum_lane X_0[lane][j] * eq(lane, r)
//! ```
//!
//! and `r` is the same tensor the queries have to be folded by.
//! There is no separate folding phase because there is nothing left to fold.
//!
//! # Why an opened row is a claim about the next message
//!
//! Every lane is an independent codeword of one code, which
//! [`binius_math::ntt::subspace_polys`] pins against `ReedSolomonCode::encode_batch`.
//! Encoding is linear, so folding across lanes and encoding commute:
//!
//! ```text
//!     sum_lane cw_lane[x] * eq(lane, r)  =  (G . X_1)[x]
//! ```
//!
//! An opened row at position `q`, folded by `r`, is therefore exactly `<G_q, X_1>`.
//! That is one linear claim about `X_1` per query, and [`InducedBasis`] batches them into one.
//!
//! # Why the next commitment goes out before the queries
//!
//! Whatever a level reduces to is bound before any of that level's query positions are sampled.
//! For an intermediate level that is the next level's Merkle root.
//! For the last level it is the residual, sent in the clear against its own commitment.
//! A prover therefore has to fix what it folded to without knowing which rows will be checked
//! against it, which is the entire soundness argument for a cleartext residual.
//!
//! # Where the proof of work stands
//!
//! [`LigeritoParams::grinding`](super::LigeritoParams::grinding) fixes two difficulties.
//! Each of them has exactly one place in the transcript where it belongs.
//!
//! ```text
//!     level i:  round message -> GRIND challenge_bits -> fold challenge     (log_lanes times)
//!               next commitment -> GRIND query_bits -> query positions
//! ```
//!
//! The fold challenge is the one the correlated-agreement term bounds.
//! By the time it is drawn, the only thing a prover could still vary is this round's coefficients.
//! Those are already sent, so a grind there is the entire cost of asking for a second challenge.
//!
//! The query positions are drawn once the level's commitment is fixed.
//! A grind before them taxes re-rolling the positions rather than the challenge.
//! That is what lets a security target be reached with fewer rows opened.
//!
//! The two are not interchangeable, and [`Grinding`](crate::soundness::Grinding) keeps them apart.
//! The first raises a ceiling that no number of queries can touch.
//! The second only shortens the query phase.
//!
//! # Why only level 0 gets the MLE-check shortcut
//!
//! An MLE-check is a sumcheck whose weight is known to be an equality indicator, which lets the
//! verifier recover a round polynomial's missing coefficient from the evaluation point.
//! Level 0's weight is `eq(., z)`, so it qualifies.
//! Every level below it carries a glued query claim, making the weight
//!
//! ```text
//!     W = eq_scale * eq(., z) + sum_i beta_i * w_i
//! ```
//!
//! which is not an equality indicator.
//! Those levels run a plain degree-2 sumcheck over the product of the message and the weight, and
//! the equality factor each round strips has to be carried by hand.

use binius_field::{BinaryField, FieldOps};
use binius_ip::{
	mlecheck,
	sumcheck::{self, RoundCoeffs},
};
use binius_math::{
	multilinear::{evaluate::evaluate_inplace_scalars, hypercube::Hypercube},
	ntt::domain_context::GaoMateerOnTheFly,
};

use super::{InducedBasis, LigeritoParams, error::Error};
use crate::{
	channel::grinding::GrindingVerifierChannel,
	fri::batch::{BrakedownOracle, ProxTestOracle},
	merkle_channel::MerkleIPVerifierChannel,
};

/// Level 0's round polynomial is degree 1, since the MLE-check factors the equality weight out.
///
/// The proof-size estimate reads this rather than restating it, so the two cannot drift.
/// Recovering the missing coefficient is what the equality indicator buys.
pub(super) const MLECHECK_DEGREE: usize = 1;

/// Every later level's round polynomial is degree 2, the product of message and weight.
///
/// A glued weight is not an equality indicator, so nothing recovers the missing coefficient.
/// It has to be sent, and that is the extra element per fold round a deeper level pays.
pub(super) const PRODUCT_DEGREE: usize = 2;

/// Verifies a Ligerito opening against a committed ladder of Reed-Solomon codewords.
///
/// Holds the parameters and the commitment to level 0's codeword.
/// The caller receives that commitment, so it stays in charge of when the first oracle arrives.
/// Every deeper commitment arrives at a point the protocol fixes, so this reads those itself.
#[derive(Debug, Clone)]
pub struct LigeritoVerifier<'a, C> {
	/// The ladder's shape, one [`super::LigeritoLevel`] per committed level.
	params: &'a LigeritoParams,
	/// The commitment to level 0's interleaved codeword.
	commitment: C,
}

impl<'a, C: Clone> LigeritoVerifier<'a, C> {
	/// Binds a verifier to a ladder and the commitment to its outermost codeword.
	pub const fn new(params: &'a LigeritoParams, commitment: C) -> Self {
		Self { params, commitment }
	}

	/// Verifies the opening of `<X_0, eq(eval_point)> = eval_claim`.
	///
	/// Walks the ladder, and closes with the two checks the residual makes possible.
	/// The running sumcheck claim must equal the residual paired with the accumulated weight.
	/// And the last level's opened rows, batched by [`InducedBasis`], must equal that level's
	/// basis paired with the residual.
	///
	/// Both are asserted through the channel rather than compared, so a channel that builds a
	/// circuit records them as constraints.
	///
	/// The proof of work the parameters fix is checked at the two points named above.
	/// A difficulty of zero costs the transcript nothing.
	///
	/// ## Preconditions
	///
	/// * `eval_point` has `params.log_msg_len()` coordinates, in low-to-high variable order.
	pub fn verify<F, Channel>(
		&self,
		eval_point: &[Channel::Elem],
		eval_claim: Channel::Elem,
		channel: &mut Channel,
	) -> Result<(), Error>
	where
		F: BinaryField,
		Channel: MerkleIPVerifierChannel<F, Commitment = C> + GrindingVerifierChannel,
		Channel::Elem: From<F>,
	{
		let levels = self.params.levels();
		let grinding = self.params.grinding();
		assert_eq!(
			eval_point.len(),
			self.params.log_msg_len(),
			"precondition: eval_point must have one coordinate per message variable"
		);

		let mut sum = eval_claim;
		// The equality factor the plain rounds strip, which the MLE-check level would have divided
		// out for itself.
		let mut eq_scale = Channel::Elem::one();
		// The query claims glued in so far, each still weighing the message it was induced on.
		let mut glued = Vec::<(Channel::Elem, InducedBasis<Channel::Elem>)>::new();
		let mut commitment = self.commitment.clone();
		let mut residual = None;

		for (i, level) in levels.iter().enumerate() {
			// Only level 0 carries an equality indicator alone, so only it gets the shortcut.
			let is_outermost = i == 0;
			// The message this level commits spans its columns and its lanes.
			let n_vars = level.log_msg_len();
			let mut challenges = Vec::with_capacity(level.log_lanes);
			for round in 0..level.log_lanes {
				// Rounds bind the highest variable first.
				let z = eval_point[n_vars - 1 - round].clone();
				let coeffs = if is_outermost {
					mlecheck::RoundProof(RoundCoeffs(channel.recv_many(MLECHECK_DEGREE)?))
						.recover(sum, z.clone())
				} else {
					sumcheck::RoundProof(RoundCoeffs(channel.recv_many(PRODUCT_DEGREE)?))
						.recover(sum)
				};
				// The round message is out, so this is the last moment before the challenge it
				// decides. A prover asking for another one redoes the search from here.
				channel.verify_grind(grinding.challenge_bits())?;
				let challenge = channel.sample();
				sum = coeffs.evaluate(&challenge);
				// The MLE-check divides its round's equality factor out; a plain round does not.
				if !is_outermost {
					eq_scale *= Hypercube::One.eq_one_var(challenge.clone(), z);
				}
				challenges.push(challenge);
			}

			// Whatever this level folded to is bound here, before a query position exists.
			let next_commitment = match levels.get(i + 1) {
				Some(next) => Some(
					channel.recv_merkle_commitment(1 << next.log_lanes, next.log_codeword_len())?,
				),
				None => {
					let handle = channel.recv_merkle_commitment(1, level.log_msg_cols)?;
					residual = Some(channel.recv_committed_vector(&handle)?);
					None
				}
			};

			// A glued basis weighs the message this level just folded, so it follows the fold.
			for (_, basis) in &mut glued {
				*basis = basis.fold_high(&challenges);
			}

			// The commitment above is fixed, and no position exists yet. A prover that dislikes
			// the positions it is about to see pays for the next draw here.
			channel.verify_grind(grinding.query_bits())?;

			let indices = (0..level.n_queries)
				.map(|_| channel.sample_bits(level.log_codeword_len()))
				.collect::<Vec<_>>();

			// `open_queries` returns each opened coset already folded by the challenges, which is
			// the value the module documentation identifies with `<G_q, X_{i+1}>`.
			let oracle = BrakedownOracle::new(challenges, commitment.clone(), 0);
			let folded_rows = oracle.open_queries(&indices, channel)?;

			let alpha = channel.sample();
			let domain_context = GaoMateerOnTheFly::generate(level.log_codeword_len());
			let basis = InducedBasis::from_query_words(
				&domain_context,
				level.log_msg_cols,
				&indices,
				&alpha,
				channel,
			);

			match next_commitment {
				Some(next) => {
					// No message in the clear to pair against, so the row claim joins the running
					// sumcheck at a fresh challenge rather than being checked on its own.
					let beta = channel.sample();
					sum += beta.clone() * basis.enforced_sum(&folded_rows);
					glued.push((beta, basis));
					commitment = next;
				}
				None => {
					// The residual is in the clear, so the row claim is checked directly against
					// it and never enters the sumcheck.
					let residual = residual.as_ref().expect("the last level sets the residual");
					channel.assert_zero(basis.pair(residual) - basis.enforced_sum(&folded_rows))?;
				}
			}
		}

		let residual = residual.expect("levels is non-empty, so the last one sets the residual");

		// The sumcheck half: the reduced claim against the residual paired with the weight the
		// ladder accumulated, which is the equality indicator plus every glued basis.
		let mut paired = Channel::Elem::zero();
		for (beta, basis) in &glued {
			paired += beta.clone() * basis.pair(&residual);
		}
		let residual_eval =
			evaluate_inplace_scalars(residual, &eval_point[..self.params.log_residual_dim()]);
		channel.assert_zero(paired + eq_scale * residual_eval - sum)?;

		Ok(())
	}
}
