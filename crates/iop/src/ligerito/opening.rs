// Copyright 2026 The Binius Developers

//! One committed Ligerito level, on the verifier side.
//!
//! The level proves an evaluation claim `<X_0, eq(z)> = y` about the multilinear it committed.
//! `X_0` is held as `2^log_lanes` interleaved lanes of `2^log_msg_cols` columns, and each lane is
//! a Reed-Solomon codeword of the same code.
//!
//! # Why the lane fold and the sumcheck are the same challenges
//!
//! An MLE-check binds one variable per round, highest first.
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
//! # Why an opened row is a claim about `X_1`
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
//! # Why the residual goes out before the queries
//!
//! The residual is the whole of `X_1`, sent in the clear because this level is terminal.
//! Its commitment is observed before any query position is sampled.
//! A prover therefore has to fix `X_1` without knowing which rows will be checked against it,
//! which is the entire soundness argument for sending it in the clear.

use binius_field::BinaryField;
use binius_ip::{mlecheck, sumcheck::RoundCoeffs};
use binius_math::{
	multilinear::evaluate::evaluate_inplace_scalars, ntt::domain_context::GaoMateerOnTheFly,
};

use super::{InducedBasis, LigeritoLevel, error::Error};
use crate::{
	fri::batch::{BrakedownOracle, ProxTestOracle},
	merkle_channel::MerkleIPVerifierChannel,
};

/// The MLE-check round polynomial is degree 1, since the composite is the multilinear itself.
///
/// The proof-size estimate reads this rather than restating it, so the two cannot drift.
/// Recovering the missing coefficient is what the equality indicator buys.
/// A weight that is not an equality indicator would have to send it.
pub(super) const DEGREE: usize = 1;

/// Verifies one committed Ligerito level whose residual is sent in the clear.
///
/// Holds the level's shape and the commitment to its interleaved codeword.
/// The caller receives that commitment, so it stays in charge of when the level's oracle arrives.
#[derive(Debug, Clone)]
pub struct LevelVerifier<C> {
	/// The level's shape: lane count, column count, rate, and query count.
	level: LigeritoLevel,
	/// The commitment to the level's interleaved codeword.
	commitment: C,
}

impl<C: Clone> LevelVerifier<C> {
	/// Binds a verifier to a level shape and the commitment to its codeword.
	pub const fn new(level: LigeritoLevel, commitment: C) -> Self {
		Self { level, commitment }
	}

	/// Verifies the opening of `<X_0, eq(eval_point)> = eval_claim`.
	///
	/// The two checks it closes with are the two halves of the level.
	/// The MLE-check's reduced sum must equal the residual's own evaluation at the coordinates the
	/// fold left unbound.
	/// And the opened rows, batched by [`InducedBasis`], must equal that basis paired with the
	/// residual.
	///
	/// Both are asserted through the channel rather than compared, so a channel that builds a
	/// circuit records them as constraints.
	///
	/// ## Preconditions
	///
	/// * `eval_point` has `level.log_msg_len()` coordinates, in low-to-high variable order.
	pub fn verify<F, Channel>(
		&self,
		eval_point: &[Channel::Elem],
		eval_claim: Channel::Elem,
		channel: &mut Channel,
	) -> Result<(), Error>
	where
		F: BinaryField,
		Channel: MerkleIPVerifierChannel<F, Commitment = C>,
		Channel::Elem: From<F>,
	{
		let n_vars = self.level.log_msg_len();
		assert_eq!(
			eval_point.len(),
			n_vars,
			"precondition: eval_point must have one coordinate per message variable"
		);

		// MLE-check rounds. These bind the lane variables, highest first, and their challenges are
		// what the opened rows get folded by.
		let mut sum = eval_claim;
		let mut challenges = Vec::with_capacity(self.level.log_lanes);
		for round in 0..self.level.log_lanes {
			let round_proof = mlecheck::RoundProof(RoundCoeffs(channel.recv_many(DEGREE)?));
			let alpha = eval_point[n_vars - 1 - round].clone();
			let round_coeffs = round_proof.recover(sum, alpha);
			let challenge = channel.sample();
			sum = round_coeffs.evaluate(&challenge);
			challenges.push(challenge);
		}

		// The residual is bound here, before a single query position exists.
		let residual_commitment = channel.recv_merkle_commitment(1, self.level.log_msg_cols)?;
		let residual = channel.recv_committed_vector(&residual_commitment)?;

		let indices = (0..self.level.n_queries)
			.map(|_| channel.sample_bits(self.level.log_codeword_len()))
			.collect::<Vec<_>>();

		// `open_queries` returns each opened coset already folded by the challenges, which is the
		// value the module documentation identifies with `<G_q, X_1>`.
		let oracle = BrakedownOracle::new(challenges, self.commitment.clone(), 0);
		let folded_rows = oracle.open_queries(&indices, channel)?;

		let alpha = channel.sample();
		let domain_context = GaoMateerOnTheFly::generate(self.level.log_codeword_len());
		let basis = InducedBasis::from_query_words(
			&domain_context,
			self.level.log_msg_cols,
			&indices,
			&alpha,
			channel,
		);

		// The query half: the batched rows against the basis paired with the residual.
		//
		// Pairing against a message in the clear is a terminal move. A recursive level never holds
		// its residual, so it glues the induced claim into a sumcheck over the next level and
		// reaches the basis through `InducedBasis::evaluate` alone.
		channel.assert_zero(basis.pair(&residual) - basis.enforced_sum(&folded_rows))?;

		// The sumcheck half: the reduced claim against the residual's own evaluation at the
		// coordinates the lane fold left unbound.
		let residual_eval =
			evaluate_inplace_scalars(residual, &eval_point[..self.level.log_msg_cols]);
		channel.assert_zero(residual_eval - sum)?;

		Ok(())
	}
}
