// Copyright 2026 The Binius Developers

//! The Ligerito opening protocol on the prover side.
//!
//! The counterpart of [`binius_iop::ligerito::LigeritoVerifier`], which describes the protocol.
//! Three facts recorded there drive everything here.
//! The sumcheck challenges of a level are its lane fold.
//! Whatever a level folds to is committed before that level's queries are drawn.
//! And only level 0 may use the MLE-check shortcut.

use std::iter::zip;

use binius_compute::{Allocator, GlobalAllocator};
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_iop::ligerito::{InducedBasis, LigeritoLevel, LigeritoParams};
use binius_ip::mlecheck;
use binius_ip_prover::sumcheck::{
	bivariate_product_evaluator::bivariate_product_prover,
	common::{MleCheckProver, SumcheckProver},
	multilinear_eval::multilinear_eval_prover,
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec,
	inner_product::inner_product_packed,
	multilinear::{MultilinearMut, hypercube::Hypercube},
	ntt::{AdditiveNTT, domain_context::GaoMateerOnTheFly},
	reed_solomon::ReedSolomonCode,
};

use crate::{
	fri::{BrakedownOracleProver, ProxTestOracleProver},
	merkle_channel::MerkleIPProverChannel,
};

/// Encodes `message` lane by lane, Merkle-commits the result, and sends the commitment.
///
/// `message` is the level's multilinear in variable order, so its high `log_lanes` variables carry
/// the lane index and its low `log_msg_cols` variables carry the column.
/// That is already the layout [`ReedSolomonCode::encode_batch`] reads, which takes lane `lane`'s
/// columns from
///
/// ```text
///     message[(reverse_bits(lane, log_lanes) << log_msg_cols) | j]
/// ```
///
/// as [`binius_math::ntt::subspace_polys`] pins.
/// The bit reversal is what makes an opened coset fold by the sumcheck challenges in sampling
/// order, which [`binius_iop::ligerito::LigeritoVerifier`] spells out.
/// So no reshaping happens here, and a caller must not do any either.
///
/// One leaf is one codeword position across every lane, so a leaf is `2^log_lanes` scalars.
///
/// The codeword is allocated globally rather than from an arena.
/// A level's codeword outlives the call that builds it, and nothing here is on a hot path yet.
///
/// ## Preconditions
///
/// * `message` has `level.log_msg_len()` variables.
/// * `ntt`'s domain covers the level's codeword domain.
fn commit_level<F, P, NTT, Channel>(
	level: &LigeritoLevel,
	ntt: &NTT,
	message: FieldSlice<'_, P>,
	channel: &mut Channel,
) -> BrakedownOracleProver<P, Channel::Commitment>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
{
	assert_eq!(
		message.log_len(),
		level.log_msg_len(),
		"precondition: message must have one variable per message variable of the level"
	);

	let code = ReedSolomonCode::<F>::new(level.log_msg_cols, level.log_inv_rate);
	let codeword = code.encode_batch(ntt, message, level.log_lanes, &GlobalAllocator);
	let commitment = channel.send_merkle_commitment(codeword.as_view(), 1 << level.log_lanes);

	BrakedownOracleProver::new(codeword, commitment, 0)
}

/// The weight vector `level`'s opened rows induce, built over that level's codeword domain.
///
/// The prover needs every entry of it, unlike the verifier, which reaches the same weight through
/// [`InducedBasis::evaluate`] alone.
/// So this is the one place the ladder holds a query position as a number rather than a word, and
/// the only reason [`LigeritoProver::prove`] pins the channel's word type.
///
/// [`InducedBasis::to_dense`] expands one row at a time, costing `O(t * 2^log_msg_cols)`.
/// `AdditiveNTT::transpose_transform` computes the same vector as one encode, independent of `t`.
/// Which wins turns on the row count, so substituting it needs a selection rule rather than an
/// edit, and that is left to the pass that benchmarks the ladder.
fn induced_weight<F: BinaryField>(level: &LigeritoLevel, indices: &[Word], alpha: F) -> Vec<F> {
	let domain_context = GaoMateerOnTheFly::generate(level.log_codeword_len());
	let indices = indices
		.iter()
		.map(|index| index.as_u64() as usize)
		.collect::<Vec<_>>();
	InducedBasis::new(&domain_context, level.log_msg_cols, &indices, alpha).to_dense()
}

/// Proves a Ligerito opening against a committed ladder of Reed-Solomon codewords.
///
/// Holds the ladder's shape, the transform its levels encode over, and the oracle that answers
/// queries against level 0's codeword.
/// Deeper levels only exist once the folds above them have run, so [`Self::prove`] commits those.
pub struct LigeritoProver<'a, P: PackedField, C, NTT> {
	/// The ladder's shape, one [`LigeritoLevel`] per committed level.
	params: &'a LigeritoParams,
	/// The transform every level encodes over, sized for the largest of them.
	ntt: &'a NTT,
	/// Level 0's committed interleaved codeword, and the handle that opens it.
	oracle: BrakedownOracleProver<P, C>,
}

impl<'a, F, P, C, NTT> LigeritoProver<'a, P, C, NTT>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	/// Encodes and commits level 0, sending its Merkle root.
	///
	/// A single `ntt` serves the whole ladder.
	/// The levels encode over nested Gao-Mateer domains, so the largest of them covers them all.
	/// That is normally level 0, whose codeword is the longest, but a level that drops the rate
	/// without folding any lanes has a longer one still.
	///
	/// ## Preconditions
	///
	/// * `message` has `params.log_msg_len()` variables.
	/// * `ntt`'s domain covers every level's codeword domain.
	pub fn commit<Channel>(
		params: &'a LigeritoParams,
		ntt: &'a NTT,
		message: FieldSlice<'_, P>,
		channel: &mut Channel,
	) -> Self
	where
		Channel: MerkleIPProverChannel<F, Commitment = C>,
	{
		let level = &params.levels()[0];
		Self {
			params,
			ntt,
			oracle: commit_level(level, ntt, message, channel),
		}
	}

	/// Proves the opening of `<message, eq(eval_point)> = eval_claim`.
	///
	/// Per level: runs the fold rounds, commits what they folded to, and only then answers the
	/// queries the channel draws against that level's codeword.
	/// An intermediate level's query claim is glued into the running sumcheck at a fresh challenge;
	/// the last level's is left for the verifier to check against the cleartext residual.
	///
	/// ## Preconditions
	///
	/// * `message` is the multilinear [`Self::commit`] committed.
	/// * `eval_point` has `params.log_msg_len()` coordinates, in low-to-high variable order.
	pub fn prove<A, Channel>(
		&self,
		message: FieldSlice<'_, P>,
		eval_point: &[F],
		eval_claim: F,
		alloc: &A,
		channel: &mut Channel,
	) where
		A: Allocator,
		Channel: MerkleIPProverChannel<F, Commitment = C, Word = Word>,
	{
		let levels = self.params.levels();
		assert_eq!(
			message.log_len(),
			self.params.log_msg_len(),
			"precondition: message must have one variable per message variable of the ladder"
		);
		assert_eq!(
			eval_point.len(),
			self.params.log_msg_len(),
			"precondition: eval_point must have one coordinate per message variable"
		);

		// The message the current level commits, folded down the ladder into the residual.
		let mut current = FieldBuffer::from_view_in(alloc, message);
		// The running sumcheck weight, absent while the MLE-check carries it implicitly.
		let mut weight: Option<FieldVec<P, A>> = None;
		let mut sum = eval_claim;
		// Level 0's oracle lives in `self`; every deeper level's is built in this loop.
		let mut deeper: Option<BrakedownOracleProver<P, C>> = None;

		for (i, level) in levels.iter().enumerate() {
			let n_vars = level.log_msg_len();
			let witness = FieldBuffer::from_view_in(alloc, current.as_view());

			match weight.take() {
				None => {
					// Level 0's weight is the equality indicator, which the MLE-check factors out.
					// So it interpolates a degree-1 polynomial and truncates its constant term.
					let mut prover =
						multilinear_eval_prover(alloc, witness, &eval_point[..n_vars], sum);
					for _ in 0..level.log_lanes {
						let coeffs = prover.execute().pop().expect("the prover has one claim");
						channel.send_many(mlecheck::RoundProof::truncate(coeffs.clone()).coeffs());
						let challenge = channel.sample();
						prover.fold(challenge);
						// The full round polynomial at the challenge is the reduced claim, which
						// is what the verifier recovers from the truncated one.
						sum = coeffs.evaluate(&challenge);
						current.fold_highest_var(challenge);
					}
				}
				Some(mut running) => {
					// A glued weight is not an equality indicator, so the round polynomial is the
					// full degree-2 product and the truncated coefficient is the leading one.
					let mut prover = bivariate_product_prover(
						alloc,
						[witness, FieldBuffer::from_view_in(alloc, running.as_view())],
						sum,
					);
					for _ in 0..level.log_lanes {
						let coeffs = prover.execute().pop().expect("the prover has one claim");
						channel.send_many(coeffs.clone().truncate().coeffs());
						let challenge = channel.sample();
						prover.fold(challenge);
						sum = coeffs.evaluate(&challenge);
						current.fold_highest_var(challenge);
						running.fold_highest_var(challenge);
					}
					weight = Some(running);
				}
			}

			// Whatever this level folded to is bound here, before a query position exists.
			let next = match levels.get(i + 1) {
				Some(next_level) => {
					Some(commit_level(next_level, self.ntt, current.as_view(), channel))
				}
				None => {
					let commitment = channel.send_merkle_commitment(current.as_view(), 1);
					channel.send_committed_vector(&commitment, current.as_view());
					None
				}
			};

			let indices = (0..level.n_queries)
				.map(|_| channel.sample_bits(level.log_codeword_len()))
				.collect::<Vec<_>>();
			deeper
				.as_ref()
				.unwrap_or(&self.oracle)
				.open_queries(&indices, channel);

			// The row-batching challenge.
			// The last level's is drawn only to keep the two transcripts in step, since its rows
			// meet the residual in the clear rather than a weight this prover has to build.
			let alpha: F = channel.sample();

			if next.is_some() {
				let beta: F = channel.sample();
				let mut glued =
					FieldBuffer::from_values_in(alloc, &induced_weight(level, &indices, alpha));

				// The claim the opened rows make is the induced weight paired with the message
				// they folded to, which is what the verifier reads off as `enforced_sum`.
				sum += beta
					* inner_product_packed::<F, P>(
						level.log_msg_cols,
						glued.as_ref().iter().copied(),
						current.as_ref().iter().copied(),
					);

				// Level 0 leaves no running weight, having carried its equality indicator inside
				// the MLE-check, so it is materialized here at the unbound coordinates.
				let running = weight.take().unwrap_or_else(|| {
					Hypercube::One
						.expand(&eval_point[..level.log_msg_cols])
						.build_in(alloc)
				});
				let beta = P::broadcast(beta);
				for (entry, &carried) in zip(glued.as_mut(), running.as_ref()) {
					*entry = carried + beta * *entry;
				}
				weight = Some(glued);
			}
			deeper = next;
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Ghash128b as B128, Random};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		ligerito::{Error, LigeritoVerifier},
		merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
		merkle_tree::BinaryMerkleTreeScheme,
		soundness::SoundnessRegime,
	};
	use binius_math::{
		multilinear::Multilinear,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::merkle_channel::ProverMerkleTranscriptChannel;

	type StdChallenger = HasherChallenger<StdDigest>;

	/// A ladder whose level `i` commits at inverse rate `2^(i + 1)` and opens `n_queries` rows.
	///
	/// `lanes[i]` is level `i`'s fold amount, and `log_msg_cols` is level 0's column count.
	/// Level `i + 1` takes what level `i` folds to, so its columns are `cols_i - lanes_{i+1}`.
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
		LigeritoParams::new(levels, SoundnessRegime::default(), 32)
	}

	/// Commits `committed`, then proves and verifies `<opened, eq(z)> = opened(z) + claim_offset`.
	///
	/// Splitting the committed message from the opened one is how a dishonest prover is expressed
	/// here: every value it sends in the clear is consistent with `opened`, while the codeword
	/// level 0's queries land in encodes `committed`.
	/// An honest run passes the same buffer twice and a zero offset.
	///
	/// Returns the finished proof's length in bytes, so a byte count is only ever taken from a
	/// transcript that convinced the verifier.
	fn run(
		params: &LigeritoParams,
		committed: &FieldBuffer<B128>,
		opened: &FieldBuffer<B128>,
		claim_offset: B128,
	) -> Result<usize, Error> {
		let n_vars = params.log_msg_len();
		let mut rng = StdRng::seed_from_u64(n_vars as u64);
		let eval_point = (0..n_vars)
			.map(|_| B128::random(&mut rng))
			.collect::<Vec<_>>();
		let eval_claim = opened.evaluate(&eval_point) + claim_offset;

		// One transform for the whole ladder, sized for its longest codeword.
		let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(
			params.max_log_codeword_len(),
		));

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			ProverMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
				&mut prover_transcript,
			);
		let prover = LigeritoProver::commit(params, &ntt, committed.as_view(), &mut prover_channel);
		prover.prove(
			opened.as_view(),
			&eval_point,
			eval_claim,
			&GlobalAllocator,
			&mut prover_channel,
		);
		prover_channel.into_transcript();

		let proof = prover_transcript.finalize();
		let proof_size = proof.len();

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let mut verifier_channel =
			VerifierMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
				&mut verifier_transcript,
			);
		let level = &params.levels()[0];
		let commitment = verifier_channel
			.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())?;
		LigeritoVerifier::new(params, commitment).verify(
			&eval_point,
			eval_claim,
			&mut verifier_channel,
		)?;

		Ok(proof_size)
	}

	/// A random message of the ladder's shape.
	fn message(params: &LigeritoParams, seed: u64) -> FieldBuffer<B128> {
		random_field_buffer(&mut StdRng::seed_from_u64(seed), params.log_msg_len())
	}

	/// Every ladder an honest prover can produce must verify.
	///
	/// The shapes vary the depth, the fold amounts and the column count independently.
	/// One level is the terminal case, where no claim is ever glued and the whole protocol is the
	/// MLE-check plus one query round.
	/// A level folding no lanes is the degenerate case, where a level recommits at a lower rate
	/// without reducing anything.
	#[test]
	fn an_honest_opening_verifies() {
		let shapes: &[(usize, &[usize])] = &[
			(3, &[1]),
			(4, &[2]),
			(4, &[1, 1]),
			(5, &[2, 1]),
			(6, &[2, 2]),
			(6, &[1, 1, 1]),
			(6, &[2, 2, 2]),
			(7, &[3, 1, 1, 1]),
			(4, &[2, 0, 1]),
		];
		for &(log_msg_cols, lanes) in shapes {
			let params = ladder(log_msg_cols, lanes, 5);
			let msg = message(&params, 0);
			run(&params, &msg, &msg, B128::ZERO)
				.unwrap_or_else(|err| panic!("{log_msg_cols} {lanes:?}: {err}"));
		}
	}

	/// A codeword that does not encode the message the prover reduced must be caught.
	///
	/// This is the check the query round exists for, at one level.
	/// The MLE-check, the residual and the claim are all consistent with each other; only the
	/// codeword the queries open disagrees, and the terminal check pairs it against the residual.
	#[test]
	fn a_codeword_unrelated_to_the_residual_is_rejected() {
		let params = ladder(3, &[2], 5);
		let err = run(&params, &message(&params, 0), &message(&params, 1), B128::ZERO)
			.expect_err("the opened rows encode a different message than the residual");
		assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
	}

	/// The same mismatch, deep enough that the glue is what has to catch it.
	///
	/// Level 0's rows no longer meet a message in the clear.
	/// Their claim is folded into the running sumcheck at a challenge the prover cannot predict,
	/// and only the final pairing against the residual sees it.
	#[test]
	fn a_glued_row_claim_that_does_not_hold_is_rejected() {
		let params = ladder(6, &[2, 2, 2], 5);
		let err = run(&params, &message(&params, 0), &message(&params, 1), B128::ZERO)
			.expect_err("level 0's rows encode a different message than the ladder reduced");
		assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
	}
	/// A claim the message does not satisfy must be caught, at every depth.
	///
	/// This is the check the sumcheck exists for: the reduced sum no longer matches the residual
	/// paired with the accumulated weight, even though every value the prover sent is internally
	/// consistent.
	#[test]
	fn a_claim_the_message_does_not_satisfy_is_rejected() {
		for lanes in [&[2usize] as &[usize], &[2, 2], &[2, 2, 2]] {
			let params = ladder(6, lanes, 5);
			let msg = message(&params, 0);
			let err = run(&params, &msg, &msg, B128::ONE)
				.expect_err("the evaluation claim is off by one");
			assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
		}
	}

	/// The proof-size estimate must equal the transcript, not merely approximate it.
	///
	/// [`LigeritoParams::proof_size`] calls itself exact, and the ladder search minimizes it.
	/// An estimate that undercounts therefore picks the shape of a proof nobody produces.
	///
	/// The sweep moves all four dimensions of a level independently.
	/// `n_queries` is swept because it alone picks the Merkle layer the prover decommits at, and
	/// the estimate has to pick the same one.
	#[test]
	fn the_estimate_equals_the_proof_the_prover_writes() {
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();

		for log_msg_cols in 2..6 {
			for log_lanes in 0..4 {
				for log_inv_rate in 1..3 {
					for n_queries in [1, 3, 5, 12, 31] {
						let level = LigeritoLevel {
							log_msg_cols,
							log_lanes,
							log_inv_rate,
							n_queries,
						};
						// A level never opens more rows than its codeword has positions.
						if !level.is_feasible() {
							continue;
						}

						// The query count is pinned on the level, so the security target only has
						// to be one the constructor's feasibility check accepts.
						let params =
							LigeritoParams::new(vec![level], SoundnessRegime::UniqueDecoding, 8);
						let msg = message(&params, 0);
						let written = run(&params, &msg, &msg, B128::ZERO).unwrap_or_else(|err| {
							panic!("{level:?}: honest proof rejected: {err}")
						});

						assert_eq!(params.proof_size(&scheme), written, "{level:?}");
					}
				}
			}
		}
	}
	/// The estimate must stay exact once the ladder recurses.
	///
	/// Level 0 discounts its round message and no deeper level does.
	/// A one-level sweep cannot tell those two prices apart, so this walks real ladders.
	/// It is the only thing that measures [`PRODUCT_DEGREE`] rather than arguing for it.
	#[test]
	fn the_estimate_stays_exact_down_the_ladder() {
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();

		for lanes in [&[2usize] as &[usize], &[2, 2], &[1, 2, 1], &[2, 2, 2]] {
			for n_queries in [1, 5, 12] {
				let params = ladder(6, lanes, n_queries);
				let msg = message(&params, 0);
				let written = run(&params, &msg, &msg, B128::ZERO)
					.unwrap_or_else(|err| panic!("{lanes:?}: honest proof rejected: {err}"));

				assert_eq!(params.proof_size(&scheme), written, "{lanes:?} n_queries={n_queries}");
			}
		}
	}
}
